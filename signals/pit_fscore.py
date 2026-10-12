"""Point-in-time Piotroski F-score (Piotroski 2000, JAR 38, "Value Investing:
The Use of Historical Financial Statement Information to Separate Winners from
Losers").

Nine binary signals of fundamental health, each 1 = good, summed to 0-9.
Higher is better in the original paper (high-F value stocks beat low-F value
stocks), so NOTHING here is negated -- f_score and every f1..f9 already point
the right way.

Construction, and why each choice was made:

  * ANNUAL, not TTM.  The measure is defined fiscal-year vs fiscal-year, and in
    this panel it has to be: NetCashProvidedByUsedInOperatingActivities is filed
    year-to-date, so only ~Q1 rows survive a dur.between(60,115) filter (38,945
    rows of which 32,378 are Q1).  A 4-quarter TTM sum of operating cash flow is
    therefore not buildable for most names, while annual OCF covers 6,897.
  * Every comparison is made inside ONE ticker's own series: fiscal year t
    against the annual row 300-430 days earlier.  No calendar-quarter pivot, so
    off-calendar fiscal years are handled correctly.
  * Restatements: for a given (ticker, period_end) the most recently FILED row
    wins.  Since `f` is pre-filtered to filed_date <= D that is exactly the
    "as known on D" view, and year t-1 figures usually arrive from the year-t
    10-K, which is what Piotroski compares against.
  * Balance-sheet items are matched to the fiscal-year-end date itself
    (+/-20 days), and beginning-of-year assets to fy_end minus one year
    (+/-20 days), because Piotroski scales by beginning-of-year total assets.
  * Staleness guard: the latest fiscal year must end within 550 days of D.
    At D = 2026-01-30 a December filer's newest 10-K is still FY2024 (396 days
    old), so the cutoff cannot be tighter; 550 days keeps delinquent filers out.

Deviations from the literal Compustat recipe, all forced by the concept panel:

  * ebit/sale (the OSAP variant of component 8) is not buildable -- no interest
    expense concept.  Component 8 is the ORIGINAL Piotroski gross margin,
    GrossProfit/Revenue, with OperatingIncomeLoss/Revenue as an explicit
    fallback when GrossProfit is absent in either year.  The same source is used
    for both years; the two are never mixed inside one comparison.
  * dltt (component 5) is LongTermDebtNoncurrent where tagged (2,557 tickers),
    otherwise non-current liabilities, Liabilities - LiabilitiesCurrent
    (6,000+ tickers).  Again one source for both years, never mixed.
  * Share issuance (component 7) uses fiscal-year-end share counts.  Stock
    splits are the trap: NVDA's dei cover-page count goes 2.46e9 -> 2.45e10 in
    2024 and would read as a 10x equity offering.  Preference order is
    CommonStockSharesOutstanding then CommonStockSharesIssued from `sh` (SEC
    frames, restated, hence split-consistent) and only then the dei cover-page
    tag from `f`, which is as-filed and therefore split-contaminated.  The dei
    series is de-split by scanning each ticker's own consecutive observations
    for jumps within 3% of a round split factor and dividing them out.

Outputs (all higher = better):
    f1  net income > 0
    f2  operating cash flow > 0
    f3  return on assets rose
    f4  operating cash flow > net income (accruals)
    f5  long-term-debt leverage fell (or stayed flat)
    f6  current ratio rose
    f7  shares outstanding did not rise
    f8  gross margin rose
    f9  asset turnover rose
    f_score        sum of f1..f9, NaN unless all nine are present (Piotroski
                   excludes firms missing any input)
    f_score_avail  same information rescaled onto 0-9 using whichever components
                   exist, requiring at least 6 of 9.  Roughly doubles coverage;
                   the strict score is the faithful one.
"""
import numpy as np
import pandas as pd

ANNUAL_MIN, ANNUAL_MAX = 300, 430      # fiscal-year duration window (52/53-week years incl.)
YEAR_MIN, YEAR_MAX = 300, 430          # allowed gap between consecutive fiscal-year ends
BS_TOL = pd.Timedelta(days=20)         # balance-sheet date vs fiscal-year end
DEI_TOL = pd.Timedelta(days=110)       # cover-page date after fiscal-year end
MAX_STALE = pd.Timedelta(days=550)     # newest fiscal year must be at most this old
MIN_AVAIL = 6                          # components required for f_score_avail

# round split factors, used only to de-split the as-filed dei share count
SPLIT_FACTORS = np.array([1.5, 2., 2.5, 3., 4., 5., 6., 7., 8., 10., 12.,
                          15., 20., 25., 30., 40., 50., 100.])
SPLIT_TOL = 0.03


# ---------------------------------------------------------------- helpers ---
def _dedup(d):
    """One row per (ticker, period_end): the most recently filed version wins."""
    key = ["ticker", "period_end"]
    srt = key + (["filed_date"] if "filed_date" in d.columns else [])
    return d.sort_values(srt).drop_duplicates(key, keep="last")


def _annual(f, concept, name):
    """Annual duration rows for one concept, one row per ticker/fiscal-year end."""
    d = f[(f.concept == concept) & (~f.instantaneous) &
          f.dur.between(ANNUAL_MIN, ANNUAL_MAX)]
    if len(d) == 0:
        return pd.DataFrame({"ticker": [], "period_end": [], name: []})
    return _dedup(d)[["ticker", "period_end", "val"]].rename(columns={"val": name})


def _inst(frame, concept):
    """Instantaneous (balance-sheet / cover-page) rows, deduped, sorted by date."""
    d = frame[(frame.concept == concept) & frame.instantaneous]
    if len(d) == 0:
        return pd.DataFrame({"ticker": [], "period_end": [], "val": []})
    return _dedup(d)[["ticker", "period_end", "val"]].sort_values("period_end")


def _asof(left, right, name, on="period_end", tol=BS_TOL, direction="nearest"):
    """Attach `right` (ticker, period_end, val) to `left` at left[on] +/- tol."""
    if len(right) == 0:
        left[name] = np.nan
        return left
    l = left.sort_values(on)
    r = right.rename(columns={"period_end": on, "val": name}).sort_values(on)
    out = pd.merge_asof(l, r[["ticker", on, name]], on=on, by="ticker",
                        direction=direction, tolerance=tol)
    return out


def _desplit_dei(dei):
    """Split-adjust the as-filed dei share count inside each ticker's own series.

    A stock split shows up as one consecutive-observation ratio sitting within
    3% of a round factor (2x, 10x, or a reverse 1/10).  Divide those out
    cumulatively; genuine issuance and buybacks are untouched.
    """
    d = dei.sort_values(["ticker", "period_end"]).copy()
    prev = d.groupby("ticker").val.shift()
    r = (d.val / prev).replace([np.inf, -np.inf], np.nan)

    fac = pd.Series(1.0, index=d.index)
    up = np.abs(r.values[:, None] / SPLIT_FACTORS[None, :] - 1.0) <= SPLIT_TOL
    dn = np.abs(r.values[:, None] * SPLIT_FACTORS[None, :] - 1.0) <= SPLIT_TOL
    hit_up, hit_dn = up.any(1), dn.any(1)
    fac[hit_up] = SPLIT_FACTORS[up[hit_up].argmax(1)]
    fac[hit_dn] = 1.0 / SPLIT_FACTORS[dn[hit_dn].argmax(1)]
    fac[r.isna().values] = 1.0

    d["cum"] = fac.groupby(d.ticker).cumprod()
    d["val"] = d.val / d["cum"]                       # common (latest) basis
    return d[["ticker", "period_end", "val"]]


def _gt(a, b):
    """Strict >, NaN-preserving (NaN in either operand -> NaN, not False)."""
    out = (a > b).astype(float)
    return out.where(a.notna() & b.notna())


def _le(a, b):
    out = (a <= b).astype(float)
    return out.where(a.notna() & b.notna())


def _pos(s):
    """Positive-denominator mask: strictly > 0 and finite."""
    return s.notna() & (s > 0)


def _safe(num, den):
    """num/den, NaN wherever den is not strictly positive (rule 3: no blow-ups)."""
    return (num / den.where(_pos(den))).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- compute ---
def compute(f, sh, px, hist, D):
    f = f.copy()
    if "dur" not in f.columns:
        f["dur"] = (pd.to_datetime(f.period_end) - pd.to_datetime(f.period_start)).dt.days

    # ---- annual flows, one row per ticker / fiscal-year end -----------------
    parts = [_annual(f, "NetIncomeLoss", "ni"),
             _annual(f, "NetCashProvidedByUsedInOperatingActivities", "ocf"),
             _annual(f, "Revenues", "rev_a"),
             _annual(f, "RevenueFromContractWithCustomerExcludingAssessedTax", "rev_b"),
             _annual(f, "GrossProfit", "gp"),
             _annual(f, "OperatingIncomeLoss", "oi")]
    T = parts[0]
    for p in parts[1:]:
        T = T.merge(p, on=["ticker", "period_end"], how="outer")
    T = T[T.ni.notna() | T.ocf.notna()]
    if len(T) == 0:
        return pd.DataFrame()

    # only the two newest fiscal years per ticker matter; drop anything stale
    T = T[T.period_end <= D]
    T = T.sort_values(["ticker", "period_end"]).groupby("ticker").tail(3)

    # ---- balance sheet at the fiscal-year end, and one year before it -------
    T["beg_key"] = T.period_end - pd.Timedelta(days=365)
    for cpt, col in [("Assets", "at"), ("AssetsCurrent", "act"),
                     ("LiabilitiesCurrent", "lct"),
                     ("LongTermDebtNoncurrent", "ltd_a"),
                     ("Liabilities", "liab")]:
        T = _asof(T, _inst(f, cpt), col)
    T = _asof(T, _inst(f, "Assets"), "at_beg", on="beg_key")

    # non-current liabilities as the long-term-debt fallback
    ncl = T.liab - T.lct
    T["ltd_b"] = ncl.where(ncl >= 0)

    # ---- fiscal-year-end share counts, split-consistent where possible -----
    shares = []
    for src in ["CommonStockSharesOutstanding", "CommonStockSharesIssued"]:
        shares.append(_inst(sh, src))
    shares.append(_desplit_dei(_inst(f, "EntityCommonStockSharesOutstanding")))
    for i, s in enumerate(shares[:2]):
        T = _asof(T, s, f"sh{i}")
    T = _asof(T, shares[2], "sh2", tol=DEI_TOL, direction="forward")

    # ---- pair each fiscal year with the previous one -----------------------
    T = T.sort_values(["ticker", "period_end"])
    g = T.groupby("ticker")
    P = g.shift(1)                                     # previous fiscal year
    T["gap"] = (T.period_end - P.period_end).dt.days
    ok = T.gap.between(YEAR_MIN, YEAR_MAX)
    T = T[ok]
    P = P[ok]
    T = T.groupby("ticker").tail(1)                    # newest usable pair
    P = P.loc[T.index]
    T = T[T.period_end >= D - MAX_STALE]
    P = P.loc[T.index]
    if len(T) == 0:
        return pd.DataFrame()
    T = T.set_index("ticker")
    P.index = T.index

    # ---- pick one revenue / margin / leverage source, same in both years ---
    rev_t = T.rev_a.where(_pos(T.rev_a) & _pos(P.rev_a), T.rev_b)
    rev_p = P.rev_a.where(_pos(T.rev_a) & _pos(P.rev_a), P.rev_b)
    rev_t = rev_t.where(_pos(rev_t) & _pos(rev_p))
    rev_p = rev_p.where(rev_t.notna())

    both_gp = T.gp.notna() & P.gp.notna()
    marg_t = T.gp.where(both_gp, T.oi)
    marg_p = P.gp.where(both_gp, P.oi)

    both_ltd = T.ltd_a.notna() & P.ltd_a.notna()
    ltd_t = T.ltd_a.where(both_ltd, T.ltd_b)
    ltd_p = P.ltd_a.where(both_ltd, P.ltd_b)

    shares_t = shares_p = None
    for c in ["sh0", "sh1", "sh2"]:
        st, sp = T[c].where(_pos(T[c]) & _pos(P[c])), P[c].where(_pos(T[c]) & _pos(P[c]))
        shares_t = st if shares_t is None else shares_t.combine_first(st)
        shares_p = sp if shares_p is None else shares_p.combine_first(sp)

    # ---- the nine components ----------------------------------------------
    roa_t = _safe(T.ni, T.at_beg)                       # beginning-of-year assets
    roa_p = _safe(P.ni, P.at_beg)
    avg_at_t = ((T["at"] + T.at_beg) / 2).where(_pos(T["at"]) & _pos(T.at_beg))
    avg_at_p = ((P["at"] + P.at_beg) / 2).where(_pos(P["at"]) & _pos(P.at_beg))
    lev_t = _safe(ltd_t, avg_at_t)
    lev_p = _safe(ltd_p, avg_at_p)
    cr_t = _safe(T.act, T.lct)
    cr_p = _safe(P.act, P.lct)
    gm_t = _safe(marg_t, rev_t)
    gm_p = _safe(marg_p, rev_p)
    turn_t = _safe(rev_t, T.at_beg)
    turn_p = _safe(rev_p, P.at_beg)

    out = pd.DataFrame(index=T.index)
    out["f1"] = (T.ni > 0).astype(float).where(T.ni.notna())
    out["f2"] = (T.ocf > 0).astype(float).where(T.ocf.notna())
    out["f3"] = _gt(roa_t, roa_p)
    out["f4"] = _gt(T.ocf, T.ni)                        # denominators cancel
    out["f5"] = _le(lev_t, lev_p)
    out["f6"] = _gt(cr_t, cr_p)
    out["f7"] = _le(shares_t, shares_p)
    out["f8"] = _gt(gm_t, gm_p)
    out["f9"] = _gt(turn_t, turn_p)

    comps = [f"f{i}" for i in range(1, 10)]
    n = out[comps].notna().sum(axis=1)
    out["f_score"] = out[comps].sum(axis=1).where(n == 9)
    out["f_score_avail"] = (out[comps].mean(axis=1) * 9).where(n >= MIN_AVAIL)

    # only names with a live, sane price (px/hist are pre-filtered to >= $1)
    out = out.loc[out.index.intersection(px.dropna().index)]
    return out.replace([np.inf, -np.inf], np.nan)
