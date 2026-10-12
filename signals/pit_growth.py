"""Investment & asset-growth signals, point-in-time.

Family: the "real investment" anomalies. Firms that expand their balance sheet
aggressively subsequently underperform (Cooper-Gulen-Schill 2008 asset growth,
Hirshleifer-Hou-Teoh-Zhang 2004 net operating assets, Titman-Wei-Xie 2004
investment). All five signals below are returned already sign-flipped so that
HIGHER = BETTER EXPECTED RETURN, per the module contract.

  asset_growth     -(A_t/A_{t-1y} - 1)                          low growth wins
  capex_intensity  -((A_t - A_{t-1y} + DDA_ttm) / A_{t-1y})     low investment wins
  noa              -(NOA_t / A_{t-1y})                          low operating assets win
  d_noa            -((NOA_t - NOA_{t-1y}) / A_{t-1y})           low NOA growth wins
  sales_growth      see SALES_GROWTH_SIGN below (measured, not assumed)

  NOA = (Assets - Cash) - (Liabilities - LongTermDebtNoncurrent)
        with Cash and LongTermDebtNoncurrent filled to 0 when the tag is absent,
        exactly as Hirshleifer et al. specify ("all items except at and ceq
        replaced with 0 if missing").

CONCEPTS THE PANEL DOES NOT HAVE (documented, not silently proxied)
  PaymentsToAcquirePropertyPlantAndEquipment (capx)  -- the real capex line
  PropertyPlantAndEquipmentNet / Gross (ppent/ppegt)
  InventoryNet (invt), AccountsReceivableNetCurrent (rect), AccountsPayable (ap)
  ShortTermInvestments (che ex-cash), MinorityInterest (mib), PreferredStock (pstk)
Consequences, stated plainly:
  * capex_intensity is a PROXY, not capex. Gross investment is approximated by
    the change in the whole balance sheet plus depreciation added back
    (dA + DDA), because dPPE_net = capx - dep is unavailable. It therefore
    picks up working-capital and acquisition growth as well as real capex, and
    is ~0.9 correlated with asset_growth. It is a variant of asset growth, not
    an independent read on capex. InvestPPEInv, grcapx and GrLTNOA cannot be
    built at all on this panel.
  * NOA omits minority interest, deferred charges and preferred stock, and
    "cash" is CashAndCashEquivalents only -- short-term investments are not
    tagged, so cash-rich firms holding T-bills look more operating-asset heavy
    than they are.

POINT-IN-TIME DISCIPLINE
  f and sh arrive pre-filtered to filed_date <= D, so BOTH legs of every
  year-over-year change are automatically already-filed. On top of that:
  * one row per (ticker, period_end, concept), keeping the LAST-FILED value,
    so a 10-K restatement supersedes the original 10-Q without lookahead
    (both were filed <= D by construction).
  * the lag is matched inside each ticker's OWN period_end series at ~365 days
    with a 45-day tolerance -- never a calendar-quarter pivot, which silently
    misaligns off-calendar fiscal years.
  * balance-sheet components are aligned on a single (ticker, period_end) grid
    so NOA_t and NOA_{t-1y} are each built from one consistent balance sheet.
  * STALE_DAYS drops names whose most recent balance sheet is older than
    ~14 months at D (defunct filers still sitting in the panel).

GUARDS (rule 3: near-zero denominators score a perfect rank)
  every ratio divides by LAGGED TOTAL ASSETS, floored at MIN_DENOM ($1m), and
  the result is clipped to economically sane bounds before it is returned.
  No market cap is computed anywhere in this family, so the ADR / preferred
  share-count trap does not apply here.
"""
import numpy as np
import pandas as pd

# ---- tuning constants, all in one place ---------------------------------
LAG_DAYS      = 365      # target lag for the year-ago observation
LAG_TOL       = 45       # +/- tolerance; quarters sit 91 days apart so this cannot
                         # accidentally match an adjacent quarter
STALE_DAYS    = 430      # drop names whose newest balance sheet predates D by more
MIN_DENOM     = 1e6      # $1m floor on lagged assets / prior-year revenue
CLIP_GROWTH   = (-0.90, 5.0)   # ratio-change bounds
CLIP_LEVEL    = (-2.0,  3.0)   # NOA / assets bounds (net-cash firms go negative)
CLIP_DELTA    = (-2.0,  2.0)   # change-in-NOA / assets bounds

# Sign for sales_growth. MEASURED on this system's own point-in-time panel at
# 2025-01-31 / 2025-07-31 / 2026-01-30 with a 63-trading-day forward return and
# sector-relative quintiles: HIGH TTM sales growth earned the HIGHER forward
# return at all three dates (Q5-Q1 about +5.5% raw, monotone 4/4 on the pooled
# mean). That is the OPPOSITE of Lakonishok-Shleifer-Vishny glamour, and it is
# a 3-date, one-regime, survivor-biased sample -- see the report. So we ship
# +1 (high growth = better) but flag it as the one sign in this family that is
# fitted to data rather than imposed by the literature.
SALES_GROWTH_SIGN = +1

_BS_CONCEPTS = ["Assets", "AssetsCurrent", "CashAndCashEquivalentsAtCarryingValue",
                "Liabilities", "LongTermDebtNoncurrent"]
_REV_CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]


# ------------------------------------------------------------------ helpers
def _dedup(d):
    """One row per (ticker, period_end, concept): the last-filed value wins.

    Every row here was filed <= D already, so preferring the later filing is a
    restatement, not lookahead.
    """
    return (d.sort_values(["filed_date", "accn"])
             .drop_duplicates(["ticker", "period_end", "concept"], keep="last"))


def _balance_panel(f):
    """(ticker, period_end) grid with the four balance-sheet columns aligned."""
    d = f[f.instantaneous & f.concept.isin(_BS_CONCEPTS)]
    d = d[d.val.notna() & d.period_end.notna()]
    d = _dedup(d[["ticker", "period_end", "concept", "val", "filed_date", "accn"]])
    p = (d.pivot_table(index=["ticker", "period_end"], columns="concept",
                       values="val", aggfunc="last")
           .reset_index())
    for c in _BS_CONCEPTS:                       # a concept can be wholly absent
        if c not in p.columns:
            p[c] = np.nan
    return p.sort_values("period_end")


def _quarterly(f, concept):
    """Standalone-quarter duration rows (dur 60-115) for one concept."""
    d = f[(f.concept == concept) & (~f.instantaneous) & f.dur.between(60, 115)]
    d = d[d.val.notna() & d.period_end.notna()]
    d = _dedup(d[["ticker", "period_end", "concept", "val", "filed_date", "accn"]])
    return d[["ticker", "period_end", "val"]].sort_values("period_end")


def _annual(f, concept):
    """Full-fiscal-year duration rows (dur 300-400) for one concept."""
    d = f[(f.concept == concept) & (~f.instantaneous) & f.dur.between(300, 400)]
    d = d[d.val.notna() & d.period_end.notna() & d.period_start.notna()]
    d = _dedup(d[["ticker", "period_end", "period_start", "concept",
                  "val", "filed_date", "accn"]])
    return d[["ticker", "period_start", "period_end", "val"]].sort_values("period_end")


def _qseries(f, concept):
    """Quarterly series with the MISSING FOURTH QUARTER reconstructed.

    This is the single largest correctness fix in the module. Almost no US
    filer tags a standalone Q4: the 10-K carries the full-year duration and the
    10-Qs carry Q1-Q3, so a naive quarterly series has one hole per fiscal year.
    Measured on this panel, the last 8 quarterly revenue rows span a MEDIAN OF
    913 days -- ten quarters, not eight. Summing them as a "TTM" silently
    splices 2023 revenue onto 2025 revenue; rejecting them on span left 65
    tickers with a sales-growth number out of ~5,500 with revenue tags.

    Fix: for every annual row, sum the quarterly rows that fall strictly inside
    its own fiscal window; if exactly three are present, Q4 = annual - those
    three, dated at the fiscal year end. Real quarterly rows always win over a
    synthesised one.
    """
    q = _quarterly(f, concept)
    a = _annual(f, concept)
    if q.empty or a.empty:
        return q
    j = a.merge(q.rename(columns={"period_end": "q_end", "val": "q_val"}),
                on="ticker", how="inner")
    j = j[(j.q_end > j.period_start) & (j.q_end < j.period_end - pd.Timedelta(days=30))]
    g = j.groupby(["ticker", "period_end"]).agg(q_sum=("q_val", "sum"),
                                                n=("q_val", "size")).reset_index()
    g = g[g.n == 3]
    syn = a.merge(g, on=["ticker", "period_end"], how="inner")
    syn["val"] = syn.val - syn.q_sum
    syn = syn[["ticker", "period_end", "val"]]
    both = pd.concat([q.assign(_real=1), syn.assign(_real=0)], ignore_index=True)
    both = (both.sort_values(["_real"])
                .drop_duplicates(["ticker", "period_end"], keep="last")   # real wins
                .drop(columns="_real"))
    return both.sort_values("period_end")


def _matched_growth(q, min_pairs=2, window=372):
    """Like-for-like YoY growth from a quarterly series, tolerant of a hole.

    A strict trailing-8-quarter TTM cannot be used on this panel. Fiscal
    YEAR-END facts here carry the filing date of the NEXT year's report (the
    comparative column overwrote the original accession), so at any historic D
    the most recent FY-end quarter is simply absent: the last 8 rows span 731+
    days and a span check rejects almost everyone. Measured at 2025-07-31, the
    strict version returned 65 tickers.

    Instead: take the up-to-4 most recent quarters inside one year of the
    ticker's own newest period_end, match EACH of them to the same fiscal
    quarter a year earlier (merge_asof, +/-LAG_TOL days, within the ticker's
    own series), and require at least `min_pairs` matched quarters. Growth is
    then sum(current) / sum(matched prior) - 1: identical calendar quarters on
    both sides, so seasonality cancels the way a TTM would, and one missing Q4
    costs precision rather than the whole observation.
    """
    if q.empty:
        return pd.DataFrame(columns=["cur", "prv", "period_end", "n"])
    q = q.sort_values("period_end")
    anchor = q.groupby("ticker").period_end.max()
    cur = q[q.period_end > q.ticker.map(anchor) - pd.Timedelta(days=window)]
    prev = q.copy()
    prev["join_end"] = prev.period_end + pd.Timedelta(days=LAG_DAYS)
    prev = (prev[["ticker", "join_end", "val"]].rename(columns={"val": "val_ly"})
                .sort_values("join_end"))
    m = pd.merge_asof(cur.sort_values("period_end"), prev, left_on="period_end",
                      right_on="join_end", by="ticker", direction="nearest",
                      tolerance=pd.Timedelta(days=LAG_TOL))
    m = m[m.val_ly.notna()]
    if m.empty:
        return pd.DataFrame(columns=["cur", "prv", "period_end", "n"])
    g = m.groupby("ticker")
    out = pd.DataFrame({"cur": g.val.sum(), "prv": g.val_ly.sum(),
                        "period_end": g.period_end.max(), "n": g.size()})
    return out[out.n >= min_pairs]


def _annualized(q, a, min_q=3, window=372):
    """Annual-rate flow: the quarters inside the last year, grossed up to four.

    Same reason as above -- demanding a clean 4-quarter TTM loses every filer
    whose Q4 is only inside the (late-dated) 10-K. Three quarters scaled by 4/3
    is a far smaller error than dropping the name. Falls back to the annual
    duration row when no usable quarters exist.
    """
    out = pd.Series(dtype=float)
    if not q.empty:
        q = q.sort_values("period_end")
        anchor = q.groupby("ticker").period_end.max()
        w = q[q.period_end > q.ticker.map(anchor) - pd.Timedelta(days=window)]
        g = w.groupby("ticker")
        n, tot = g.size(), g.val.sum()
        n = n[n.between(min_q, 4)]
        out = (tot.reindex(n.index) * 4.0 / n)
    if not a.empty:
        out = out.combine_first(a.sort_values("period_end")
                                 .groupby("ticker").val.last())
    return out


def _yoy(panel, cols, on="period_end"):
    """Match each observation to the one ~LAG_DAYS earlier in the SAME ticker.

    merge_asof inside the ticker's own series -- the calendar-quarter pivot this
    replaces produced 553 names of all-zero data for off-calendar fiscal years.
    """
    cur = panel.sort_values(on)
    prev = panel.copy()
    prev["join_end"] = prev[on] + pd.Timedelta(days=LAG_DAYS)
    prev = prev[["ticker", "join_end"] + cols].rename(
        columns={c: c + "_ly" for c in cols}).sort_values("join_end")
    m = pd.merge_asof(cur, prev, left_on=on, right_on="join_end", by="ticker",
                      direction="nearest", tolerance=pd.Timedelta(days=LAG_TOL))
    return m


def _safe_div(num, den, floor=MIN_DENOM):
    """Division with a hard floor on the denominator (rule 3)."""
    den = den.where(den.abs() >= floor)
    return (num / den).replace([np.inf, -np.inf], np.nan)


# -------------------------------------------------------------------- main
def compute(f, sh, px, hist, D):
    """See module docstring. `sh` and `hist` are unused by this family: every
    signal is scaled by lagged total assets, so no share count and no price
    path is required (and therefore no market cap to get wrong)."""
    D = pd.Timestamp(D)

    # ---------------- balance sheet, current vs one year earlier ----------
    bp = _balance_panel(f)
    bp = bp[bp.Assets.notna() & (bp.Assets > 0)]
    m = _yoy(bp, _BS_CONCEPTS)
    m = m[m.Assets_ly.notna() & (m.Assets_ly > 0)]
    m = m.sort_values("period_end").groupby("ticker").last()      # newest matched pair
    m = m[m.index.notna()]
    m = m[(D - m.period_end).dt.days <= STALE_DAYS]               # no zombie filers

    A, A_ly = m.Assets, m.Assets_ly
    cash    = m.CashAndCashEquivalentsAtCarryingValue.fillna(0.0)
    cash_ly = m.CashAndCashEquivalentsAtCarryingValue_ly.fillna(0.0)
    ltd     = m.LongTermDebtNoncurrent.fillna(0.0)                # missing debt -> 0
    ltd_ly  = m.LongTermDebtNoncurrent_ly.fillna(0.0)
    L, L_ly = m.Liabilities, m.Liabilities_ly

    # 1) asset growth -- Cooper, Gulen & Schill (2008). Sign: -1, literature.
    asset_growth = _safe_div(A - A_ly, A_ly).clip(*CLIP_GROWTH)

    # 2) capex intensity -- gross-investment proxy, see docstring caveat.
    #    Built on NON-CURRENT assets (Assets - AssetsCurrent ~ PP&E + intangibles
    #    + long-term investments), not on the whole balance sheet: for a
    #    long-lived asset dLTA = capx - depreciation, so dLTA + DDA is a real
    #    gross-investment estimate. Doing it on TOTAL assets instead -- which is
    #    the literal reading of "depreciation-adjusted asset change" -- gives a
    #    series that is 0.99-1.00 correlated with asset_growth and therefore
    #    carries no independent information; measured, not assumed.
    dda = _annualized(_qseries(f, "DepreciationDepletionAndAmortization"),
                      _annual(f, "DepreciationDepletionAndAmortization"))
    dda = dda.reindex(m.index)
    lta    = A - m.AssetsCurrent                                   # NaN for banks etc.
    lta_ly = A_ly - m.AssetsCurrent_ly
    capex_proxy = (lta - lta_ly) + dda            # NaN unless BOTH DDA and current assets
    capex_intensity = _safe_div(capex_proxy, A_ly).clip(*CLIP_GROWTH)

    # 3/4) net operating assets -- Hirshleifer, Hou, Teoh & Zhang (2004).
    noa_t  = (A - cash) - (L - ltd)
    noa_ly = (A_ly - cash_ly) - (L_ly - ltd_ly)
    noa_t  = noa_t.where(L.notna())                                # Liabilities required
    noa_ly = noa_ly.where(L_ly.notna())
    noa   = _safe_div(noa_t, A_ly).clip(*CLIP_LEVEL)
    d_noa = _safe_div(noa_t - noa_ly, A_ly).clip(*CLIP_DELTA)

    # 5) TTM sales growth. Per-ticker the revenue TAG is fixed (Revenues where
    #    the firm reports it, else the ASC 606 contract-revenue tag) so the two
    #    TTM windows are never built from different definitions.
    rev = None
    for c in _REV_CONCEPTS:                       # Revenues first, ASC 606 tag second
        p = _matched_growth(_qseries(f, c))       # Q4 reconstructed where possible
        rev = p if rev is None else rev.combine_first(p)
    if rev is None or rev.empty:
        sales_growth = pd.Series(dtype=float)
    else:
        rev = rev[(D - rev.period_end).dt.days <= STALE_DAYS]
        sales_growth = _safe_div(rev.cur - rev.prv,
                                 rev.prv.where(rev.prv > 0)).clip(*CLIP_GROWTH)

    out = pd.DataFrame({
        "asset_growth":    -asset_growth,      # negate: high growth -> low returns
        "capex_intensity": -capex_intensity,   # negate: high investment -> low returns
        "noa":             -noa,               # negate: high NOA -> low returns
        "d_noa":           -d_noa,             # negate: NOA growth -> low returns
        "sales_growth":     SALES_GROWTH_SIGN * sales_growth,
    })
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(how="all")
    out.index.name = "ticker"

    # keep only names we can actually trade at D (rule 4: sub-penny junk out)
    if px is not None and len(px):
        live = px[px >= 1.0].index
        out = out.loc[out.index.intersection(live)]
    return out.sort_index()
