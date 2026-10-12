"""Point-in-time profitability-quality signals (Novy-Marx family).

    gross_prof     12m GrossProfit / Assets            (Novy-Marx GP/AT)
    cash_op_prof   12m OperatingCashFlow / Assets
    roic           12m OperatingIncomeLoss / (Assets - LiabilitiesCurrent)
    op_margin      12m OperatingIncomeLoss / 12m Revenue
    d_op_margin    YoY change in op_margin
    gp_stability   1 / stdev of GrossProfit/Assets across fiscal years

All six are already oriented so that HIGHER = BETTER expected return, so nothing
is negated.  gp_stability is an inverse-stdev, which is the negation implicitly.

------------------------------------------------------------------------------
WHY THIS MODULE USES ANNUAL FRAMES, NOT 4-QUARTER TTM
------------------------------------------------------------------------------
data/sec/fundamentals.md: the XBRL *frames* API returns, for each (concept,
period), the value from the MOST RECENTLY FILED accession containing that fact.
So the 2025Q2 revenue of a typical filer carries the filed_date of the 2026
10-K that repeats it as a comparative.  Gating on `filed_date <= D` (correct,
no-lookahead) therefore deletes the most recent quarters at any past D.

Measured, at D = 2025-07-31, tickers with 4 CONSECUTIVE quarterly observations
(period_end span 240-330 days):

    GrossProfit    207        annual frames  2,642
    OperatingIncome 161       annual frames  4,900
    OperatingCF       0       annual frames  6,197

OperatingCF is structurally zero: cash-flow statements are filed year-to-date,
so a discrete 3-month cash-flow period only ever exists for fiscal Q1 and the
frames panel holds nothing else (see judgement call 2 in fundamentals.md).  A
naive `groupby.tail(4)` on quarterly rows therefore sums FOUR DIFFERENT YEARS'
Q1s and calls it TTM.  That is what `signals/build_all.py::ttm()` and
`research/pit_test.py::ttm()` currently do for cfp/ocf.

So the 12-month flow here is resolved as: the CY annual frame, unless a
genuinely consecutive 4-quarter window ends LATER, in which case that window
wins.  Both are 12-month, non-overlapping quantities.

Honest limitation, stated once: because of the filed_date gate above, the
freshest annual frame visible at a past D is old -- median period_end lag is
~940 days at the 2025 test dates and ~760-1100 days at 2026-01-30.  These
signals are lookahead-free but they are STALE at historical dates, so a
backtest of them understates what the same code sees today (median lag 231 days
at 2026-08-19).  Quality ratios are highly persistent, which is the only reason
this is usable at all.  A true as-filed history needs data/sec/companyconcept.
"""
import numpy as np
import pandas as pd

Q_LO, Q_HI = 60, 115        # a "quarter" duration in days
A_LO, A_HI = 330, 400       # an "annual" duration in days
TTM_SPAN = (240, 330)       # period_end span of 4 consecutive quarters
ASSET_FLOOR = 1e6           # $1m: guard against near-zero denominators
REV_FLOOR = 1e7             # $10m of 12m revenue before a margin is meaningful
IC_FRAC = 0.05              # invested capital must be >=5% of assets
MAX_ASSET_RATIO = 10.0      # |flow/assets| above this is a tagging error
MARGIN_CLIP = (-2.0, 1.5)   # a -3,500% margin is a denominator artefact, not information
STAB_MIN_OBS = 3            # fiscal years required for a stdev
STAB_MAX_OBS = 5
STAB_MIN_LEVEL = 2e-3       # drop shells whose GP/AT is ~0 every year (stdev~0)
YOY_TOL = 75                # days of tolerance on the year-ago observation


# --------------------------------------------------------------------------
# frame extraction
# --------------------------------------------------------------------------
def _dedup(d):
    """One row per (ticker, period_end): the most recently filed restatement."""
    return (d.sort_values(["period_end", "filed_date"])
             .drop_duplicates(["ticker", "period_end"], keep="last"))


def _annual(f, concept):
    d = f[(f.concept == concept) & (~f.instantaneous) & f.dur.between(A_LO, A_HI)]
    if d.empty:
        return pd.DataFrame(columns=["ticker", "period_end", "val"])
    return _dedup(d)[["ticker", "period_end", "val"]]


def _quarterly(f, concept):
    d = f[(f.concept == concept) & (~f.instantaneous) & f.dur.between(Q_LO, Q_HI)]
    if d.empty:
        return pd.DataFrame(columns=["ticker", "period_end", "val"])
    return _dedup(d)[["ticker", "period_end", "val"]]


def _ttm_windows(f, concept):
    """Every 4-CONSECUTIVE-quarter window, not merely the last 4 rows.

    The span test is the whole point: without it a ticker that reports one
    discrete quarter per year yields a 'TTM' spanning four calendar years.
    """
    q = _quarterly(f, concept).sort_values(["ticker", "period_end"])
    if len(q) < 4:
        return pd.DataFrame(columns=["ticker", "period_end", "val"])
    g = q.groupby("ticker")
    s = g.val.rolling(4).sum().reset_index(level=0, drop=True)
    n = g.val.rolling(4).count().reset_index(level=0, drop=True)
    span = (q.period_end - g.period_end.shift(3)).dt.days
    ok = (n == 4) & span.between(*TTM_SPAN)
    return q.assign(val=s).loc[ok, ["ticker", "period_end", "val"]]


def _flow12(f, concept):
    """All 12-month flow observations per ticker, annual frames + valid TTMs.

    Returns long frame (ticker, period_end, val) sorted by period_end.  Where
    both sources describe the same period_end the annual frame wins.
    """
    a = _annual(f, concept)
    t = _ttm_windows(f, concept)
    out = pd.concat([a.assign(_p=0), t.assign(_p=1)], ignore_index=True)
    if out.empty:
        return out.drop(columns=["_p"], errors="ignore")
    out = (out.sort_values(["ticker", "period_end", "_p"])
              .drop_duplicates(["ticker", "period_end"], keep="first")
              .drop(columns="_p"))
    return out.sort_values("period_end")


def _latest(long_df):
    """Freshest 12-month observation per ticker -> DataFrame[val, period_end]."""
    if long_df.empty:
        return pd.DataFrame(columns=["val", "period_end"])
    return long_df.sort_values("period_end").groupby("ticker").last()[["val", "period_end"]]


def _inst(f, concept):
    """Long frame of instantaneous (balance-sheet) observations."""
    d = f[(f.concept == concept) & f.instantaneous]
    if d.empty:
        return pd.DataFrame(columns=["ticker", "period_end", "val"])
    return _dedup(d)[["ticker", "period_end", "val"]].sort_values("period_end")


def _at_rows(inst_long, rows, tol=120):
    """Balance-sheet value AS OF each row's period_end, positionally aligned.

    Matching the denominator to the numerator's window matters here: the flow
    can be two years stale (see module docstring) while the newest balance
    sheet is not, and GP_2023 / Assets_2026 is not a profitability ratio.
    Falls back to the ticker's latest available value when no dated match
    exists (~2% of rows).  `rows` may hold many observations per ticker.
    """
    out = np.full(len(rows), np.nan)
    if inst_long.empty or len(rows) == 0:
        return out
    r = rows[["ticker", "period_end"]].reset_index(drop=True)
    r["_i"] = np.arange(len(r))
    r = r.dropna(subset=["period_end"]).sort_values("period_end")
    if r.empty:
        return out
    m = pd.merge_asof(r, inst_long.rename(columns={"val": "x"}),
                      on="period_end", by="ticker", direction="nearest",
                      tolerance=pd.Timedelta(days=tol))
    latest = inst_long.groupby("ticker").last().val
    m["x"] = m.x.fillna(m.ticker.map(latest))
    out[m._i.values] = m.x.values
    return out


def _at(inst_long, latest_flow, tol=120):
    """Same, for a one-row-per-ticker frame; returns a Series keyed by ticker."""
    rows = pd.DataFrame({"ticker": latest_flow.index,
                         "period_end": latest_flow.period_end.values})
    return pd.Series(_at_rows(inst_long, rows, tol), index=latest_flow.index)


def _div(num, den, floor):
    """Every division in this module goes through here."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    den = den.where(den >= floor)
    return (num / den).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
def compute(f, sh, px, hist, D):
    """Profitability-quality signals as of D.  Higher = better, all six.

    `sh`, `px` and `hist` are unused: none of these ratios touch price or
    share count, which is deliberate -- it keeps the whole family clear of the
    market-cap / preferred-line / ADR trap that produced a $36,480bn Fannie Mae.
    """
    f = f.copy()
    if "dur" not in f.columns:
        f["dur"] = (f.period_end - f.period_start).dt.days

    assets_l = _inst(f, "Assets")
    lcur_l = _inst(f, "LiabilitiesCurrent")

    gp = _latest(_flow12(f, "GrossProfit"))
    ocf = _latest(_flow12(f, "NetCashProvidedByUsedInOperatingActivities"))
    oi_long = _flow12(f, "OperatingIncomeLoss")
    oi = _latest(oi_long)

    out = {}

    # ---- gross_prof: GP / Assets ---------------------------------------
    A = _at(assets_l, gp)
    v = _div(gp.val, A, ASSET_FLOOR)
    out["gross_prof"] = v.where(v.abs() <= MAX_ASSET_RATIO)

    # ---- cash_op_prof: OCF / Assets ------------------------------------
    A = _at(assets_l, ocf)
    v = _div(ocf.val, A, ASSET_FLOOR)
    out["cash_op_prof"] = v.where(v.abs() <= MAX_ASSET_RATIO)

    # ---- roic: OI / (Assets - LiabilitiesCurrent) ----------------------
    # Invested capital goes to zero (or negative) for banks, insurers and any
    # firm running a large working-capital float; an unguarded division there
    # is exactly the Athene-Holding failure mode, so require IC to be a real
    # fraction of the balance sheet rather than merely positive.
    A = _at(assets_l, oi)
    L = _at(lcur_l, oi)
    ic = (A - L).where(A.notna() & L.notna())
    ic = ic.where(ic >= (IC_FRAC * A))
    v = _div(oi.val, ic, ASSET_FLOOR)
    out["roic"] = v.where(v.abs() <= MAX_ASSET_RATIO)

    # ---- op_margin / d_op_margin ---------------------------------------
    # Numerator and denominator must cover the SAME 12-month window, so the
    # two flows are joined on period_end before either is used.
    # Revenue = MAX of the two tags, not a preference order.  General Mills
    # tags Revenues = $1.96bn (a segment line) alongside
    # RevenueFromContractWithCustomer = $20.09bn for FY2023; preferring
    # Revenues, as build_all.py does, hands GIS an 175% operating margin.
    # The two tags disagree by >1.5x on 21% of the rows that carry both.
    rev_long = pd.concat([_flow12(f, "Revenues"),
                          _flow12(f, "RevenueFromContractWithCustomerExcludingAssessedTax")],
                         ignore_index=True)
    rev_long = (rev_long.groupby(["ticker", "period_end"], as_index=False).val.max()
                if len(rev_long) else rev_long)
    m = oi_long.merge(rev_long.rename(columns={"val": "rev"}),
                      on=["ticker", "period_end"], how="inner")
    m["margin"] = _div(m.val, m.rev, REV_FLOOR).clip(*MARGIN_CLIP)
    m = m.dropna(subset=["margin"]).sort_values("period_end")

    last = m.groupby("ticker").last()
    out["op_margin"] = last.margin

    # Year-ago alignment against the ticker's OWN series -- never a calendar
    # pivot.  Off-calendar fiscal years (Jan/Jun/Sep year-ends) land in the
    # wrong calendar bucket and silently produce all-zero deltas.
    prev = m[["ticker", "period_end", "margin"]].copy()
    prev["join_end"] = prev.period_end + pd.Timedelta(days=365)
    cur = last.reset_index()[["ticker", "period_end", "margin"]].sort_values("period_end")
    j = pd.merge_asof(cur, prev[["ticker", "join_end", "margin"]]
                             .rename(columns={"margin": "margin_ly"})
                             .sort_values("join_end"),
                      left_on="period_end", right_on="join_end", by="ticker",
                      direction="nearest", tolerance=pd.Timedelta(days=YOY_TOL))
    j = j.set_index("ticker")
    out["d_op_margin"] = (j.margin - j.margin_ly).dropna()

    # ---- gp_stability ---------------------------------------------------
    # Spec asks for 8 quarters.  Not available: at the three test dates only
    # 137-328 tickers have >=6 quarterly GrossProfit observations spanning
    # 400-800 days (same filed_date-gate cause as above).  Substituted the
    # last <=5 FISCAL YEARS of GP/Assets, minimum 3.  Annual frames only, so
    # the observations cannot overlap and understate the stdev.
    gp_a = _annual(f, "GrossProfit").sort_values("period_end")
    if len(gp_a):
        gp_a = gp_a.assign(_y=gp_a.period_end.dt.year).drop_duplicates(
            ["ticker", "_y"], keep="last")
        ga = _at_rows(assets_l, gp_a)
        gp_a["ratio"] = _div(pd.Series(gp_a.val.values),
                             pd.Series(ga), ASSET_FLOOR).values
        gp_a = gp_a[gp_a.ratio.abs() <= MAX_ASSET_RATIO].dropna(subset=["ratio"])
        w = gp_a.sort_values("period_end").groupby("ticker").tail(STAB_MAX_OBS)
        st = w.groupby("ticker").ratio.agg(["std", "count", "mean"])
        st = st[(st["count"] >= STAB_MIN_OBS) & (st["mean"].abs() >= STAB_MIN_LEVEL)]
        # 1/(sd + eps): rank-identical to -sd for sd>0, and the eps stops a
        # dormant shell with an identical GP every year scoring +inf.
        out["gp_stability"] = 1.0 / (st["std"] + 1e-4)

    res = pd.DataFrame(out)
    res.index.name = "ticker"
    return res.replace([np.inf, -np.inf], np.nan).dropna(how="all")
