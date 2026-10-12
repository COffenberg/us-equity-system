"""Accounting red flags — point-in-time, fiscal-year aligned.

Five deltas/ratios that say whether reported earnings are backed by cash and by
tax paid.  Every column is oriented HIGHER = BETTER EXPECTED RETURN.

    accrual_ratio    -(NI - OCF) / avg assets            (Sloan 1996; OSAP sign -1)
    d_accrual        -(YoY change in that ratio)
    tax_book_gap     -(book pretax - tax-implied pretax) / avg assets
    d_tax_book_gap   -(YoY change in that gap)           ("widening" reading)
    cash_conv        OCF / NI, profitable names only
    d_nwc            -(YoY change in net working capital / assets)  (OSAP ChNWC)

WHAT IS NOT HERE — dso_change
-----------------------------
Days sales outstanding needs a receivables balance.  The panel's 22 concepts do
NOT include AccountsReceivableNetCurrent (or any receivables tag), and neither
does shares_fallback.parquet.  DSO is therefore not computable from this data
and no column claims to be it.  `d_nwc` is included instead as a SEPARATE,
independently published signal (OSAP ChNWC) — it is a whole-working-capital
measure, not a DSO proxy, and it is missing OSAP's `dlc` (debt in current
liabilities) term because that concept is also absent from the panel.

TWO DATA TRAPS THIS MODULE WORKS AROUND
---------------------------------------
1. CASH-FLOW STATEMENTS ARE FILED YEAR-TO-DATE.  A 10-Q's OCF covers Q1, H1,
   Q1-Q3 — never a discrete Q2 or Q3.  So `dur.between(60,115)` catches only the
   Q1 stub, and a "trailing 4 quarters" of OCF is Q1 of FOUR DIFFERENT YEARS
   (measured: last-4 span median = 1096 days).  Summing that is not a TTM.
2. CALENDAR-Q4 INCOME IS THIN.  10-Ks tag the annual figure, not the Q4 stub, so
   a trailing-4-quarter NetIncomeLoss window actually spans ~366 days with one
   quarter missing (measured: last-4 span median = 366 days).

Both are documented in data/sec/fundamentals.md ("Use fp == 'FY' rows for any
cash-flow or full-year quantity").  This module therefore builds every FLOW from
ANNUAL rows (dur 340-400) and aligns every BALANCE-SHEET item to that fiscal
year end, so NI, OCF and tax always cover the same 12 months of the same
company's own fiscal calendar.  Rule 6 (off-calendar fiscal years) is handled by
comparing each observation with the one ~365d earlier in that ticker's OWN
series via merge_asof, never with a calendar pivot.

NO LOOKAHEAD: every value — current year, prior year, prior-prior year, and the
balance sheets attached to them — is read from `f`, which the caller has already
gated on filed_date <= D.  Nothing is sourced from outside that frame.

Market cap is deliberately never computed here: all six signals are scaled by
assets or by another income-statement line, so no price-times-shares denominator
(and none of its preferred-share / ADR failure modes) enters this family.
"""
import numpy as np
import pandas as pd

# US federal statutory rate, post-TCJA (2018+).  tax_book_gap is a linear
# combination of pretax/assets and tax/assets, so this constant does affect the
# cross-sectional ordering; 0.21 is the right number for every test date here.
STAT_RATE = 0.21

MIN_ASSETS = 1e7        # $10m. NOT cosmetic: at a $1m floor, 2.7% of names had
                        # |accruals| > 1x assets -- median avg assets $6m, median
                        # loss 2.15x assets. Those shells alone moved accrual_ratio's
                        # measured quintile spread from -1.1% to -3.0%.
BS_TOL = 20             # days: balance-sheet date vs fiscal year end
YR_TOL = 75             # days: tolerance when matching the year-earlier obs
MAX_STALE = 1150        # days: drop names whose newest annual data predates this
ACC_CLIP = 1.0          # |accruals| > 1x assets is a data error, not a signal
GAP_CLIP = 1.0
CC_CLIP = (-5.0, 15.0)
NWC_CLIP = (-2.0, 1.0)   # (CA-cash-CL)/A: cannot exceed +1; distress makes it negative


def _sortcols(f):
    return ["period_end"] + (["filed_date"] if "filed_date" in f.columns else [])


def _annual(f, concept):
    """One annual (fiscal-year) observation per ticker/period_end."""
    d = f[(f.concept == concept) & (~f.instantaneous) & f.dur.between(340, 400)]
    if not len(d):
        return pd.DataFrame(columns=["ticker", "period_end", "val"])
    return (d.sort_values(_sortcols(f))
             .drop_duplicates(["ticker", "period_end"], keep="last")
             [["ticker", "period_end", "val"]])


def _inst(f, concept):
    """One balance-sheet observation per ticker/period_end."""
    d = f[(f.concept == concept) & f.instantaneous]
    if not len(d):
        return pd.DataFrame(columns=["ticker", "period_end", "val"])
    return (d.sort_values(_sortcols(f))
             .drop_duplicates(["ticker", "period_end"], keep="last")
             [["ticker", "period_end", "val"]])


def _attach_bs(base, bs, name):
    """Value of a balance-sheet series AT each fiscal year end in `base`."""
    if not len(bs):
        base[name] = np.nan
        return base
    L = base.sort_values("period_end")
    R = bs.sort_values("period_end").rename(columns={"val": name})
    out = pd.merge_asof(L, R, on="period_end", by="ticker",
                        direction="nearest", tolerance=pd.Timedelta(days=BS_TOL))
    return out


def _lag(base, cols, days, suffix):
    """For each row, the same columns from the obs ~`days` earlier, same ticker.

    merge_asof on the ticker's own series — never a calendar-quarter pivot, so
    off-calendar fiscal years line up correctly (hard-won rule 6).
    """
    prev = base[["ticker", "period_end"] + cols].copy()
    prev["join_end"] = prev.period_end + pd.Timedelta(days=days)
    prev = (prev.drop(columns="period_end")
                .rename(columns={c: c + suffix for c in cols})
                .sort_values("join_end"))
    out = pd.merge_asof(base.sort_values("period_end"), prev,
                        left_on="period_end", right_on="join_end", by="ticker",
                        direction="nearest", tolerance=pd.Timedelta(days=YR_TOL))
    return out.drop(columns="join_end")


def _safe(num, den, floor):
    """Divide, killing near-zero denominators before they mint a perfect rank."""
    den = den.where(den.abs() >= floor)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def compute(f, sh, px, hist, D):
    f = f.copy()
    if "dur" not in f.columns:
        f["dur"] = (pd.to_datetime(f.period_end) - pd.to_datetime(f.period_start)).dt.days
    f["period_end"] = pd.to_datetime(f.period_end)

    # ---- annual flows, joined on the company's own fiscal year end ----------
    flow = None
    for concept, name in [("NetIncomeLoss", "ni"),
                          ("NetCashProvidedByUsedInOperatingActivities", "ocf"),
                          ("IncomeTaxExpenseBenefit", "tax")]:
        a = _annual(f, concept).rename(columns={"val": name})
        flow = a if flow is None else flow.merge(a, on=["ticker", "period_end"], how="outer")
    if flow is None or not len(flow):
        return pd.DataFrame()
    flow = flow.dropna(subset=["ticker", "period_end"])

    # ---- balance sheet as of each of those fiscal year ends -----------------
    for concept, name in [("Assets", "A"),
                          ("AssetsCurrent", "CA"),
                          ("CashAndCashEquivalentsAtCarryingValue", "CASH"),
                          ("LiabilitiesCurrent", "CL")]:
        flow = _attach_bs(flow, _inst(f, concept), name)

    # working capital ratio at each fiscal year end (OSAP ChNWC, minus dlc)
    flow["nwc"] = _safe(flow.CA - flow.CASH - flow.CL, flow.A, MIN_ASSETS).clip(*NWC_CLIP)

    # ---- year-earlier and two-year-earlier observations --------------------
    cols = ["ni", "ocf", "tax", "A", "nwc"]
    flow = _lag(flow, cols, 365, "_1")
    flow = _lag(flow, ["A"], 730, "_2")

    # ---- anchor each signal on its own newest COMPLETE fiscal year ---------
    flow = flow.sort_values("period_end")

    def pick(need):
        """Newest fiscal year per ticker in which EVERY input this signal needs
        is present.

        Two things this must not do.  (a) groupby.last() skips NaN per column,
        so a ticker whose newest year lacks OCF would get that year's net income
        paired with an OCF from an earlier year -- the accrual of a year that
        never happened.  tail(1) on a dropna'd frame keeps whole rows, so every
        input of a given signal always comes from the SAME fiscal year (and that
        year's own prior year).  (b) taking one anchor year per TICKER for all
        signals would throw away a usable tax reading just because OCF is
        missing that year, which cost ~1,100 names of tax_book_gap coverage.
        The anchor is therefore per signal, not per ticker.
        """
        d = flow.dropna(subset=need).groupby("ticker").tail(1).set_index("ticker")
        return d[(D - d.period_end).dt.days.between(0, MAX_STALE)]

    def avg_assets(d, a="A", b="A_1"):
        """Average total assets; falls back to the single year when the prior
        balance sheet is missing rather than dropping the name."""
        return pd.concat([d[a], d[b]], axis=1).mean(axis=1, skipna=True)

    out = {}

    # 1 & 5 share an anchor: both need NI, OCF and the balance sheet ---------
    d = pick(["ni", "ocf", "A"])
    if len(d):
        aa = avg_assets(d)

        # 1. accrual_ratio --------------------------------------------------
        # (NI - OCF)/avg assets.  High accruals = earnings not backed by cash =
        # LOW future returns (Sloan 1996, OSAP sign -1), so NEGATE.
        out["accrual_ratio"] = -_safe(d.ni - d.ocf, aa, MIN_ASSETS).clip(-ACC_CLIP, ACC_CLIP)

        # 5. cash_conv ------------------------------------------------------
        # OCF/NI for profitable names.  NOT negated: more cash per dollar of
        # reported profit is better.  Two guards on the denominator — NI must be
        # positive AND at least 0.5% of assets — because a near-zero NI sends
        # the ratio to infinity and mints a perfect rank (the Athene failure).
        ni_ok = d.ni.where((d.ni > 0) & (d.ni >= 0.005 * aa))
        out["cash_conv"] = (d.ocf / ni_ok).replace([np.inf, -np.inf], np.nan).clip(*CC_CLIP)

    # 2. d_accrual ----------------------------------------------------------
    # Same ratio a year ago, scaled by THAT year's average assets.  Rising
    # accruals = deteriorating earnings quality = bad, so NEGATE.
    d = pick(["ni", "ocf", "ni_1", "ocf_1", "A"])
    if len(d):
        aa = avg_assets(d)
        aa_1 = avg_assets(d, "A_1", "A_2").fillna(aa)
        accr = _safe(d.ni - d.ocf, aa, MIN_ASSETS).clip(-ACC_CLIP, ACC_CLIP)
        accr_1 = _safe(d.ni_1 - d.ocf_1, aa_1, MIN_ASSETS).clip(-ACC_CLIP, ACC_CLIP)
        out["d_accrual"] = -(accr - accr_1)

    # 3. tax_book_gap -------------------------------------------------------
    # Book pretax income = NI + tax expense.  Tax-implied pretax = tax/0.21.
    # A large positive gap = book earnings the tax authority never saw = bad,
    # so NEGATE.  Loss-makers are excluded: the book-tax gap of a company with
    # negative pretax income is uninterpretable (OSAP's Tax signal likewise
    # special-cases negative income).
    d = pick(["ni", "tax", "A"])
    if len(d):
        pretax = d.ni + d.tax
        gap = _safe(pretax - d.tax / STAT_RATE, avg_assets(d), MIN_ASSETS)
        out["tax_book_gap"] = -gap.where(pretax > 0).clip(-GAP_CLIP, GAP_CLIP)

    # 4. d_tax_book_gap -----------------------------------------------------
    # The "widening" reading of the same idea.  Same denominator both years so
    # the delta measures the gap, not a change in the asset base.  NEGATE.
    d = pick(["ni", "tax", "ni_1", "tax_1", "A"])
    if len(d):
        aa = avg_assets(d)
        pretax, pretax_1 = d.ni + d.tax, d.ni_1 + d.tax_1
        gap = _safe(pretax - d.tax / STAT_RATE, aa, MIN_ASSETS)
        gap = gap.where(pretax > 0).clip(-GAP_CLIP, GAP_CLIP)
        gap_1 = _safe(pretax_1 - d.tax_1 / STAT_RATE, aa, MIN_ASSETS)
        gap_1 = gap_1.where(pretax_1 > 0).clip(-GAP_CLIP, GAP_CLIP)
        out["d_tax_book_gap"] = -(gap - gap_1)

    # 6. d_nwc --------------------------------------------------------------
    # YoY change in (current assets - cash - current liabilities)/assets.
    # Working-capital buildup = revenue booked ahead of collection = bad, so
    # NEGATE (OSAP ChNWC sign -1).  This is NOT dso_change; see module docstring.
    d = pick(["nwc", "nwc_1"])
    if len(d):
        out["d_nwc"] = -(d.nwc - d.nwc_1).clip(-ACC_CLIP, ACC_CLIP)

    if not out:
        return pd.DataFrame()
    order = ["accrual_ratio", "d_accrual", "tax_book_gap", "d_tax_book_gap",
             "cash_conv", "d_nwc"]
    out = pd.DataFrame(out)[[c for c in order if c in out]]
    out.index.name = "ticker"
    return out.replace([np.inf, -np.inf], np.nan)
