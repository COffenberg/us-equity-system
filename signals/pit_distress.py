"""Distress & safety signals, point-in-time.

Exposes compute(f, sh, px, hist, D) -> DataFrame indexed by ticker.
HIGHER IS ALWAYS BETTER (safer / higher expected return) in every column.

Columns produced
----------------
  altman_z        Altman (1968) 5-factor public-firm Z-score.  Higher = safer.
  ohlson_o        NEGATED Ohlson (1980) 9-term O-score.  Higher = less distress.
  cash_to_assets  cash / total assets.  Higher = safer.
  earnings_vol    NEGATED std-dev of quarterly NetIncomeLoss/Assets over 8q.
  equity_neg      0 when book equity >= 0, -1 when negative.  Higher = safer.
  ebit_to_liab    EBIT / total liabilities.  See "interest coverage" note below.

interest_cover is NOT produced.  See INTEREST COVERAGE below.

INTEREST COVERAGE -- the requested `interest_cover` signal is impossible here.
The fundamentals panel contains 22 concepts and NONE of them is an interest
expense tag (no InterestExpense, InterestExpenseDebt, InterestAndDebtExpense,
InterestIncomeExpenseNet, ...).  Rather than manufacture an interest number
(e.g. an assumed coupon on LongTermDebtNoncurrent, which would be a fabricated
input dressed up as a filed one) the signal is skipped.  `ebit_to_liab` is
supplied in its place and is deliberately named differently: it is EBIT over
TOTAL liabilities, a debt-service-capacity ratio, NOT interest coverage.  Drop
the column if the family is meant to contain only the six named signals.

Specifications used (from data/osap/SignalDoc.csv 'Detailed Definition',
which reproduces the published formulas):

  ZScore = 1.2*(act-lct)/at + 1.4*re/at + 3.3*(ni+xint+txt)/at
           + 0.6*mve/lt + 1.0*revt/at
  OScore = -1.32 - 0.407*log(at/deflator) + 6.03*lt/at - 1.43*(act-lct)/at
           + 0.076*lct/act - 1.72*I(lt>at) - 2.37*ib/at - 1.83*fopt/lt
           + 0.285*INTWO - 0.521*(ib-ib_lag)/(|ib|+|ib_lag|)
           with fopt = oancf when fopt is missing.

Departures from the literal Compustat spec, and why:

  * xint (interest expense) is unavailable, so Altman's EBIT term uses
    OperatingIncomeLoss directly (the cleanest EBIT tag in the panel), falling
    back to NetIncomeLoss + IncomeTaxExpenseBenefit (= pre-tax income, an EBIT
    understatement by exactly the missing interest expense) when operating
    income is not tagged.
  * The GNP deflator in Ohlson's SIZE term is a single number at any one date,
    so it shifts every firm's O-score by the same constant and cannot change a
    cross-sectional rank.  It is dropped; assets are expressed in $mm so raw
    scores stay on the published scale.
  * OSAP renders Ohlson's INTWO as I(ib + ib_t-12 + ib_t-24 < 0), which is not
    Ohlson's own definition.  The published indicator is used instead:
    1 when net income was negative in BOTH of the last two annual periods.
  * earnings_vol is -std rather than 1/std.  Rank-identical for positive std,
    and immune to the near-zero-denominator explosion that 1/std invites when a
    firm has eight almost identical quarters.
  * SIC-code screens and the "drop the bottom quintile" instructions in the
    OSAP notes are portfolio-construction rules, not signal definitions, and
    are left to the caller.

Bug guards applied throughout:
  * every division is denominator-guarded, and every ratio is clipped to a
    range that is wide for a real firm but finite for a shell company, so a
    near-zero denominator cannot buy a top rank;
  * TTM sums require FOUR CONTIGUOUS quarters, verified by the durations
    summing to ~365 days AND the first period_start to last period_end span
    being ~365 days.  Without the second check a filer that tags only Q1-Q3
    (common: Q4 arrives only inside the annual figure) silently produces
    Q3(t-1)+Q1+Q2+Q3 as a "TTM";
  * year-ago comparisons are made against the observation ~365 days earlier in
    that ticker's OWN series (merge_asof by ticker), never a calendar pivot, so
    off-calendar fiscal years stay aligned;
  * every input row is already filed_date <= D, so the earlier leg of every
    change is point-in-time safe as well as the later one;
  * stale filers are dropped: the latest observation must be no more than 460
    days old at D;
  * market cap uses price x shares with the dei -> CommonStockSharesOutstanding
    -> diluted-weighted fallback chain, requires a plain A-Z ticker (preferred
    lines and unit tickers carry the parent's share count) and > $50m.
"""
import numpy as np
import pandas as pd

MAX_AGE_DAYS = 460          # a filing older than this means the filer went dark
MIN_ASSETS = 1e6            # $1m; below this every ratio is noise
MIN_MKTCAP = 5e7


def compute(f, sh, px, hist, D):
    f = f.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    f["period_start"] = pd.to_datetime(f["period_start"])
    if "dur" not in f.columns:
        f["dur"] = (f["period_end"] - f["period_start"]).dt.days
    dur = pd.to_numeric(f["dur"], errors="coerce")

    flow = f[~f["instantaneous"].astype(bool)]
    stock = f[f["instantaneous"].astype(bool)]
    q_rows = flow[dur.reindex(flow.index).between(60, 115)]
    a_rows = flow[dur.reindex(flow.index).between(330, 400)]

    # ---------------------------------------------------------------- helpers
    def inst(concept):
        """Latest instantaneous value per ticker, stale filers dropped."""
        d = stock[stock.concept == concept]
        if not len(d):
            return pd.Series(dtype="float64")
        d = d.sort_values("period_end").groupby("ticker").last()
        d = d[(D - d["period_end"]).dt.days <= MAX_AGE_DAYS]
        return pd.to_numeric(d["val"], errors="coerce")

    def annualized(concept):
        """Series of annualised values per ticker: [ticker, period_end, val].

        Prefers a validated trailing-four-quarter sum (fresher), falls back to
        the tagged annual duration.  Both legs are already filed_date <= D.
        """
        parts = []
        q = q_rows[q_rows.concept == concept][
            ["ticker", "period_start", "period_end", "val"]
        ]
        if len(q):
            q = (q.sort_values(["ticker", "period_end"])
                   .drop_duplicates(["ticker", "period_end"], keep="last")
                   .reset_index(drop=True))
            q["val"] = pd.to_numeric(q["val"], errors="coerce")
            q["d"] = (q["period_end"] - q["period_start"]).dt.days
            g = q.groupby("ticker", sort=False)
            q["s4"] = g["val"].transform(lambda s: s.rolling(4).sum())
            q["n4"] = g["val"].transform(lambda s: s.rolling(4).count())
            q["d4"] = g["d"].transform(lambda s: s.rolling(4).sum())
            q["span"] = (q["period_end"] - g["period_start"].shift(3)).dt.days
            ok = (q["n4"] == 4) & q["d4"].between(330, 400) & q["span"].between(330, 400)
            t = q.loc[ok, ["ticker", "period_end", "s4"]].rename(columns={"s4": "val"})
            t["src"] = 0                                   # TTM wins ties
            parts.append(t)
        a = a_rows[a_rows.concept == concept][["ticker", "period_end", "val"]].copy()
        if len(a):
            a["val"] = pd.to_numeric(a["val"], errors="coerce")
            a = (a.sort_values(["ticker", "period_end"])
                  .drop_duplicates(["ticker", "period_end"], keep="last"))
            a["src"] = 1
            parts.append(a)
        if not parts:
            return pd.DataFrame(columns=["ticker", "period_end", "val"])
        out = pd.concat(parts, ignore_index=True)
        out = (out.sort_values(["ticker", "period_end", "src"])
                  .drop_duplicates(["ticker", "period_end"], keep="first"))
        out = out.dropna(subset=["val"])
        return out.sort_values("period_end")[["ticker", "period_end", "val"]]

    def latest(series_df):
        if not len(series_df):
            return pd.Series(dtype="float64")
        d = series_df.groupby("ticker").last()
        d = d[(D - d["period_end"]).dt.days <= MAX_AGE_DAYS]
        return d["val"]

    def year_ago(series_df, tol=75):
        """Value ~365d before that ticker's OWN latest observation."""
        if not len(series_df):
            return pd.Series(dtype="float64")
        cur = series_df.sort_values("period_end").groupby("ticker").tail(1)
        prev = series_df.copy()
        prev["join_end"] = prev["period_end"] + pd.Timedelta(days=365)
        prev = (prev[["ticker", "join_end", "val"]]
                .rename(columns={"val": "val_ly"})
                .sort_values("join_end"))
        m = pd.merge_asof(cur.sort_values("period_end"), prev,
                          left_on="period_end", right_on="join_end", by="ticker",
                          direction="nearest", tolerance=pd.Timedelta(days=tol))
        return m.set_index("ticker")["val_ly"].dropna()

    def safe_div(num, den, floor, lo, hi):
        """Aligned division; denominators at or below `floor` become NaN and the
        result is clipped so a small survivor cannot rank first on arithmetic."""
        num, den = num.align(den, join="inner")
        den = den.where(den.abs() > floor)
        r = num / den
        return r.replace([np.inf, -np.inf], np.nan).clip(lo, hi)

    # ------------------------------------------------------------- ingredients
    A = inst("Assets")
    A = A[A > MIN_ASSETS]
    L = inst("Liabilities")
    AC = inst("AssetsCurrent")
    LC = inst("LiabilitiesCurrent")
    RE = inst("RetainedEarningsAccumulatedDeficit")
    CASH = inst("CashAndCashEquivalentsAtCarryingValue")
    EQ = inst("StockholdersEquity")

    ni_s = annualized("NetIncomeLoss")
    ni = latest(ni_s)
    ni_ly = year_ago(ni_s)
    ocf = latest(annualized("NetCashProvidedByUsedInOperatingActivities"))
    tax = latest(annualized("IncomeTaxExpenseBenefit"))
    opinc = latest(annualized("OperatingIncomeLoss"))
    rev = latest(annualized("Revenues"))
    rev = rev.combine_first(
        latest(annualized("RevenueFromContractWithCustomerExcludingAssessedTax")))

    # EBIT: operating income first; pre-tax income as fallback.  Interest
    # expense is absent from the panel, so the fallback understates EBIT by it.
    ebit = opinc.combine_first(
        (ni.reindex(ni.index.union(tax.index)).fillna(np.nan)
         + tax.reindex(ni.index.union(tax.index)).fillna(np.nan)).dropna())

    # ------------------------------------------------------------- market cap
    p = pd.to_numeric(px, errors="coerce").dropna()
    p = p[p >= 1.0]                                   # sub-penny prices -> +inf
    s = inst("EntityCommonStockSharesOutstanding")
    if sh is not None and len(sh):
        _sh = sh.copy()
        _sh["period_end"] = pd.to_datetime(_sh["period_end"])
        _sh = _sh[(D - _sh["period_end"]).dt.days <= MAX_AGE_DAYS]
        for c in ["CommonStockSharesOutstanding",
                  "WeightedAverageNumberOfDilutedSharesOutstanding"]:
            d = _sh[_sh.concept == c].sort_values("period_end")
            if len(d):
                s = s.combine_first(
                    pd.to_numeric(d.groupby("ticker").last()["val"], errors="coerce"))
    s = s[s > 0]
    mc = (p * s).dropna()
    # preferred lines, units and warrants carry the parent's share count and
    # once produced a $36,480bn Fannie Mae.  Keep plain A-Z tickers only.
    mc = mc[pd.Series(mc.index.astype(str), index=mc.index)
            .str.fullmatch(r"[A-Z]+").fillna(False)]
    mc = mc[mc > MIN_MKTCAP]

    # ------------------------------------------------------------- Altman Z
    x1 = safe_div((AC - LC).dropna(), A, MIN_ASSETS, -5, 5)      # working capital/TA
    x2 = safe_div(RE, A, MIN_ASSETS, -10, 5)                     # retained earnings/TA
    x3 = safe_div(ebit, A, MIN_ASSETS, -5, 5)                    # EBIT/TA
    x4 = safe_div(mc, L, 1e5, 0, 20)                             # mkt equity/total liab
    x5 = safe_div(rev, A, MIN_ASSETS, 0, 20)                     # sales/TA
    z = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}).dropna()
    altman_z = 1.2 * z.x1 + 1.4 * z.x2 + 3.3 * z.x3 + 0.6 * z.x4 + 1.0 * z.x5

    # ------------------------------------------------------------- Ohlson O
    size = np.log((A / 1e6).where(A > MIN_ASSETS))               # $mm, deflator dropped
    tlta = safe_div(L, A, MIN_ASSETS, 0, 5)
    wcta = x1
    clca = safe_div(LC, AC, MIN_ASSETS, 0, 20)
    _la = pd.DataFrame({"L": L, "A": A}).dropna()
    oeneg = (_la.L > _la.A).astype(float)                        # negative book equity
    nita = safe_div(ni, A, MIN_ASSETS, -5, 5)
    futl = safe_div(ocf, L, 1e5, -5, 5)
    _ni2 = pd.DataFrame({"a": ni, "b": ni_ly}).dropna()
    intwo = ((_ni2.a < 0) & (_ni2.b < 0)).astype(float)
    chin = ((_ni2.a - _ni2.b) / (_ni2.a.abs() + _ni2.b.abs())
            .where(lambda d: d > 0)).clip(-1, 1)
    o = pd.DataFrame({"size": size, "tlta": tlta, "wcta": wcta, "clca": clca,
                      "oeneg": oeneg, "nita": nita, "futl": futl,
                      "intwo": intwo, "chin": chin}).dropna()
    o_score = (-1.32 - 0.407 * o["size"] + 6.03 * o.tlta - 1.43 * o.wcta
               + 0.076 * o.clca - 1.72 * o.oeneg - 2.37 * o.nita
               - 1.83 * o.futl + 0.285 * o.intwo - 0.521 * o.chin)
    ohlson_o = -o_score                     # O rises with distress -> negate

    # ------------------------------------------------------------- simple ones
    cash_to_assets = safe_div(CASH, A, MIN_ASSETS, 0, 1.5)
    equity_neg = pd.Series(np.where(EQ < 0, -1.0, 0.0), index=EQ.index)

    # ---------------------------------------------------- earnings volatility
    qni = q_rows[q_rows.concept == "NetIncomeLoss"][["ticker", "period_end", "val"]]
    earnings_vol = pd.Series(dtype="float64")
    if len(qni):
        qni = (qni.sort_values(["ticker", "period_end"])
                  .drop_duplicates(["ticker", "period_end"], keep="last"))
        qni["val"] = pd.to_numeric(qni["val"], errors="coerce")
        # scale each quarter by the assets reported at THAT balance-sheet date,
        # not by today's assets, so an acquisition does not fake volatility
        ab = (stock[stock.concept == "Assets"][["ticker", "period_end", "val"]]
              .sort_values(["ticker", "period_end"])
              .drop_duplicates(["ticker", "period_end"], keep="last")
              .rename(columns={"val": "A_t", "period_end": "a_end"}))
        ab["A_t"] = pd.to_numeric(ab["A_t"], errors="coerce")
        m = pd.merge_asof(qni.sort_values("period_end"), ab.sort_values("a_end"),
                          left_on="period_end", right_on="a_end", by="ticker",
                          direction="nearest", tolerance=pd.Timedelta(days=45))
        m["roa"] = m["val"] / m["A_t"].where(m["A_t"] > MIN_ASSETS)
        m["roa"] = m["roa"].replace([np.inf, -np.inf], np.nan).clip(-5, 5)
        m = m[m.period_end >= D - pd.Timedelta(days=830)]        # last ~8 quarters
        last8 = m.dropna(subset=["roa"]).groupby("ticker").tail(8)
        g = last8.groupby("ticker")["roa"].agg(["std", "count"])
        earnings_vol = -g["std"].where(g["count"] >= 6)           # -std == rank(1/std)

    # -------------------------------- EBIT / total liabilities (NOT int. cover)
    ebit_to_liab = safe_div(ebit, L, 1e5, -5, 20)

    out = pd.DataFrame({
        "altman_z": altman_z,
        "ohlson_o": ohlson_o,
        "cash_to_assets": cash_to_assets,
        "earnings_vol": earnings_vol,
        "equity_neg": equity_neg,
        "ebit_to_liab": ebit_to_liab,
    })
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "ticker"
    return out.dropna(how="all")
