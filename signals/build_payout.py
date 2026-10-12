#!/usr/bin/env python3
"""Payout & external-financing signal group.

NetPayoutYield, ShareIss1Y, ShareIss5Y, NetEquityFinance, XFIN, CompositeDebtIssuance.

Formulas are taken verbatim from the 'Detailed Definition' column of
data/osap/SignalDoc.csv (Chen-Zimmermann Open Source Asset Pricing) and the
Compustat mnemonics in them are mapped to XBRL concepts here.

    NetPayoutYield        (dvc + prstkc - sstk) / mve
    ShareIss1Y            shares(t) / shares(t-1y) - 1          (split-adjusted)
    ShareIss5Y            shares(t) / shares(t-5y) - 1          (split-adjusted)
    NetEquityFinance      (sstk - prstkc) / avg(at_t, at_t-1)   |ratio| <= 1
    XFIN                  (sstk - dv - prstkc + dltis - dltr) / at
    CompositeDebtIssuance log(dltt_t + dlc_t) - log(dltt_t-5y + dlc_t-5y)

Output per signal (OUTPUT CONVENTION):
    signals/<name>.parquet   [ticker, value, rank_pct]
    signals/<name>.md        formula, concepts, coverage, top/bottom 5, judgement calls

`value`    the signal in its natural units, winsorised at 1%/99% (rule 4).
`rank_pct` cross-sectional percentile in [0,1] of (osap_sign * value), so
           rank_pct -> 1 is ALWAYS the attractive end regardless of the sign
           the literature attaches to the raw signal. Ranks are unaffected by
           winsorisation (it is monotone); winsorisation is applied so the
           stored `value` is also safe to z-score downstream.

Run:  python3 signals/build_payout.py
Deps: signals/fetch_payout_frames.py and signals/fetch_splits.py must have run.
"""
import pathlib
import gzip
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
FRAMES = os.path.join(ROOT, "data/sec/frames")
OUTDIR = os.path.join(ROOT, "signals")
ASOF = pd.Timestamp("2026-08-18")

# ---------------------------------------------------------------- eligibility
MIN_DOLLAR_VOL = 1_000_000.0     # median 60-session dollar volume
MIN_PRICE = 1.00
MIN_MKTCAP = 50_000_000.0
MIN_LISTING_DAYS = 730           # OSAP NetPayoutYield wants >= 2 years in CRSP
NONCOMMON_SUFFIX = {"WT", "W", "WS", "R", "RT", "RI", "U", "UN"}   # + P* handled separately

WINSOR = (0.01, 0.99)
FY_MIN_END = pd.Timestamp("2025-01-01")   # anchor fiscal year must end on/after this
DAY = np.timedelta64(1, "D")


def log(*a):
    print(*a, flush=True)


# =============================================================== data loading
def load_frame(tax, concept, unit, fid):
    """One cached frames response -> DataFrame[cik, period_end, period_start, val, accn]."""
    p = os.path.join(FRAMES, "%s_%s_%s_%s.json.gz" % (tax, concept, unit, fid))
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        js = json.load(fh)
    rows = js.get("data", [])
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["cik"] = d["cik"].astype(int).astype(str).str.zfill(10)
    d["period_end"] = pd.to_datetime(d["end"])
    d["period_start"] = pd.to_datetime(d["start"]) if "start" in d.columns else pd.NaT
    return d[["cik", "period_end", "period_start", "val", "accn"]]


def load_concept(concept, unit, fids, tax="us-gaap"):
    """Stack several frames of one concept. Keeps the latest-filed row per (cik, period_end)."""
    parts = [x for x in (load_frame(tax, concept, unit, f) for f in fids) if x is not None]
    if not parts:
        return pd.DataFrame(columns=["cik", "period_end", "period_start", "val", "accn"])
    d = pd.concat(parts, ignore_index=True)
    return d.drop_duplicates(["cik", "period_end"], keep="last")


ANNUAL_FIDS = ["CY2024", "CY2025", "CY2026"]
INST_FIDS = (["CY2020Q4I"] + ["CY2021Q%dI" % q for q in (1, 2, 3, 4)]
             + ["CY2025Q%dI" % q for q in (1, 2, 3, 4)] + ["CY2026Q1I", "CY2026Q2I"])
ALL_INST_FIDS = ["CY%dQ%dI" % (y, q) for y in range(2020, 2027) for q in (1, 2, 3, 4)
                 if not (y == 2026 and q > 2)]


def panel_concept(fund, concept, annual=True):
    """Concept series from the prebuilt fundamentals panel (has filed_date), CIK level."""
    d = fund[fund.concept == concept]
    if annual:
        d = d[d.fp == "FY"]
    cols = ["cik", "period_end", "period_start", "val", "accn", "filed_date"]
    return d[cols].drop_duplicates(["cik", "period_end"], keep="last")


def asof_join(base, series, target_col, tol_days, valname):
    """For each base row pick the series obs closest to base[target_col] within tol_days."""
    if series.empty:
        base[valname] = np.nan
        return base
    b = base[["cik", target_col]].dropna().sort_values(target_col)
    s = series[["cik", "period_end", "val"]].dropna(subset=["period_end"]).sort_values("period_end")
    m = pd.merge_asof(b, s, left_on=target_col, right_on="period_end", by="cik",
                      tolerance=pd.Timedelta(days=tol_days), direction="nearest")
    m = m.rename(columns={"val": valname})[["cik", valname]]
    return base.merge(m.drop_duplicates("cik"), on="cik", how="left")


# ================================================================== universe
def build_universe(fund):
    uni = pd.read_csv(os.path.join(ROOT, "data/universe/tickers_2026-08-18.csv"), dtype={"cik": str})
    px = pd.read_parquet(os.path.join(ROOT, "data/snapshots/ohlcv_2026-08-18.parquet"))
    last = px.sort_values("date").groupby("ticker").tail(1)[["ticker", "close"]]
    der = pd.read_parquet(os.path.join(ROOT, "data/prices/derived.parquet"))
    der = der[["ticker", "dollar_vol_60d", "listing_age_days"]]
    dg = pd.read_parquet(os.path.join(ROOT, "data/prices/derived_diagnostics.parquet"))
    dg = dg[["ticker", "quarantined", "unadjustable"]]

    m = uni.merge(last, on="ticker").merge(der, on="ticker", how="left").merge(dg, on="ticker", how="left")
    n0 = len(m)

    suf = m.ticker.str.extract(r"-([A-Z]{1,3})$")[0]
    noncommon = suf.isin(NONCOMMON_SUFFIX) | suf.str.match(r"^P[A-Z]?$").fillna(False)
    m = m[~noncommon].copy()
    n1 = len(m)

    m["quarantined"] = m.quarantined.fillna(False)
    keep = ((m.dollar_vol_60d >= MIN_DOLLAR_VOL) & (m.close >= MIN_PRICE)
            & (m.listing_age_days >= MIN_LISTING_DAYS) & (~m.quarantined))
    m = m[keep].copy()
    n2 = len(m)

    # one row per issuer: the most liquid line. Share classes are the same company;
    # scoring both would double-weight it in any downstream composite.
    m = m.sort_values("dollar_vol_60d", ascending=False).drop_duplicates("cik", keep="first")
    n3 = len(m)

    log("universe: %d roster+price -> %d common-stock forms -> %d past liquidity floor -> %d issuers"
        % (n0, n1, n2, n3))
    return m.reset_index(drop=True)


# ================================================================== shares
def split_factors():
    """cum multiplier that puts a share count observed at date d on TODAY's split basis."""
    parts = []
    p = os.path.join(ROOT, "data/prices/splits_yf.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        s["date"] = pd.to_datetime(s["date"])
        parts.append(s[["ticker", "date", "ratio"]])
        log("splits: %d yfinance events on %d tickers" % (len(s), s.ticker.nunique()))
    r = os.path.join(ROOT, "data/prices/splits_repaired.csv")
    if os.path.exists(r):
        d = pd.read_csv(r)
        d["date"] = pd.to_datetime(d["date"])
        # that file stores the raw PRICE ratio at the split; yfinance stores the SHARE
        # ratio, which is its reciprocal (1-for-10 reverse split: price x10, ratio 0.1).
        d["ratio"] = 1.0 / d["factor"]
        parts.append(d[["ticker", "date", "ratio"]])
        log("splits: %d Yahoo-unapplied events on %d tickers (price-ratio file, inverted)"
            % (len(d), d.ticker.nunique()))
    if not parts:
        return pd.DataFrame(columns=["ticker", "date", "ratio"])
    s = pd.concat(parts, ignore_index=True)
    # the same split can appear in both files; collapse near-duplicates (same ticker, same month)
    s["key"] = s.ticker + "|" + s.date.dt.to_period("M").astype(str)
    s = s.drop_duplicates("key").drop(columns="key")
    return s[s.ratio > 0]


def cum_split_factor(splits, ticker, obs_date):
    """product of split ratios strictly after obs_date (vectorised by caller)."""
    raise NotImplementedError  # replaced by the vectorised path below


def adjust_shares(obs, splits):
    """obs: [cik, ticker, obs_date, raw]. Returns raw * prod(ratio for splits after obs_date)."""
    if splits.empty:
        obs["adj"] = obs["raw"]
        obs["n_splits"] = 0
        return obs
    j = obs.merge(splits, on="ticker", how="left")
    j["applies"] = (j["date"] > j["obs_date"]).fillna(False)
    j["r"] = np.where(j["applies"], j["ratio"], 1.0)
    g = j.groupby(["cik", "ticker", "obs_date"], as_index=False).agg(
        f=("r", "prod"), n_splits=("applies", "sum"))
    out = obs.merge(g, on=["cik", "ticker", "obs_date"], how="left")
    out["f"] = out["f"].fillna(1.0)
    out["n_splits"] = out["n_splits"].fillna(0).astype(int)
    out["adj"] = out["raw"] * out["f"]
    return out


def shares_series(fund):
    """Priority-ordered share-count sources. Both ends of a growth ratio must come
    from the SAME source: a level shift between sources is indistinguishable from
    issuance. Each entry -> DataFrame[cik, period_end, val]."""
    srcs = []
    dei = panel_concept(fund, "EntityCommonStockSharesOutstanding", annual=False)
    srcs.append(("dei:EntityCommonStockSharesOutstanding", dei[["cik", "period_end", "val"]]))
    for c in ("CommonStockSharesOutstanding", "CommonStockSharesIssued"):
        d = load_concept(c, "shares", INST_FIDS)
        srcs.append(("us-gaap:" + c, d[["cik", "period_end", "val"]]))
    return srcs


def share_growth(uni, fund, splits, years, tol_days):
    """shares(t)/shares(t-`years`y) - 1, split-adjusted, single source for both ends."""
    srcs = shares_series(fund)
    out = {}
    used = {}
    for name, s in srcs:
        if s.empty:
            continue
        s = s.dropna(subset=["period_end", "val"])
        s = s[s.val > 0].sort_values("period_end")
        latest = s.sort_values("period_end").groupby("cik").tail(1)
        latest = latest.rename(columns={"period_end": "t_now", "val": "raw_now"})
        latest = latest[latest.t_now >= ASOF - pd.Timedelta(days=270)]      # must be current
        base = uni[["cik", "ticker"]].merge(latest, on="cik", how="inner")
        if base.empty:
            continue
        base["t_then"] = base.t_now - pd.Timedelta(days=int(round(365.25 * years)))
        b = base[["cik", "t_then"]].sort_values("t_then")
        then = pd.merge_asof(b, s[["cik", "period_end", "val"]].sort_values("period_end"),
                             left_on="t_then", right_on="period_end", by="cik",
                             tolerance=pd.Timedelta(days=tol_days), direction="nearest")
        then = then.rename(columns={"period_end": "t_obs", "val": "raw_then"})
        base = base.merge(then[["cik", "t_obs", "raw_then"]].drop_duplicates("cik"), on="cik", how="left")
        base = base.dropna(subset=["raw_then"])
        if base.empty:
            continue
        adj = adjust_shares(base.rename(columns={"t_obs": "obs_date", "raw_then": "raw"})
                            [["cik", "ticker", "obs_date", "raw"]], splits)
        base = base.merge(adj[["cik", "adj", "n_splits"]], on="cik", how="left")
        base["value"] = base.raw_now / base.adj - 1.0
        for _, r in base.iterrows():
            if r.cik in out:
                continue
            out[r.cik] = r.value
            used[r.cik] = name
    res = pd.DataFrame({"cik": list(out.keys()), "value": list(out.values())})
    res["src"] = res.cik.map(used)
    return res


# ============================================================== annual flows
def annual_flow_table(uni, fund):
    """Anchor every cash-flow leg to the SAME fiscal year as the company's own
    annual operating-cash-flow line, so the legs of a formula can never be mixed
    across fiscal years."""
    ocf = panel_concept(fund, "NetCashProvidedByUsedInOperatingActivities", annual=True)
    ocf = ocf[(ocf.period_end >= FY_MIN_END) & (ocf.filed_date <= ASOF)]
    anchor = ocf.sort_values("period_end").groupby("cik").tail(1)
    anchor = anchor.rename(columns={"period_end": "fy_end", "period_start": "fy_start",
                                    "filed_date": "fy_filed", "val": "ocf"})
    base = uni[["cik", "ticker", "close", "dollar_vol_60d", "title"]].merge(
        anchor[["cik", "fy_end", "fy_start", "fy_filed", "ocf"]], on="cik", how="inner")
    log("annual anchor: %d issuers have an annual cash-flow statement ending >= %s"
        % (len(base), FY_MIN_END.date()))
    return base


PANEL_DUR = {"PaymentsOfDividendsCommonStock", "PaymentsForRepurchaseOfCommonStock",
             "ProceedsFromIssuanceOfCommonStock", "ProceedsFromIssuanceOfLongTermDebt",
             "RepaymentsOfLongTermDebt"}


def attach_flow(base, fund, concept):
    """Value of `concept` for exactly the anchor fiscal year (period_end within 7 days)."""
    if concept in PANEL_DUR:
        s = panel_concept(fund, concept, annual=True)
        s = s[s.filed_date <= ASOF]
    else:
        s = load_concept(concept, "USD", ANNUAL_FIDS)
    if s.empty:
        base[concept] = np.nan
        return base
    s = s[["cik", "period_end", "val"]].dropna()
    m = base[["cik", "fy_end"]].merge(s, on="cik", how="left")
    m = m[(m.period_end - m.fy_end).abs() <= pd.Timedelta(days=7)]
    m = m.sort_values("period_end").drop_duplicates("cik", keep="last")
    return base.merge(m[["cik", "val"]].rename(columns={"val": concept}), on="cik", how="left")


def c0(df, *cols):
    """coalesce: first non-null across cols."""
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        if c in df.columns:
            out = out.where(out.notna(), df[c])
    return out


def s0(df, *cols):
    """sum treating missing as zero; NaN only if every col is missing."""
    present = pd.Series(False, index=df.index)
    tot = pd.Series(0.0, index=df.index)
    for c in cols:
        if c in df.columns:
            v = df[c]
            tot = tot + v.fillna(0.0)
            present = present | v.notna()
    return tot.where(present, np.nan)


# ================================================================ debt levels
DEBT_CONCEPTS = ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations",
                 "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
                 "ShortTermBorrowings", "OtherShortTermBorrowings", "NotesPayableCurrent",
                 "DebtCurrent", "DebtLongtermAndShorttermCombinedAmount", "LongTermNotesPayable",
                 "LinesOfCreditCurrent", "ConvertibleDebtNoncurrent"]


def debt_at(fund, ciks, target):
    """Total debt (dltt + dlc) per CIK at the balance-sheet date nearest `target`
    (a Series indexed like ciks). Returns [cik, debt, recipe, obs_date]."""
    wide = pd.DataFrame({"cik": ciks.values, "tgt": target.values})
    for c in DEBT_CONCEPTS:
        if c == "LongTermDebtNoncurrent":
            s = panel_concept(fund, c, annual=False)
            s = s[s.filed_date <= ASOF][["cik", "period_end", "val"]]
        else:
            s = load_concept(c, "USD", INST_FIDS)[["cik", "period_end", "val"]]
        if s.empty:
            wide[c] = np.nan
            wide[c + "__d"] = pd.NaT
            continue
        s = s.dropna().sort_values("period_end")
        b = wide[["cik", "tgt"]].dropna().sort_values("tgt")
        m = pd.merge_asof(b, s, left_on="tgt", right_on="period_end", by="cik",
                          tolerance=pd.Timedelta(days=100), direction="nearest")
        m = m.drop_duplicates("cik")
        wide = wide.merge(m[["cik", "val", "period_end"]].rename(
            columns={"val": c, "period_end": c + "__d"}), on="cik", how="left")

    # recipe ladder, most explicit first. dlc = current debt + short-term borrowings.
    dlc = s0(wide, "LongTermDebtCurrent", "ShortTermBorrowings", "OtherShortTermBorrowings",
             "NotesPayableCurrent", "LinesOfCreditCurrent")
    dlc_alt = s0(wide, "LongTermDebtAndCapitalLeaseObligationsCurrent", "ShortTermBorrowings",
                 "OtherShortTermBorrowings", "NotesPayableCurrent", "LinesOfCreditCurrent")
    dlc_any = dlc.where(dlc.notna(), dlc_alt)
    dlc_any = dlc_any.where(dlc_any.notna(), wide.get("DebtCurrent"))

    r3 = wide["DebtLongtermAndShorttermCombinedAmount"]
    r2 = wide["LongTermDebt"] + dlc_any.fillna(0.0)                       # LongTermDebt incl current
    r2 = r2.where(wide["LongTermDebt"].notna())
    ltdnc = c0(wide, "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
               "LongTermNotesPayable", "ConvertibleDebtNoncurrent")
    r1 = ltdnc + dlc_any.fillna(0.0)
    r1 = r1.where(ltdnc.notna())
    r0 = dlc_any                                                          # current debt only

    recipe = pd.Series("none", index=wide.index)
    debt = pd.Series(np.nan, index=wide.index)
    for name, ser in (("dlc_only", r0), ("ltdnc+dlc", r1), ("longtermdebt+stb", r2),
                      ("combined_total", r3)):
        take = ser.notna()
        debt = debt.where(~take, ser)
        recipe = recipe.where(~take, name)

    dcols = [c + "__d" for c in DEBT_CONCEPTS if c + "__d" in wide.columns]
    obs = wide[dcols].max(axis=1)
    return pd.DataFrame({"cik": wide.cik, "debt": debt.values, "recipe": recipe.values,
                         "obs_date": obs.values})


# ================================================================== output
def winsorise(v):
    lo, hi = v.quantile(WINSOR[0]), v.quantile(WINSOR[1])
    return v.clip(lo, hi), lo, hi


def write_signal(name, df, sign, formula, concepts, judgement, extra_stats=""):
    """df: [ticker, value, ...]. sign: +1 high-is-good, -1 low-is-good."""
    d = df.dropna(subset=["value"]).copy()
    d = d[np.isfinite(d.value)]
    raw = d.value.copy()
    d["value"], lo, hi = winsorise(d.value)
    d["rank_pct"] = (sign * d.value).rank(pct=True, method="average")
    d = d.sort_values("rank_pct", ascending=False).reset_index(drop=True)
    d["_raw"] = raw.reindex(d.index).values if False else np.nan
    # keep raw alongside for the report
    d = d.drop(columns=["_raw"])
    rawmap = dict(zip(df.ticker, df.value))
    out = d[["ticker", "value", "rank_pct"]]
    out.to_parquet(os.path.join(OUTDIR, name + ".parquet"), index=False)

    top = d.head(10)
    bot = d.tail(10).iloc[::-1]

    def tbl(x):
        rows = []
        for _, r in x.iterrows():
            rows.append("| %-8s | %14s | %14s | %.4f |" % (
                r.ticker, fmt(rawmap.get(r.ticker)), fmt(r.value), r.rank_pct))
        return "\n".join(rows)

    hdr = "| ticker | raw value | winsorised | rank_pct |\n|---|---:|---:|---:|"
    md = f"""# {name}

**Coverage: {len(d)} tickers.**  Sign = {sign:+d} ({'high' if sign > 0 else 'low'} raw value is attractive).
`rank_pct` is the percentile of `{'+' if sign > 0 else '-'}value`, so **rank_pct -> 1.00 is always the attractive end**.
`value` is winsorised at 1%/99% (bounds {fmt(lo)} / {fmt(hi)}); ranking is unaffected by that (monotone).

## Formula
OSAP `Detailed Definition`:

> {formula}

## XBRL concepts used
{concepts}

## Most attractive (rank_pct ~ 1)
{hdr}
{tbl(top)}

## Least attractive (rank_pct ~ 0)
{hdr}
{tbl(bot)}

## Judgement calls
{judgement}
{extra_stats}
"""
    with open(os.path.join(OUTDIR, name + ".md"), "w") as fh:
        fh.write(md)

    log("\n=== %s  coverage=%d  sign=%+d  winsor=[%s, %s]" % (name, len(d), sign, fmt(lo), fmt(hi)))
    log("  TOP 10 (attractive):")
    for _, r in top.iterrows():
        log("    %-8s raw=%-14s w=%-14s rank=%.4f" % (r.ticker, fmt(rawmap.get(r.ticker)),
                                                      fmt(r.value), r.rank_pct))
    log("  BOTTOM 10 (unattractive):")
    for _, r in bot.iterrows():
        log("    %-8s raw=%-14s w=%-14s rank=%.4f" % (r.ticker, fmt(rawmap.get(r.ticker)),
                                                      fmt(r.value), r.rank_pct))
    if len(d) < 1500:
        log("  !! COVERAGE UNDER 1,500 -- investigate")
    return len(d)


def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    ax = abs(x)
    if ax >= 1e9:
        return "%.3fB" % (x / 1e9)
    if ax >= 1e6:
        return "%.3fM" % (x / 1e6)
    return "%.5f" % x


# ===================================================================== main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fund = pd.read_parquet(os.path.join(ROOT, "data/sec/fundamentals.parquet"))
    fund = fund.drop(columns=["ticker"]).drop_duplicates(["cik", "concept", "period_end", "fp"])

    uni = build_universe(fund)
    splits = split_factors()

    # ---------------- market cap -------------------------------------------
    shsrc = shares_series(fund)
    mk = uni[["cik", "ticker", "close"]].copy()
    mk["shares"] = np.nan
    mk["sh_src"] = None
    mk["sh_date"] = pd.NaT
    for name, s in shsrc:
        s = s.dropna(subset=["period_end", "val"])
        s = s[s.val > 0].sort_values("period_end").groupby("cik").tail(1)
        s = s.rename(columns={"val": "v", "period_end": "d"})
        j = mk.merge(s[["cik", "v", "d"]], on="cik", how="left")
        fresh = j.d >= ASOF - pd.Timedelta(days=270)
        take = mk.shares.isna() & j.v.notna() & fresh
        mk.loc[take, "shares"] = j.loc[take, "v"]
        mk.loc[take, "sh_src"] = name
        mk.loc[take, "sh_date"] = j.loc[take, "d"]
    mk["mktcap"] = mk.shares * mk.close
    log("market cap: %d/%d issuers priced; source mix %s"
        % (mk.mktcap.notna().sum(), len(mk), mk.sh_src.value_counts().to_dict()))
    for t in ("AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "XOM"):
        r = mk[mk.ticker == t]
        if len(r):
            log("   sanity %-6s shares=%.3fB px=%.2f mktcap=$%.0fB  (%s)"
                % (t, r.shares.iloc[0] / 1e9, r.close.iloc[0], r.mktcap.iloc[0] / 1e9,
                   r.sh_src.iloc[0]))

    # ---------------- annual flows -----------------------------------------
    base = annual_flow_table(uni, fund)
    FLOWS = ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
             "PaymentsOfDividendsPreferredStockAndPreferenceStock",
             "PaymentsOfDividendsMinorityInterest",
             "PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity",
             "PaymentsForRepurchaseOfPreferredStockAndPreferenceStock",
             "ProceedsFromIssuanceOfCommonStock", "ProceedsFromIssuanceOrSaleOfEquity",
             "ProceedsFromIssuanceOfPrivatePlacement",
             "ProceedsFromIssuanceOfPreferredStockAndPreferenceStock",
             "ProceedsFromStockOptionsExercised", "ProceedsFromIssuanceOfWarrants",
             "ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt",
             "ProceedsFromNotesPayable", "ProceedsFromLinesOfCredit",
             "ProceedsFromLongTermLinesOfCredit", "ProceedsFromIssuanceOfSeniorLongTermDebt",
             "ProceedsFromIssuanceOfUnsecuredDebt", "ProceedsFromIssuanceOfSecuredDebt",
             "ProceedsFromConvertibleDebt", "ProceedsFromRelatedPartyDebt",
             "RepaymentsOfLongTermDebt", "RepaymentsOfDebt", "RepaymentsOfNotesPayable",
             "RepaymentsOfLinesOfCredit", "RepaymentsOfLongTermLinesOfCredit",
             "RepaymentsOfSeniorDebt", "RepaymentsOfConvertibleDebt",
             "RepaymentsOfRelatedPartyDebt", "RepaymentsOfSecuredDebt",
             "RepaymentsOfUnsecuredDebt"]
    for c in FLOWS:
        base = attach_flow(base, fund, c)
    log("flow tag hit rates (share of %d anchored issuers):" % len(base))
    for c in FLOWS:
        log("   %-58s %5d  %4.1f%%" % (c, base[c].notna().sum(), 100 * base[c].notna().mean()))

    # ---- Compustat mnemonic reconstruction --------------------------------
    # dvc: common dividends paid
    dvc_direct = base["PaymentsOfDividendsCommonStock"]
    dvc_resid = (base["PaymentsOfDividends"].fillna(np.nan)
                 - base["PaymentsOfDividendsPreferredStockAndPreferenceStock"].fillna(0.0)
                 - base["PaymentsOfDividendsMinorityInterest"].fillna(0.0))
    base["dvc"] = dvc_direct.where(dvc_direct.notna(), dvc_resid)
    # dv: total dividends paid
    dv_direct = base["PaymentsOfDividends"]
    dv_sum = s0(base, "PaymentsOfDividendsCommonStock",
                "PaymentsOfDividendsPreferredStockAndPreferenceStock",
                "PaymentsOfDividendsMinorityInterest")
    base["dv"] = dv_direct.where(dv_direct.notna(), dv_sum)
    # prstkc: purchase of common AND preferred stock
    prstkc_direct = s0(base, "PaymentsForRepurchaseOfCommonStock",
                       "PaymentsForRepurchaseOfPreferredStockAndPreferenceStock")
    base["prstkc"] = prstkc_direct.where(prstkc_direct.notna(), base["PaymentsForRepurchaseOfEquity"])
    # sstk: sale of common and preferred stock
    sstk_total = base["ProceedsFromIssuanceOrSaleOfEquity"]        # explicit total: do not add to it
    sstk_parts = s0(base, "ProceedsFromIssuanceOfCommonStock", "ProceedsFromIssuanceOfPrivatePlacement",
                    "ProceedsFromIssuanceOfPreferredStockAndPreferenceStock",
                    "ProceedsFromIssuanceOfWarrants", "ProceedsFromStockOptionsExercised")
    base["sstk"] = sstk_total.where(sstk_total.notna(), sstk_parts)
    # dltis / dltr: matched tiers so gross revolver churn cannot enter one side only
    t1i, t1r = base["ProceedsFromIssuanceOfLongTermDebt"], base["RepaymentsOfLongTermDebt"]
    t2i, t2r = base["ProceedsFromIssuanceOfDebt"], base["RepaymentsOfDebt"]
    t3i = s0(base, "ProceedsFromNotesPayable", "ProceedsFromLinesOfCredit",
             "ProceedsFromLongTermLinesOfCredit", "ProceedsFromIssuanceOfSeniorLongTermDebt",
             "ProceedsFromIssuanceOfUnsecuredDebt", "ProceedsFromIssuanceOfSecuredDebt",
             "ProceedsFromConvertibleDebt", "ProceedsFromRelatedPartyDebt")
    t3r = s0(base, "RepaymentsOfNotesPayable", "RepaymentsOfLinesOfCredit",
             "RepaymentsOfLongTermLinesOfCredit", "RepaymentsOfSeniorDebt",
             "RepaymentsOfConvertibleDebt", "RepaymentsOfRelatedPartyDebt",
             "RepaymentsOfSecuredDebt", "RepaymentsOfUnsecuredDebt")
    tier = pd.Series("none", index=base.index)
    dltis = pd.Series(np.nan, index=base.index)
    dltr = pd.Series(np.nan, index=base.index)
    for nm, i_, r_ in (("granular", t3i, t3r), ("debt", t2i, t2r), ("longterm", t1i, t1r)):
        take = i_.notna() | r_.notna()
        dltis = dltis.where(~take, i_.fillna(0.0))
        dltr = dltr.where(~take, r_.fillna(0.0))
        tier = tier.where(~take, nm)
    base["dltis"], base["dltr"], base["debt_flow_tier"] = dltis, dltr, tier
    log("dltis/dltr tier mix: %s" % tier.value_counts().to_dict())

    # A company that filed an annual cash-flow statement but did not tag a payout
    # line did not make that payment. Missing -> 0, gated on the anchor existing.
    for c in ("dvc", "dv", "prstkc", "sstk", "dltis", "dltr"):
        n_missing = base[c].isna().sum()
        base[c] = base[c].fillna(0.0).abs() if c in ("dvc", "dv", "prstkc", "dltr") else base[c].fillna(0.0)
        log("   %-6s missing->0 for %d issuers (%.1f%%)" % (c, n_missing, 100 * n_missing / len(base)))
    base["sstk"] = base["sstk"].abs()
    base["dltis"] = base["dltis"].abs()

    # ---------------- balance sheet ----------------------------------------
    assets = panel_concept(fund, "Assets", annual=False)
    assets = assets[assets.filed_date <= ASOF][["cik", "period_end", "val"]]
    base = asof_join(base, assets, "fy_end", 20, "at")
    base["fy_end_lag"] = base.fy_end - pd.Timedelta(days=365)
    base = asof_join(base, assets, "fy_end_lag", 45, "at_lag")
    eq = panel_concept(fund, "StockholdersEquity", annual=False)
    eq = eq[eq.filed_date <= ASOF][["cik", "period_end", "val"]]
    base = asof_join(base, eq, "fy_end", 20, "ceq")
    base = base.merge(mk[["cik", "mktcap", "shares", "sh_src"]], on="cik", how="left")
    log("balance sheet: at=%d at_lag=%d ceq=%d mktcap=%d (of %d)"
        % (base["at"].notna().sum(), base.at_lag.notna().sum(), base.ceq.notna().sum(),
           base.mktcap.notna().sum(), len(base)))

    base = base[base.mktcap >= MIN_MKTCAP].copy()
    log("after $%.0fM market-cap floor: %d issuers" % (MIN_MKTCAP / 1e6, len(base)))

    # =================================================== 1. NetPayoutYield
    npy = base.copy()
    npy["value"] = (npy.dvc + npy.prstkc - npy.sstk) / npy.mktcap
    npy = npy[npy.ceq > 0]
    write_signal(
        "NetPayoutYield", npy[["ticker", "value"]], +1,
        "Dividends (dvc) plus purchase of common and preferred stock (prstkc) minus sale of common "
        "and preferred stock (sstk), divided by market value of equity lagged 6 months. Exclude if "
        "NetPayoutYield is 0, financial firm based on SIC code, ceq <= 0, or less than 2 years in CRSP.",
        """`dvc` = `PaymentsOfDividendsCommonStock`, else `PaymentsOfDividends` - `PaymentsOfDividendsPreferredStockAndPreferenceStock` - `PaymentsOfDividendsMinorityInterest`, else 0.
`prstkc` = `PaymentsForRepurchaseOfCommonStock` + `PaymentsForRepurchaseOfPreferredStockAndPreferenceStock`, else `PaymentsForRepurchaseOfEquity`, else 0.
`sstk` = `ProceedsFromIssuanceOrSaleOfEquity` (explicit total) else the sum of `ProceedsFromIssuanceOfCommonStock` + `ProceedsFromIssuanceOfPrivatePlacement` + `ProceedsFromIssuanceOfPreferredStockAndPreferenceStock` + `ProceedsFromIssuanceOfWarrants` + `ProceedsFromStockOptionsExercised`.
`mve` = latest cover-page shares outstanding (`dei:EntityCommonStockSharesOutstanding`, else `us-gaap:CommonStockSharesOutstanding`, else `us-gaap:CommonStockSharesIssued`) x last close 2026-08-17.
`ceq` proxied by `StockholdersEquity` at the fiscal year end.""",
        """1. **Market value of equity is measured today, not lagged 6 months.** OSAP lags it to line
   up with a historical Compustat data release. We are scoring today's cross-section, so the
   current market cap is the correct and more accurate denominator; there is no lookahead in it.
2. **Zero-payout firms are KEPT.** OSAP drops `NetPayoutYield == 0`. Dropping them here would
   discard a large, economically meaningful block (firms that neither return nor raise capital)
   and would leave `rank_pct` defined on a self-selected subset. Zeros sit exactly where they
   belong in the ordering: below payers, above issuers.
3. **Financial firms are KEPT.** OSAP drops them by SIC code; the SEC frames API carries no SIC
   and getting one would cost a per-company `submissions` call for every issuer. Banks and
   insurers are large, genuine net-payout names, so excluding them is a portfolio-construction
   choice rather than a data-quality one. **This is an unresolved deviation** - if the composite
   later wants the OSAP-faithful version, SIC must be sourced separately.
4. `ceq > 0` is enforced with `StockholdersEquity`, which includes preferred equity; Compustat
   `ceq` is common equity only. The two differ only for preferred-heavy issuers.
5. Sign conventions differ across filers, so all payout legs are taken as absolute values -
   a few filers tag repurchases as negative cash flows and a few as positive.""")

    # =================================================== 2/3. ShareIss
    for nm, yrs, tol in (("ShareIss1Y", 1, 120), ("ShareIss5Y", 5, 200)):
        g = share_growth(uni, fund, splits, yrs, tol)
        g = g.merge(base[["cik", "ticker"]], on="cik", how="inner")
        log("%s source mix: %s" % (nm, g.src.value_counts().to_dict()))
        write_signal(
            nm, g[["ticker", "value"]], -1,
            ("Growth in number of shares between t-18 and t-6. Number of shares is calculated as "
             "shrout/cfacshr to adjust for splits."
             if yrs == 1 else
             "5-year growth in number of shares. Number of shares is calculated as shrout/cfacshr "
             "to adjust for splits."),
            f"""Shares from, in priority order, `dei:EntityCommonStockSharesOutstanding` (cover page, the closest analogue to CRSP `shrout`), then `us-gaap:CommonStockSharesOutstanding`, then `us-gaap:CommonStockSharesIssued`.
Both ends of the ratio are forced to come from the **same** concept - a level difference between two share concepts is indistinguishable from issuance.
Split adjustment (`cfacshr` analogue) from Yahoo split events (`data/prices/splits_yf.csv`) unioned with the {len(pd.read_csv(os.path.join(ROOT, 'data/prices/splits_repaired.csv')))} Yahoo-unapplied splits already detected in `data/prices/splits_repaired.csv`.
Source mix used: {g.src.value_counts().to_dict()}""",
            f"""1. **Split adjustment is the whole ballgame here.** `dei:EntityCommonStockSharesOutstanding`
   is the as-reported cover-page count and is NOT retroactively restated for splits. Unadjusted,
   a 1-for-10 reverse split reads as a 90% share retirement and lands at the very top of the buy
   list - and reverse splits concentrate in exactly the distressed microcaps this signal should
   be shorting. Every historical share count is multiplied by the product of split ratios that
   occurred after it, putting it on today's basis.
2. **Window is t to t-{yrs}y, not OSAP's t-6m to t-{12 * yrs + 6}m.** OSAP lags by 6 months to
   emulate a Compustat release delay. Cover-page share counts are public the day the filing hits
   EDGAR, so using the freshest observation adds information without any lookahead.
3. Observations are matched to the target date by nearest balance-sheet date within
   +/-{tol} days; issuers with no observation in that window are dropped rather than extrapolated.
4. **Multi-class issuers.** `dei:EntityCommonStockSharesOutstanding` is absent for issuers that
   tag the cover page per share class (Alphabet, Fox, Berkshire). The `us-gaap` fallbacks recover
   most of them at the consolidated level; Berkshire tags every share concept per class and is
   dropped entirely.
5. One row per issuer (most liquid share class), so a dual-class company is scored once.""")

    # =================================================== 4. NetEquityFinance
    nef = base.copy()
    nef["avg_at"] = (nef["at"] + nef["at_lag"]) / 2.0
    nef["value"] = (nef.sstk - nef.prstkc) / nef.avg_at
    n_before = nef.value.notna().sum()
    nef = nef[nef.value.abs() <= 1.0]
    write_signal(
        "NetEquityFinance", nef[["ticker", "value"]], -1,
        "Sale of common stock (sstk) minus purchase of common stock (prstkc), scaled by average "
        "total assets (at) from years t and t-1. Exclude if absolute value of ratio is greater than 1.",
        """`sstk` and `prstkc` exactly as in NetPayoutYield (same anchor fiscal year).
`at` = `us-gaap:Assets` at the fiscal year end; `at_{t-1}` = `Assets` at the nearest balance-sheet
date within 45 days of one year earlier. Denominator is the average of the two.""",
        f"""1. The OSAP `|ratio| > 1` exclusion is applied literally: {n_before - len(nef)} issuers dropped.
   Those are almost all microcaps that issued more equity in one year than they hold in assets.
2. Both `at` and `at_{{t-1}}` are required - no single-year fallback, because scaling a two-year
   flow by a one-year asset base changes the units.
3. Missing payout tags are read as zero, gated on the issuer having actually filed an annual
   cash-flow statement for the anchor year (its own `NetCashProvidedByUsedInOperatingActivities`
   annual figure). A company that filed a cash-flow statement and did not tag stock issuance did
   not issue stock; this is how Compustat treats the same absence.""")

    # =================================================== 5. XFIN
    xf = base.copy()
    xf["value"] = (xf.sstk - xf.dv - xf.prstkc + xf.dltis - xf.dltr) / xf["at"]
    write_signal(
        "XFIN", xf[["ticker", "value"]], -1,
        "Sale of common stock (sstk) minus dividends (dv) minus purchase of common stock (prstkc) "
        "plus long-term debt issuance (dltis) minus long-term debt reductions (dltr). Scaled by "
        "total assets (at).",
        """`sstk`, `prstkc` as in NetPayoutYield. `dv` = `PaymentsOfDividends`, else the sum of the
common + preferred + minority-interest dividend tags.
`dltis`/`dltr` are drawn from **matched tiers**, tried in order and never mixed:
(1) `ProceedsFromIssuanceOfLongTermDebt` / `RepaymentsOfLongTermDebt`;
(2) `ProceedsFromIssuanceOfDebt` / `RepaymentsOfDebt`;
(3) the granular sums - `ProceedsFromNotesPayable` + `ProceedsFromLinesOfCredit` +
`ProceedsFromLongTermLinesOfCredit` + `ProceedsFromIssuanceOfSeniorLongTermDebt` +
`ProceedsFromIssuanceOfUnsecuredDebt` + `ProceedsFromIssuanceOfSecuredDebt` +
`ProceedsFromConvertibleDebt` + `ProceedsFromRelatedPartyDebt`, against the matching
`RepaymentsOf*` sums.
`at` = `us-gaap:Assets` at the fiscal year end.""",
        f"""1. **Issuance and repayment must come from the same tier.** Revolver draws and repayments are
   gross flows that can each be many times total assets and they very nearly cancel. Taking
   proceeds from the granular tier while repayments came from the long-term tier would leave a
   huge one-sided number. Tier mix actually used: {tier.value_counts().to_dict()}.
2. `dltis`/`dltr` include line-of-credit flows in tier 3. Compustat's `dltis` is long-term only
   and would push short-term revolver churn into `dlcch`; because the tier is matched on both
   sides the churn nets out, but the gross numbers are larger than Compustat's for revolver-heavy
   issuers.
3. Scaled by `at` at the fiscal year end (not the two-year average) - that is what the
   definition specifies, and it differs from NetEquityFinance on purpose.
4. Missing flow tags read as zero, gated on the annual cash-flow-statement anchor (see
   NetEquityFinance note 3).""")

    # =================================================== 6. CompositeDebtIssuance
    cdi_base = base[["cik", "ticker", "fy_end"]].copy()
    d_now = debt_at(fund, cdi_base.cik, cdi_base.fy_end)
    d_old = debt_at(fund, cdi_base.cik, cdi_base.fy_end - pd.Timedelta(days=int(round(365.25 * 5))))
    cdi = (cdi_base.merge(d_now.rename(columns={"debt": "d_now", "recipe": "r_now",
                                                "obs_date": "dt_now"}), on="cik", how="left")
                   .merge(d_old.rename(columns={"debt": "d_old", "recipe": "r_old",
                                                "obs_date": "dt_old"}), on="cik", how="left"))
    same = cdi.r_now == cdi.r_old
    log("CompositeDebtIssuance: recipe available now=%d then=%d, same recipe both ends=%d"
        % ((cdi.r_now != "none").sum(), (cdi.r_old != "none").sum(), same.sum()))
    log("  recipe mix now: %s" % cdi.r_now.value_counts().to_dict())
    cdi = cdi[same & (cdi.d_now > 0) & (cdi.d_old > 0)].copy()
    cdi["value"] = np.log(cdi.d_now) - np.log(cdi.d_old)
    write_signal(
        "CompositeDebtIssuance", cdi[["ticker", "value"]], -1,
        "Log of long-term debt (dltt) plus debt in current liabilties (dlc) minus log of the same "
        "variable 5 years ago.",
        """Total debt = `dltt + dlc`, built from the first recipe available at BOTH ends:
(a) `DebtLongtermAndShorttermCombinedAmount`;
(b) `LongTermDebt` (includes current maturities) + short-term borrowings;
(c) `LongTermDebtNoncurrent` (else `LongTermDebtAndCapitalLeaseObligations`, `LongTermNotesPayable`,
`ConvertibleDebtNoncurrent`) + current debt;
(d) current debt only.
Current debt = `LongTermDebtCurrent` (else `LongTermDebtAndCapitalLeaseObligationsCurrent`, else
`DebtCurrent`) + `ShortTermBorrowings` + `OtherShortTermBorrowings` + `NotesPayableCurrent` +
`LinesOfCreditCurrent`.""",
        f"""1. **This is the weakest-covered signal in the group and the honest reason is XBRL tagging.**
   `LongTermDebtNoncurrent` exists for only ~2,550 tickers across all frames, and `dlc` has no
   single dominant tag at all. Coverage below is what survives requiring a positive debt balance
   at *both* ends five years apart.
2. **The same recipe is required at both ends.** A firm that tagged `LongTermDebtNoncurrent` in
   2021 and `LongTermDebt` in 2026 would otherwise show a debt jump that is pure tag migration.
   {int(same.sum())} of {len(cdi_base)} issuers clear that test.
3. **Zero-debt firms are dropped, not scored.** `log(0)` is undefined and OSAP/Lyandres treat the
   ratio as undefined too. This biases the scored universe toward levered firms - debt-free
   companies, which the economics would call attractive, simply do not appear. Any composite
   using this signal must handle its absence rather than impute a neutral rank.
4. Balance-sheet dates are matched to the fiscal year end and to five years before it, nearest
   within 100 days.""")

    log("\nDONE")


if __name__ == "__main__":
    main()
