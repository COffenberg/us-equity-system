#!/usr/bin/env python3
"""
Valuation & profitability signal group.

Definitions taken verbatim from data/osap/SignalDoc.csv, 'Detailed Definition':

    roaq  "Quarterly net income (ibq) divided by lagged total assets (atq)."
    cfp   "Operating cash-flow (oancf) divided by market value of equity. ..."
    SP    "Ratio of annual sales (sale) to market value of equity."

All three carry Sign = +1 in SignalDoc, so higher value == more attractive, and `rank_pct`
is an ascending cross-sectional percentile (1.0 == most attractive name in the universe).

Inputs -- all pre-existing, nothing here is re-derived:
    data/universe/tickers_2026-08-18.csv      cik -> ticker roster
    data/snapshots/ohlcv_2026-08-18.parquet   latest close (primary price, per spec)
    data/prices/derived_diagnostics.parquet   last_close   (covers the tickers the snapshot
                                              wrongly recorded as no_data, incl. MSFT/TSLA)
    data/prices/derived.parquet               dollar_vol_60d -> liquidity floor
    data/sec/fundamentals.parquet             XBRL frames panel with real filed_date
    data/sec/frames/*.json.gz                 raw frames incl. fetch_valprof_fallbacks.py's
    data/sec/accn_index.parquet               accn -> form, filed_date
    signals/_audit_shares_yf.csv              independent share count (audit_shares.py)

Usage:
    python3 signals/build_valprof.py
    python3 signals/build_valprof.py --no-audit    # SEC only; market caps are then untrusted
"""
import argparse
import gzip
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "signals")
AS_OF = pd.Timestamp("2026-08-18")     # point-in-time gate: nothing filed after this is usable
AUDIT_CSV = os.path.join(SIG, "_audit_shares_yf.csv")

# ---- universe / validity parameters ------------------------------------------------------
MIN_PRICE = 1.00        # OSAP's own roaq filter is abs(prc)>1
MIN_DOLLAR_VOL = 1e6    # median 60d dollar volume. Without a floor the tails of all three
                        # signals fill with shells, de-SPACs and $0.30 tape.
LOOSE_DOLLAR_VOL = 1e5  # reported as a sensitivity only
MIN_SHARES = 1e5        # below this the "issuer" is a wholly-owned financing subsidiary whose
                        # listed line is a baby bond, not equity (Qwest Corp: 1 share)
MAX_SHARE_AGE_D = 400   # a share count from 2023 is not today's share count
MIN_MKTCAP = 2e7        # $20m. Below this, at >=$1m/day of volume, it is a data artifact
MAX_FUNDAMENTAL_AGE_D = 730
AUDIT_TOL = 0.25        # |shares_sec/shares_yf - 1| beyond this => market cap not trusted
MIN_ASSETS = 1e6        # roaq denominator floor
ASSET_JUMP_TOL = 10.0   # lagged vs concurrent total assets must agree within this factor

# Revenue tag priority. `Revenues` is the broadest total-revenue tag and the closest analogue
# of Compustat `sale`; the contract-revenue tags are what most post-ASC606 filers use; the last
# two are the bank / broker top lines.
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "RevenuesNetOfInterestExpense",
    "InterestAndDividendIncomeOperating",
]
# dei cover-page count is the true outstanding count. The weighted-average tags are the fallback
# for multi-class issuers, whose dei fact carries a class axis and is therefore invisible to the
# frames API (Alphabet, Berkshire, Fox, ...). The fallback is noticeably dirtier -- several
# filers tag it in thousands -- so it only survives if the audit confirms it.
SHARE_TAGS_FALLBACK = [
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
]

log = print
NOTES = []          # judgement calls / diagnostics collected for the .md files


def hr(t):
    log("\n" + "=" * 86 + f"\n{t}\n" + "=" * 86)


def fmt(x):
    return f"{x:,.6g}"


# ==========================================================================================
# loaders
# ==========================================================================================
def is_common_equity(t):
    """Warrants / rights / units / preferreds inherit their issuer's fundamentals through the
    CIK join but trade at a completely unrelated price."""
    if "-" in t:
        suf = t.rsplit("-", 1)[1]
        if suf in ("WT", "WS", "UN", "RI", "RT", "U", "W", "R") or suf.startswith("P"):
            return False
    return True


def strip_spac_derivatives(u):
    """SPAC units/warrants/rights list as <base>+U/W/R/Z under the same CIK as the common."""
    drop = set()
    for _, g in u.groupby("cik"):
        tk = set(g.ticker)
        for t in tk:
            if len(t) >= 5 and t[-1] in "UWRZ" and t[:-1] in tk:
                drop.add(t)
    return u[~u.ticker.isin(drop)]


def load_prices():
    px = pd.read_parquet(os.path.join(ROOT, "data/snapshots/ohlcv_2026-08-18.parquet"))
    px["date"] = pd.to_datetime(px["date"])
    snap = (px.dropna(subset=["close"]).sort_values("date").groupby("ticker").tail(1)
              [["ticker", "date", "close"]].rename(columns={"close": "px_snap", "date": "px_date"}))
    dg = pd.read_parquet(os.path.join(ROOT, "data/prices/derived_diagnostics.parquet"))
    dv = pd.read_parquet(os.path.join(ROOT, "data/prices/derived.parquet"))[["ticker", "dollar_vol_60d"]]
    p = (snap.merge(dg[["ticker", "last_close"]], on="ticker", how="outer")
             .merge(dv, on="ticker", how="left"))
    p["price"] = p.px_snap.fillna(p.last_close)
    p["px_source"] = np.where(p.px_snap.notna(), "snapshot", "derived_last_close")
    return p[["ticker", "price", "px_date", "px_source", "dollar_vol_60d"]]


def load_fundamentals():
    f = pd.read_parquet(os.path.join(ROOT, "data/sec/fundamentals.parquet"))
    n0 = len(f)
    # The ticker join in fundamentals.parquet is many-to-many (1,449 CIKs carry >1 ticker), so
    # the same fact appears once per ticker. Everything below is per-CIK; dedupe first.
    f = f.drop_duplicates(subset=["cik", "concept", "period_end", "period_start", "fp", "accn"])
    n1 = len(f)
    # PIT gate on the ACTUAL filing date. The panel's own note warns that an assumed 45-day lag
    # holds for only 45% of rows, so an assumed lag is not usable -- gate on filed_date.
    f = f[f.filed_date.notna() & (f.filed_date <= AS_OF)]
    log(f"  fundamentals: {n0:,} rows -> {n1:,} after ticker-dup collapse -> {len(f):,} after "
        f"PIT gate filed_date <= {AS_OF.date()}")
    return f


def load_extra_frames(concepts, unit):
    """Raw frames cached by fetch_valprof_fallbacks.py, shaped like the fundamentals panel."""
    cache = os.path.join(ROOT, "data/sec/frames")
    rows = []
    for fn in sorted(os.listdir(cache)):
        if not fn.endswith(".json.gz"):
            continue
        parts = fn[: -len(".json.gz")].split("_")
        if len(parts) < 4:
            continue
        concept, u_ = "_".join(parts[1:-2]), parts[-2]
        if concept not in concepts or u_ != unit:
            continue
        with gzip.open(os.path.join(cache, fn), "rt") as fh:
            df = pd.DataFrame(json.load(fh).get("data", []))
        if df.empty:
            continue
        df["concept"] = concept
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    o = pd.concat(rows, ignore_index=True)
    o["cik"] = o.cik.astype(str).str.zfill(10)
    o["period_end"] = pd.to_datetime(o["end"])
    o["period_start"] = pd.to_datetime(o["start"]) if "start" in o else pd.NaT
    ai = pd.read_parquet(os.path.join(ROOT, "data/sec/accn_index.parquet"))
    o = o.merge(ai, on="accn", how="left")
    o["filed_date"] = pd.to_datetime(o.filed_date)
    o = o[o.filed_date.notna() & (o.filed_date <= AS_OF)]
    return o[["cik", "concept", "period_end", "period_start", "val", "accn", "form", "filed_date"]] \
        .drop_duplicates(subset=["cik", "concept", "period_end", "period_start", "accn"])


# ==========================================================================================
# shares outstanding
# ==========================================================================================
def build_shares(f):
    hr("SHARES OUTSTANDING")
    dei = f[(f.concept == "EntityCommonStockSharesOutstanding") & (f.val >= MIN_SHARES)].copy()
    dei = dei.sort_values(["period_end", "filed_date"]).groupby("cik").tail(1)
    dei = dei[["cik", "val", "period_end", "form"]].rename(columns={"val": "shares",
                                                                   "period_end": "shares_asof"})
    dei["shares_src"] = "dei:EntityCommonStockSharesOutstanding"
    log(f"  dei:EntityCommonStockSharesOutstanding (>= {MIN_SHARES:,.0f} shares): {len(dei):,} CIKs")

    extra = load_extra_frames(SHARE_TAGS_FALLBACK, "shares")
    got, add = set(dei.cik), []
    for tag in SHARE_TAGS_FALLBACK:
        s = extra[(extra.concept == tag) & (extra.val >= MIN_SHARES) & (~extra.cik.isin(got))]
        if s.empty:
            continue
        s = s.sort_values(["period_end", "filed_date"]).groupby("cik").tail(1)
        s = s[["cik", "val", "period_end", "form"]].rename(columns={"val": "shares",
                                                                   "period_end": "shares_asof"})
        s["shares_src"] = f"us-gaap:{tag}"
        add.append(s)
        got |= set(s.cik)
        log(f"  fallback us-gaap:{tag:50s} +{len(s):,} CIKs")
    sh = pd.concat([dei] + add, ignore_index=True) if add else dei

    n0 = len(sh)
    sh = sh[(AS_OF - sh.shares_asof).dt.days <= MAX_SHARE_AGE_D]
    log(f"  freshness gate (share count <= {MAX_SHARE_AGE_D}d old): {len(sh):,} CIKs "
        f"({n0 - len(sh):,} dropped as stale)")
    return sh


def apply_audit(base, use_audit):
    """Gate market cap on an independent share count. See audit_shares.py for why."""
    hr("MARKET-CAP AUDIT")
    base["mktcap"] = base.price * base.shares
    base["audit"] = "not_run"
    base["shares_yf"] = np.nan
    base["share_ratio"] = np.nan
    if not use_audit:
        NOTES.append("Audit skipped (--no-audit): market caps are unvalidated.")
        base["mcap_ok"] = base.mktcap.notna() & (base.mktcap >= MIN_MKTCAP)
        return base

    a = pd.read_csv(AUDIT_CSV)[["ticker", "shares_yf"]]
    base = base.drop(columns=["shares_yf"]).merge(a, on="ticker", how="left")
    base["share_ratio"] = base.shares / base.shares_yf

    has_sec = base.shares.notna()
    has_yf = base.shares_yf.notna()
    agree = base.share_ratio.between(1 - AUDIT_TOL, 1 + AUDIT_TOL)

    base.loc[has_sec & ~has_yf, "audit"] = "no_independent_count"
    base.loc[has_sec & has_yf & ~agree, "audit"] = "MISMATCH"
    base.loc[has_sec & has_yf & agree, "audit"] = "agree"
    base.loc[~has_sec, "audit"] = "no_sec_count"
    log(base.audit.value_counts().to_string())

    mis = base[base.audit == "MISMATCH"].copy()
    mis["foreign"] = mis.form.astype(str).str.startswith(("20-F", "40-F"))
    log(f"\n  {len(mis):,} mismatches; {int(mis.foreign.sum()):,} are foreign-private-issuer "
        f"filings (20-F/40-F -> ADR ratio), {int((~mis.foreign).sum()):,} domestic "
        f"(mostly a share count tagged in thousands).")
    log("\n  20 largest FAKE market caps caught by the audit:")
    show = mis.nlargest(20, "mktcap")[["ticker", "title", "form", "price", "shares", "shares_yf",
                                       "share_ratio", "mktcap"]]
    log(show.to_string(index=False, float_format=fmt))

    nof = base[base.audit == "no_independent_count"]
    log(f"\n  {len(nof):,} lines have an SEC share count but no independent one. These are "
        f"ETFs / commodity + crypto trusts / baby bonds / preferreds / depositary shares -- "
        f"they inherit an operating company's fundamentals through the CIK join but are not "
        f"common equity. Sample: {', '.join(nof.ticker.head(14))}")

    base["mcap_ok"] = (base.audit == "agree") & base.mktcap.notna() & (base.mktcap >= MIN_MKTCAP)
    log(f"\n  market caps trusted: {int(base.mcap_ok.sum()):,} / {len(base):,}   "
        f"(also requires mktcap >= ${MIN_MKTCAP:,.0f})")
    NOTES.append(
        f"Market cap = latest close x latest SEC share count, then gated on an independent "
        f"share count: {int(base.mcap_ok.sum()):,} of {len(base):,} liquid lines survive. "
        f"{len(mis):,} were rejected as MISMATCH (ADR ratios such as BSAC 400:1, LTM 1961:1, "
        f"TSM 5:1, and domestic counts tagged in thousands such as TEM) and {len(nof):,} as "
        f"non-common-equity lines with no independent share count.")
    return base


# ==========================================================================================
# signals
# ==========================================================================================
def build_roaq(f, base):
    """OSAP: "Quarterly net income (ibq) divided by lagged total assets (atq)."

    ibq is income before extraordinary items; the panel carries NetIncomeLoss, the nearest
    available tag (ASU 2015-01 eliminated the extraordinary-items category, so post-2015 the
    two coincide for the overwhelming majority of filers).
    lagged atq = total assets at the balance-sheet date immediately preceding the income
    quarter, i.e. period_end == quarter's period_start - 1 day.
    """
    hr("roaq  =  quarterly NetIncomeLoss / lagged Assets")
    ni = f[(f.concept == "NetIncomeLoss") & (f.fp != "FY") & f.period_start.notna()].copy()
    ni["dur"] = (ni.period_end - ni.period_start).dt.days
    ni = ni[ni.dur.between(80, 100)]                      # a true 3-month stub, not a YTD figure
    ni = ni.sort_values(["period_end", "filed_date"]).groupby("cik").tail(1)
    ni = ni[["cik", "val", "period_start", "period_end"]].rename(
        columns={"val": "ni_q", "period_end": "ni_end", "period_start": "ni_start"})
    log(f"  latest quarterly NetIncomeLoss: {len(ni):,} CIKs   median quarter-end "
        f"{ni.ni_end.median().date()}")

    at = f[(f.concept == "Assets") & (f.val > 0)][["cik", "val", "period_end"]] \
        .rename(columns={"val": "assets", "period_end": "at_end"})

    ni["target"] = ni.ni_start - pd.Timedelta(days=1)
    j = ni.merge(at, on="cik", how="left")
    j["gap"] = (j.at_end - j.target).dt.days.abs()
    j = j[j.gap <= 20].sort_values("gap").groupby("cik").head(1) \
                      .rename(columns={"assets": "assets_lag"})
    log(f"  matched to a lagged Assets balance (|gap| <= 20d): {len(j):,} CIKs")

    # Guard the denominator. Some filers tag a placeholder total-assets value: Yesway reported
    # Assets = 1 for 2026-03-31 between real balances of 1.9bn and 2.1bn, which turns roaq into
    # 1.6e7 and puts the name at the very top of the signal.
    cur = at.rename(columns={"assets": "assets_cur", "at_end": "cur_end"})
    j = j.merge(cur, left_on=["cik", "ni_end"], right_on=["cik", "cur_end"], how="left")
    n0 = len(j)
    j = j[j.assets_lag >= MIN_ASSETS]
    n1 = len(j)
    ratio = j.assets_lag / j.assets_cur
    j = j[ratio.isna() | ratio.between(1 / ASSET_JUMP_TOL, ASSET_JUMP_TOL)]
    log(f"  denominator guards: {n0:,} -> {n1:,} (assets_lag >= ${MIN_ASSETS:,.0f}) "
        f"-> {len(j):,} (lagged vs concurrent assets within {ASSET_JUMP_TOL:g}x)")
    NOTES.append(
        f"roaq denominator guards removed {n0 - len(j):,} CIKs whose lagged total assets were a "
        f"placeholder or a pre-IPO shell balance (Yesway tagged Assets = 1 between real balances "
        f"of $1.9bn and $2.1bn, which scored roaq = 1.6e7 and ranked #1 before the guard).")

    j["value"] = j.ni_q / j.assets_lag
    j = j[np.isfinite(j.value)]
    d = base.merge(j[["cik", "value", "ni_end", "assets_lag"]], on="cik", how="inner")
    return d[(AS_OF - d.ni_end).dt.days <= MAX_FUNDAMENTAL_AGE_D]


def build_cfp(f, base):
    """OSAP: "Operating cash-flow (oancf) divided by market value of equity."

    Cash-flow statements are filed year-to-date, so the quarterly calendar frames only ever
    populate Q1; the annual CY frames (fp == 'FY') are the usable ones, and they do carry
    non-calendar fiscal years (period_end is spread across all 12 months).
    """
    hr("cfp  =  annual NetCashProvidedByUsedInOperatingActivities / market cap")
    o = f[(f.concept == "NetCashProvidedByUsedInOperatingActivities") & (f.fp == "FY")].copy()
    o["dur"] = (o.period_end - o.period_start).dt.days
    o = o[o.dur.between(330, 400)]
    o = o.sort_values(["period_end", "filed_date"]).groupby("cik").tail(1)
    o = o[["cik", "val", "period_end"]].rename(columns={"val": "ocf", "period_end": "ocf_end"})
    o = o[(AS_OF - o.ocf_end).dt.days <= MAX_FUNDAMENTAL_AGE_D]
    log(f"  latest annual operating cash flow: {len(o):,} CIKs   median FY-end "
        f"{o.ocf_end.median().date()}")

    d = base.merge(o, on="cik", how="inner")
    d["value"] = d.ocf / d.mktcap
    return d[np.isfinite(d.value)]


def build_sp(f, base):
    """OSAP: "Ratio of annual sales (sale) to market value of equity." """
    hr("SP  =  annual revenue / market cap")
    have = f[f.concept.isin(REVENUE_TAGS) & (f.fp == "FY")][
        ["cik", "concept", "val", "period_end", "period_start", "filed_date"]].copy()
    extra = load_extra_frames(REVENUE_TAGS, "USD")
    if not extra.empty:
        have = pd.concat([have, extra[["cik", "concept", "val", "period_end", "period_start",
                                       "filed_date"]]], ignore_index=True)
    have = have[have.period_start.notna() & have.val.notna()]
    have["dur"] = (have.period_end - have.period_start).dt.days
    have = have[have.dur.between(330, 400)]
    have = have[(AS_OF - have.period_end).dt.days <= MAX_FUNDAMENTAL_AGE_D]
    have["prio"] = have.concept.map({c: i for i, c in enumerate(REVENUE_TAGS)})
    # newest fiscal year first, then the broadest top-line tag available for that year
    have = have.sort_values(["period_end", "prio"], ascending=[False, True]).groupby("cik").head(1)
    log("  revenue tag chosen per CIK:")
    for c, n in have.concept.value_counts().items():
        log(f"    {c:55s} {n:6,}")
    log(f"  latest annual revenue: {len(have):,} CIKs   median FY-end {have.period_end.median().date()}")

    have = have[["cik", "val", "period_end", "concept"]].rename(
        columns={"val": "sales", "period_end": "sales_end", "concept": "rev_tag"})
    d = base.merge(have, on="cik", how="inner")
    d["value"] = d.sales / d.mktcap
    return d[np.isfinite(d.value)]


# ==========================================================================================
# rank, report, write
# ==========================================================================================
def finalise(name, d):
    """Winsorise 1/99 then rank cross-sectionally. Ranks are monotone so winsorising cannot
    reorder anything; what it does is collapse the extreme 1% of each tail onto a single
    clipped value, which is the point -- no single microcap ratio can dominate the composite."""
    d = d.dropna(subset=["value"]).copy()
    lo, hi = d.value.quantile(0.01), d.value.quantile(0.99)
    d["value"] = d.value.clip(lo, hi)
    d["rank_pct"] = d.value.rank(pct=True, method="average")
    out = d[["ticker", "value", "rank_pct"]].sort_values("rank_pct", ascending=False) \
                                            .reset_index(drop=True)
    p = os.path.join(SIG, f"{name}.parquet")
    out.to_parquet(p, index=False)
    log(f"\n  [{name}] coverage {len(out):,} tickers | winsorised to [{lo:.5g}, {hi:.5g}] -> {p}")
    return d, lo, hi


def report(name, d, cols):
    log(f"\n  --- {name}: TOP 10 (most attractive) ---")
    log(d.nlargest(10, "value")[cols].to_string(index=False, float_format=fmt))
    log(f"\n  --- {name}: BOTTOM 10 ---")
    log(d.nsmallest(10, "value")[cols].to_string(index=False, float_format=fmt))


def write_md(name, formula, concepts, d, lo, hi, notes, cols):
    top = d.nlargest(5, "value")[cols]
    bot = d.nsmallest(5, "value")[cols]

    def tbl(t):
        hdr = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
        for r in t.itertuples(index=False):
            cells = [f"{v:,.6g}" if isinstance(v, (int, float, np.floating)) else str(v) for v in r]
            hdr += "| " + " | ".join(cells) + " |\n"
        return hdr

    md = f"""# {name}

**Formula**

```
{formula}
```

OSAP `SignalDoc.csv` 'Detailed Definition' (verbatim):

> {OSAP_DEF[name]}

Sign = +1, so a higher value is more attractive. `rank_pct` is the ascending cross-sectional
percentile within this file's universe; `rank_pct` = 1.0 is the most attractive name.

**XBRL concepts used**

{concepts}

**Coverage: {len(d):,} tickers.**

Universe: SEC roster 2026-08-18, common-equity lines only (warrants / rights / units /
preferreds and SPAC derivative tickers removed), latest close > ${MIN_PRICE:.2f}, median 60-day
dollar volume >= ${MIN_DOLLAR_VOL:,.0f}, and — for the two market-cap ratios — a market cap that
passed the independent share-count audit.

Winsorised at 1% / 99% before ranking: value clipped to [{lo:.5g}, {hi:.5g}].

**Top 5**

{tbl(top)}
**Bottom 5**

{tbl(bot)}
**Judgement calls**

{notes}
"""
    p = os.path.join(SIG, f"{name}.md")
    open(p, "w").write(md)
    log(f"  wrote {p}")


OSAP_DEF = {
    "roaq": "Quarterly net income (ibq) divided by lagged total assets (atq).",
    "cfp": "Operating cash-flow (oancf) divided by market value of equity. If operating "
           "cash-flow is missing, replace by difference betwee net income (ib) and level of "
           "accruals, where the latter is the annual change in current assets (act) minus the "
           "annual change in cash and short-term investments (che), minus the annual change in "
           "current liabilities (lct) plus the annual change in debt in current liabilities "
           "(dlc) plus the annual change in payable income taxes (txp) plus depreciation (dp).",
    "SP": "Ratio of annual sales (sale) to market value of equity.",
}


# ==========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    hr("UNIVERSE / PRICES")
    u = pd.read_csv(os.path.join(ROOT, "data/universe/tickers_2026-08-18.csv"), dtype={"cik": str})
    u = u.dropna(subset=["ticker"]).copy()
    u["ticker"] = u.ticker.str.strip().str.upper()
    log(f"  SEC roster: {len(u):,} tickers / {u.cik.nunique():,} CIKs")
    u = u[u.ticker.map(is_common_equity)]
    log(f"  drop -WT/-UN/-RI/-P* lines            -> {len(u):,}")
    u = strip_spac_derivatives(u)
    log(f"  drop SPAC unit/warrant/right siblings -> {len(u):,}")

    base = u.merge(load_prices(), on="ticker", how="inner")
    log(f"  with a price                          -> {len(base):,}   "
        f"(price date {base.px_date.max().date()}, "
        f"{int((base.px_source == 'derived_last_close').sum()):,} via derived last_close)")
    base = base[base.price > MIN_PRICE]
    log(f"  price > ${MIN_PRICE:.2f}                         -> {len(base):,}")
    loose = int((base.dollar_vol_60d >= LOOSE_DOLLAR_VOL).sum())
    base = base[base.dollar_vol_60d >= MIN_DOLLAR_VOL]
    log(f"  dollar_vol_60d >= ${MIN_DOLLAR_VOL:,.0f}            -> {len(base):,}   "
        f"(a looser ${LOOSE_DOLLAR_VOL:,.0f} floor would give {loose:,})")

    f = load_fundamentals()
    base = base.merge(build_shares(f), on="cik", how="left")
    log(f"\n  liquid lines with an SEC share count: {int(base.shares.notna().sum()):,} / {len(base):,}")
    base = apply_audit(base, use_audit=not args.no_audit)

    hr("TOP 20 BY MARKET CAP  (must read as the actual largest US companies)")
    t = base[base.mcap_ok].nlargest(20, "mktcap")[
        ["ticker", "title", "price", "shares", "shares_src", "mktcap", "audit"]].copy()
    t["mktcap_$bn"] = (t.mktcap / 1e9).round(1)
    t["shares_src"] = t.shares_src.str.replace("us-gaap:WeightedAverageNumberOf", "wavg:", regex=False) \
                                  .str.replace("dei:EntityCommonStockSharesOutstanding", "dei:cover", regex=False)
    log(t.drop(columns=["mktcap"]).to_string(index=False, float_format=fmt))

    b_mc = base[base.mcap_ok]
    cols_mc = ["ticker", "title", "value", "mktcap", "dollar_vol_60d"]
    cols_r = ["ticker", "title", "value", "assets_lag", "dollar_vol_60d"]

    # ---- roaq. Does not need a market cap, but does need the line to be an operating company:
    # ETFs and commodity trusts inherit a "NetIncomeLoss" that is mark-to-market P&L. The audit's
    # eligibility flag is the cleanest available filter for that, so all three share one universe.
    roaq = build_roaq(f, b_mc)
    log(f"  coverage: {len(roaq):,} tickers")
    report("roaq", roaq, cols_r)
    d, lo, hi = finalise("roaq", roaq)
    write_md("roaq", "roaq = NetIncomeLoss[latest 3-month quarter] / Assets[quarter start - 1 day]",
             "- `us-gaap:NetIncomeLoss` — quarterly calendar frames, duration 80–100 days\n"
             "- `us-gaap:Assets` — instantaneous frame at the preceding balance-sheet date",
             d, lo, hi, ROAQ_NOTES, cols_r)

    cfp = build_cfp(f, b_mc)
    log(f"  coverage: {len(cfp):,} tickers")
    report("cfp", cfp, cols_mc)
    d, lo, hi = finalise("cfp", cfp)
    write_md("cfp", "cfp = NetCashProvidedByUsedInOperatingActivities[latest fiscal year] / market cap",
             "- `us-gaap:NetCashProvidedByUsedInOperatingActivities` — annual `CY####` frames "
             "(`fp == 'FY'`), duration 330–400 days\n"
             "- `dei:EntityCommonStockSharesOutstanding` (+ weighted-average fallback) x latest close",
             d, lo, hi, CFP_NOTES, cols_mc)

    sp = build_sp(f, b_mc)
    log(f"  coverage: {len(sp):,} tickers")
    report("SP", sp, cols_mc)
    d, lo, hi = finalise("SP", sp)
    write_md("SP", "SP = annual revenue[latest fiscal year] / market cap",
             "- `us-gaap:Revenues`, else `RevenueFromContractWithCustomerExcludingAssessedTax`, "
             "else `...IncludingAssessedTax`, else `RevenuesNetOfInterestExpense`, else "
             "`InterestAndDividendIncomeOperating` — annual frames, duration 330–400 days\n"
             "- `dei:EntityCommonStockSharesOutstanding` (+ weighted-average fallback) x latest close",
             d, lo, hi, SP_NOTES, cols_mc)

    base.to_parquet(os.path.join(SIG, "_valprof_base.parquet"), index=False)
    hr("NOTES")
    for n in NOTES:
        log("  * " + n)


ROAQ_NOTES = """1. **`ibq` is not directly taggable.** Compustat's `ibq` is income before extraordinary
   items; the frames panel carries `NetIncomeLoss`. ASU 2015-01 eliminated the
   extraordinary-items category in 2015, so for current filings the two coincide for the
   overwhelming majority of issuers. This is the one substantive deviation from the OSAP
   definition.
2. **"Lagged" total assets** is read as the balance-sheet date immediately preceding the income
   quarter (`period_end == quarter period_start − 1 day`, matched within ±20 days), not the
   same-quarter balance.
3. **Denominator guards.** Some filers tag a placeholder total-assets value — Yesway (YSWY)
   reported `Assets = 1` for 2026-03-31 sitting between real balances of $1.95bn and $2.07bn,
   which produced roaq = 1.6e7 and ranked #1 in the first build. Rows are dropped unless
   lagged assets are ≥ $1m and within 10× of the concurrent quarter-end balance.
4. **Calendar-Q4 net income is sparse** (≈2,700 tickers vs ≈6,600 for Q1–Q3) because 10-Ks tag
   the annual figure rather than the Q4 stub. Taking the latest *available* 3-month stub per
   CIK rather than a fixed calendar quarter avoids that hole; the median quarter used is
   2026-06-30.
5. roaq needs no market cap, but it is computed on the same audited universe as `cfp`/`SP`.
   Without that, ETFs and commodity trusts enter with a "net income" that is mark-to-market
   P&L on fund assets — Amplify Commodity Trust and Grayscale Zcash Trust both ranked top-10
   before the eligibility gate."""

CFP_NOTES = """1. **Annual, not quarterly, cash flow.** Cash-flow statements are filed year-to-date, so the
   SEC's quarterly calendar frames only ever populate Q1 (5,837 tickers in Q1 vs ~600 in
   Q2/Q3). The annual `CY####` frames are the usable series and they do carry non-calendar
   fiscal years — `period_end` is spread across all twelve months. OSAP's Portfolio Period for
   `cfp` is 12 months, so an annual input is faithful to the definition anyway.
2. **The accruals fallback in the OSAP definition was not implemented.** It requires `dlc`
   (debt in current liabilities) and `txp` (income taxes payable), neither of which is in the
   fundamentals panel. It would have added 0 tickers: every name in this universe with a
   market cap and no operating-cash-flow figure also lacks the fallback's inputs.
3. **Market value of equity is today's**, not the fiscal-year-end value — the system scores the
   current cross-section, so the current price is the correct denominator.
4. **Negative values are real and are kept.** Roughly a fifth of the universe burns cash; those
   names sit at the unattractive end of the rank, which is the intended behaviour."""

SP_NOTES = """1. **Revenue tag priority.** Compustat's `sale` has no single XBRL equivalent. `Revenues` is
   taken first as the broadest total-revenue tag, then the ASC 606 contract-revenue tags, then
   the bank/broker top lines. The tag actually used is reported in the build log; roughly half
   the universe resolves on `Revenues` and the rest mostly on
   `RevenueFromContractWithCustomerExcludingAssessedTax`.
2. **`RevenueFromContractWithCustomerExcludingAssessedTax` can understate the top line** for
   issuers with material non-contract revenue (interest, lease, investment income) that also
   fail to tag `Revenues`. Those names score a slightly low SP. There is no free way to detect
   this per filer.
3. **`SalesRevenueNet` is gone** — the SEC frames API returns 404 for it for CY2024/CY2025; the
   tag was deprecated with ASC 606 and no current filer uses it.
4. **Market value of equity is today's**, matching `cfp`.
5. Financial issuers whose revenue nets out fair-value losses can print negative annual
   revenue (Rithm Property Trust). Those are left in at the unattractive end rather than
   nulled — a negative top line is information."""


if __name__ == "__main__":
    main()
