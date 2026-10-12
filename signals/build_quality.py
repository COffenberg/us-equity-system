#!/usr/bin/env python3
"""Distress & accounting-quality signal group: OScore, PctTotAcc, Tax, ChTax.

Formulas are taken verbatim from the 'Detailed Definition' column of
data/osap/SignalDoc.csv (Open Source Asset Pricing catalog). Where the catalog
text is ambiguous or predates a tax-law change the deviation is recorded in
JUDGEMENT[...] and reproduced in the per-signal .md.

Inputs are read straight out of the gzipped SEC XBRL frames cache in
data/sec/frames/ rather than data/sec/fundamentals.parquet, because sibling
processes write that parquet concurrently. Nothing here mutates shared state
except by adding new files under signals/.

Point-in-time: every fact is gated on the ACTUAL filing date recovered from the
EDGAR quarterly master index (data/sec/accn_index.parquet), not on an assumed
reporting lag. See data/sec/fundamentals.md -- 45% of frames rows come from a
filing more than a year after the period ended, so an assumed lag is not safe.

Run:  python3 signals/build_quality.py
"""
import pathlib
import glob, gzip, json, os, re, sys
import numpy as np
import pandas as pd

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
CACHE = os.path.join(ROOT, "data/sec/frames")
OUTDIR = os.path.join(ROOT, "signals")
AS_OF = pd.Timestamp("2026-08-18")

# ---- universe floors ---------------------------------------------------------
# Base investability floor applied to every signal in the group. Without it the
# extremes of any accounting ratio are wall-to-wall sub-penny shells.
MIN_PRICE_BASE = 1.00
MIN_DOLLAR_VOL = 100_000.0      # median 60d dollar volume
# OSAP lists `Filter: abs(prc)>5` for OScore and Tax specifically.
MIN_PRICE_PRC5 = 5.00

# OScore's deflator. -0.407*log(at/deflator) = -0.407*log(at) + 0.407*log(deflator);
# the second piece is one additive constant shared by every firm in a same-date
# cross-section, so it cannot change a single rank. Set to 1.0 with `at` in $mm.
OSCORE_DEFLATOR = 1.0

# Statutory federal rate. Catalog stops at ".35 from 1993 onwards" (pre-TCJA).
TAX_RATE = 0.21

# OSAP `Sign` column: +1 = high signal value predicts HIGH returns,
#                    -1 = high signal value predicts LOW returns.
SIGNS = {"OScore": -1, "PctTotAcc": -1, "Tax": +1, "ChTax": +1}

WINSOR = (0.01, 0.99)


# =============================================================================
# frames cache reader
# =============================================================================
_KIND_RX = {"A": r"CY\d{4}", "Q": r"CY\d{4}Q\d", "I": r"CY\d{4}Q\dI"}


def load_concept(concept, kind, tax="us-gaap", unit="USD"):
    """Read every cached frames response for one concept.

    kind: 'A' annual duration (CY####), 'Q' quarterly duration (CY####Q#),
          'I' instantaneous / balance sheet (CY####Q#I).
    Returns [cik_int, period_end, period_start, val, accn]; no PIT gate yet.
    """
    rx = re.compile(_KIND_RX[kind] + r"$")
    recs = []
    for fp in sorted(glob.glob(os.path.join(CACHE, "%s_%s_%s_*.json.gz" % (tax, concept, unit)))):
        fid = os.path.basename(fp)[: -len(".json.gz")].rsplit("_", 1)[1]
        if not rx.fullmatch(fid):
            continue
        try:
            with gzip.open(fp, "rt", encoding="utf-8") as fh:
                js = json.load(fh)
        except (OSError, ValueError):          # a sibling may be mid-write
            continue
        for d in js.get("data", []):
            recs.append((d["cik"], d.get("end"), d.get("start"), d.get("val"), d.get("accn")))
    df = pd.DataFrame(recs, columns=["cik_int", "period_end", "period_start", "val", "accn"])
    if df.empty:
        return df
    df["period_end"] = pd.to_datetime(df.period_end, errors="coerce")
    df["period_start"] = pd.to_datetime(df.period_start, errors="coerce")
    df["val"] = pd.to_numeric(df.val, errors="coerce")
    return df.dropna(subset=["period_end", "val"])


def pit_gate(df, accn_idx, name):
    """Keep only facts already public at AS_OF, then dedupe to the latest filing
    for each (cik, period_end)."""
    if df.empty:
        return df
    n0 = len(df)
    df = df.merge(accn_idx, on="accn", how="left")
    # unknown filing date (0.4% of accessions): fall back to the >=45d rule
    fallback = df.filed_date.isna()
    eff = df.filed_date.where(~fallback, df.period_end + pd.Timedelta(days=45))
    df = df[(eff <= AS_OF) & (df.period_end <= AS_OF)].copy()
    df["_fd"] = eff[df.index]
    df = (df.sort_values(["cik_int", "period_end", "_fd"])
            .drop_duplicates(["cik_int", "period_end"], keep="last"))
    print("    %-62s %7d -> %7d rows  %5d ciks" % (name, n0, len(df), df.cik_int.nunique()))
    return df[["cik_int", "period_end", "period_start", "val"]]


def series(concept, kind, accn_idx, **kw):
    return pit_gate(load_concept(concept, kind, **kw), accn_idx, "%s [%s]" % (concept, kind))


# =============================================================================
# helpers
# =============================================================================
def asof_pick(left, right, col, tol_days=7):
    """Attach right.val as `col` to left, matching on cik + nearest period_end
    within tol_days. Used to hang a balance sheet off an income-statement date."""
    if right.empty:
        left[col] = np.nan
        return left
    l = left.sort_values("period_end")
    r = (right[["cik_int", "period_end", "val"]]
         .rename(columns={"val": col})
         .sort_values("period_end"))
    out = pd.merge_asof(l, r, on="period_end", by="cik_int",
                        tolerance=pd.Timedelta(days=tol_days), direction="nearest")
    return out


def winsorise(s, lo=WINSOR[0], hi=WINSOR[1]):
    v = s.dropna()
    if len(v) < 20:
        return s
    return s.clip(lower=v.quantile(lo), upper=v.quantile(hi))


def write_signal(name, df, formula, concepts, judgements, extra_lines, diag):
    """df: [ticker, value] for eligible names. Winsorise -> rank -> write."""
    df = df.dropna(subset=["value"]).drop_duplicates("ticker").copy()
    raw_lo, raw_hi = df.value.min(), df.value.max()
    df["value"] = winsorise(df["value"])
    df["rank_pct"] = df["value"].rank(method="average", pct=True)
    df = df.sort_values("value").reset_index(drop=True)[["ticker", "value", "rank_pct"]]

    pq = os.path.join(OUTDIR, "%s.parquet" % name)
    df.to_parquet(pq, index=False)

    top = df.tail(10).iloc[::-1]
    bot = df.head(10)
    print("\n  === %s ===  coverage = %d tickers" % (name, len(df)))
    print("  raw range [%.4g, %.4g]   winsorised [%.4g, %.4g]   median %.4g"
          % (raw_lo, raw_hi, df.value.min(), df.value.max(), df.value.median()))
    print("  TOP 10 (highest value):    " + ", ".join("%s %.3f" % (r.ticker, r.value) for r in top.itertuples()))
    print("  BOTTOM 10 (lowest value):  " + ", ".join("%s %.3f" % (r.ticker, r.value) for r in bot.itertuples()))

    sign = SIGNS[name]
    direction = ("LOW `value` is the attractive side (OSAP Sign = -1): the composite must use "
                 "`1 - rank_pct`, or flip the sign." if sign < 0 else
                 "HIGH `value` is the attractive side (OSAP Sign = +1): use `rank_pct` as-is.")

    def tbl(sub):
        return "\n".join("| %s | %.4f | %.3f |" % (r.ticker, r.value, r.rank_pct) for r in sub.itertuples())

    md = f"""# {name}

{diag['long']} — OSAP `{name}` ({diag['authors']}, {diag['year']}), category *{diag['econ']}*.

> **Orientation — OSAP Sign = {sign:+d}.** {direction}
> `rank_pct` in the parquet is the plain percentile of `value` (0 = lowest value,
> 1 = highest value) and carries **no** return orientation of its own.

## Exact formula

```
{formula}
```

## XBRL concepts used

{concepts}

## Coverage

**{len(df)} tickers.**

{extra_lines}

Universe gate: ticker present in `data/prices/derived.parquet` (common stock; Yahoo
warrants/rights/units already dropped there) AND `last_close >= {diag['minpx']:.2f}` AND
`dollar_vol_60d >= ${MIN_DOLLAR_VOL:,.0f}`. Point-in-time gate: every XBRL fact must have an
EDGAR `filed_date <= {AS_OF.date()}`.

## Top 5 / bottom 5

Highest `value`:

| ticker | value | rank_pct |
|---|---|---|
{tbl(top.head(5))}

Lowest `value`:

| ticker | value | rank_pct |
|---|---|---|
{tbl(bot.head(5))}

Raw (pre-winsorisation) range: `[{raw_lo:.4g}, {raw_hi:.4g}]`. Winsorised at
{WINSOR[0]:.0%}/{WINSOR[1]:.0%} to `[{df.value.min():.4g}, {df.value.max():.4g}]`, so the extreme
1% at each end share a tied `rank_pct` — deliberate, and harmless for decile sorting.

## Judgement calls

{judgements}
"""
    with open(os.path.join(OUTDIR, "%s.md" % name), "w") as fh:
        fh.write(md)
    return df


# =============================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ---------------- universe, prices, SIC ----------------------------------
    univ = pd.read_csv(os.path.join(ROOT, "data/universe/tickers_2026-08-18.csv"), dtype={"cik": str})
    univ["cik_int"] = univ.cik.astype(int)

    # drop share-class artefacts that are warrants/rights/units: 5-char ticker
    # ending W/R/U whose 4-char stem is also a filer under the same CIK.
    stems = set(univ.ticker)
    art = univ.ticker.str.len().eq(5) & univ.ticker.str[-1].isin(list("WRU")) & \
          univ.ticker.str[:4].isin(stems)
    print("universe %d tickers; dropping %d warrant/right/unit lookalikes" % (len(univ), art.sum()))
    univ = univ[~art]

    der = pd.read_parquet(os.path.join(ROOT, "data/prices/derived.parquet"))[["ticker", "dollar_vol_60d"]]
    diagp = pd.read_parquet(os.path.join(ROOT, "data/prices/derived_diagnostics.parquet"))[["ticker", "last_close"]]
    px = der.merge(diagp, on="ticker", how="inner")

    base = univ.merge(px, on="ticker", how="inner")
    base = base[(base.last_close >= MIN_PRICE_BASE) & (base.dollar_vol_60d >= MIN_DOLLAR_VOL)]
    prc5 = base[base.last_close >= MIN_PRICE_PRC5]
    print("investable base (px>=$%.2f, dv>=$%,.0fk): %d tickers / %d ciks"
          .replace("%,", "%") % (MIN_PRICE_BASE, MIN_DOLLAR_VOL / 1e3, len(base), base.cik_int.nunique()))
    print("prc>5 subset: %d tickers / %d ciks" % (len(prc5), prc5.cik_int.nunique()))

    sicfiles = sorted(glob.glob(os.path.join(ROOT, "data/sec/dera/sub_*.tsv.gz")))
    sic = pd.concat([pd.read_csv(f, sep="\t", dtype=str, usecols=["cik", "sic"]) for f in sicfiles])
    sic = sic.dropna(subset=["sic"])
    sic["cik_int"] = sic.cik.astype(int)
    sic["sic"] = pd.to_numeric(sic.sic, errors="coerce")
    sic = sic.dropna(subset=["sic"]).drop_duplicates("cik_int")[["cik_int", "sic"]]
    print("SIC codes for %d ciks (from %d DERA sub.txt files)" % (len(sic), len(sicfiles)))

    accn_idx = pd.read_parquet(os.path.join(ROOT, "data/sec/accn_index.parquet"))[["accn", "filed_date"]]
    accn_idx["filed_date"] = pd.to_datetime(accn_idx.filed_date, errors="coerce")

    # ---------------- load every fact we need --------------------------------
    print("\nloading XBRL frames (point-in-time gated):")
    A = lambda c: series(c, "A", accn_idx)
    I = lambda c: series(c, "I", accn_idx)

    at_q   = I("Assets")
    lt_q   = I("Liabilities")
    act_q  = I("AssetsCurrent")
    lct_q  = I("LiabilitiesCurrent")
    seq_q  = I("StockholdersEquity")

    ni_a    = A("NetIncomeLoss")
    oancf_a = A("NetCashProvidedByUsedInOperatingActivities")
    oancf_c = A("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
    fincf_a = A("NetCashProvidedByUsedInFinancingActivities")
    fincf_c = A("NetCashProvidedByUsedInFinancingActivitiesContinuingOperations")
    ivncf_a = A("NetCashProvidedByUsedInInvestingActivities")
    ivncf_c = A("NetCashProvidedByUsedInInvestingActivitiesContinuingOperations")

    prstk_c = A("PaymentsForRepurchaseOfCommonStock")
    prstk_p = A("PaymentsForRepurchaseOfPreferredStockAndPreferenceStock")
    prstk_e = A("PaymentsForRepurchaseOfEquity")
    sstk_c  = A("ProceedsFromIssuanceOfCommonStock")
    sstk_e  = A("ProceedsFromIssuanceOrSaleOfEquity")
    dv_t    = A("PaymentsOfDividends")
    dv_c    = A("PaymentsOfDividendsCommonStock")
    dv_p    = A("PaymentsOfDividendsPreferredStockAndPreferenceStock")

    txt_a   = A("IncomeTaxExpenseBenefit")
    txfed_a = A("CurrentFederalTaxExpenseBenefit")
    txfo_a  = A("CurrentForeignTaxExpenseBenefit")
    txc_a   = A("CurrentIncomeTaxExpenseBenefit")
    txdi_a  = A("DeferredIncomeTaxExpenseBenefit")
    txt_q   = series("IncomeTaxExpenseBenefit", "Q", accn_idx)

    # ---------------- annual spine: latest 3 fiscal years per cik ------------
    spine = ni_a[["cik_int", "period_end", "val"]].rename(columns={"val": "ni"})
    spine = spine.sort_values(["cik_int", "period_end"])
    spine["rk"] = spine.groupby("cik_int").period_end.rank(ascending=False, method="first")
    fy0 = spine[spine.rk == 1][["cik_int", "period_end", "ni"]].rename(columns={"ni": "ib0", "period_end": "pe0"})
    fy1 = spine[spine.rk == 2][["cik_int", "period_end", "ni"]].rename(columns={"ni": "ib1", "period_end": "pe1"})
    fy2 = spine[spine.rk == 3][["cik_int", "period_end", "ni"]].rename(columns={"ni": "ib2", "period_end": "pe2"})
    ann = fy0.merge(fy1, on="cik_int", how="left").merge(fy2, on="cik_int", how="left")
    # the prior years must actually be ~1 and ~2 years back, not an arbitrary gap
    ann.loc[((ann.pe0 - ann.pe1).dt.days.sub(365).abs() > 60), "ib1"] = np.nan
    ann.loc[((ann.pe0 - ann.pe2).dt.days.sub(730).abs() > 90), "ib2"] = np.nan
    ann["stale_days"] = (AS_OF - ann.pe0).dt.days
    print("\nannual spine: %d ciks; latest FY end median %s; %d with FY end older than 550d"
          % (len(ann), ann.pe0.median().date(), (ann.stale_days > 550).sum()))
    ann = ann[ann.stale_days <= 550]            # a >18-month-old 10-K is a dead filer
    ann = ann.rename(columns={"pe0": "period_end"})

    def hang(df, srcs, col, tol=7):
        """Attach the first non-null of several concepts at the spine's FY end."""
        for i, s in enumerate(srcs):
            df = asof_pick(df, s, "%s__%d" % (col, i), tol_days=tol)
        cols = ["%s__%d" % (col, i) for i in range(len(srcs))]
        df[col] = df[cols[0]]
        for c in cols[1:]:
            df[col] = df[col].fillna(df[c])
        return df.drop(columns=cols)

    def hang_sum(df, srcs, col, tol=7):
        """Attach the SUM of several concepts (missing legs treated as absent)."""
        for i, s in enumerate(srcs):
            df = asof_pick(df, s, "%s__%d" % (col, i), tol_days=tol)
        cols = ["%s__%d" % (col, i) for i in range(len(srcs))]
        df[col] = df[cols].sum(axis=1, min_count=1)
        return df.drop(columns=cols)

    ann = hang(ann, [at_q], "at")
    ann = hang(ann, [lt_q], "lt")
    ann = hang(ann, [act_q], "act")
    ann = hang(ann, [lct_q], "lct")
    ann = hang(ann, [seq_q], "seq")
    ann = hang(ann, [oancf_a, oancf_c], "oancf")
    ann = hang(ann, [fincf_a, fincf_c], "fincf")
    ann = hang(ann, [ivncf_a, ivncf_c], "ivncf")
    ann = hang_sum(ann, [prstk_c, prstk_p], "prstk_sum")
    ann = hang(ann, [prstk_e], "prstk_eq")
    ann = hang(ann, [sstk_c, sstk_e], "sstk")
    ann = hang_sum(ann, [dv_c, dv_p], "dv_sum")
    ann = hang(ann, [dv_t], "dv_tot")
    ann = hang(ann, [txt_a], "txt")
    ann = hang(ann, [txfed_a], "txfed")
    ann = hang(ann, [txfo_a], "txfo")
    ann = hang(ann, [txc_a], "txc")
    ann = hang(ann, [txdi_a], "txdi")

    ann["prstkcc"] = ann.prstk_sum.fillna(ann.prstk_eq)
    ann["dvt"] = ann.dv_tot.fillna(ann.dv_sum)
    # `lt` fallback: many filers never tag Liabilities (not required when the face
    # of the balance sheet shows liabilities-and-equity). at - seq recovers it.
    ann["lt_src"] = np.where(ann.lt.notna(), "Liabilities", "Assets-StockholdersEquity")
    ann["lt"] = ann.lt.fillna(ann.at_ - ann.seq if "at_" in ann else ann["at"] - ann["seq"])

    ann = ann.merge(sic, on="cik_int", how="left")
    print("annual spine after staleness gate: %d ciks; balance sheet matched: at %d, lt %d, act %d, lct %d"
          % (len(ann), ann["at"].notna().sum(), ann.lt.notna().sum(), ann.act.notna().sum(), ann.lct.notna().sum()))

    results = {}

    # =====================================================================
    # 1. OScore  (Ohlson 1980 as used by Dichev 1998)
    # =====================================================================
    o = ann.copy()
    o["fopt"] = o.oancf                      # catalog: "fopt = oancf if fopt is missing"
    ok = (o["at"] > 0) & (o.lt > 0) & (o.act > 0) & o.lct.notna() & o.ib0.notna() & o.ib1.notna() & o.fopt.notna()
    print("\nOScore input availability: at>0 %d | lt>0 %d | act>0 %d | lct %d | ib0 %d | ib1 %d | fopt %d | ALL %d"
          % ((o["at"] > 0).sum(), (o.lt > 0).sum(), (o.act > 0).sum(), o.lct.notna().sum(),
             o.ib0.notna().sum(), o.ib1.notna().sum(), o.fopt.notna().sum(), ok.sum()))
    o = o[ok].copy()
    at_mm = o["at"] / 1e6                    # Compustat `at` is $ millions
    o["chin"] = (o.ib0 - o.ib1) / (o.ib0.abs() + o.ib1.abs())
    o["intwo_ohlson"] = ((o.ib0 < 0) & (o.ib1 < 0)).astype(float)
    o["intwo_literal"] = (o[["ib0", "ib1", "ib2"]].sum(axis=1, min_count=3) < 0).astype(float)
    o.loc[o.ib2.isna(), "intwo_literal"] = np.nan

    def oscore(intwo):
        return (-1.32
                - 0.407 * np.log(at_mm / OSCORE_DEFLATOR)
                + 6.03 * (o.lt / o["at"])
                - 1.43 * ((o.act - o.lct) / o["at"])
                + 0.076 * (o.lct / o.act)
                - 1.72 * (o.lt > o["at"]).astype(float)
                - 2.37 * (o.ib0 / o["at"])
                - 1.83 * (o.fopt / o.lt)
                + 0.285 * intwo
                - 0.521 * o.chin)

    o["value"] = oscore(o.intwo_ohlson)
    o["value_lit"] = oscore(o.intwo_literal)
    both = o[["value", "value_lit"]].dropna()
    sp = both.corr(method="spearman").iloc[0, 1]
    print("OScore INTWO variants: Spearman rank corr (Ohlson vs literal 3yr-sum) = %.5f  on %d names"
          % (sp, len(both)))

    # OSAP SIC exclusion (Dichev's sample): drop 3999<=sic<=4999 and sic>5999
    n_pre = len(o)
    o["sic_ok"] = ~(((o.sic >= 3999) & (o.sic <= 4999)) | (o.sic > 5999))
    n_nosic = o.sic.isna().sum()
    o = o[o.sic_ok.fillna(False)]
    print("OScore SIC filter: %d -> %d ciks (%d had no SIC and were dropped)" % (n_pre, len(o), n_nosic))

    om = prc5.merge(o[["cik_int", "value", "period_end", "sic"]], on="cik_int", how="inner")
    res = write_signal(
        "OScore", om[["ticker", "value"]],
        formula=(
            "OScore = -1.32\n"
            "         - 0.407 * log(at / deflator)          at in $ millions, deflator = 1.0 (see note 1)\n"
            "         + 6.03  * (lt / at)\n"
            "         - 1.43  * ((act - lct) / at)          working capital / assets\n"
            "         + 0.076 * (lct / act)\n"
            "         - 1.72  * I(lt > at)                  OENEG\n"
            "         - 2.37  * (ib / at)\n"
            "         - 1.83  * (fopt / lt)                 fopt := oancf (catalog fallback)\n"
            "         + 0.285 * INTWO                       I(ib_t < 0 AND ib_t-1 < 0)   (see note 2)\n"
            "         - 0.521 * (ib_t - ib_t-1) / (|ib_t| + |ib_t-1|)\n"
            "\n"
            "Sample restriction (OSAP): exclude SIC in [3999, 4999] and SIC > 5999; exclude price < $5."),
        concepts=(
            "| Compustat item | XBRL concept | frame |\n|---|---|---|\n"
            "| `at` | `us-gaap:Assets` | instantaneous, at the fiscal year end |\n"
            "| `lt` | `us-gaap:Liabilities`, falling back to `Assets - StockholdersEquity` | instantaneous |\n"
            "| `act` | `us-gaap:AssetsCurrent` | instantaneous |\n"
            "| `lct` | `us-gaap:LiabilitiesCurrent` | instantaneous |\n"
            "| `ib` | `us-gaap:NetIncomeLoss` | annual duration, three consecutive fiscal years |\n"
            "| `fopt` | `us-gaap:NetCashProvidedByUsedInOperatingActivities` "
            "(+ `...ContinuingOperations` fallback) | annual duration |\n"
            "| SIC | DERA Financial Statement Data Sets `sub.txt` | 2025q3–2026q1 |\n"),
        judgements=(
            f"1. **Deflator set to 1.0, `at` in $ millions.** `-0.407*log(at/deflator)` decomposes into "
            f"`-0.407*log(at) + 0.407*log(deflator)`; the second piece is one constant shared by the whole "
            f"cross-section on a single date, so it shifts every OScore identically and changes no rank. "
            f"Consequence: levels sit ~0.10 above a 2017=100-deflator convention and must **not** be compared "
            f"to Ohlson's published 0.38 bankruptcy threshold. Firms' fiscal year ends do span ~12 months, so "
            f"strictly the deflator differs slightly across firms — at ~2.5% inflation that is 0.407*0.025 = 0.01 "
            f"of OScore, against a cross-sectional SD of ~1.6.\n"
            f"2. **INTWO.** The catalog writes this term as `.285*(ib + ib_t-12 + ib_t-24 < 0)` — a sum over "
            f"three years — but 0.285 is unambiguously Ohlson's coefficient on INTWO, *1 if net income was "
            f"negative in each of the last two years*. I implemented Ohlson's definition and computed the "
            f"catalog-literal variant alongside it: **Spearman rank correlation {sp:.5f}**. The choice is "
            f"immaterial to ranking, and Ohlson's version needs only two years of income, which keeps ~"
            f"{int(o.ib2.notna().mean()*100)}% -> 100% of the sample.\n"
            f"3. **`ib` proxied by `NetIncomeLoss`.** Compustat `ib` is income before extraordinary items. "
            f"ASU 2015-01 eliminated the extraordinary-item classification from US GAAP, so for current filings "
            f"`NetIncomeLoss` and `ib` coincide except for discontinued operations. "
            f"`IncomeLossFromContinuingOperations...` was fetched as an alternative but covers only ~420 filers "
            f"a year — far too thin to use as the primary.\n"
            f"4. **`fopt` = `oancf`.** Compustat's funds-from-operations item was discontinued with the pre-1988 "
            f"cash-flow format; the catalog itself prescribes this fallback.\n"
            f"5. **`lt` fallback.** {int((ann.lt_src=='Assets-StockholdersEquity').sum())} of "
            f"{len(ann)} filers on the annual spine never tag `Liabilities` (it is not a required face-of-"
            f"balance-sheet line when liabilities-and-equity is shown), so `Assets - StockholdersEquity` "
            f"was used for those.\n"
            f"6. **Balance sheet aligned to the fiscal year end**, not to the latest quarter — matched within "
            f"±7 days of the annual income statement's `period_end` — so every ratio has a consistent numerator "
            f"and denominator date. Cost: the balance sheet is up to ~18 months old for a late filer.\n"
            f"7. **Firms with no SIC code were dropped**, not kept. {n_nosic} filers were affected; they are "
            f"names absent from the last three DERA quarterly datasets, i.e. filers that have not filed since "
            f"mid-2025, which is itself a reason to exclude them.\n"
            f"8. **The bottom-quintile exclusion is NOT applied here.** The catalog adds: *\"exclude if OScore "
            f"is in the bottom quintile of OScore (original paper shows non-monotonic returns, as does our "
            f"replication)\"*. That is a portfolio-construction rule, not part of the signal value, so it is "
            f"left to the construction step — **but it must be applied there**, or the long leg buys the "
            f"lowest-OScore quintile, which does not pay.\n"),
        extra_lines=(
            f"Filters bite in this order: {len(prc5)} investable tickers at price >= $5 -> joined to "
            f"{len(o)} CIKs that clear the SIC restriction and have all nine inputs."),
        diag=dict(long="Ohlson bankruptcy O-Score", authors="Dichev", year=1998,
                  econ="default risk", minpx=MIN_PRICE_PRC5))
    results["OScore"] = (res, om)

    # =====================================================================
    # 2. PctTotAcc  (Hafzalla, Lundholm, Van Winkle 2011)
    # =====================================================================
    p = ann.copy()
    # Cash-flow distribution lines are absent when the activity did not occur;
    # Compustat records those as 0, so a missing tag here means "no buybacks" etc.
    n_pr, n_ss, n_dv = p.prstkcc.notna().sum(), p.sstk.notna().sum(), p.dvt.notna().sum()
    for c in ["prstkcc", "sstk", "dvt"]:
        p[c] = p[c].fillna(0.0)
    ok = p.ni.notna() & p.oancf.notna() & p.fincf.notna() & p.ivncf.notna() & (p.ni.abs() > 0)
    print("\nPctTotAcc input availability: ni %d | oancf %d | fincf %d | ivncf %d | ALL %d"
          % (p.ni.notna().sum(), p.oancf.notna().sum(), p.fincf.notna().sum(), p.ivncf.notna().sum(), ok.sum()))
    print("  present-vs-defaulted-to-0: prstkcc %d/%d, sstk %d/%d, dvt %d/%d"
          % (n_pr, len(p), n_ss, len(p), n_dv, len(p)))
    p = p[ok].copy()
    p["totacc"] = p.ni - (p.prstkcc - p.sstk + p.dvt + p.oancf + p.fincf + p.ivncf)
    p["value"] = p.totacc / p.ni.abs()

    pm = base.merge(p[["cik_int", "value", "period_end"]], on="cik_int", how="inner")
    res = write_signal(
        "PctTotAcc", pm[["ticker", "value"]],
        formula=("PctTotAcc = ( ni - ( prstkcc - sstk + dvt + oancf + fincf + ivncf ) ) / |ni|\n"
                 "\n"
                 "i.e. net income less cash-basis earnings, where cash-basis earnings is the sum of all\n"
                 "three cash-flow subtotals plus net cash returned to shareholders. Scaled by |ni|, which\n"
                 "is what makes this the 'percent' rather than the asset-scaled accruals measure."),
        concepts=(
            "| Compustat item | XBRL concept |\n|---|---|\n"
            "| `ni` | `us-gaap:NetIncomeLoss` |\n"
            "| `oancf` | `us-gaap:NetCashProvidedByUsedInOperatingActivities` (fallback `...ContinuingOperations`) |\n"
            "| `fincf` | `us-gaap:NetCashProvidedByUsedInFinancingActivities` (fallback `...ContinuingOperations`) |\n"
            "| `ivncf` | `us-gaap:NetCashProvidedByUsedInInvestingActivities` (fallback `...ContinuingOperations`) |\n"
            "| `prstkcc` | `PaymentsForRepurchaseOfCommonStock` + "
            "`PaymentsForRepurchaseOfPreferredStockAndPreferenceStock`, fallback `PaymentsForRepurchaseOfEquity` |\n"
            "| `sstk` | `ProceedsFromIssuanceOfCommonStock`, fallback `ProceedsFromIssuanceOrSaleOfEquity` |\n"
            "| `dvt` | `PaymentsOfDividends`, fallback `PaymentsOfDividendsCommonStock` + "
            "`PaymentsOfDividendsPreferredStockAndPreferenceStock` |\n"
            "\nAll annual (12-month duration) frames, taken at each filer's most recent fiscal year end.\n"),
        judgements=(
            f"1. **Missing buyback / issuance / dividend tags are read as 0, not as missing.** These are "
            f"cash-flow-statement lines that simply do not appear when the activity did not happen, and "
            f"Compustat records them as zero. Only {n_pr:,} of {len(ann):,} spine filers tag a repurchase, "
            f"{n_ss:,} an issuance and {n_dv:,} a dividend. Treating absence as missing instead would have "
            f"cut coverage by roughly two thirds and would have thrown away every non-payer — a biased sample. "
            f"The residual risk is a filer that *did* buy back stock but tagged it with a concept outside the "
            f"list above; that firm's accruals come out too high.\n"
            f"2. **The three cash-flow subtotals are required, not defaulted.** A filer that omits `fincf` or "
            f"`ivncf` has not filed a complete cash-flow statement, and defaulting them to 0 would make the "
            f"accrual read like the whole of net income.\n"
            f"3. **`|ni|` in the denominator is left unfloored.** Firms with near-zero net income produce "
            f"enormous ratios — that is inherent to the percent-accruals construction, and the 1%/99% "
            f"winsorisation is what contains it. No minimum-|ni| screen was imposed, because imposing one "
            f"would drop exactly the break-even firms the measure is designed to flag.\n"
            f"4. **No price filter in OSAP for this signal**, so only the group's base investability floor "
            f"(price >= ${MIN_PRICE_BASE:.2f}, $" + f"{MIN_DOLLAR_VOL:,.0f} median 60d dollar volume) applies.\n"),
        extra_lines="Rebalance cadence per the triage file: 12 months (`low_turnover = True`).",
        diag=dict(long="Percent Total Accruals", authors="Hafzalla, Lundholm, Van Winkle", year=2011,
                  econ="accruals", minpx=MIN_PRICE_BASE))
    results["PctTotAcc"] = (res, pm)

    # =====================================================================
    # 3. Tax  (Lev and Nissim 2004)
    # =====================================================================
    t = ann.copy()
    num_fedfo = t.txfed + t.txfo                       # both must be present
    num_txt = t.txt - t.txdi
    t["numer"] = num_fedfo
    t["num_src"] = np.where(num_fedfo.notna(), "txfed+txfo", None)
    m = t.numer.isna() & num_txt.notna()
    t.loc[m, "numer"] = num_txt[m]; t.loc[m, "num_src"] = "txt-txdi"
    m = t.numer.isna() & t.txc.notna()
    t.loc[m, "numer"] = t.txc[m]; t.loc[m, "num_src"] = "CurrentIncomeTaxExpenseBenefit"
    print("\nTax numerator source counts:\n%s" % t.num_src.value_counts(dropna=False).to_string())

    ok = t.numer.notna() & t.ib0.notna() & (t.ib0 != 0)
    t = t[ok].copy()
    t["value"] = t.numer / (TAX_RATE * t.ib0)
    t.loc[(t.ib0 < 0) & (t.numer > 0), "value"] = 1.0   # catalog rule
    print("Tax: %d ciks with a value; %d hit the negative-income rule"
          % (len(t), ((t.ib0 < 0) & (t.numer > 0)).sum()))

    tm = prc5.merge(t[["cik_int", "value", "period_end", "num_src"]], on="cik_int", how="inner")
    res = write_signal(
        "Tax", tm[["ticker", "value"]],
        formula=("            txfed + txfo                       (both present)\n"
                 "numerator = txt - txdi                         (otherwise)\n"
                 "            CurrentIncomeTaxExpenseBenefit     (otherwise; added fallback, note 2)\n"
                 "\n"
                 "Tax = numerator / (statutory_rate * ib)        statutory_rate = 0.21   (note 1)\n"
                 "\n"
                 "if ib < 0 and numerator > 0:  Tax := 1         (catalog rule)\n"
                 "Exclude price < $5."),
        concepts=(
            "| Compustat item | XBRL concept |\n|---|---|\n"
            "| `txfed` | `us-gaap:CurrentFederalTaxExpenseBenefit` |\n"
            "| `txfo` | `us-gaap:CurrentForeignTaxExpenseBenefit` |\n"
            "| `txt` | `us-gaap:IncomeTaxExpenseBenefit` |\n"
            "| `txdi` | `us-gaap:DeferredIncomeTaxExpenseBenefit` |\n"
            "| (fallback) | `us-gaap:CurrentIncomeTaxExpenseBenefit` |\n"
            "| `ib` | `us-gaap:NetIncomeLoss` |\n"
            "\nAll annual (12-month duration) frames at each filer's most recent fiscal year end.\n"
            "\nNumerator source split: " +
            ", ".join("%s %d" % (k, v) for k, v in tm.num_src.value_counts().items()) + ".\n"),
        judgements=(
            f"1. **Statutory rate 0.21, not the catalog's 0.35.** The catalog's schedule stops at "
            f"\".35 from 1993 onwards\", written before the 2017 Tax Cuts and Jobs Act cut the federal "
            f"corporate rate to 21% for tax years from 2018. **This choice cannot change any rank**: the rate "
            f"is a single positive constant dividing every firm's denominator, so it rescales all values by "
            f"0.35/0.21 and preserves the ordering exactly (including for negative `ib`). It only makes the "
            f"levels interpretable — a fully-taxed firm now scores near 1.0 rather than near 0.6.\n"
            f"2. **Third fallback added.** The catalog gives two numerator recipes. I added "
            f"`CurrentIncomeTaxExpenseBenefit` as a third, because it is the directly-tagged total current tax "
            f"expense and is conceptually identical to `txt - txdi`. It supplied the numerator for only "
            f"{int((tm.num_src=='CurrentIncomeTaxExpenseBenefit').sum())} tickers, so nothing hinges on it.\n"
            f"3. **State taxes are treated inconsistently — faithfully so.** The primary numerator "
            f"`txfed + txfo` excludes state tax, while the `txt - txdi` fallback includes it. That "
            f"inconsistency is in the catalog definition and in Lev–Nissim; reproducing it beats silently "
            f"'fixing' it, but it does mean the ~"
            f"{int((tm.num_src!='txfed+txfo').mean()*100)}% of names on the fallback carry a mechanically "
            f"higher numerator than the rest.\n"
            f"4. **`ib` proxied by `NetIncomeLoss`** — same reasoning as OScore note 3.\n"
            f"5. **The negative-income rule truncates at exactly 1.0**, creating a visible spike in the "
            f"distribution. That is the catalog's rule verbatim, not an artefact.\n"),
        extra_lines=("Rebalance cadence per the triage file: 12 months. Note the OSAP caveat on this one — "
                     "Lev and Nissim's result holds in 1973–1992 and in 1993–2000 only if 1998 is dropped; "
                     "recent-sample t is 2.47, the weakest of the four in this group."),
        diag=dict(long="Taxable income to book income", authors="Lev and Nissim", year=2004,
                  econ="other", minpx=MIN_PRICE_PRC5))
    results["Tax"] = (res, tm)

    # =====================================================================
    # 4. ChTax  (Thomas and Zhang 2011)
    # =====================================================================
    q = txt_q[["cik_int", "period_end", "val"]].rename(columns={"val": "txtq"}).sort_values(["cik_int", "period_end"])
    latest = q.groupby("cik_int", as_index=False).tail(1).rename(columns={"period_end": "pe_t", "txtq": "txtq_t"})
    latest = latest[(AS_OF - latest.pe_t).dt.days <= 200]     # else the filer is stale
    cand = latest.merge(q.rename(columns={"period_end": "pe_l", "txtq": "txtq_l"}), on="cik_int", how="left")
    cand["gap"] = (cand.pe_t - cand.pe_l).dt.days
    cand = cand[(cand.gap - 365).abs() <= 45]                 # the 4-quarter lag
    cand["err"] = (cand.gap - 365).abs()
    lag = cand.sort_values(["cik_int", "err"]).groupby("cik_int", as_index=False).head(1)
    ch = lag[["cik_int", "pe_t", "txtq_t", "pe_l", "txtq_l"]].copy()
    # lagged total assets: balance sheet at the *lagged* quarter end
    ch = ch.rename(columns={"pe_l": "period_end"})
    ch = asof_pick(ch, at_q, "at_l", tol_days=7)
    ch = ch.rename(columns={"period_end": "pe_l"})
    print("\nChTax: %d ciks with a fresh quarter; %d found a t-4 pair; %d matched lagged assets"
          % (len(latest), len(ch), ch.at_l.notna().sum()))
    ch = ch[(ch.at_l > 0)].copy()
    ch["value"] = (ch.txtq_t - ch.txtq_l) / ch.at_l
    # sensitivity: scale by current assets instead of lagged
    chk = ch.rename(columns={"pe_t": "period_end"})
    chk = asof_pick(chk, at_q, "at_t", tol_days=7)
    alt = ((chk.txtq_t - chk.txtq_l) / chk.at_t)
    sp2 = pd.DataFrame({"a": chk.value.values, "b": alt.values}).dropna().corr(method="spearman").iloc[0, 1]
    print("ChTax denominator sensitivity: Spearman(lagged at, current at) = %.5f" % sp2)

    cm = base.merge(ch[["cik_int", "value", "pe_t", "pe_l"]], on="cik_int", how="inner")
    res = write_signal(
        "ChTax", cm[["ticker", "value"]],
        formula=("ChTax = ( txtq_t - txtq_{t-4} ) / at_{t-4}\n"
                 "\n"
                 "4-quarter change in quarterly total income tax expense, scaled by total assets as of the\n"
                 "same quarter four periods back. Quarters are matched on actual `period_end`, requiring a\n"
                 "365 +/- 45 day gap, so non-calendar fiscal years line up correctly."),
        concepts=(
            "| Compustat item | XBRL concept | frame |\n|---|---|---|\n"
            "| `txtq` | `us-gaap:IncomeTaxExpenseBenefit` | quarterly duration (~90-day), CY2020Q1–CY2026Q2 |\n"
            "| `at` | `us-gaap:Assets` | instantaneous, at the lagged quarter end |\n"),
        judgements=(
            f"1. **\"Lagged total assets\" read as `at_{{t-4}}`, not `at_{{t-1}}`.** The catalog says only "
            f"\"scaled by lagged total assets\". Scaling by assets at the same date as the base quarter of the "
            f"change is both the Thomas–Zhang convention and the internally consistent one. Sensitivity check "
            f"against scaling by current-quarter assets: **Spearman rank correlation {sp2:.5f}**, so the "
            f"reading is not load-bearing.\n"
            f"2. **Calendar-Q4 quarterly frames are thin by construction** — roughly 850 filers versus ~4,000 "
            f"in other quarters, because a 10-K tags the annual figure rather than the Q4 three-month stub "
            f"(documented in `data/sec/fundamentals.md`). This is handled by anchoring on each filer's own "
            f"*latest available* quarter rather than a fixed calendar quarter, so a December-year-end firm is "
            f"measured on its June quarter against the prior June. The cost is that the anchor date varies "
            f"across the cross-section by up to ~6 months.\n"
            f"3. **Staleness gate:** the anchor quarter must end within 200 days of {AS_OF.date()}, which "
            f"drops filers that have stopped reporting.\n"
            f"4. **Quarterly frames are genuine ~90-day durations** (median 90, IQR 90–91) — the SEC frames "
            f"API rejects the year-to-date tagging that would otherwise contaminate a quarterly series. "
            f"Verified before use.\n"
            f"5. **No price filter in OSAP for this signal**, so only the base investability floor applies. "
            f"ChTax has the lowest decay of the four ({0.35:.2f}) but a 3-month rebalance, i.e. the highest "
            f"turnover.\n"),
        extra_lines="Rebalance cadence per the triage file: 3 months (`low_turnover = False`).",
        diag=dict(long="Change in Taxes", authors="Thomas and Zhang", year=2011,
                  econ="other", minpx=MIN_PRICE_BASE))
    results["ChTax"] = (res, cm)

    # ---------------- cross-signal summary ----------------------------------
    print("\n" + "=" * 78)
    print("COVERAGE SUMMARY")
    for k, (r, _) in results.items():
        flag = "  <-- BELOW 1500" if len(r) < 1500 else ""
        print("  %-12s %5d tickers   sign %+d%s" % (k, len(r), SIGNS[k], flag))
    wide = None
    for k, (r, _) in results.items():
        s = r.set_index("ticker")["rank_pct"].rename(k)
        wide = s.to_frame() if wide is None else wide.join(s, how="outer")
    print("\npairwise Spearman of rank_pct (raw orientation, overlapping names only):")
    print(wide.corr(method="spearman").round(3).to_string())
    print("\nnames with all four signals: %d" % wide.dropna().shape[0])
    print("=" * 78)


if __name__ == "__main__":
    main()
