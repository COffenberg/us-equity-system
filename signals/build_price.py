#!/usr/bin/env python3
"""
Build the "Price & technical" signal group.

Source of truth for the features: data/prices/derived.parquet (one row per ticker,
built 2026-08-18 off history_3y.parquet, anchor session 2026-08-17).
See data/prices/derived.md for the upstream repairs (splits, quarantine, adj-close sanity).

Signals produced (one parquet + one md each, columns [ticker, value, rank_pct]):

  mom_12_1        OSAP Mom12m   (Jegadeesh-Titman 1993, sign +1)
  mom_6m          OSAP Mom6m    (Jegadeesh-Titman 1993, sign +1)
  px_to_52w_high  OSAP High52   (George-Hwang 2004,     sign +1)
  vol_60d         OSAP RealizedVol analog (Ang et al 2006, sign -1)
  max_ret_1m      OSAP MaxRet   (Bali-Cakici-Whitelaw 2011, sign -1)  <- LOTTERY MARKER
  volume_trend    OSAP VolumeTrend (Haugen-Baker 1996,  sign -1)
  rel_strength    not an OSAP signal; ret_12m minus EW universe median ret_12m

No lookahead is possible here: every input is a realised price/volume observation on
or before the 2026-08-17 anchor session. Survivorship is not a concern - we score
today's cross-section and can only buy what exists.

Usage:  python3 signals/build_price.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "signals"
OUT.mkdir(exist_ok=True)

ANCHOR = "2026-08-17"          # last complete session in the price panel
ASOF = "2026-08-18"            # build date

# ---------------------------------------------------------------- universe rules
MIN_DOLLAR_VOL = 1_000_000.0   # median daily dollar volume over last 60 sessions
MIN_PRICE = 1.00               # last close; kills sub-penny tape
MIN_TRADED_60 = 30             # must actually have traded recently
WINSOR = (0.01, 0.99)

# --- non-common-equity exclusion -------------------------------------------------
# Three independent tests, because no single one is sufficient (each was checked
# against the actual roster before being adopted):
#
#  (a) BROAD_FUND_PAT + "issuer files no us-gaap XBRL".  Catches the ~180 liquid
#      closed-end funds, muni/credit funds and royalty trusts (PIMCO, Nuveen,
#      BlackRock, Eaton Vance, Cornerstone, SBR, PBT...).  The us-gaap test is what
#      makes the broad name pattern safe: REITs and banks whose legal names contain
#      "Trust"/"Bancshares" (DLR, NTRS, ESS, VNO, ARCC, MAIN...) all DO file us-gaap
#      and are kept.  Foreign private issuers (TSM, ASML, NOK, NVO) file 20-F and
#      have no us-gaap facts either - they survive because their names carry no
#      fund word.
#  (b) TIGHT_FUND_PAT.  Catches commodity/crypto trusts that DO file 10-K with
#      us-gaap and therefore slip past (a): GLD, SLV, IBIT, USO, UVXY, FBTC, ETHA...
#  (c) Ticker syntax + one-line-per-issuer.  Preferred series (-PA/-PJ/...), warrants,
#      rights and units die on syntax; baby bonds (TBB under AT&T), leveraged ETNs
#      sharing a bank's CIK (FNGU/BULZ/NRGU under Bank of Montreal, DGP under
#      Deutsche Bank) and duplicate share classes (GOOG/GOOGL, BRK-A/BRK-B) die on
#      the one-line-per-CIK rule.
BROAD_FUND_PAT = re.compile(
    r"\bFUNDS?\b|\bTRUST\b|\bETFS?\b|\bSHARES\b|\bPORTFOLIO\b|\bINDEX\b"
    r"|CURRENCYSHARES|\bETNS?\b|ROYALTY",
    re.I,
)

TIGHT_FUND_PAT = re.compile(
    r"""
      \bETFS?\b | \bETNS?\b | EXCHANGE.TRADED
    | \bISHARES\b | \bSPDR\b | \bPROSHARES\b | \bDIREXION\b | \bVANECK\b
    | \bGRAYSCALE\b | \bWISDOMTREE\s+(?:TRUST|FUNDS)\b | GLOBAL\s+X\s+FUNDS
    | \bAMPLIFY\b | \bTIDAL\b | \bGRANITESHARES\b | \bYIELDMAX\b | \bROUNDHILL\b
    | \bSPROTT\s+PHYSICAL\b | \bABRDN\b | \bBITWISE\b | \bARK\s+21SHARES\b
    | \bVS\s+TRUST\b | \bWORLD\s+GOLD\s+TRUST\b
    | INDEX\s+FUND | \bINDEX\s+TRUST\b | TRUST,?\s+SERIES | UNIT\s+INVESTMENT
    | (?:BITCOIN|ETHEREUM|SOLANA|XRP|LITECOIN|CRYPTO|GOLD|SILVER|PLATINUM|PALLADIUM|
       COPPER|OIL|NATURAL\s+GAS|COMMODITY|CURRENCY)\s+(?:TRUST|FUND|MINI\s+TRUST)
    | \bFUND,?\s*(?:LP|L\.P\.|LLC)\b | \bFUNDS?\s+TRUST\b | \bTRUST\s+I{1,3}\s*$
    | \bBULLISH\s+FUND\b | \bBEARISH\s+FUND\b
    | \b(?:INCOME|EQUITY|MUNICIPAL|OPPORTUNIT\w+|STRATEGY|CREDIT|LENDING)\s+FUND\b
    """,
    re.I | re.X,
)

# operating companies whose legal name trips a pattern
FUND_WHITELIST = {"WT"}          # WisdomTree, Inc. - the asset manager itself

# Ticker syntax of non-common-equity lines: preferred series, warrants, rights, units.
NON_EQUITY_SUFFIX = re.compile(r"-(?:P[A-Z]?|WT|RT|RI|U|R|W)$")

# ETNs whose issuer CIK carries no common-stock line of its own, so the
# one-line-per-CIK rule cannot displace them. Identified by inspection of the
# multi-ticker CIKs; listed explicitly rather than pattern-matched.
KNOWN_ETN = {"VXX", "DJP", "GBUG", "GLDI", "SLVO", "USOI"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def hr(title: str) -> None:
    log("\n" + "=" * 92)
    log(title)
    log("=" * 92)


# =============================================================== load + universe
def load() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/prices/derived.parquet")
    g = pd.read_parquet(ROOT / "data/prices/derived_diagnostics.parquet")
    u = pd.read_csv(ROOT / f"data/universe/tickers_{ASOF}.csv", dtype={"cik": str})

    m = d.merge(g, on="ticker", how="left", validate="1:1")
    m = m.merge(u[["ticker", "cik", "title"]].drop_duplicates("ticker"),
                on="ticker", how="left", validate="1:1")
    m["title"] = m["title"].fillna("")
    # a ticker with no roster row gets a unique synthetic cik so the one-line-per-CIK
    # rule cannot collapse unrelated names onto each other
    m["cik"] = m["cik"].fillna(pd.Series("NOCIK_" + m["ticker"], index=m.index))
    return m


def build_universe(m: pd.DataFrame) -> pd.DataFrame:
    hr("UNIVERSE CONSTRUCTION")
    log(f"derived.parquet tickers                                  {len(m):>7,}")

    # issuers that file us-gaap XBRL facts for a period ending 2024 or later
    fund = pd.read_parquet(ROOT / "data/sec/fundamentals.parquet",
                           columns=["cik", "period_end"])
    gaap_ciks = set(fund.loc[fund["period_end"] >= "2024-01-01", "cik"].dropna().unique())

    m["is_cef"] = m["title"].str.contains(BROAD_FUND_PAT, regex=True) & ~m["cik"].isin(gaap_ciks)
    m["is_etp"] = m["title"].str.contains(TIGHT_FUND_PAT, regex=True) & ~m["ticker"].isin(FUND_WHITELIST)
    m["is_etn"] = m["ticker"].isin(KNOWN_ETN)
    m["bad_suffix"] = m["ticker"].str.contains(NON_EQUITY_SUFFIX, regex=True)
    m["fund_like"] = m["is_cef"] | m["is_etp"] | m["is_etn"]

    steps = [
        ("closed-end fund / royalty trust (name + no us-gaap)", ~m["is_cef"]),
        ("commodity / crypto / index ETP (name)", ~m["is_etp"]),
        ("ETN with no common line under its CIK", ~m["is_etn"]),
        ("preferred / warrant / right / unit (ticker syntax)", ~m["bad_suffix"]),
        (f"median 60d dollar volume >= ${MIN_DOLLAR_VOL:,.0f}", m["dollar_vol_60d"].fillna(0) >= MIN_DOLLAR_VOL),
        (f"last close >= ${MIN_PRICE:,.2f}", m["last_close"].fillna(0) >= MIN_PRICE),
        (f"traded on >= {MIN_TRADED_60} of last 60 sessions", m["traded_days_60"].fillna(0) >= MIN_TRADED_60),
        ("adj_close usable (not quarantined)", ~m["quarantined"].fillna(False)),
        ("adj_close usable (not unadjustable)", ~m["unadjustable"].fillna(False)),
    ]
    keep = pd.Series(True, index=m.index)
    for label, cond in steps:
        before = int(keep.sum())
        keep &= cond
        log(f"  {label:<52s} {int(keep.sum()):>7,}   (-{before - int(keep.sum()):,})")

    inv = m[keep].copy()

    # one line per issuer: shortest symbol first (the primary listing; also demotes
    # ETNs like FNGU under BMO and baby bonds like TBB under T), then most liquid.
    inv["_sl"] = inv["ticker"].str.len()
    inv = inv.sort_values(["_sl", "dollar_vol_60d"], ascending=[True, False])
    before = len(inv)
    dupes = inv[inv.duplicated("cik", keep="first")][["ticker", "cik", "title", "dollar_vol_60d"]]
    inv = inv.drop_duplicates("cik", keep="first")
    log(f"  {'one line per issuer CIK (secondary classes dropped)':<52s} {len(inv):>7,}   (-{before - len(inv):,})")
    dupes.to_csv(OUT / "_excluded_secondary_lines.csv", index=False)

    inv = inv.drop(columns=["_sl"]).reset_index(drop=True)
    log(f"\nINVESTABLE UNIVERSE                                      {len(inv):>7,}")

    excl = m[m["fund_like"] | m["bad_suffix"]][["ticker", "title", "dollar_vol_60d"]] \
        .sort_values("dollar_vol_60d", ascending=False)
    excl.to_csv(OUT / "_excluded_funds.csv", index=False)
    liq_excl = int((excl["dollar_vol_60d"].fillna(0) >= MIN_DOLLAR_VOL).sum())
    log(f"non-common-equity names excluded: {len(excl):,} ({liq_excl} of them liquid) "
        f"-> signals/_excluded_funds.csv")
    log(f"secondary share classes / ETNs / baby bonds dropped: {len(dupes)} "
        f"-> signals/_excluded_secondary_lines.csv")
    return inv


# =============================================================== volume trend (OSAP)
def osap_volume_trend(inv_tickers: set[str]) -> pd.DataFrame:
    """
    OSAP VolumeTrend (Haugen & Baker 1996), Detailed Definition:
      "Rolling coefficient from regressing monthly trading volume on a linear time
       trend over a window of 60 months (require that at least 30 exist).
       Scale coefficient by 60-month average of trading volume."

    Deviations, both forced and documented in the .md:
      * window is 35 months (2023-09..2026-07, the full months in the 3y panel), not 60;
        minimum observations scaled proportionally to 24 (same 60% ratio as 30/60).
      * monthly DOLLAR volume (close x volume) instead of share volume. Share volume
        is not split-invariant and history_3y.parquet is the unrepaired record
        (447 unapplied splits detected upstream); dollar volume is invariant.
    """
    h = pd.read_parquet(ROOT / "data/prices/history_3y.parquet",
                        columns=["date", "ticker", "close", "volume"])
    h = h[h["ticker"].isin(inv_tickers)]
    h["ym"] = h["date"].dt.to_period("M")

    first_full, last_full = pd.Period("2023-09", "M"), pd.Period("2026-07", "M")
    h = h[(h["ym"] >= first_full) & (h["ym"] <= last_full)]      # drop partial edge months

    h["dv"] = h["close"] * h["volume"]
    mon = h.groupby(["ticker", "ym"], observed=True)["dv"].sum().reset_index()
    mon = mon[mon["dv"] > 0]

    # t = calendar month index so gaps in the series are respected, not collapsed
    mon["t"] = (mon["ym"] - first_full).apply(lambda x: x.n).astype(float)

    mon["tv"] = mon["t"] * mon["dv"]
    mon["tt"] = mon["t"] ** 2
    agg = mon.groupby("ticker").agg(n=("dv", "size"), st=("t", "sum"), sv=("dv", "sum"),
                                    stv=("tv", "sum"), stt=("tt", "sum"),
                                    last_m=("ym", "max"))
    MIN_MONTHS = 24
    agg = agg[(agg["n"] >= MIN_MONTHS) & (agg["last_m"] == last_full)]

    denom = agg["n"] * agg["stt"] - agg["st"] ** 2
    slope = (agg["n"] * agg["stv"] - agg["st"] * agg["sv"]) / denom
    meanv = agg["sv"] / agg["n"]
    out = pd.DataFrame({"ticker": agg.index, "value": (slope / meanv).values})
    return out.replace([np.inf, -np.inf], np.nan).dropna()


# =============================================================== signal plumbing
def winsorise(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR[0]), s.quantile(WINSOR[1])
    return s.clip(lo, hi)


def emit(name: str, df: pd.DataFrame, meta: dict, inv: pd.DataFrame) -> pd.DataFrame:
    """df: [ticker, value] raw. Winsorise 1/99, rank ascending, write parquet + md."""
    df = df.dropna(subset=["value"]).copy()
    df["value_raw"] = df["value"]
    lo, hi = df["value"].quantile(WINSOR[0]), df["value"].quantile(WINSOR[1])
    df["value"] = df["value"].clip(lo, hi)
    df["rank_pct"] = df["value"].rank(pct=True, method="average")
    # NB: display order is by RAW value. Sorting by rank_pct would order the
    # winsorised ties arbitrarily and make the eyeball test meaningless.
    df = df.sort_values("value_raw", ascending=False).reset_index(drop=True)

    df[["ticker", "value", "rank_pct"]].to_parquet(OUT / f"{name}.parquet", index=False)

    info = inv.set_index("ticker")
    top = df.head(10)
    bot = df.tail(10).iloc[::-1]

    def tbl(part: pd.DataFrame) -> str:
        rows = []
        for _, r in part.iterrows():
            t = r["ticker"]
            title = str(info.at[t, "title"])[:38] if t in info.index else ""
            dv = info.at[t, "dollar_vol_60d"] if t in info.index else np.nan
            px = info.at[t, "last_close"] if t in info.index else np.nan
            rows.append(f"| {t} | {r['value_raw']:.4f} | {r['value']:.4f} | {r['rank_pct']:.4f} | "
                        f"${px:,.2f} | ${dv/1e6:,.1f}M | {title} |")
        return "\n".join(rows)

    hdr = ("| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |\n"
           "|---|---|---|---|---|---|---|")

    sign = meta["sign"]
    sign_txt = ("**+1** — high value = high expected return, use `rank_pct` directly."
                if sign > 0 else
                "**-1** — high value = *low* expected return, the composite must use "
                "`1 - rank_pct`.")

    md = f"""# {name}

**{meta['headline']}**

- OSAP signal: `{meta['osap']}` — {meta['osap_desc']}
- Sign in OSAP: {sign_txt}
- Coverage: **{len(df):,}** tickers of {len(inv):,} in the investable universe \
({len(df)/len(inv):.1%})
- Built {ASOF} from the price panel anchored on session {ANCHOR}.

## Exact formula

```
{meta['formula'].strip()}
```

**Inputs:** {meta['inputs']}
No XBRL / SEC concepts are used by this signal group — it is price and volume only.

## Distribution (raw, pre-winsorisation)

```
{meta['dist']}
```

Winsorisation: values clipped to the 1st/99th percentile of the investable universe
(`[{lo:.4f}, {hi:.4f}]`) **before** ranking, per the build rules. `value` in the parquet is
the winsorised number; `rank_pct` is the ascending percentile rank of it. Consequence worth
knowing: the clipped 1% tails become ties, so the ~{max(1, round(len(df)*0.01))} most extreme
names on each end share a rank. Ordering is untouched everywhere else.

## Top 10 (highest value)

{hdr}
{tbl(top)}

## Bottom 10 (lowest value)

{hdr}
{tbl(bot)}

## Judgement calls

{meta['judgement'].strip()}

## Universe

Investable universe = **{len(inv):,} names**, one line per issuer CIK: SEC-roster common
equity with median 60d dollar volume >= ${MIN_DOLLAR_VOL:,.0f}, last close >= ${MIN_PRICE:.2f},
traded on >= {MIN_TRADED_60} of the last 60 sessions, and adj_close not quarantined or
unadjustable upstream. Closed-end funds, commodity/crypto trusts, ETFs, ETNs, royalty
trusts, preferred series, warrants, rights and units are excluded
(`_excluded_funds.csv`), as are secondary share classes and baby bonds
(`_excluded_secondary_lines.csv`). Ranks are computed **within** this universe, so
`rank_pct` is comparable only to other signals ranked on the same universe.
"""
    (OUT / f"{name}.md").write_text(md)
    return df


def describe(s: pd.Series) -> str:
    q = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return q.to_string()


# =============================================================== main
def main() -> None:
    m = load()
    inv = build_universe(m)
    inv_t = set(inv["ticker"])

    hr("INPUT CONSISTENCY CHECKS")
    chk = inv.dropna(subset=["ret_12m", "ret_1m", "ret_12m_ex1m"])
    implied = (1 + chk["ret_12m"]) / (1 + chk["ret_1m"]) - 1
    err = (implied - chk["ret_12m_ex1m"]).abs()
    log(f"ret_12m_ex1m vs (1+ret_12m)/(1+ret_1m)-1 : n={len(chk):,} "
        f"max abs err={err.max():.2e}  median={err.median():.2e}")
    log(f"px_to_52w_high max (must be <= 1.0)      : {inv['px_to_52w_high'].max():.6f}")
    log(f"px_to_52w_high min                       : {inv['px_to_52w_high'].min():.6f}")
    log(f"vol_60d < 0 count (must be 0)            : {(inv['vol_60d'] < 0).sum()}")

    signals: dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------- mom_12_1
    s = inv[["ticker", "ret_12m_ex1m"]].rename(columns={"ret_12m_ex1m": "value"})
    signals["mom_12_1"] = emit("mom_12_1", s, dict(
        headline="12-month price momentum skipping the most recent month. The workhorse "
                 "cross-sectional momentum signal; skipping month t-1 removes the "
                 "short-term-reversal contamination that otherwise cancels part of it.",
        osap="Mom12m", osap_desc="Momentum (12 month), Jegadeesh and Titman 1993, "
             "Detailed Definition: \"Stock return between months t-12 and t-1\"",
        sign=+1,
        formula="mom_12_1 = P[t-21] / P[t-252] - 1\n\n"
                "P = split- and dividend-adjusted close (adj_close).\n"
                "t-21 / t-252 are positions on a shared trading calendar (757 sessions,\n"
                "dates where >=25% of the panel printed), so \"21 sessions ago\" is the same\n"
                "calendar date for every ticker. t = 2026-08-17.",
        inputs="`ret_12m_ex1m` from data/prices/derived.parquet.",
        dist=describe(inv["ret_12m_ex1m"].dropna()),
        judgement="""
1. **Months are 21/252 trading sessions, not calendar months.** OSAP says "months t-12
   to t-1"; the panel is daily, so the standard 21-session month is used. Positions are
   on a shared calendar, so no ticker gets a different lookback window because it halted.
2. **Skipping is done in the price ratio, not by compounding returns.** `ret_12m_ex1m` is
   `P[t-21]/P[t-252]-1` directly, which is exact; it is *not* `(1+ret_12m)/(1+ret_1m)-1`
   accumulated from separate windows. The two agree to ~1e-16 (checked at build time).
3. **Adjusted prices.** Uses adj_close, so dividends are in the return. OSAP's Mom12m is
   also a total return (CRSP `ret`). Consistent.
4. Names whose adj_close was quarantined upstream (523 tickers, sub-penny tape and
   unexplained >5x jumps) are out of the universe entirely rather than carrying a
   fabricated +10,000% momentum. Asymmetric on purpose: a missed split puts a fake
   monster return at the *top* of a buy list.
""",
    ), inv)

    # ---------------------------------------------------------------- mom_6m
    # OSAP Mom6m == return between t-6 and t-1, i.e. it SKIPS the current month.
    # derived.parquet's ret_6m does not skip; recover the skipped version exactly:
    #   (1+ret_6m)/(1+ret_1m) = (P[t]/P[t-126]) / (P[t]/P[t-21]) = P[t-21]/P[t-126]
    sub = inv.dropna(subset=["ret_6m", "ret_1m"]).copy()
    sub["value"] = (1 + sub["ret_6m"]) / (1 + sub["ret_1m"]) - 1
    signals["mom_6m"] = emit("mom_6m", sub[["ticker", "value"]], dict(
        headline="6-month price momentum skipping the most recent month, per the OSAP "
                 "definition (return between t-6 and t-1).",
        osap="Mom6m", osap_desc="Momentum (6 month), Jegadeesh and Titman 1993, "
             "Detailed Definition: \"Stock return between months t-6 and t-1\". OSAP's note: "
             "\"OP doesn't skip the current month in the signal ... We follow McLean and "
             "Pontiff, Green-Hand-Zhang, and others in skipping the current month instead.\"",
        sign=+1,
        formula="mom_6m = P[t-21] / P[t-126] - 1\n\n"
                "computed from derived.parquet as the exact algebraic identity\n"
                "  (1 + ret_6m) / (1 + ret_1m) - 1\n"
                "  = (P[t]/P[t-126]) / (P[t]/P[t-21]) - 1\n"
                "  = P[t-21]/P[t-126] - 1",
        inputs="`ret_6m` and `ret_1m` from data/prices/derived.parquet.",
        dist=describe(sub["value"]),
        judgement="""
1. **This signal SKIPS the most recent month; the raw `ret_6m` column in derived.parquet
   does not.** The catalog is explicit that Mom6m is the t-6..t-1 return, and OSAP's own
   note says they deliberately skip the current month. Using the unskipped `ret_6m` would
   blend momentum with one month of short-term reversal (OSAP `STreversal`, sign -1) and
   partially cancel the signal. Spearman(mom_6m, raw ret_6m) is printed in the build log.
2. The skip is recovered by *algebra on two existing columns*, not by re-deriving prices,
   so it inherits exactly the same trading-calendar alignment and repairs as the rest of
   the panel. The identity is exact, not an approximation.
3. Requires both `ret_6m` and `ret_1m` present, which is a slightly stricter data
   requirement than `ret_6m` alone.
""",
    ), inv)

    # ---------------------------------------------------------------- px_to_52w_high
    s = inv[["ticker", "px_to_52w_high"]].rename(columns={"px_to_52w_high": "value"})
    signals["px_to_52w_high"] = emit("px_to_52w_high", s, dict(
        headline="Proximity to the 52-week high. George & Hwang (2004): stocks trading "
                 "near their 52-week high subsequently outperform, and the ratio "
                 "dominates raw momentum in horse races. 1.00 = at the high.",
        osap="High52", osap_desc="52 week high, George and Hwang 2004, Detailed Definition: "
             "\"Price (prc/cfacshr) divided by the maximum price over the previous 12 months.\" "
             "OSAP tests it as a 30/70 split (LS Quantile 0.3), not deciles, held 6 months.",
        sign=+1,
        formula="px_to_52w_high = P[t] / max(P[t-251 .. t])\n\n"
                "P = adjusted close. `prc/cfacshr` in the OSAP definition is CRSP's\n"
                "split-adjusted price, which is what adj_close is here.",
        inputs="`px_to_52w_high` from data/prices/derived.parquet.",
        dist=describe(inv["px_to_52w_high"].dropna()),
        judgement="""
1. **Close-to-close, not close-to-intraday-high.** The panel carries no adjusted high, so
   the denominator is the max *adjusted close* over 252 sessions. Values are therefore
   very slightly higher than the true close-to-intraday-high ratio and cap at exactly
   1.0000 for a name that closed at its 52-week high today. OSAP's definition uses
   `prc`, i.e. closing price, so this matches the catalog rather than deviating from it.
2. **This is a bounded ratio in (0, 1], which is why it behaves so differently from
   momentum.** The 1%/99% winsorisation barely bites at the top (many names sit at 1.00)
   and clips only the deepest-drawdown tail at the bottom.
3. The top of this ranking is a large tie-block of names sitting exactly at their high.
   Winsorisation does not create that tie - the data does. Downstream, treat rank_pct
   near the top as "at the high", not as a fine ordering.
""",
    ), inv)

    # ---------------------------------------------------------------- vol_60d
    s = inv[["ticker", "vol_60d"]].rename(columns={"vol_60d": "value"})
    signals["vol_60d"] = emit("vol_60d", s, dict(
        headline="Annualised realised volatility of daily returns over the last 60 "
                 "sessions. Low-volatility names have historically earned higher "
                 "risk-adjusted returns; the sign is NEGATIVE.",
        osap="RealizedVol", osap_desc="Realized (Total) Volatility, Ang et al. 2006, "
             "Detailed Definition: \"Standard deviation of residuals from CAPM regressions "
             "using the past month of daily data. Value weighted\"",
        sign=-1,
        formula="vol_60d = stdev(r_d over the last 60 sessions) * sqrt(252)\n"
                "  where r_d = adj_close[d]/adj_close[d-1] - 1, kept only on sessions with a\n"
                "  real print (a non-trading day contributes no fake 0% return),\n"
                "  and at least 30 real prints are required in the window.",
        inputs="`vol_60d` from data/prices/derived.parquet.",
        dist=describe(inv["vol_60d"].dropna()),
        judgement="""
1. **This is TOTAL volatility, not the CAPM-residual volatility OSAP specifies, and the
   window is 60 sessions rather than 21.** Two deviations, both deliberate:
   - Residualising needs a market factor and a per-ticker beta regression, which the
     current panel does not carry. Total and idiosyncratic vol correlate ~0.95
     cross-sectionally in US equities, and OSAP itself carries both variants
     (`RealizedVol` total, `IdioVol3F` / `IdioVolCAPM` residual) with the same sign.
   - 60 sessions is a materially less noisy estimate than 21. The cost is slower
     response to a genuine vol regime change.
   If the residual version is wanted later, it needs a daily market-return series and a
   rolling regression - not available from derived.parquet, which is ticker-level.
2. **Sign is -1.** High vol = low expected return. A composite that adds `rank_pct`
   raw will get this backwards; it must use `1 - rank_pct`.
3. Vol is computed on forward-filled prices but only kept on real-print sessions, so a
   halted microcap does not get an artificially calm vol estimate from a run of zeros.
""",
    ), inv)

    # ---------------------------------------------------------------- max_ret_1m
    s = inv[["ticker", "max_ret_1m"]].rename(columns={"max_ret_1m": "value"})
    signals["max_ret_1m"] = emit("max_ret_1m", s, dict(
        headline="LOTTERY MARKER - NOT A BUY SIGNAL. The single largest daily return in "
                 "the last 21 sessions. Bali, Cakici & Whitelaw (2011) document that "
                 "high-MAX stocks EARN LESS on average: investors overpay for "
                 "lottery-like right-tail payoffs. The top of this ranking is the "
                 "sell/avoid end.",
        osap="MaxRet", osap_desc="Maximum return over month, Bali, Cakici, and Whitelaw 2011, "
             "Detailed Definition: \"Maximum of daily returns (ret) over the previous month\". "
             "OSAP notes: \"Doesn't work as well EW. Very nonlinear return vs decile\" - the "
             "effect is concentrated in the extreme decile, and needs value weighting.",
        sign=-1,
        formula="max_ret_1m = max(r_d over the last 21 sessions)\n"
                "  where r_d = adj_close[d]/adj_close[d-1] - 1 on real-print sessions only.",
        inputs="`max_ret_1m` from data/prices/derived.parquet.",
        dist=describe(inv["max_ret_1m"].dropna()),
        judgement="""
> [!warning] **Sample-definition variable, not a buy signal.**
> This is built and ranked because the composite needs it as a *negative* screen and
> because it defines the lottery-stock sample. Reading a high `rank_pct` here as
> "attractive" inverts the documented effect. Sign is **-1**: the composite must use
> `1 - rank_pct`, and a defensible alternative use is a hard exclusion of the top decile
> rather than a linear tilt.

1. **The effect is very nonlinear (OSAP's own note) and works value-weighted, not
   equal-weighted.** A linear rank tilt across the whole cross-section is the *weakest*
   way to use it. Treat the top decile as the live part of the signal.
2. **It is mechanically entangled with `vol_60d`.** A single-day max return is largely a
   draw from the same distribution that produces high realised vol; the Spearman
   correlation between the two is printed in the build log and is high. Loading both into
   a composite double-counts volatility. Cluster analysis (`research/out/pool_clusters.csv`)
   should decide which one survives.
3. **21 sessions, not a calendar month.** Same convention as everything else here.
4. Upstream split repair matters more for this signal than any other: one unapplied split
   is a single +10,000% "daily" return, which would put a clean stock at rank 1.0000 on a
   lottery screen. 447 unapplied splits were repaired and 523 tickers quarantined before
   this was computed.
""",
    ), inv)

    # ---------------------------------------------------------------- volume_trend
    vt = osap_volume_trend(inv_t)
    ratio = inv[["ticker", "volume_trend"]].rename(columns={"volume_trend": "ratio"})
    cmp_ = vt.merge(ratio, on="ticker", how="inner").dropna()
    rho_vt = cmp_["value"].corr(cmp_["ratio"], method="spearman")
    signals["volume_trend"] = emit("volume_trend", vt, dict(
        headline="Trend in trading volume: the slope of monthly dollar volume regressed "
                 "on a linear time trend, scaled by average monthly volume. Haugen & "
                 "Baker (1996) find rising volume predicts LOWER returns; sign is NEGATIVE.",
        osap="VolumeTrend", osap_desc="Volume Trend, Haugen and Baker 1996, Detailed "
             "Definition: \"Rolling coefficient from regressing monthly trading volume on a "
             "linear time trend over a window of 60 months (require that at least 30 exist). "
             "Scale coefficient by 60-month average of trading volume.\" Rebalanced annually "
             "(Portfolio Period 12) in OSAP - this is a slow signal.",
        sign=-1,
        formula="V_m  = sum of (close x volume) over calendar month m      [monthly dollar volume]\n"
                "m    = 2023-09 .. 2026-07 (the 35 complete months in the 3y panel)\n"
                "b    = OLS slope of V_m on the calendar month index t_m\n"
                "volume_trend = b / mean(V_m)\n\n"
                "Requires >= 24 months present with non-zero volume and requires the final\n"
                "month (2026-07) to be present, so the trend is not measured on a stale name.\n"
                "Units: fraction of average monthly volume added per month.",
        inputs="`data/prices/history_3y.parquet` (date, ticker, close, volume) - the raw "
               "recorder output, aggregated to monthly dollar volume.",
        dist=describe(vt["value"]),
        judgement=f"""
1. **The window is 35 months, not OSAP's 60 - the panel is only 3 years deep.** The
   minimum-observation requirement was scaled proportionally (24 of 35 = 69%, vs OSAP's
   30 of 60 = 50%, so this is if anything stricter). A 35-month slope is a noisier estimate
   of a "volume trend" than a 60-month one and this is the single largest deviation from
   the catalog in this signal group. It cannot be fixed without more history; the recorder
   has been live since 2026-08-17, so it fixes itself over time.
2. **Dollar volume, not share volume.** OSAP uses CRSP `vol` (shares). Yahoo's raw share
   volume is not split-invariant and `history_3y.parquet` is the *unrepaired* record - 447
   unapplied splits were detected upstream. At an unapplied split, share volume steps by
   ~1/k and would be read as a volume trend that is pure corporate action. `close x volume`
   is invariant to that. This is a correctness fix, not a preference.
3. **Not the `volume_trend` column in derived.parquet.** That column is
   `dollar_vol_60d / median(close x volume, last 250)` - a 60d-vs-250d ratio, a fast
   volume-surge measure, not the catalog's regression slope. Spearman between the two
   over the {len(cmp_):,} names holding both is **{rho_vt:.3f}**, i.e. they are related but
   are not the same signal. The catalog definition was implemented because the task
   requires the catalog definition; the ratio remains available in derived.parquet if a
   fast volume-surge variant is wanted.
4. **Partial months at both edges are dropped** (2023-08 starts on the 10th, 2026-08 ends
   on the 17th). Including them would put a large artificial down-tick at the end of every
   single ticker's series and bias the whole cross-section negative.
5. **Coverage is structurally lower than the other price signals** because 24 months of
   history is required: recent IPOs cannot have this signal at all. That is missing data,
   not a zero - it is left absent rather than imputed.
6. Sign is **-1**: rising volume predicts lower returns. The composite must use
   `1 - rank_pct`.
""",
    ), inv)

    # ---------------------------------------------------------------- rel_strength
    med12 = inv["ret_12m"].median()
    sub = inv.dropna(subset=["ret_12m"]).copy()
    sub["value"] = sub["ret_12m"] - med12
    rho_rs = (sub[["ticker", "value"]]
              .merge(inv[["ticker", "ret_12m_ex1m"]], on="ticker", how="inner")
              .dropna()[["value", "ret_12m_ex1m"]]
              .corr(method="spearman").iloc[0, 1])
    signals["rel_strength"] = emit("rel_strength", sub[["ticker", "value"]], dict(
        headline="Relative strength: the stock's 12-month total return minus the "
                 "equal-weighted median 12-month return of the investable universe. "
                 "Positive = outperformed the typical US stock over the last year.",
        osap="(none - not an OSAP signal)",
        osap_desc="This is a cross-sectional recentring of a 12-month return, not a "
                  "catalogued predictor. The nearest catalog entries are `Mom12m` "
                  "(t-12..t-1) and `IntMom` (t-12..t-6). No Detailed Definition exists to "
                  "follow, so the formula is exactly as specified in the task.",
        sign=+1,
        formula=f"rel_strength = ret_12m - median(ret_12m over the investable universe)\n"
                f"  ret_12m = P[t]/P[t-252] - 1  (does NOT skip the recent month)\n"
                f"  universe median on {ASOF} = {med12:+.6f}  ({med12:+.2%})",
        inputs="`ret_12m` from data/prices/derived.parquet, recentred on the "
               "equal-weighted universe median.",
        dist=describe(sub["value"]),
        judgement=f"""
> [!warning] **This signal carries no cross-sectional information beyond `ret_12m`.**
> Subtracting a single constant from every stock is a monotone transform, so `rank_pct`
> here is **identical** to the rank of the raw 12-month return. Its value is the *level*
> (is this stock beating the median stock, yes/no, and by how much), which is useful as a
> filter or a reporting number, not as an extra column in a rank composite.

1. **It is close to, but not the same as, `mom_12_1`.** `rel_strength` uses the full
   t-0..t-12 window; `mom_12_1` skips the most recent month. Spearman between the two is
   **{rho_rs:.3f}** on this cross-section. Loading both into a composite is close to
   double-weighting momentum. If only one 12-month momentum column survives, it should be
   `mom_12_1` - the skipped month is what removes short-term reversal contamination, and
   that is the version with the documented premium.
2. **The median is over the investable universe** ({len(inv):,} names after the liquidity
   floor), not over all 9,687 tickers in the price panel. The unfiltered median is
   dominated by illiquid microcaps and is not a benchmark anyone could hold. Both numbers
   are printed in the build log.
3. **Median, not mean**, as specified. The mean 12m return of a US microcap-heavy
   cross-section is pulled hard by right-tail names and is not a robust centre.
4. Equal-weighted, as specified - so the benchmark is "the typical stock", not "the
   market". A cap-weighted benchmark would be a different (and in 2025-26, much harder)
   bar to clear.
""",
    ), inv)

    # ---------------------------------------------------------------- sanity report
    hr("COVERAGE")
    log(f"{'signal':<18s} {'n':>7s} {'% of universe':>14s}  {'min':>10s} {'median':>10s} {'max':>12s}")
    for k, v in signals.items():
        log(f"{k:<18s} {len(v):>7,} {len(v)/len(inv):>13.1%}  "
            f"{v['value_raw'].min():>10.3f} {v['value_raw'].median():>10.3f} {v['value_raw'].max():>12.3f}")
        if len(v) < 1500:
            log(f"  *** WARNING: coverage below the 1,500-ticker gate for {k} ***")

    hr("TOP 10 / BOTTOM 10 PER SIGNAL  (eyeball test)")
    info = inv.set_index("ticker")
    for k, v in signals.items():
        log(f"\n--- {k} ---")
        for label, part in (("TOP   ", v.head(10)), ("BOTTOM", v.tail(10).iloc[::-1])):
            for _, r in part.iterrows():
                t = r["ticker"]
                log(f"  {label} {t:<7s} raw={r['value_raw']:>12.4f}  rank={r['rank_pct']:.4f}  "
                    f"px=${info.at[t,'last_close']:>9,.2f}  $vol={info.at[t,'dollar_vol_60d']/1e6:>9,.1f}M  "
                    f"{str(info.at[t,'title'])[:40]}")
            log("")

    hr("SPEARMAN RANK CORRELATION BETWEEN SIGNALS")
    wide = None
    for k, v in signals.items():
        s = v.set_index("ticker")["rank_pct"].rename(k)
        wide = s.to_frame() if wide is None else wide.join(s, how="outer")
    log(wide.corr(method="spearman").round(3).to_string())

    hr("EXTRA CROSS-CHECKS")
    raw6 = inv.set_index("ticker")["ret_6m"]
    j = signals["mom_6m"].set_index("ticker")["value_raw"].to_frame("mom_6m").join(raw6)
    log(f"Spearman(mom_6m [skips t-1], raw ret_6m [does not]) = "
        f"{j.corr(method='spearman').iloc[0,1]:.4f}   n={len(j.dropna()):,}")
    log(f"universe median ret_12m (investable, n={inv['ret_12m'].notna().sum():,})"
        f" = {med12:+.4f}")
    allm = m["ret_12m"].median()
    log(f"universe median ret_12m (ALL 9,687 panel tickers)      = {allm:+.4f}  "
        f"<- illiquid-dominated, not used")
    small = inv[inv["dollar_vol_60d"] < 5e6]
    log(f"names in universe under $5M/day: {len(small):,} of {len(inv):,} "
        f"({len(small)/len(inv):.0%}) - the floor is a floor, not a large-cap screen")
    log(f"\nwrote {len(signals)*2} files to {OUT}")


if __name__ == "__main__":
    main()
