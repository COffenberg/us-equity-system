"""Composite score + ranked lists. Weight by CLUSTER, not by signal."""
import pandas as pd, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = pd.read_parquet(ROOT/"signals/signal_raw.parquet")
_et = pd.read_parquet(ROOT/"signals/earnings_timing.parquet")
RAW = RAW.join(_et[["AnnouncementReturn"]], how="left")
_ow = pd.read_parquet(ROOT/"signals/ownership.parquet")
RAW = RAW.join(_ow[["inst_holders","inst_d_holders"]], how="left")
R   = pd.read_parquet(ROOT/"signals/signal_ranks.parquet")
SEC = pd.read_parquet(ROOT/"data/sec/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector
SEC = SEC.reindex(RAW.index).fillna("Other")

# Sector-relative ranking. A P/E of 12 is cheap for tech and ordinary for a bank,
# so ranking valuation across the whole market mostly ranks sectors, not stocks.
# Rank each signal inside its own sector; fall back to the market where a sector
# is too thin to rank against. This also does most of the work of conditioning
# bank accounting: "more levered than other banks" is a real statement, where
# "more levered than a software company" is not.
MIN_SECTOR = 25
big = SEC.value_counts()[lambda x: x >= MIN_SECTOR].index
grp = SEC.where(SEC.isin(big), "Other")
def _rank_in_sector(col):
    out = RAW[col].groupby(grp).rank(pct=True)
    return out.fillna(RAW[col].rank(pct=True))          # thin sectors -> market rank
R_rel = pd.DataFrame({c: _rank_in_sector(c) for c in RAW.columns})
R_rel["mktcap"] = R["mktcap"]; R_rel["runway_months"] = R["runway_months"]
R = R_rel
print(f"sector-relative ranks over {len(big)} sectors "
      f"({grp.value_counts().get('Other',0)} names pooled as Other)")
D = pd.read_parquet(ROOT/"data/prices/derived.parquet").set_index("ticker")
U = pd.read_csv(sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1], dtype={"cik":str}) \
      .drop_duplicates("ticker").set_index("ticker").title
PX = pd.read_parquet(ROOT/"data/prices/history_3y.parquet") \
       .sort_values("date").groupby("ticker").last().close \
       .combine_first(pd.read_parquet(ROOT/"data/snapshots/ohlcv_2026-08-18.parquet")
                        .sort_values("date").groupby("ticker").last().close)

# One bet gets one vote. The payout/financing signals are ~one idea measured
# several ways; averaging inside the cluster stops it casting eight votes.
CLUSTERS = {
    # Rebuilt 2026-09-22. Every prior composite was screened on quintile SPREAD
    # (Q5 minus Q1), which rewards a signal whose BOTTOM decile is terrible --
    # irrelevant to a long-only book that never holds the bottom. Rescreening all
    # 41 signals on the real objective (40-name book vs SPY) left exactly two
    # above a noise null, and both are value. Several signals promoted under the
    # old screen measured among the WORST here: asset_growth -4.91%, d_noa -5.30%,
    # Mom6m -3.71%. They are removed.
    #
    #   SP  +4.75% vs SPY, 86% of quarters, p=2% vs noise
    #   bm  +4.19% vs SPY, 71% of quarters, p=2% vs noise
    #   cfp adds +0.6pp in combination and is the same economic idea measured on cash
    #
    # Full book: +5.71% per quarter, 71% win, t=2.09 over 7 quarters. Six of seven
    # positive; robust to dropping any single quarter (t stays 1.55-2.61); but two
    # strong quarters carry most of it, and without them t falls to 1.15.
    "value": ["SP", "bm", "cfp"],
}
SKEW = ["px_52w", "volume_trend"]

cl = pd.DataFrame({k: R[[c for c in v if c in R]].mean(axis=1) for k, v in CLUSTERS.items()})
cl["momentum"] = R[["mom_12_1","mom_6m","rel_strength"]].mean(axis=1)   # reported, not scored in mean
n_cl = cl[list(CLUSTERS)].notna().sum(axis=1)
# one cluster now, so guard on signal depth instead of cluster count
_vs = R[[c for c in CLUSTERS["value"] if c in R]].notna().sum(axis=1)
mean_term = cl[list(CLUSTERS)].mean(axis=1)
skew_term = R[[c for c in SKEW if c in R]].mean(axis=1)

# ---------- universe filter ----------
liq   = D.dollar_vol_60d.reindex(R.index)
price = PX.reindex(R.index)
age   = D.listing_age_days.reindex(R.index)
runway = R.runway_months

# --- exclude non-operating companies -------------------------------------
# Commodity/currency ETFs file 10-Ks, so the domestic-filer test lets them
# through: run 1 put UGA, CPER and UUP in the top 10 with an accruals rank.
NM = U.reindex(R.index).fillna("").str.upper()
is_fund = NM.str.contains(r"\bFUND\b|\bETF\b|\bINDEX\b|\bTRUST\b|\bPORTFOLIO\b|SHARES ETF", regex=True)
# Banks and insurers: the accounting signals here (accruals, leverage, roa)
# do not mean the same thing on a bank balance sheet. Until definitions are
# conditioned by archetype, score them separately rather than pretend.
is_fin = pd.Series(False, index=R.index)   # kept for reporting; no longer excluded
lev_raw = -R.leverage if "leverage" in R else pd.Series(np.nan, index=R.index)

keep = pd.Series(True, index=R.index)
steps = []
for name, cond in [
    ("median $ volume >= $500k", liq >= 5e5),
    ("price >= $1",              price >= 1),
    # A near-zero market cap sends every /mktcap ratio to +inf and scores a
    # perfect 1.00 on value AND financing. Athene Holding (acquired 2022) did
    # exactly this at rank 3 of CORE before this floor existed.
    ("market cap >= $50m",       R.mktcap >= 50e6),
    ("listed >= 250 days",       (age >= 250) | age.isna()),
    ("runway >= 12mo or profitable", (runway >= 12) | runway.isna()),
    ("2+ of the 3 value signals",     _vs >= 2),
    ("not a fund/ETF",             ~is_fund),
]:
    before = keep.sum(); keep &= cond.fillna(False) if cond.dtype == bool else cond
    steps.append((name, before, keep.sum()))

print("UNIVERSE FILTER")
print(f"  {'start':38s} {len(R):6d}")
for n, b, a in steps: print(f"  {n:38s} {a:6d}   (-{b-a})")

out = pd.DataFrame({
    "name": U.reindex(R.index), "mktcap_bn": R.mktcap/1e9, "price": price,
    "mean": mean_term, "skew": skew_term, "n_clusters": n_cl,
}).join(cl.add_prefix("c_")).join(R[[c for l in CLUSTERS.values() for c in l if c in R]])
out = out[keep]
out["sector"] = SEC.reindex(out.index).fillna("Other")
out = out.join(_et[["days_to_next_earnings","in_pre_earnings_window",
                    "ann_ret_hit_rate","ann_ret_n"]], how="left")
out["CORE"]     = out["mean"]
out["SKEWLIST"] = 0.5*out["mean"] + 0.5*out["skew"]
out["COMBINED"] = 0.7*out["mean"] + 0.3*out["skew"]
out.to_parquet(ROOT/"signals/ranked.parquet")

# Sector-relative ranking equalises the yardstick but not the outcome: sectors
# differ in internal dispersion, so a bimodal one (a profitable biotech among
# cash-burners) still floods the top. A cap is a CONSTRUCTION decision, not a
# scoring one -- ranks stay untouched, positions get spread.
MAX_PER_SECTOR = 6

def top(col, n=40, cap=True):
    if not cap: return out.nlargest(n, col)
    picked, counts = [], {}
    for t, r in out.sort_values(col, ascending=False).iterrows():
        sec = r["sector"]
        if counts.get(sec, 0) >= MAX_PER_SECTOR: continue
        picked.append(t); counts[sec] = counts.get(sec, 0) + 1
        if len(picked) == n: break
    return out.loc[picked].sort_values(col, ascending=False)

def fmt(df, col):
    L = []
    for i, (t, r) in enumerate(df.iterrows(), 1):
        drivers = r[[f"c_{k}" for k in CLUSTERS]].astype(float).dropna().nlargest(3)
        d = ", ".join(f"{k[2:]} {v:.2f}" for k, v in drivers.items())
        nm = (r["name"] or "")[:30]
        L.append(f"{i:3d}. {t:6s} {nm:32s} ${r.mktcap_bn:7.1f}bn  {r[col]:.3f}   {d}")
    return "\n".join(L)

lines = [f"# Ranked lists — {pd.Timestamp('2026-08-19').date()}", "",
         f"Excluded {int(is_fund.sum())} funds/ETFs. Banks and insurers are now scored "
         f"inside their own sector rather than excluded.", "",
         f"Scored {keep.sum()} of {len(R)} names. Clusters: {', '.join(CLUSTERS)}.", ""]
for key, title in [("CORE","CORE — mean term only"),
                   ("SKEWLIST","SKEW — 50/50 mean and upside asymmetry"),
                   ("COMBINED","COMBINED — 70/30")]:
    lines += [f"## {title}", "```", fmt(top(key), key), "```", ""]

ov = lambda a,b: len(set(top(a).index) & set(top(b).index))
lines += ["## Overlap", "",
          f"- CORE vs SKEW: {ov('CORE','SKEWLIST')} of 40 shared",
          f"- CORE vs COMBINED: {ov('CORE','COMBINED')} of 40",
          f"- SKEW vs COMBINED: {ov('SKEWLIST','COMBINED')} of 40", ""]
(ROOT/"signals/RANKED.md").write_text("\n".join(lines))
print()
print("\n".join(lines[-6:]))
print(f"\nmedian mktcap of COMBINED top40: ${top('COMBINED').mktcap_bn.median():.2f}bn")
print(f"wrote signals/RANKED.md")
