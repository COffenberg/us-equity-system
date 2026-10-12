"""Coverage + drop taxonomy + liquidity distribution for the price panel."""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "prices"

snap = pd.read_parquet(ROOT / "data/snapshots/ohlcv_2026-08-18.parquet", columns=["ticker"])
rec = pd.read_csv(ROOT/"data/snapshots/status_2026-08-18.csv")
uni = sorted(set(snap.ticker.astype(str)) | set(pd.read_parquet(P/"history_3y.parquet", columns=["ticker"]).ticker.astype(str)))
hist = pd.read_parquet(P / "history_3y.parquet")
der = pd.read_parquet(P / "derived.parquet")
diag = pd.read_parquet(P / "derived_diagnostics.parquet")
st = pd.read_csv(P / "history_3y_status.csv")
d = der.merge(diag, on="ticker")

print(f"universe (snapshot)      {len(uni)}")
print(f"history_3y rows          {len(hist):,}")
print(f"history_3y tickers       {hist.ticker.nunique()}")
print(f"date range               {hist.date.min().date()} .. {hist.date.max().date()}")
print(f"derived rows             {len(der)}")

# ---- drop taxonomy (mutually exclusive, evaluated in order) ----
reason = pd.Series("usable_12m", index=pd.Index(uni, name="ticker"))
thr = set(st.loc[st.reason.astype(str) == "throttled", "ticker"])
nod = set(st.loc[st.reason.astype(str) == "no_data", "ticker"])
have = set(der.ticker)
di = d.set_index("ticker")

reason[reason.index.isin(nod)] = "no_3y_data_from_yfinance"
reason[reason.index.isin(thr)] = "throttled_untrusted"
reason[[t not in have and reason[t] == "usable_12m" for t in reason.index]] = "absent_from_panel"

idx = reason.index[reason == "usable_12m"]
sub = di.reindex(idx)
reason[sub.index[sub.unadjustable.fillna(False)]] = "corrupt_adjustment"
idx = reason.index[reason == "usable_12m"]
sub = di.reindex(idx)
reason[sub.index[sub.quarantined.fillna(False)]] = "quarantined_unexplained_jump"
idx = reason.index[reason == "usable_12m"]
sub = di.reindex(idx)
young = sub.index[sub.listing_age_days < 365]
reason[young] = "listed_under_12m"
idx = reason.index[reason == "usable_12m"]
sub = di.reindex(idx)
gap = sub.index[sub.ret_12m.isna()]
reason[gap] = "gap_at_12m_endpoint"

vc = reason.value_counts()
print("\n--- drop taxonomy ---")
for k, v in vc.items():
    print(f"{k:28s} {v:6d}")
print(f"\nusable ret_12m (non-null)  {int(der.ret_12m.notna().sum())}")
print(f"usable ret_12m_ex1m        {int(der.ret_12m_ex1m.notna().sum())}")

print("\n--- coverage per column ---")
for c in der.columns:
    if c == "ticker":
        continue
    print(f"{c:22s} {int(der[c].notna().sum()):6d}  ({der[c].notna().mean():5.1%})")

dv = der.dollar_vol_60d.dropna()
print(f"\n--- dollar_vol_60d (USD/day, median of last 60 sessions), n={len(dv)} ---")
for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    print(f"p{int(q*100):02d}  {dv.quantile(q):>18,.0f}")
print(f"zero / non-trading        {int((dv <= 0).sum())}")
for f in [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]:
    print(f"survive >= ${f:>12,.0f}   {int((dv >= f).sum()):5d}  ({(dv>=f).mean():5.1%})")

print("\n--- sanity: extremes ---")
for c in ["ret_12m", "vol_60d", "px_to_52w_high", "drawdown_from_peak", "volume_trend", "max_ret_1m"]:
    s = der[c].dropna()
    print(f"{c:20s} min {s.min():12.4f}  p1 {s.quantile(.01):10.4f}  med {s.median():10.4f}  p99 {s.quantile(.99):12.4f}  max {s.max():14.4f}")

reason.rename("drop_reason").reset_index().to_csv(P / "coverage_reasons.csv", index=False)
print(f"\nwrote {P/'coverage_reasons.csv'}")
