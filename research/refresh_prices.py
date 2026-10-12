"""Fold the recorder's daily snapshots into the 3-year price panel.

The panel was pulled once on 2026-08-18 and never refreshed, so the ranking ran
on 19-day-old prices and a forward record was nearly frozen with stale entries.
The recorder already captures everything needed; it just was not being read.
Now the snapshots ARE the update mechanism, which is what they were for.
"""
import pandas as pd, pathlib, glob, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

hist = pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
hist["date"] = pd.to_datetime(hist.date)
before_max, before_rows = hist.date.max(), len(hist)

snaps = []
for f in sorted(glob.glob(str(ROOT/"data/snapshots/ohlcv_*.parquet"))):
    d = pd.read_parquet(f); d["date"] = pd.to_datetime(d.date)
    snaps.append(d[[c for c in hist.columns if c in d.columns]])
S = pd.concat(snaps, ignore_index=True)

merged = (pd.concat([hist, S], ignore_index=True)
            .sort_values(["ticker", "date"])
            .drop_duplicates(["ticker", "date"], keep="last"))
merged.to_parquet(ROOT/"data/prices/history_3y.parquet")
print(f"price panel: {before_rows:,} -> {len(merged):,} rows")
print(f"  last date  {before_max.date()} -> {merged.date.max().date()}")
print(f"  tickers    {hist.ticker.nunique():,} -> {merged.ticker.nunique():,}")

# derived features are computed FROM the panel, so they are stale too
import numpy as np
P = merged.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
C = merged.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
V = merged.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").sort_index()
last = P.iloc[-1]
def back(n): return P.iloc[max(0, len(P)-n-1)]
d = pd.DataFrame({
    "ret_1m":  last/back(21)-1,
    "ret_6m":  last/back(126)-1,
    "ret_12m": last/back(252)-1,
})
d["ret_12m_ex1m"] = back(21)/back(252)-1
r = P.pct_change()
d["vol_60d"]     = r.iloc[-60:].std()*np.sqrt(252)
d["max_ret_1m"]  = r.iloc[-21:].max()
d["px_to_52w_high"]    = last/P.iloc[-252:].max()
d["drawdown_from_peak"]= last/P.max()-1
dv = (C*V)
d["dollar_vol_60d"]  = dv.iloc[-60:].median()
d["volume_trend"]    = dv.iloc[-60:].median()/dv.iloc[-250:].median()
d["listing_age_days"]= P.notna().sum()*(365/252)
d = d.replace([np.inf,-np.inf], np.nan).reset_index().rename(columns={"index":"ticker"})
d.to_parquet(ROOT/"data/prices/derived.parquet")
print(f"derived features rebuilt: {len(d):,} tickers, {d.ret_12m.notna().sum():,} with 12m history")
