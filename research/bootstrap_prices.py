"""Build the 3-year price panel from scratch when it is absent.

GitHub runners are stateless: every run starts from the repo. refresh_prices.py
appends to an existing panel, so on a cold runner there is nothing to append to.
The panel is too large and too churny to keep in git (97MB rewritten daily), so
it lives in the Actions cache and this rebuilds it on a cache miss.

A panel that exists is not the same as a panel that is whole. The first cloud
bootstrap (2026-10-06) seeded 3y history only for the 4,897 tickers in the latest
weekday snapshot; every other name got a handful of snapshot rows. That panel
passed "present", and orders.py silently skipped top-ranked names it could not
price, so the cloud book differed from the Mac's by five names. So depth is
checked, and a shallow panel is replaced, not trusted.

The replacement is the Mac's own panel, published as the `panel-seed` release
asset: the 2026-08-18 full pull plus every snapshot since, 10,207 tickers, 7,777
with 500+ days. Re-downloading from Yahoo cannot reproduce it -- delisted names
are purged there. refresh_prices.py folds the committed snapshots on top.
"""
import pandas as pd, numpy as np, pathlib, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT/"data/prices/history_3y.parquet"
MIN_DEEP = 7000      # tickers with 500+ days; the seed has 7,777, the bad panel had <4,900

def deep(d):
    return int((d.groupby("ticker").size() >= 500).sum())

if OUT.exists():
    d = pd.read_parquet(OUT, columns=["ticker"])
    n = deep(d)
    if n >= MIN_DEEP:
        print(f"panel present: {len(d):,} rows, {d.ticker.nunique():,} tickers, {n:,} with 500+ days — nothing to do")
        sys.exit(0)
    print(f"panel SHALLOW: {len(d):,} rows, {d.ticker.nunique():,} tickers, only {n:,} with 500+ days "
          f"(need {MIN_DEEP:,}) — replacing it")
    OUT.unlink()

OUT.parent.mkdir(parents=True, exist_ok=True)
seed = OUT.with_name("history_3y_seed.parquet")
r = subprocess.run(["gh", "release", "download", "panel-seed", "--pattern", seed.name,
                    "--dir", str(seed.parent), "--clobber"], capture_output=True, text=True)
if r.returncode == 0 and seed.exists():
    d = pd.read_parquet(seed)
    n = deep(d)
    if n >= MIN_DEEP:
        seed.rename(OUT)
        print(f"seeded from release panel-seed: {len(d):,} rows, {d.ticker.nunique():,} tickers, "
              f"{n:,} with 500+ days, through {pd.to_datetime(d.date).max().date()}")
        sys.exit(0)
    print(f"release seed is itself shallow ({n:,} deep tickers) — falling back to Yahoo")
else:
    print(f"release seed unavailable ({r.stderr.strip()[:200]}) — falling back to Yahoo")

import yfinance as yf
snaps = sorted((ROOT/"data/snapshots").glob("ohlcv_*.parquet"))
if not snaps:
    print("no snapshots to seed from"); sys.exit(1)
# every name any snapshot ever saw -- the latest weekday snapshot alone is the
# ~5,100 tradable names and leaves the rest of the panel without history
tickers = sorted(set().union(*(set(pd.read_parquet(s, columns=["ticker"]).ticker.astype(str)) for s in snaps)))
tickers = [t for t in tickers if t.isalpha()] + ["SPY", "QQQ", "IWM"]
print(f"bootstrapping 3y history for {len(tickers):,} tickers")

CHUNK, PAUSE, MIN_TRUST = 100, 5.0, 0.50
frames = []
for i in range(0, len(tickers), CHUNK):
    c = tickers[i:i+CHUNK]
    for attempt, backoff in enumerate((0, 30, 90), 1):
        if backoff: time.sleep(backoff)
        try:
            raw = yf.download(c, period="3y", progress=False, threads=True,
                              group_by="ticker", auto_adjust=False)
        except Exception as e:
            print(f"  chunk {i//CHUNK+1}: {type(e).__name__}"); continue
        lv = set(raw.columns.get_level_values(0)) if hasattr(raw.columns, "get_level_values") else set()
        got = [t for t in c if t in lv and not raw[t].dropna(how="all").empty]
        if len(got)/len(c) >= MIN_TRUST or attempt == 3:
            for t in got:
                df = raw[t].dropna(how="all").reset_index()
                df.columns = [str(x).lower().replace(" ", "_") for x in df.columns]
                df["ticker"] = t
                frames.append(df[[x for x in ("date","open","high","low","close","adj_close","volume","ticker") if x in df.columns]])
            print(f"  chunk {i//CHUNK+1}/{(len(tickers)+CHUNK-1)//CHUNK}: {len(got)}/{len(c)}")
            break
    time.sleep(PAUSE)

if not frames:
    print("bootstrap produced nothing"); sys.exit(1)
P = pd.concat(frames, ignore_index=True)
P["date"] = pd.to_datetime(P.date)
P = P.sort_values(["ticker","date"]).drop_duplicates(["ticker","date"], keep="last")
P.to_parquet(OUT)
print(f"wrote {len(P):,} rows, {P.ticker.nunique():,} tickers, {P.date.min().date()} -> {P.date.max().date()}")
# Not exit 1: the snapshot step after this is the irreplaceable one and must still run.
# But a shallow Yahoo rebuild is the exact failure this file exists to catch, so say so.
if deep(P) < MIN_DEEP:
    print(f"::warning::Yahoo bootstrap is shallow ({deep(P):,} tickers with 500+ days, need {MIN_DEEP:,}). "
          f"The book will skip unpriced names until the panel-seed release is restored.")
