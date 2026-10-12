"""Refetch the tickers the 2026-08-18 snapshot marked `no_data`.

The snapshot's trust floor rejects a chunk below 50% hit rate, but a chunk that
came back at, say, 65% still passes -- and every ticker missing from it is
recorded as `no_data`. MSFT and TSLA were lost exactly that way. `no_data` from
a single pass is therefore a hypothesis, not a fact; it is only credible after
an independent refetch also returns nothing.
"""
import datetime as dt, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from history_3y import fetch_chunk, CHUNK, PAUSE, YEARS, log

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "prices"
PARTS = P / "parts_recover"
PARTS.mkdir(parents=True, exist_ok=True)

st = pd.read_csv(ROOT / "data/snapshots/status_2026-08-18.csv")
missing = sorted(st.loc[st.reason.astype(str) == "no_data", "ticker"].astype(str).unique())
log(f"recovery: {len(missing)} tickers the snapshot called no_data")

end = dt.date.today() + dt.timedelta(days=1)
start = end - dt.timedelta(days=int(YEARS * 365.25) + 10)
chunks = [missing[i:i + CHUNK] for i in range(0, len(missing), CHUNK)]
for i, ch in enumerate(chunks, 1):
    pq, stp = PARTS / f"chunk_{i:04d}.parquet", PARTS / f"chunk_{i:04d}_status.csv"
    if pq.exists() and stp.exists():
        continue
    pr, stt = fetch_chunk(ch, i, len(chunks), start, end)
    pr.to_parquet(pq, index=False)
    pd.DataFrame(stt).to_csv(stp, index=False)
    time.sleep(PAUSE)

parts = [pd.read_parquet(p) for p in sorted(PARTS.glob("chunk_*.parquet"))]
parts = [p for p in parts if not p.empty]
rec = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
if rec.empty:
    log("recovery: nothing came back — the no_data verdict stands")
    sys.exit(0)

rec["date"] = pd.to_datetime(rec["date"], utc=True).dt.tz_localize(None).dt.normalize()
try:
    from zoneinfo import ZoneInfo
    now_et = dt.datetime.now(ZoneInfo("America/New_York"))
except Exception:
    now_et = dt.datetime.utcnow() - dt.timedelta(hours=4)
if now_et.hour < 16:
    rec = rec[rec["date"] < pd.Timestamp(now_et.date())]

rec = rec[["date", "ticker", "close", "adj_close", "volume"]]
rec = rec.drop_duplicates(subset=["ticker", "date"], keep="last")
log(f"recovery: {rec.ticker.nunique()} of {len(missing)} came back ({len(rec):,} rows)")

hist = pd.read_parquet(P / "history_3y.parquet")
merged = pd.concat([hist, rec[~rec.ticker.isin(set(hist.ticker))]], ignore_index=True)
merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
merged.to_parquet(P / "history_3y.parquet", index=False)
log(f"history_3y.parquet now {len(merged):,} rows · {merged.ticker.nunique()} tickers")
