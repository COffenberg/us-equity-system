"""Pull 3 years of daily history for the snapshot universe.

Same throttle discipline as recorder/snapshot.py: a chunk whose hit rate falls
below MIN_TRUST_RATE is NOT trusted to assert absence. It is retried with
escalating backoff and, failing that, recorded as `throttled` so a later re-run
refetches it. Chunk parts are written incrementally -> the run resumes.

OUTPUT
    data/prices/parts_3y/chunk_NNNN.parquet     resume state
    data/prices/parts_3y/chunk_NNNN_status.csv
    data/prices/history_3y.parquet              [date, ticker, close, adj_close, volume]
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "snapshots" / "ohlcv_2026-08-18.parquet"
OUT_DIR = ROOT / "data" / "prices"
PARTS = OUT_DIR / "parts_3y"
LOG_PATH = ROOT / "data" / "history_3y.log"

CHUNK = 150
PAUSE = 2.0
MIN_TRUST_RATE = 0.50
BACKOFFS = (30, 90, 240)
YEARS = 3


def log(msg: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(line + "\n")


def _normalise(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for tkr in tickers:
            if tkr not in available:
                continue
            sub = raw[tkr].dropna(how="all")
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub["ticker"] = tkr
            frames.append(sub)
    else:
        sub = raw.dropna(how="all").reset_index()
        sub["ticker"] = tickers[0]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    keep = [c for c in ["date", "ticker", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep]
    # a row with no close is not an observation
    if "close" in df.columns:
        df = df[df["close"].notna()]
    return df


def _download(chunk: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        chunk,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    return _normalise(raw, chunk)


def fetch_chunk(chunk, idx, total, start, end):
    best, best_rate = pd.DataFrame(), -1.0
    for attempt in range(len(BACKOFFS) + 1):
        try:
            part = _download(chunk, start, end)
        except Exception as exc:
            log(f"  chunk {idx}/{total} attempt {attempt+1} error: {type(exc).__name__}: {exc}")
            part = pd.DataFrame()
        got = set(part["ticker"].unique()) if not part.empty else set()
        rate = len(got) / len(chunk)
        if rate > best_rate:
            best, best_rate = part, rate
        if rate >= MIN_TRUST_RATE:
            tag = f"recovered on attempt {attempt+1} " if attempt else ""
            log(f"  chunk {idx}/{total}: {tag}{len(got)}/{len(chunk)} · {len(part)} rows")
            return part, [
                {"ticker": t, "ok": t in got, "reason": "" if t in got else "no_data"} for t in chunk
            ]
        if attempt < len(BACKOFFS):
            wait = BACKOFFS[attempt]
            log(f"  chunk {idx}/{total}: hit rate {rate:.0%} below trust floor — backing off {wait}s")
            time.sleep(wait)
    got = set(best["ticker"].unique()) if not best.empty else set()
    log(f"  chunk {idx}/{total}: STILL THROTTLED after {len(BACKOFFS)} retries — marking throttled")
    return best, [
        {"ticker": t, "ok": t in got, "reason": "" if t in got else "throttled"} for t in chunk
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-throttled", action="store_true",
                    help="delete parts whose status has any throttled row, then resume")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.force and PARTS.exists():
        shutil.rmtree(PARTS)
    PARTS.mkdir(parents=True, exist_ok=True)

    if args.retry_throttled:
        for st in sorted(PARTS.glob("chunk_*_status.csv")):
            s = pd.read_csv(st)
            if (s["reason"].astype(str) == "throttled").any():
                st.unlink()
                pq = st.with_name(st.name.replace("_status.csv", ".parquet"))
                if pq.exists():
                    pq.unlink()
                log(f"cleared throttled part {st.name}")

    snap = pd.read_parquet(SNAP, columns=["ticker"])
    tickers = sorted(snap["ticker"].astype(str).unique().tolist())
    if args.limit:
        tickers = tickers[: args.limit]

    chunks = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=int(YEARS * 365.25) + 10)
    log(f"history: {len(tickers)} tickers, {len(chunks)} chunks, {start} -> {end}")

    t0 = time.time()
    for i, chunk in enumerate(chunks, 1):
        pq = PARTS / f"chunk_{i:04d}.parquet"
        st = PARTS / f"chunk_{i:04d}_status.csv"
        if pq.exists() and st.exists():
            continue
        prices, status = fetch_chunk(chunk, i, len(chunks), start, end)
        prices.to_parquet(pq, index=False)
        pd.DataFrame(status).to_csv(st, index=False)
        time.sleep(PAUSE)

    parts = [pd.read_parquet(p) for p in sorted(PARTS.glob("chunk_*.parquet"))]
    parts = [p for p in parts if not p.empty]
    status = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob("chunk_*_status.csv"))],
                       ignore_index=True)
    if not parts:
        log("ERROR: no data at all")
        return 1

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df[["date", "ticker", "close", "adj_close", "volume"]]
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

    # Drop the in-progress session. Yahoo serves a live partial bar while the US
    # market is open: understated volume and a close that is really a last trade.
    # Recording it would corrupt every 60d dollar-volume median and the anchor
    # return. Rule: if it is not yet 16:00 ET, no bar dated today is complete.
    try:
        from zoneinfo import ZoneInfo
        now_et = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = dt.datetime.utcnow() - dt.timedelta(hours=4)
    if now_et.hour < 16:
        cutoff = pd.Timestamp(now_et.date())
        n_drop = int((df["date"] >= cutoff).sum())
        if n_drop:
            log(f"dropping {n_drop} rows dated >= {cutoff.date()} "
                f"(market open at {now_et:%H:%M} ET — partial bar)")
            df = df[df["date"] < cutoff]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(OUT_DIR / "history_3y.parquet", index=False)
    status.to_csv(OUT_DIR / "history_3y_status.csv", index=False)

    thr = int((status["reason"].astype(str) == "throttled").sum())
    nod = int((status["reason"].astype(str) == "no_data").sum())
    log(f"WROTE history_3y.parquet: {len(df):,} rows, {df.ticker.nunique()} tickers, "
        f"{df.date.min().date()}..{df.date.max().date()} | no_data {nod} · throttled {thr} "
        f"| {(time.time()-t0)/60:.1f} min")
    if thr:
        log(f"  {thr} throttled — re-run with --retry-throttled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
