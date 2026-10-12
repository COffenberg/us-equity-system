"""Daily point-in-time snapshot of the US equity cross-section.

WHY THIS EXISTS
---------------
The selection system is validated forward, not by backtest, because no free
source provides survivorship-bias-free US prices (a claim verified
exhaustively in an earlier project). Forward validation only works
if point-in-time history actually accumulates -- and the *only* way to get that
for free is to record it as it happens.

SCOPING RULE: record only what cannot be reconstructed later.

  reconstructable  -> SEC filings. Permanent, timestamped, free. Fundamentals,
                      shares outstanding, insider transactions, filing text can
                      all be pulled from EDGAR at any future date, exactly as
                      they stood. NOT recorded here.
  perishable       -> prices and volume of names that later delist (Yahoo purges
                      delisted symbols entirely), and universe membership itself.
                      Recorded here, daily, never deleted.

That rule is what keeps this script small. It captures OHLCV + the roster, and
nothing else.

THROTTLING IS A CORRECTNESS PROBLEM, NOT A SPEED PROBLEM
--------------------------------------------------------
Observed 2026-08-17 on the first full run: Yahoo served ~94% hit rates for six
chunks, then collapsed to 7% and kept serving empty results fast. Recording that
naively would have marked ~7,000 tickers as having no data -- and in this
system's own semantics "no data" is how a delisting becomes visible. Because the
universe is alphabetical, it would also have corrupted the same slice every day.

So a chunk whose hit rate falls below MIN_TRUST_RATE is never trusted. It is
retried with escalating backoff, and if it still will not come back it is
recorded as `throttled`, which is explicitly NOT `no_data`. Only a chunk that
returned a plausible hit rate is allowed to assert that a ticker has no data.

Chunks are written incrementally, so a killed or throttled run can simply be
re-run: completed chunks are skipped and only the missing ones are refetched.

USAGE
-----
    python3 recorder/snapshot.py                # today's snapshot (resumes)
    python3 recorder/snapshot.py --limit 50     # smoke test
    python3 recorder/snapshot.py --force        # discard today's parts, refetch all

OUTPUT
------
    data/universe/tickers_YYYY-MM-DD.csv        roster as SEC reported it
    data/snapshots/ohlcv_YYYY-MM-DD.parquet     one row per (ticker, date)
    data/snapshots/status_YYYY-MM-DD.csv        per-ticker outcome
    data/snapshots/parts_YYYY-MM-DD/            per-chunk intermediates (resume state)
"""
from __future__ import annotations

import os
import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_DIR = ROOT / "data" / "universe"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
LOG_PATH = ROOT / "data" / "recorder.log"

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC requires a descriptive UA with contact info on every automated request.
SEC_UA = {"User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"]}

CHUNK = 100             # tickers per yfinance call; smaller = gentler on the endpoint
PAUSE = 5.0             # seconds between chunks (raised from 2.0 after Yahoo began
                        # throttling ~80% of chunks from 2026-08-25)
LOOKBACK_DAYS = 45      # NOT just "today with slack". A long window makes missed days
                        # self-healing: if the job fails or the Mac is off for a week,
                        # the next successful run back-fills every name that still
                        # exists. The only irreducible loss is a company that delisted
                        # inside the gap -- which is the whole point of the recorder,
                        # but it is a far smaller loss than the entire cross-section.
                        # (2026-08-29..09-05 were lost this way before this change.)
MIN_TRUST_RATE = 0.50   # below this, assume throttling rather than genuine absence
BACKOFFS = (30, 90, 240)  # seconds; escalating retry delays for a distrusted chunk


def log(msg: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(line + "\n")


def fetch_universe(today: str) -> pd.DataFrame:
    """SEC's own ticker->CIK roster. Free, no key, ~10k entries.

    Recorded verbatim rather than filtered: membership change is exactly the
    perishable thing we are here to capture, so filtering happens at read time,
    never at write time. Caching it per-day also keeps chunk boundaries stable
    across resumed runs.
    """
    out = UNIVERSE_DIR / f"tickers_{today}.csv"
    if out.exists():
        return pd.read_csv(out, dtype={"cik": str})

    log("fetching SEC company_tickers.json")
    resp = requests.get(SEC_TICKERS_URL, headers=SEC_UA, timeout=60)
    resp.raise_for_status()
    raw = json.loads(resp.text)

    rows = [
        {
            "cik": str(rec["cik_str"]).zfill(10),
            "ticker": str(rec["ticker"]).strip().upper(),
            "title": rec.get("title", ""),
        }
        for rec in raw.values()
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset="ticker").sort_values("ticker")

    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log(f"universe: {len(df)} tickers -> {out.name}")
    return df


def _normalise(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yfinance's MultiIndex shape varies by version and by how many tickers
    came back. Flatten defensively."""
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
    else:  # a single ticker survived the chunk
        sub = raw.dropna(how="all").reset_index()
        sub["ticker"] = tickers[0]
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    return df[keep]


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


def fetch_chunk(chunk: list[str], idx: int, total: int, start: dt.date, end: dt.date):
    """Fetch one chunk, retrying while the hit rate looks like throttling.

    Returns (prices, status_rows). A chunk that never clears MIN_TRUST_RATE is
    reported as `throttled` -- never as `no_data`, because only a trusted chunk
    is allowed to assert that a ticker has genuinely stopped trading.
    """
    best, best_rate = pd.DataFrame(), -1.0

    for attempt in range(len(BACKOFFS) + 1):
        try:
            part = _download(chunk, start, end)
        except Exception as exc:
            log(f"  chunk {idx}/{total} attempt {attempt + 1} error: {type(exc).__name__}: {exc}")
            part = pd.DataFrame()

        got = set(part["ticker"].unique()) if not part.empty else set()
        rate = len(got) / len(chunk)
        if rate > best_rate:
            best, best_rate = part, rate

        if rate >= MIN_TRUST_RATE:
            if attempt:
                log(f"  chunk {idx}/{total}: recovered on attempt {attempt + 1} ({len(got)}/{len(chunk)})")
            else:
                log(f"  chunk {idx}/{total}: {len(got)}/{len(chunk)}")
            trusted_got = got
            return part, [
                {"ticker": t, "ok": t in trusted_got, "reason": "" if t in trusted_got else "no_data"}
                for t in chunk
            ]

        if attempt < len(BACKOFFS):
            wait = BACKOFFS[attempt]
            log(
                f"  chunk {idx}/{total}: hit rate {rate:.0%} below trust floor "
                f"— assuming throttling, backing off {wait}s"
            )
            time.sleep(wait)

    got = set(best["ticker"].unique()) if not best.empty else set()
    log(f"  chunk {idx}/{total}: STILL THROTTLED after {len(BACKOFFS)} retries — marking throttled, not no_data")
    return best, [
        {"ticker": t, "ok": t in got, "reason": "" if t in got else "throttled"} for t in chunk
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="only first N tickers (smoke test)")
    ap.add_argument("--force", action="store_true", help="discard today's parts and refetch")
    ap.add_argument("--full", action="store_true", help="record the full SEC roster regardless of weekday")
    args = ap.parse_args()

    today = dt.date.today().isoformat()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    parts_dir = SNAPSHOT_DIR / f"parts_{today}"
    if args.force and parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    universe = fetch_universe(today)
    tickers = universe["ticker"].tolist()
    # Yahoo throttled ~80% of full-roster runs from 2026-08-25 and killed the job
    # outright on 09-08. Only ~5,000 of 10,400 names are tradable at any size, so
    # record those daily and sweep the full roster on Saturdays for delistings.
    if not args.full and dt.date.today().weekday() != 5:
        try:
            dv = pd.read_parquet(SNAPSHOT_DIR.parent / "prices" / "derived.parquet") \
                   .set_index("ticker").dollar_vol_60d
            keep = set(dv.index[dv >= 2.5e5])
            import json, glob
            for f in glob.glob(str(SNAPSHOT_DIR.parent.parent / "forward" / "prediction_*.json")):
                for lst in json.load(open(f))["lists"].values():
                    keep.update(n["ticker"] for n in lst)
            keep.update(["SPY", "QQQ", "IWM"])
            tickers = [t for t in tickers if t in keep] + [t for t in ("SPY","QQQ","IWM") if t not in set(tickers)]
            log(f"tradable-universe day: {len(tickers)} tickers (full roster on Saturdays)")
        except Exception as e:
            log(f"could not narrow universe ({e}); recording full roster")
    if args.limit:
        tickers = tickers[: args.limit]

    chunks = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    log(f"snapshot {today}: {len(tickers)} tickers, {len(chunks)} chunks of {CHUNK}")

    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    t0 = time.time()

    for i, chunk in enumerate(chunks, 1):
        pq = parts_dir / f"chunk_{i:04d}.parquet"
        st = parts_dir / f"chunk_{i:04d}_status.csv"
        if pq.exists() and st.exists():
            continue  # resume: already fetched

        prices, status = fetch_chunk(chunk, i, len(chunks), start, end)
        prices.to_parquet(pq, index=False)          # may be empty; presence marks completion
        pd.DataFrame(status).to_csv(st, index=False)
        time.sleep(PAUSE)

    price_parts = [pd.read_parquet(p) for p in sorted(parts_dir.glob("chunk_*.parquet"))]
    status_parts = [pd.read_csv(p) for p in sorted(parts_dir.glob("chunk_*_status.csv"))]
    prices = pd.concat([p for p in price_parts if not p.empty], ignore_index=True) if price_parts else pd.DataFrame()
    status = pd.concat(status_parts, ignore_index=True) if status_parts else pd.DataFrame()

    if prices.empty:
        log("ERROR: no price data at all — not writing an empty snapshot")
        return 1

    prices["date"] = pd.to_datetime(prices["date"]).dt.date.astype(str)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices.to_parquet(SNAPSHOT_DIR / f"ohlcv_{today}.parquet", index=False)
    status.to_csv(SNAPSHOT_DIR / f"status_{today}.csv", index=False)

    ok = int(status["ok"].sum())
    throttled = int((status["reason"] == "throttled").sum())
    no_data = int((status["reason"] == "no_data").sum())
    mins = (time.time() - t0) / 60

    log(
        f"WROTE ohlcv_{today}.parquet: {len(prices)} rows, {prices['ticker'].nunique()} tickers "
        f"| ok {ok} · no_data {no_data} · throttled {throttled} | {mins:.1f} min"
    )
    if throttled:
        log(f"  {throttled} tickers throttled — re-run later today to fill them (parts resume automatically)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
