#!/usr/bin/env python3
"""Split history for the whole roster, cached to data/prices/splits_yf.csv.

ShareIss1Y/5Y compare a raw cover-page share count today against one 1y/5y ago.
dei:EntityCommonStockSharesOutstanding is as-reported and is NOT retroactively
restated for splits, so without this a 1-for-10 reverse split reads as a 90%
share retirement -- i.e. it lands at the very top of the buy list. Monthly bars
are enough: the chart endpoint returns split events regardless of interval, and
month-level timing is immaterial over a 1y/5y window.
"""
import pathlib
import os, sys, time
import pandas as pd, yfinance as yf

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
OUT = os.path.join(ROOT, "data/prices/splits_yf.csv")
CHUNK = 200


def main():
    if os.path.exists(OUT):
        print("cached:", OUT, len(pd.read_csv(OUT)), "split events"); return
    uni = pd.read_csv(os.path.join(ROOT, "data/universe/tickers_2026-08-18.csv"), dtype={"cik": str})
    px = pd.read_parquet(os.path.join(ROOT, "data/snapshots/ohlcv_2026-08-18.parquet"), columns=["ticker"])
    tickers = sorted(set(uni.ticker) & set(px.ticker))
    print(len(tickers), "tickers")
    rows, t0 = [], time.time()
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            d = yf.download(batch, period="7y", interval="1mo", actions=True, group_by="ticker",
                            progress=False, auto_adjust=False, threads=True)
        except Exception as e:
            print("  chunk %d FAILED %r" % (i, e), flush=True); continue
        for t in batch:
            try:
                s = d[(t, "Stock Splits")].dropna()
            except Exception:
                continue
            s = s[s > 0]
            for dt, v in s.items():
                rows.append({"ticker": t, "date": pd.Timestamp(dt).date(), "ratio": float(v)})
        print("[%5d/%d] %.0fs  events=%d" % (min(i + CHUNK, len(tickers)), len(tickers),
                                             time.time() - t0, len(rows)), flush=True)
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(OUT, index=False)
    print("wrote", OUT, len(df), "events on", df.ticker.nunique(), "tickers")


if __name__ == "__main__":
    main()
