#!/usr/bin/env python3
"""Independent share count per ticker, used by build_valprof.py as a *validity gate* on
market cap -- never as a data source.  Signal values stay 100% SEC-derived.

Two things it buys us, neither obtainable from EDGAR alone:

1. ADR ratios.  SEC reports *ordinary* shares for a 20-F filer while the US line trades as
   an ADR.  Banco Santander Chile: 188.4bn ordinary vs 471m ADRs -> ratio exactly 400.
   Multiplying the ADR price by the ordinary count is what manufactures a $6.5tn market cap.
2. Non-common-equity lines.  ETFs, commodity/crypto trusts, baby bonds, preferreds and
   depositary shares have no share count here at all, while they *do* inherit their
   issuer's fundamentals through the CIK join on EDGAR.

Writes signals/_audit_shares_yf.csv  [ticker, shares_yf, attempts].
Resumable: existing rows are kept, only missing/failed tickers are re-tried.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "signals")
OUT = os.path.join(SIG, "_audit_shares_yf.csv")

MIN_PRICE = 1.00
MIN_DOLLAR_VOL = 1e5      # deliberately looser than the build's floor so the cache is a
                          # superset and filter tuning never needs a re-pull
RETRIES = 2


def candidates():
    u = pd.read_csv(os.path.join(ROOT, "data/universe/tickers_2026-08-18.csv"), dtype={"cik": str})
    u = u.dropna(subset=["ticker"])
    px = pd.read_parquet(os.path.join(ROOT, "data/snapshots/ohlcv_2026-08-18.parquet"))
    px["date"] = pd.to_datetime(px["date"])
    snap = (px.dropna(subset=["close"]).sort_values("date").groupby("ticker").tail(1)
              [["ticker", "close"]].rename(columns={"close": "px_snap"}))
    dg = pd.read_parquet(os.path.join(ROOT, "data/prices/derived_diagnostics.parquet"))
    dv = pd.read_parquet(os.path.join(ROOT, "data/prices/derived.parquet"))[["ticker", "dollar_vol_60d"]]
    p = snap.merge(dg[["ticker", "last_close"]], on="ticker", how="outer").merge(dv, on="ticker", how="left")
    p["price"] = p.px_snap.fillna(p.last_close)
    p = p[(p.price > MIN_PRICE) & (p.dollar_vol_60d >= MIN_DOLLAR_VOL)]
    return sorted(set(p.ticker) & set(u.ticker))


def main():
    import yfinance as yf
    want = candidates()
    done = pd.DataFrame(columns=["ticker", "shares_yf", "attempts"])
    if os.path.exists(OUT):
        done = pd.read_csv(OUT)
    have_ok = set(done.loc[done.shares_yf.notna(), "ticker"])
    tried = dict(zip(done.ticker, done.attempts)) if "attempts" in done else {}
    todo = [t for t in want if t not in have_ok and tried.get(t, 0) < RETRIES]
    print(f"candidates {len(want):,} | cached-ok {len(have_ok):,} | to fetch {len(todo):,}", flush=True)

    rows = {r.ticker: {"ticker": r.ticker, "shares_yf": r.shares_yf,
                       "attempts": getattr(r, "attempts", 1)} for r in done.itertuples()}
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        s = np.nan
        try:
            s = yf.Ticker(t).fast_info.get("shares")
            s = float(s) if s else np.nan
        except Exception:
            pass
        prev = rows.get(t, {}).get("attempts", 0) or 0
        rows[t] = {"ticker": t, "shares_yf": s, "attempts": prev + 1}
        if i % 250 == 0 or i == len(todo):
            pd.DataFrame(rows.values()).to_csv(OUT, index=False)
            el = time.time() - t0
            print(f"  {i:,}/{len(todo):,}  {el:,.0f}s  eta {el / i * (len(todo) - i):,.0f}s", flush=True)
    pd.DataFrame(rows.values()).to_csv(OUT, index=False)
    d = pd.DataFrame(rows.values())
    print(f"done. {d.shares_yf.notna().sum():,} with a share count, "
          f"{d.shares_yf.isna().sum():,} without -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
