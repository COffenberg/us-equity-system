"""Triage the Open Source Asset Pricing catalog into a usable candidate pool.

WHY NOT JUST READ THE PAPERS
-----------------------------
The catalog's median sample END year is 2001. Every one of these predictors was
published on data that stopped roughly 25 years ago, so the original t-stats say
nothing about whether the effect still exists. Chen & Zimmermann's own work finds
substantial post-publication decay.

We do not have to guess: OSAP ships monthly long-short returns through the
present, so the out-of-sample test already ran. This script measures it.

For each predictor:
    in-sample   -- through the original paper's SampleEndYear
    post-sample -- everything after it (the real out-of-sample evidence)
    recent      -- last 15 years, i.e. is it alive in a regime we'd trade in

Then filters to what we can actually build for free and trade under realisation tax:

    buildable   Accounting (SEC XBRL), Price/Trading (our own recorder),
                13F and Event (SEC). NOT Analyst (needs paid consensus) or
                Options (needs an options feed, and we have no options access).
    tradable    12-month rebalance preferred; 1-month signals must clear a much
                higher bar to survive turnover cost + realisation tax.

Output is a ranked candidate table, not a decision. Mechanism review comes next
and is a human step.

    python3 research/triage_osap.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OSAP = ROOT / "data" / "osap"
OUT = ROOT / "research" / "out"

RECENT_FROM = 2011          # "is it alive now" window
MIN_MONTHS = 60             # need 5y in a window before trusting its t-stat

# Data sources we can build for free. Analyst and Options are excluded on
# access grounds, not on evidence grounds -- they may well work, we just
# cannot compute them.
BUILDABLE = {"Accounting", "Price", "Trading", "13F", "Event", "Other"}


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < MIN_MONTHS or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def window_stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    if len(r) < MIN_MONTHS:
        return {f"{label}_ann": np.nan, f"{label}_t": np.nan, f"{label}_n": len(r)}
    return {
        f"{label}_ann": float(r.mean() * 12),      # annualised, in percent
        f"{label}_t": tstat(r),
        f"{label}_n": len(r),
    }


def main() -> int:
    doc = pd.read_csv(OSAP / "SignalDoc.csv")
    ret = pd.read_csv(OSAP / "ls_returns.raw")

    ret["date"] = pd.to_datetime(ret["date"])
    ret = ret.set_index("date").sort_index()
    print(f"returns: {ret.shape[1]} signals, {ret.index.min():%Y-%m} to {ret.index.max():%Y-%m}")

    doc = doc[doc["Cat.Signal"] == "Predictor"].copy()
    doc = doc.set_index("Acronym")

    rows = []
    for sig in ret.columns:
        if sig not in doc.index:
            continue
        meta = doc.loc[sig]
        if isinstance(meta, pd.DataFrame):       # duplicate acronym guard
            meta = meta.iloc[0]

        r = ret[sig].dropna()
        if r.empty:
            continue

        end = meta.get("SampleEndYear")
        end = int(end) if pd.notna(end) else 2000

        row = {
            "signal": sig,
            "authors": str(meta.get("Authors", ""))[:38],
            "year": meta.get("Year"),
            "sample_end": end,
            "econ": meta.get("Cat.Economic"),
            "data": meta.get("Cat.Data"),
            "quality": meta.get("Signal Rep Quality"),
            "rebal_months": meta.get("Portfolio Period"),
            "orig_t": meta.get("T-Stat"),
        }
        row.update(window_stats(r[r.index.year <= end], "is"))
        row.update(window_stats(r[r.index.year > end], "oos"))
        row.update(window_stats(r[r.index.year >= RECENT_FROM], "recent"))
        rows.append(row)

    df = pd.DataFrame(rows)

    # decay: what fraction of the in-sample premium survived publication
    df["decay"] = np.where(
        df["is_ann"].abs() > 1e-9, df["oos_ann"] / df["is_ann"], np.nan
    )

    df["buildable"] = df["data"].isin(BUILDABLE)
    df["low_turnover"] = df["rebal_months"].fillna(1) >= 12

    OUT.mkdir(parents=True, exist_ok=True)
    df.sort_values("recent_t", ascending=False).to_csv(OUT / "osap_triage.csv", index=False)

    # ---- summary ----
    n = len(df)
    print(f"\nmatched predictors: {n}")
    print(f"  buildable free:      {int(df['buildable'].sum())}")
    print(f"  12mo+ rebalance:     {int(df['low_turnover'].sum())}")

    alive = df["oos_t"] > 2
    still = df["recent_t"] > 2
    print(f"\nsurvived out-of-sample (t>2 after original sample end): {int(alive.sum())}/{n}")
    print(f"still working since {RECENT_FROM} (t>2):                   {int(still.sum())}/{n}")

    med = df["decay"].median()
    print(f"median decay (oos premium / in-sample premium):        {med:.2f}")

    pool = df[df["buildable"] & (df["recent_t"] > 2)].copy()
    print(f"\n=== CANDIDATE POOL: buildable AND alive since {RECENT_FROM} — {len(pool)} signals ===")
    show = pool.sort_values("recent_t", ascending=False)[
        ["signal", "econ", "data", "rebal_months", "quality",
         "is_ann", "oos_ann", "recent_ann", "recent_t", "decay"]
    ]
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(show.round(2).to_string(index=False))

    print(f"\nfull table -> {OUT / 'osap_triage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
