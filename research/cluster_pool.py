"""How many INDEPENDENT bets are in the candidate pool?

A ranked list of 46 surviving signals overstates what you have. Seven ways of
measuring "companies that issue equity underperform" is one trade expressed
seven times -- the failure mode summed up as "gross kills you, not net."

This clusters the pool on the correlation of their monthly long-short returns,
so the grouping is empirical rather than taken from the catalog's own category
labels. It also flags the two things that would otherwise silently corrupt the
pool:

  short-leg dependence -- OSAP returns are LONG-SHORT. We do not short
      (long-only), so any signal whose premium lives in the short leg is
      unusable for us. The "short sale constraints" category is definitionally
      that, and its members are marked accordingly.

  multiple testing  -- ranking 212 signals by recent t-stat and keeping t>2
      guarantees false positives. The expected count is reported so the pool is
      never read as 46 confirmed effects.

    python3 research/cluster_pool.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent.parent
OSAP = ROOT / "data" / "osap"
OUT = ROOT / "research" / "out"

RECENT_FROM = 2011
CORR_CUT = 0.55          # merge signals whose returns correlate above this
SHORT_LEG_CATS = {"short sale constraints", "ownership"}


def main() -> int:
    tri = pd.read_csv(OUT / "osap_triage.csv")
    ret = pd.read_csv(OSAP / "ls_returns.raw")
    ret["date"] = pd.to_datetime(ret["date"])
    ret = ret.set_index("date").sort_index()

    pool = tri[tri["buildable"] & (tri["recent_t"] > 2)].copy()
    sigs = [s for s in pool["signal"] if s in ret.columns]

    R = ret.loc[ret.index.year >= RECENT_FROM, sigs].dropna(axis=1, thresh=100)
    sigs = list(R.columns)
    C = R.corr()

    d = squareform(np.clip(1 - C.abs().values, 0, 2), checks=False)
    labels = fcluster(linkage(d, method="average"), t=1 - CORR_CUT, criterion="distance")

    pool = pool.set_index("signal")
    grp = pd.DataFrame({"signal": sigs, "cluster": labels}).set_index("signal")
    grp = grp.join(pool[["econ", "data", "rebal_months", "recent_ann", "recent_t", "decay"]])

    # ---- multiple testing context ----
    n_tested = len(tri)
    expected_false = n_tested * 0.05
    n_pass = int((tri["recent_t"] > 2).sum())

    print(f"tested {n_tested} predictors; {n_pass} pass t>2 since {RECENT_FROM}")
    print(f"expected false positives at 5%: ~{expected_false:.0f}  "
          f"-> roughly {max(0, n_pass - expected_false)/n_pass:.0%} of passes are likely real\n")

    print(f"=== {grp['cluster'].nunique()} INDEPENDENT CLUSTERS among {len(sigs)} surviving signals "
          f"(|corr| > {CORR_CUT} merged) ===\n")

    order = (grp.groupby("cluster")["recent_t"].max().sort_values(ascending=False).index)
    for c in order:
        members = grp[grp["cluster"] == c].sort_values("recent_t", ascending=False)
        econs = members["econ"].value_counts()
        theme = econs.index[0]
        shortleg = bool(set(members["econ"]) & SHORT_LEG_CATS)
        lowturn = int((members["rebal_months"].fillna(1) >= 12).sum())

        flag = "  [SHORT-LEG — unusable long-only]" if shortleg else ""
        print(f"CLUSTER {c}: {theme}  ({len(members)} signals, {lowturn} at 12mo rebalance){flag}")
        for s, row in members.iterrows():
            rb = row["rebal_months"]
            rb = f"{int(rb)}mo" if pd.notna(rb) else "  ? "
            print(f"    {s:24s} {rb:>5s}  recent {row['recent_ann']:6.1f}%/yr  "
                  f"t={row['recent_t']:.2f}  decay={row['decay']:.2f}")
        print()

    usable = grp[~grp["econ"].isin(SHORT_LEG_CATS)]
    print(f"usable long-only clusters: {usable['cluster'].nunique()} "
          f"(of {grp['cluster'].nunique()} total)")
    print(f"usable signals at 12mo rebalance: "
          f"{int((usable['rebal_months'].fillna(1) >= 12).sum())}")

    grp.reset_index().to_csv(OUT / "pool_clusters.csv", index=False)
    C.to_csv(OUT / "pool_correlations.csv")
    print(f"\n-> {OUT / 'pool_clusters.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
