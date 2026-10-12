"""Live read-across: names reporting soon whose sector peers already reported well.

Validated in research/test_readacross.py over 38,576 events: pre-announcement
hit rate runs 41.4% -> 62.2% across peer-result quintiles, and survives a
double-sort against sector momentum (+19pts vs +7pts). This is the mechanism
behind buying a supplier before its print, made systematic.
"""
import pandas as pd, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
TODAY = pd.Timestamp("2026-08-19")

et  = pd.read_parquet(ROOT/"signals/earnings_timing.parquet")
sec = pd.read_parquet(ROOT/"data/sec/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector
d   = pd.read_parquet(ROOT/"data/prices/derived.parquet").set_index("ticker")

et["sector"] = sec.reindex(et.index)
et["liquid"] = d.dollar_vol_60d.reindex(et.index) >= 5e5

# who already reported, recently, and how did the market take it
recent = et[(et.days_since_earnings <= 45) & (et.days_since_earnings >= 1) & et.liquid]
peer = recent.groupby("sector").agg(peer_ann_mean=("AnnouncementReturn","mean"),
                                    peer_ann_med=("AnnouncementReturn","median"),
                                    peer_n=("AnnouncementReturn","size"))
peer = peer[peer.peer_n >= 3]
peer["peer_pct"] = peer.peer_ann_med.rank(pct=True)

out = et.join(peer, on="sector")
out["readacross"] = out.peer_pct                      # higher = peers beat, before you report
out.to_parquet(ROOT/"signals/readacross.parquet")

print("SECTOR READ-ACROSS, as of 2026-08-19")
print(f"{'sector':24s} {'reported':>9s} {'median ann':>11s} {'pctile':>7s}")
for s, r in peer.sort_values("peer_ann_med", ascending=False).iterrows():
    print(f"{s:24s} {int(r.peer_n):9d} {r.peer_ann_med:+11.2%} {r.peer_pct:7.2f}")

# the actionable list: reporting soon, peers already beat, and liquid
watch = out[(out.in_pre_earnings_window == True) & out.liquid & (out.peer_pct >= 0.6)]
rank = pd.read_parquet(ROOT/"signals/ranked.parquet")
watch = watch.join(rank[["name","sector","COMBINED","mktcap_bn"]].drop(columns="sector"), how="inner")
watch = watch.sort_values("COMBINED", ascending=False)
print(f"\nWATCHLIST — report within 21 days, peers already beat, scored in the system")
print(f"{'ticker':7s} {'company':26s} {'sector':22s} {'days':>5s} {'peer':>6s} {'score':>6s}")
for t, r in watch.head(25).iterrows():
    print(f"{t:7s} {str(r['name'])[:25]:26s} {str(r.sector)[:21]:22s} "
          f"{r.days_to_next_earnings:5.0f} {r.peer_pct:6.2f} {r.COMBINED:6.3f}")
print(f"\n{len(watch)} names qualify")
watch.to_parquet(ROOT/"signals/watchlist_readacross.parquet")
