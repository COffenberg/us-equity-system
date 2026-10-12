"""Freeze today's ranked lists as a dated, immutable prediction.

The power problem means no backtest can validate this system. The only thing
that converts time into evidence is a forward record that cannot be edited
after the fact. Written once per run date, never modified.
"""
import pandas as pd, pathlib, json, hashlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT/"forward"
DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-08-19"

o = pd.read_parquet(ROOT/"signals/ranked.parquet")
try: w = pd.read_parquet(ROOT/"signals/watchlist_readacross.parquet")
except Exception: w = pd.DataFrame()

def top(col, n=40, cap=6):
    picked, counts = [], {}
    for t, r in o.sort_values(col, ascending=False).iterrows():
        s = r["sector"]
        if counts.get(s, 0) >= cap: continue
        picked.append(t); counts[s] = counts.get(s, 0) + 1
        if len(picked) == n: break
    return picked

lists = {"CORE": top("CORE"), "SKEW": top("SKEWLIST"), "COMBINED": top("COMBINED")}
if len(w): lists["READACROSS"] = list(w.sort_values("COMBINED", ascending=False).head(25).index)

px = (pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
        .sort_values("date").groupby("ticker").last())

rec = {"date": DATE, "universe_scored": int(len(o)),
       "n_signals": int(len([c for c in o.columns if c.startswith("c_")])),
       "benchmarks": ["SPY","QQQ","IWM"],
       "rules": {"rebalance": "not yet decided -- track buy-and-hold from this date",
                 "horizon_days": [21, 63, 126, 252],
                 "no_survivorship_edit": "entries are never removed, including delistings"},
       "lists": {}}
for k, tick in lists.items():
    rec["lists"][k] = [{"ticker": t,
                        "price": round(float(px.close.get(t, float("nan"))), 4),
                        "adj_close": round(float(px.adj_close.get(t, float("nan"))), 4),
                        "sector": str(o.sector.get(t, "")),
                        "score": round(float(o[{"CORE":"CORE","SKEW":"SKEWLIST",
                                                "COMBINED":"COMBINED","READACROSS":"COMBINED"}[k]].get(t, float("nan"))), 4)}
                       for t in tick]
for b in rec["benchmarks"]:
    rec.setdefault("benchmark_prices", {})[b] = round(float(px.close.get(b, float("nan"))), 4)

blob = json.dumps(rec, indent=1, sort_keys=True)
rec["sha256"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
f = OUT/f"prediction_{DATE}.json"
if f.exists():
    print(f"REFUSING to overwrite {f.name} — the forward record is append-only."); sys.exit(1)
f.write_text(json.dumps(rec, indent=1, sort_keys=True))
print(f"froze {sum(len(v) for v in lists.values())} positions across {len(lists)} lists -> {f.name}")
print(f"sha256 {rec['sha256']}")
for k, v in lists.items():
    got = sum(1 for t in v if pd.notna(px.close.get(t)))
    print(f"  {k:11s} {len(v):3d} names, {got} with an entry price")
print(f"\nbenchmarks: " + ", ".join(f"{b} ${rec['benchmark_prices'][b]}" for b in rec["benchmarks"]))
