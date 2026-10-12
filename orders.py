"""Turn the ranked list into an order list. The dashboard's job is to say what to
buy, not to hand over research to interpret.

One book, not three lists: COMBINED (70% fundamentals, 30% skew), equal weight,
40 names, capped at 6 per sector. Equal weight because nothing in the evidence
justifies conviction sizing -- a 40-name floor exists precisely because the
payoff shape is unpredictable name by name.
"""
import pandas as pd, numpy as np, pathlib, json, glob, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent

o = pd.read_parquet(ROOT/"signals/ranked.parquet")
H = pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
H["date"] = pd.to_datetime(H.date)
px = H.sort_values("date").groupby("ticker").close.last()

# Sizing assumes fractional shares, so it is a
# percentage question only -- no share counts, no FX, no leftover cash.
WEIGHT = None  # set below, equal across the book

def book(col="COMBINED", n=40, cap=6):
    picked, counts = [], {}
    for t, r in o.sort_values(col, ascending=False).iterrows():
        s = r.get("sector", "Other")
        if counts.get(s, 0) >= cap: continue
        if pd.isna(px.get(t)): continue
        picked.append(t); counts[s] = counts.get(s, 0) + 1
        if len(picked) == n: break
    return o.loc[picked]

B = book()
CL = [c[2:] for c in o.columns if c.startswith("c_")]
rows = []
for t, r in B.iterrows():
    d = r[[f"c_{c}" for c in CL]].astype(float).dropna().nlargest(2)
    rows.append({"ticker": t, "name": str(r.get("name") or "")[:28],
                 "sector": str(r.get("sector", "")), "price": round(float(px[t]), 2),
                 "score": round(float(r.COMBINED), 3),
                 "mktcap_bn": None if pd.isna(r.get("mktcap_bn")) else round(float(r.mktcap_bn), 1),
                 "why": ", ".join(f"{k[2:]} {v:.2f}" for k, v in d.items()),
                 "soon": bool(r.get("in_pre_earnings_window", False))})

# what changed since the last frozen book -- the only thing that generates trades
prev = sorted(glob.glob(str(ROOT/"forward/prediction_*.json")))
prev_set = set()
prev_date = None
if prev:
    rec = json.load(open(prev[-1])); prev_date = rec["date"]
    prev_set = {n["ticker"] for n in rec["lists"].get("COMBINED", [])}
cur_set = {r["ticker"] for r in rows}

w = round(100.0 / len(rows), 3)
for r in rows: r["weight"] = w
out = {"as_of": str(H.date.max().date()), "weight_pct": w,
       "n": len(rows), "holdings": rows,
       "prev_date": prev_date,
       "adds": sorted(cur_set - prev_set), "drops": sorted(prev_set - cur_set),
       "unchanged": len(cur_set & prev_set)}
json.dump(out, open(ROOT/"signals/orders.json", "w"), indent=1)
print(f"order book: {len(rows)} names at {w}% each, as of {out['as_of']}")
print(f"  vs last frozen book ({prev_date}): {len(out['adds'])} adds, {len(out['drops'])} drops, "
      f"{out['unchanged']} unchanged")
if out["adds"]: print(f"  adds:  {', '.join(out['adds'][:12])}")
if out["drops"]: print(f"  drops: {', '.join(out['drops'][:12])}")
