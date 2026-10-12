"""Is the value result real or an artefact? Five stress tests.

Survivorship was measured at 0.27%/quarter attrition -- far too small to explain
a 5.71pp edge, so that hypothesis is dead. What remains: multiple testing, small
sample, and the possibility that value simply had a good run in this window.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT=pathlib.Path(__file__).resolve().parent.parent
P=pd.read_parquet(ROOT/"research/out/pit_panel3.parquet")
H=pd.read_parquet(ROOT/"data/prices/history_3y.parquet"); H["date"]=pd.to_datetime(H.date)
PX=H.pivot_table(index="date",columns="ticker",values="adj_close",aggfunc="last").sort_index()
def bench(D,h=63):
    fi=PX.index.get_indexer([pd.Timestamp(D)+pd.Timedelta(days=int(h*1.45))],method="nearest")[0]
    return PX["SPY"].iloc[fi]/PX.loc[pd.Timestamp(D),"SPY"]-1
VAL=P[["SP","bm","cfp"]].mean(axis=1)

def run(sc,n,cap=6,minliq=5e5,h=63):
    out=[]
    for D,g in P.assign(s=sc).groupby("date"):
        g=g[(g.dollar_vol>=minliq)&g.s.notna()]
        if len(g)<150: continue
        pick,cnt=[],{}
        for t,r in g.sort_values("s",ascending=False).iterrows():
            if cap and cnt.get(r.sector,0)>=cap: continue
            pick.append(r.fwd); cnt[r.sector]=cnt.get(r.sector,0)+1
            if len(pick)==n: break
        if len(pick)<n: continue
        out.append((pd.Timestamp(D),np.mean(pick)-bench(D,h)))
    return pd.DataFrame(out,columns=["date","ex"])

print("1. CONCENTRATION — does the value edge survive breadth?")
print(f"   {'N':>5s} {'edge':>8s} {'win':>5s} {'t':>6s}")
for n in (10,20,40,80,160,320):
    a=run(VAL,n)
    if len(a)<5: continue
    print(f"   {n:5d} {a.ex.mean():+8.2%} {(a.ex>0).mean():5.0%} "
          f"{a.ex.mean()/(a.ex.std(ddof=1)/np.sqrt(len(a))):6.2f}")

print("\n2. WALK-FORWARD — first half vs second half")
a=run(VAL,40); mid=len(a)//2
for lbl,part in [("first 3 quarters",a.iloc[:mid+1]),("last 3 quarters",a.iloc[mid+1:])]:
    print(f"   {lbl:18s} edge {part.ex.mean():+7.2%}  win {(part.ex>0).mean():4.0%}  n={len(part)}")

print("\n3. LIQUIDITY — is it hiding in illiquid names?")
for lo,lab in [(5e5,"$500k+"),(2e6,"$2m+"),(1e7,"$10m+"),(5e7,"$50m+")]:
    b=run(VAL,40,minliq=lo)
    if len(b)<5: continue
    print(f"   {lab:8s} edge {b.ex.mean():+7.2%}  win {(b.ex>0).mean():4.0%}  t "
          f"{b.ex.mean()/(b.ex.std(ddof=1)/np.sqrt(len(b))):5.2f}")

print("\n4. HORIZON — does it survive a longer hold?")
for h in (21,63,126):
    c=run(VAL,40,h=h)
    if len(c)<4: continue
    print(f"   {h:3d}d  edge {c.ex.mean():+7.2%}  win {(c.ex>0).mean():4.0%}  n={len(c)}")

print("\n5. THE HONEST COMPARISON — value quintile vs top-40, same signal")
for n,lab in [(40,"top 40"),(320,"top 320 (~quintile)")]:
    d=run(VAL,n)
    print(f"   {lab:22s} edge {d.ex.mean():+7.2%}")
print("   If the edge is much larger at 40 than at 320, it lives in a handful of")
print("   extreme names -- which is where noise lives too.")
