"""If we built the other 111 drivers, could we tell the real ones from the fake?

Feed the four-test screen pure noise -- signals with no predictive content by
construction -- and count how many it certifies. Whatever passes is the screen's
false-discovery rate at this sample size. Anything below that rate among real
candidates is indistinguishable from luck.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel2.parquet")
BASE = ["mom_12_1","bm","px_52w","SP"]
P["comp"] = P[BASE].mean(axis=1)
P["cq"] = P.groupby("date").comp.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))
med = P.dollar_vol.median()
rng = np.random.default_rng(12345)

def spread(df, col, q=5):
    out=[]
    for D,g in df.groupby("date"):
        v=g[col].dropna()
        if len(v)<200: continue
        b=pd.qcut(v,q,labels=False,duplicates="drop")
        m=g.loc[v.index].groupby(b).fwd.mean()
        if len(m)==q: out.append(m.iloc[-1]-m.iloc[0])
    return np.mean(out) if out else np.nan

base_sp = spread(P,"comp")
N = 200
passes, alone_hits, results = 0, 0, []
for i in range(N):
    # a fake driver: random per name-date, with realistic coverage holes
    P["fake"] = rng.normal(size=len(P))
    P.loc[rng.random(len(P)) < 0.35, "fake"] = np.nan
    r = P.fake.groupby(P.sector).rank(pct=True)
    alone = spread(P.assign(fake=r), "fake")
    P["blend"] = 0.9*P.comp + 0.1*r.fillna(0.5)
    delta = spread(P,"blend") - base_sp
    inter=[]
    for q in range(5):
        s = P[(P.cq==q) & r.notna()]
        if len(s)<400: continue
        b = pd.qcut(r[s.index],3,labels=False,duplicates="drop")
        m = s.groupby(b).fwd.mean()
        if len(m)==3: inter.append(m.iloc[-1]-m.iloc[0])
    interact = np.mean(inter) if inter else 0
    s = P[(P.dollar_vol<=med) & r.notna()]
    illq = 0
    if len(s)>=400:
        b = pd.qcut(r[s.index],3,labels=False,duplicates="drop")
        m = s.groupby(b).fwd.mean()
        if len(m)==3: illq = m.iloc[-1]-m.iloc[0]
    hits = sum([alone>0.015, delta>0.002, interact>0.010, illq>0.010])
    results.append((alone, delta, interact, illq, hits))
    if hits>=2: passes += 1
    if alone>0.015: alone_hits += 1

R = pd.DataFrame(results, columns=["alone","delta","interact","illiq","hits"])
print(f"{N} PURE-NOISE drivers put through the same four-test screen\n")
print(f"  certified 'keep' (>=2 of 4 tests):  {passes}/{N} = {passes/N:.0%}")
print(f"  looked good alone (>1.5% spread):   {alone_hits}/{N} = {alone_hits/N:.0%}")
print(f"  passed at least one test:           {(R.hits>=1).sum()}/{N} = {(R.hits>=1).mean():.0%}")
print(f"\n  best noise driver's solo spread:    {R.alone.max():+.2%}")
print(f"  best noise driver's blend delta:    {R.delta.max():+.2%}")
print(f"  best noise driver's illiquid edge:  {R.illiq.max():+.2%}")
print(f"\n  for comparison, real signals we KEPT:")
for c,v in [("asset_growth",0.0120),("d_noa",0.0161),("insider",0.0049),("leverage",0.0017)]:
    pct = (R.alone < v).mean()
    print(f"    {c:14s} solo {v:+.2%}  -- beaten by {1-pct:.0%} of pure noise")
print(f"\n  expected false 'keeps' if we built the 111 untouched drivers: "
      f"{int(round(111*passes/N))}")
