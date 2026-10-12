"""Compare real signals to noise on ALL FOUR tests, not just the solo one.

The objection: we established a driver can be worthless alone and valuable
in combination, so judging one by its solo spread contradicts our own method.
Correct. This scores every real signal against the noise distribution on each
dimension separately, and on the combined screen.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel2.parquet")
BASE = ["mom_12_1","bm","px_52w","SP"]
P["comp"] = P[BASE].mean(axis=1)
P["cq"] = P.groupby("date").comp.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))
med = P.dollar_vol.median()

def spread(df,col,q=5):
    out=[]
    for D,g in df.groupby("date"):
        v=g[col].dropna()
        if len(v)<200: continue
        b=pd.qcut(v,q,labels=False,duplicates="drop")
        m=g.loc[v.index].groupby(b).fwd.mean()
        if len(m)==q: out.append(m.iloc[-1]-m.iloc[0])
    return np.mean(out) if out else np.nan
base_sp = spread(P,"comp")

def profile(rank_series):
    r = rank_series
    alone = spread(P.assign(x=r),"x")
    P["blend"] = 0.9*P.comp + 0.1*r.fillna(0.5)
    delta = spread(P,"blend") - base_sp
    inter=[]
    for q in range(5):
        s=P[(P.cq==q)&r.notna()]
        if len(s)<400: continue
        b=pd.qcut(r[s.index],3,labels=False,duplicates="drop"); m=s.groupby(b).fwd.mean()
        if len(m)==3: inter.append(m.iloc[-1]-m.iloc[0])
    interact=np.mean(inter) if inter else np.nan
    s=P[(P.dollar_vol<=med)&r.notna()]; illq=np.nan
    if len(s)>=400:
        b=pd.qcut(r[s.index],3,labels=False,duplicates="drop"); m=s.groupby(b).fwd.mean()
        if len(m)==3: illq=m.iloc[-1]-m.iloc[0]
    return alone, delta, interact, illq

rng=np.random.default_rng(777)
N=200
noise=[]
for i in range(N):
    f=pd.Series(rng.normal(size=len(P)),index=P.index)
    f[rng.random(len(P))<0.35]=np.nan
    noise.append(profile(f.groupby(P.sector).rank(pct=True)))
Nz=pd.DataFrame(noise,columns=["alone","delta","interact","illiq"])

REAL=["leverage","insider","asset_growth","d_noa","gp_stability","cash_conv",
      "equity_neg","d_accrual","vol_60d","roaq","buyback"]
print(f"Percentile of each real signal within {N} pure-noise drivers, per test.")
print("A signal only beats noise where its percentile is high. 50 = coin flip.\n")
print(f"{'signal':15s} {'alone':>16s} {'blend':>16s} {'interact':>16s} {'illiquid':>16s} {'best':>6s}")
for c in REAL:
    if c not in P.columns: continue
    a,d,it,il = profile(P[c])
    pc=[]
    for val,col in [(a,"alone"),(d,"delta"),(it,"interact"),(il,"illiq")]:
        pc.append(100*(Nz[col]<val).mean() if val==val else np.nan)
    f=lambda v,p: f"{v:+.2%} ({p:3.0f})" if v==v else "        n/a"
    print(f"{c:15s} {f(a,pc[0]):>16s} {f(d,pc[1]):>16s} {f(it,pc[2]):>16s} {f(il,pc[3]):>16s} "
          f"{np.nanmax(pc):6.0f}")
print("\n(number in brackets = percentile vs noise on that test)")
