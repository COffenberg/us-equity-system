"""Score every signal against a pure-noise null on all four tests.

Reports each signal's best-of-four percentile AND the fraction of noise that
matches it -- an honest p-value that already pays for the four tries.
"""
import pandas as pd, numpy as np, pathlib, warnings, sys
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel3.parquet")
BASE=["mom_12_1","bm","px_52w","SP"]
P["comp"]=P[BASE].mean(axis=1)
P["cq"]=P.groupby("date").comp.transform(lambda s:pd.qcut(s,5,labels=False,duplicates="drop"))
med=P.dollar_vol.median()
def spread(df,col,q=5):
    out=[]
    for D,g in df.groupby("date"):
        v=g[col].dropna()
        if len(v)<200: continue
        b=pd.qcut(v,q,labels=False,duplicates="drop"); m=g.loc[v.index].groupby(b).fwd.mean()
        if len(m)==q: out.append(m.iloc[-1]-m.iloc[0])
    return np.mean(out) if out else np.nan
base=spread(P,"comp")
def profile(r):
    a=spread(P.assign(x=r),"x")
    P["blend"]=0.9*P.comp+0.1*r.fillna(0.5); d=spread(P,"blend")-base
    it=[]
    for q in range(5):
        s=P[(P.cq==q)&r.notna()]
        if len(s)<400: continue
        b=pd.qcut(r[s.index],3,labels=False,duplicates="drop"); m=s.groupby(b).fwd.mean()
        if len(m)==3: it.append(m.iloc[-1]-m.iloc[0])
    s=P[(P.dollar_vol<=med)&r.notna()]; il=np.nan
    if len(s)>=400:
        b=pd.qcut(r[s.index],3,labels=False,duplicates="drop"); m=s.groupby(b).fwd.mean()
        if len(m)==3: il=m.iloc[-1]-m.iloc[0]
    return a,d,(np.mean(it) if it else np.nan),il

rng=np.random.default_rng(2718); N=300; noise=[]
for i in range(N):
    fz=pd.Series(rng.normal(size=len(P)),index=P.index); fz[rng.random(len(P))<0.35]=np.nan
    noise.append(profile(fz.groupby(P.sector).rank(pct=True)))
Nz=pd.DataFrame(noise,columns=["alone","delta","interact","illiq"])
nb=[]
for i in range(N):
    pcs=[100*(Nz[c].drop(i)<Nz[c].iloc[i]).mean() for c in Nz.columns if Nz[c].iloc[i]==Nz[c].iloc[i]]
    nb.append(max(pcs) if pcs else np.nan)
nb=pd.Series(nb).dropna()

NEW=["Mom12m","Mom6m","Mom12mOffSeason","ChTax","NetEquityFinance","ShareIssLong",
     "EBM","CompEquIss","RevenueSurprise","EarningsConsistency","InvGrowthProxy"]
print(f"{'signal':20s} {'cover':>6s} {'alone':>8s} {'blend':>8s} {'inter':>8s} {'illiq':>8s} {'best':>5s} {'p':>6s}")
res=[]
for c in NEW:
    if c not in P.columns: continue
    cov=int(P[c].notna().sum()/P.date.nunique())
    a,d,it,il=profile(P[c])
    pcs=[]
    for v,col in [(a,"alone"),(d,"delta"),(it,"interact"),(il,"illiq")]:
        pcs.append(100*(Nz[col]<v).mean() if v==v else np.nan)
    best=np.nanmax(pcs); p=(nb>=best).mean()
    res.append((c,best,p))
    f=lambda x: f"{x:+.2%}" if x==x else "     n/a"
    print(f"{c:20s} {cov:6d} {f(a):>8s} {f(d):>8s} {f(it):>8s} {f(il):>8s} {best:5.0f} {p:6.0%}")
print(f"\nnull: median noise reaches {nb.median():.0f}th percentile on its best test")
strong=[r for r in res if r[2]<=0.05]
print(f"clear the 5% bar: {', '.join(r[0] for r in strong) if strong else 'none'}")
