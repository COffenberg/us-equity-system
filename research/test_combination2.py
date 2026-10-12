"""Four-test evaluation across all 31 signals, with sign-flip detection.

A large NEGATIVE 'alone' spread is not a dead signal -- it is a signal pointing
the wrong way, which is a one-character fix. Distinguishing the two matters.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT=pathlib.Path(__file__).resolve().parent.parent
P=pd.read_parquet(ROOT/"research/out/pit_panel2.parquet")
BASE=["mom_12_1","bm","px_52w","SP"]
ALL=[c for c in P.columns if c not in ("fwd","dollar_vol","sector","date","ticker")]
NEW=[c for c in ALL if c not in BASE+["cfp","roaq","PctTotAcc","leverage","buyback","issuance","vol_60d","insider"]]
P["comp"]=P[BASE].mean(axis=1)

def spread(df,col,q=5):
    out=[]
    for D,g in df.groupby("date"):
        v=g[col].dropna()
        if len(v)<200: continue
        b=pd.qcut(v,q,labels=False,duplicates="drop")
        m=g.loc[v.index].groupby(b).fwd.mean()
        if len(m)==q: out.append(m.iloc[-1]-m.iloc[0])
    return (np.mean(out) if out else np.nan, np.mean([x>0 for x in out]) if out else np.nan, len(out))

base_sp,base_win,_=spread(P,"comp")
print(f"BASELINE composite {BASE}:  Q5-Q1 {base_sp:+.2%}  win {base_win:.0%}  "
      f"{P.date.nunique()} dates, {len(P):,} name-dates\n")
med=P.dollar_vol.median()
P["cq"]=P.groupby("date").comp.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))

print(f"{'signal':18s} {'cover':>6s} {'alone':>8s} {'blendD':>8s} {'interact':>9s} {'illiq':>8s} {'verdict':>9s}")
res=[]
for c in sorted(NEW)+[x for x in BASE]:
    cov=int(P[c].notna().sum()/P.date.nunique())
    alone,_,_=spread(P,c)
    P["blend"]=0.9*P.comp+0.1*P[c].fillna(0.5)
    inc,_,_=spread(P,"blend"); delta=inc-base_sp
    inter=[]
    for q in range(5):
        s=P[(P.cq==q)&P[c].notna()]
        if len(s)<400: continue
        b=pd.qcut(s[c],3,labels=False,duplicates="drop")
        m=s.groupby(b).fwd.mean()
        if len(m)==3: inter.append(m.iloc[-1]-m.iloc[0])
    interact=np.mean(inter) if inter else np.nan
    s=P[(P.dollar_vol<=med)&P[c].notna()]
    illq=np.nan
    if len(s)>=400:
        b=pd.qcut(s[c],3,labels=False,duplicates="drop"); m=s.groupby(b).fwd.mean()
        illq=m.iloc[-1]-m.iloc[0] if len(m)==3 else np.nan
    a=alone if alone==alone else 0
    if a<-0.015: v="FLIP SIGN"
    else:
        passes=sum([a>0.015, delta>0.002, (interact or 0)>0.010, (illq or 0)>0.010])
        v="keep" if passes>=2 else ("watch" if passes==1 else "drop")
    res.append((c,cov,alone,delta,interact,illq,v,c in BASE))

f=lambda x: f"{x:+.2%}" if x==x else "    n/a"
for c,cov,a,d,it,il,v,isbase in sorted(res,key=lambda x:-(x[3] if x[3]==x[3] else -9)):
    print(f"{c:18s}{'*' if isbase else ' '}{cov:6d} {f(a):>8s} {f(d):>8s} {f(it):>9s} {f(il):>8s} {v:>9s}")
print("\n* already in baseline.  FLIP SIGN = alone spread below -1.5%, i.e. predictive but pointed backwards.")
print("verdict: keep >=2 of 4 tests. alone>1.5% | blendD>0.2% | interact>1.0% | illiq>1.0%")
