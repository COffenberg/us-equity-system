"""Four-test evaluation of every signal. Replaces the univariate-only screen
that wrongly rejected insider buying.

  1. ALONE        quintile spread on its own
  2. INCREMENTAL  does adding it to the passing composite improve the spread?
  3. INTERACTION  does it separate names INSIDE composite quintiles?
  4. CONDITIONAL  does it work where its mechanism says it should?

A signal only has to pass ONE of these to be worth carrying. That is the whole
point -- an IC of 0.02 is worthless alone and valuable in a blend.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel.parquet")

BASE = ["mom_12_1","bm","px_52w","SP"]          # passed the univariate screen
ALL  = [c for c in P.columns if c not in ("fwd","dollar_vol","ret_3m_prior","sector","date","ticker")]
OTHER = [c for c in ALL if c not in BASE]

P["comp"] = P[BASE].mean(axis=1)
def spread(df, col, q=5):
    out=[]
    for D,g in df.groupby("date"):
        v=g[col].dropna()
        if len(v)<200: continue
        b=pd.qcut(v,q,labels=False,duplicates="drop")
        m=g.loc[v.index].groupby(b).fwd.mean()
        if len(m)==q: out.append(m.iloc[-1]-m.iloc[0])
    return (np.mean(out) if out else np.nan, np.mean([x>0 for x in out]) if out else np.nan, len(out))

base_sp,base_win,_ = spread(P,"comp")
print(f"BASELINE composite {BASE}")
print(f"  Q5-Q1 {base_sp:+.2%}   win {base_win:.0%}   {P.date.nunique()} dates, {len(P):,} name-dates\n")

print(f"{'signal':11s} {'alone':>8s} {'+10% blend':>11s} {'delta':>8s} {'interact':>9s} {'illiquid':>9s} {'liquid':>8s} {'verdict':>10s}")
res=[]
for c in OTHER + BASE:
    alone,_,_ = spread(P,c)
    P["blend"] = 0.9*P.comp + 0.1*P[c].fillna(0.5)
    inc,incwin,_ = spread(P,"blend")
    delta = inc - base_sp
    # interaction: mean insider-style spread within composite quintiles
    P["cq"]=P.groupby("date").comp.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))
    inter=[]
    for q in range(5):
        s=P[(P.cq==q)&P[c].notna()]
        if len(s)<400: continue
        b=pd.qcut(s[c],3,labels=False,duplicates="drop")
        m=s.groupby(b).fwd.mean()
        if len(m)==3: inter.append(m.iloc[-1]-m.iloc[0])
    interact=np.mean(inter) if inter else np.nan
    # conditional: illiquid vs liquid half
    med=P.dollar_vol.median()
    def sub(mask):
        s=P[mask&P[c].notna()]
        if len(s)<400: return np.nan
        b=pd.qcut(s[c],3,labels=False,duplicates="drop")
        m=s.groupby(b).fwd.mean()
        return m.iloc[-1]-m.iloc[0] if len(m)==3 else np.nan
    illq, lq = sub(P.dollar_vol<=med), sub(P.dollar_vol>med)
    passes = sum([alone>0.015, delta>0.002, (interact or 0)>0.010, (illq or 0)>0.010])
    v = "keep" if passes>=2 else ("watch" if passes==1 else "drop")
    res.append((c,alone,inc,delta,interact,illq,lq,v,c in BASE))
for c,a,i,d,it,il,l,v,isbase in sorted(res,key=lambda x:-(x[3] if x[3]==x[3] else -9)):
    tag=" *" if isbase else ""
    f=lambda x: f"{x:+.2%}" if x==x else "     n/a"
    print(f"{c:11s}{tag:2s}{f(a):>7s} {f(i):>11s} {f(d):>8s} {f(it):>9s} {f(il):>9s} {f(l):>8s} {v:>9s}")
print("\n* = already in the baseline composite.  delta = change in Q5-Q1 from adding at 10% weight.")
print("verdict: keep >=2 tests pass, watch = 1, drop = 0. Thresholds: alone >1.5%, delta >0.2%, interact/illiquid >1.0%")
