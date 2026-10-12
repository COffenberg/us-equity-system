"""Does institutional breadth predict returns here? Four tests, point-in-time.

13F is filed up to 45 days after quarter end, so at a scoring date D only the
quarter ending >=45 days before D is public. Ignoring that lag would read
holdings that had not been disclosed yet -- textbook lookahead.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel2.parquet")
H = pd.read_parquet(ROOT/"data/sec/f13_holdings.parquet")
H["avail"] = H.dt + pd.Timedelta(days=45)          # public only after the deadline

BASE = ["mom_12_1","bm","px_52w","SP"]
P["comp"] = P[BASE].mean(axis=1)

feat = []
for D, g in P.groupby("date"):
    pub = H[H.avail <= D]
    if pub.empty: continue
    periods = sorted(pub.dt.unique())
    if len(periods) < 2: continue
    cur, prv = periods[-1], periods[-2]
    c = pub[pub.dt == cur].set_index("ticker")
    p = pub[pub.dt == prv].set_index("ticker")
    d = pd.DataFrame(index=g.ticker)
    d["holders"]   = c.holders.reindex(d.index)
    d["d_holders"] = (c.holders.reindex(d.index) - p.holders.reindex(d.index))
    d["d_breadth"] = d.d_holders / p.holders.reindex(d.index).clip(lower=1)
    d["d_shares"]  = (c.shares.reindex(d.index) / p.shares.reindex(d.index).replace(0, np.nan)) - 1
    d["date"] = D
    feat.append(d.reset_index())
    print(f"  {str(D)[:10]}: using 13F {str(cur)[:10]} vs {str(prv)[:10]}, "
          f"{d.holders.notna().sum():,} names covered")

if not feat:
    print("no usable dates"); raise SystemExit
Fq = pd.concat(feat, ignore_index=True)
A = P.merge(Fq, on=["ticker","date"], how="left")
A["cq"] = A.groupby("date").comp.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))
med = A.dollar_vol.median()

def spread(df, col, q=5):
    out=[]
    for D,g in df.groupby("date"):
        v=g[col].dropna()
        if len(v)<200: continue
        b=pd.qcut(v,q,labels=False,duplicates="drop")
        m=g.loc[v.index].groupby(b).fwd.mean()
        if len(m)==q: out.append(m.iloc[-1]-m.iloc[0])
    return (np.mean(out) if out else np.nan, len(out))

base_sp,_ = spread(A,"comp")
print(f"\nbaseline composite {base_sp:+.2%}")
print(f"\n{'signal':12s} {'cover':>7s} {'alone':>8s} {'blendD':>8s} {'interact':>9s} {'illiq':>8s} {'verdict':>9s}")
for c in ["holders","d_holders","d_breadth","d_shares"]:
    cov=int(A[c].notna().sum()/A.date.nunique())
    alone,n = spread(A,c)
    if n==0: print(f"{c:12s} {cov:7d}   too few dates"); continue
    r = A[c].groupby(A.sector).rank(pct=True)
    A["blend"]=0.9*A.comp+0.1*r.fillna(0.5)
    inc,_ = spread(A,"blend"); delta=inc-base_sp
    inter=[]
    for q in range(5):
        s=A[(A.cq==q)&A[c].notna()]
        if len(s)<400: continue
        b=pd.qcut(s[c],3,labels=False,duplicates="drop"); m=s.groupby(b).fwd.mean()
        if len(m)==3: inter.append(m.iloc[-1]-m.iloc[0])
    interact=np.mean(inter) if inter else np.nan
    s=A[(A.dollar_vol<=med)&A[c].notna()]; illq=np.nan
    if len(s)>=400:
        b=pd.qcut(s[c],3,labels=False,duplicates="drop"); m=s.groupby(b).fwd.mean()
        illq=m.iloc[-1]-m.iloc[0] if len(m)==3 else np.nan
    a=alone if alone==alone else 0
    passes=sum([a>0.015, delta>0.002, (interact or 0)>0.010, (illq or 0)>0.010])
    v="keep" if passes>=2 else ("watch" if passes==1 else "drop")
    f=lambda x: f"{x:+.2%}" if x==x else "    n/a"
    print(f"{c:12s} {cov:7d} {f(alone):>8s} {f(delta):>8s} {f(interact):>9s} {f(illq):>8s} {v:>9s}")
print(f"\n{n} usable dates — 13F only became broadly covered from late 2024, so this is thin.")
