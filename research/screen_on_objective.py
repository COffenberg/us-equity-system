"""Screen every signal on the ACTUAL objective: does a 40-name long-only book
ranked on it beat SPY?

Prior screens used quintile spread (Q5 minus Q1). That rewards a signal whose
BOTTOM is terrible -- worthless when shorting is unavailable. The two-source
composite was built from four spread-winners and came out 2.3pp worse than the
old one, which is what a misaligned objective function looks like.

Same noise null as before, recomputed on this metric so the p-values are honest.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel3.parquet")
H = pd.read_parquet(ROOT/"data/prices/history_3y.parquet"); H["date"]=pd.to_datetime(H.date)
PX = H.pivot_table(index="date",columns="ticker",values="adj_close",aggfunc="last").sort_index()
HOLD, NAMES, CAP = 63, 40, 6
BENCH = {}
for D in sorted(P.date.unique()):
    D=pd.Timestamp(D)
    fi=PX.index.get_indexer([D+pd.Timedelta(days=int(HOLD*1.45))],method="nearest")[0]
    try: BENCH[D]=PX["SPY"].iloc[fi]/PX.loc[D,"SPY"]-1
    except Exception: BENCH[D]=np.nan

def book_edge(rank):
    out=[]
    for D,g in P.assign(s=rank).groupby("date"):
        g=g[(g.dollar_vol>=5e5)&g.s.notna()]
        if len(g)<200: continue
        pick,cnt=[],{}
        for t,r in g.sort_values("s",ascending=False).iterrows():
            if cnt.get(r.sector,0)>=CAP: continue
            pick.append(r.fwd); cnt[r.sector]=cnt.get(r.sector,0)+1
            if len(pick)==NAMES: break
        if len(pick)<NAMES: continue
        out.append(np.mean(pick)-BENCH[pd.Timestamp(D)])
    if len(out)<5: return np.nan, np.nan
    return np.mean(out), np.mean([x>0 for x in out])

rng=np.random.default_rng(4242); N=250; nz=[]
for i in range(N):
    f=pd.Series(rng.normal(size=len(P)),index=P.index); f[rng.random(len(P))<0.35]=np.nan
    e,_=book_edge(f.groupby(P.sector).rank(pct=True))
    if e==e: nz.append(e)
nz=np.array(nz)
print(f"NOISE NULL on the real objective: {len(nz)} random signals")
print(f"  mean {nz.mean():+.2%}, sd {nz.std():.2%}, best {nz.max():+.2%}, "
      f"{(nz>0).mean():.0%} positive\n")

SIG=[c for c in P.columns if c not in ("fwd","dollar_vol","ret_3m_prior","sector","date","ticker","comp","blend","cq","s")]
rows=[]
for c in SIG:
    e,w = book_edge(P[c])
    if e!=e: continue
    rows.append((c, e, w, (nz>=e).mean()))
rows.sort(key=lambda r:-r[1])
print(f"{'signal':22s} {'edge vs SPY':>12s} {'win':>5s} {'p vs noise':>11s}")
for c,e,w,p in rows:
    flag = "  <<" if p<=0.05 else ""
    print(f"{c:22s} {e:+12.2%} {w:5.0%} {p:11.0%}{flag}")
print(f"\n{sum(1 for r in rows if r[3]<=0.05)} of {len(rows)} beat noise at 5% on the real objective")
