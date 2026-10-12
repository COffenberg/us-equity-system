"""Backlog #3b: what do thin-name trading costs eat?

Same pipeline as test_liquidity_floor.py (23 chained months, 40 names, cap 8/sector,
floors $0.5m/$2m/$10m/$50m). Panel has close+volume only, so no Corwin-Schultz.

Per month and floor:
  traded weight  = sum |w_target - w_drifted| (equal-weight reset incl. exits/entries)
  spread cost    = sum |dw_i| * halfspread_i
  halfspread_i   = Roll (1984) from trailing 120d daily-return autocovariance,
                   s = 2*sqrt(-cov); when cov>=0 (Roll undefined) use the floor-bucket
                   median of defined estimates (not zero -- zero flatters thin names)
  impact         = sqrt-law: Y * sigma_d * sqrt(Q/ADV), Y=0.7, per side, at book sizes
                   $100k and $1m (book size is not fixed; show both)
Also reports the model-free BREAKEVEN: the uniform half-spread at which the $0.5m
book's net edge falls to the $2m book's net edge (assuming $2m names cost half as much).
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT=pathlib.Path(__file__).resolve().parent.parent
src=(ROOT/"research/value_monthly_backtest.py").read_text()
exec(src.split("months=pd.date_range")[0])

months=pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
N,CAP=40,8
FLOORS=[5e5,2e6,1e7,5e7]
BOOKS=[1e5,1e6]
# assumed half-spreads by 60d median ADV: <$2m 50bp, $2-10m 20bp, $10-50m 8bp, >$50m 3bp
BUCKET_HS=[0.0050,0.0020,0.0008,0.0003]; Y=0.7; WIN=120
RET=PX.pct_change(fill_method=None)

def pick_from(order, sec, fwd):
    pick,cnt=[],{}
    for t in order:
        s_=sec.get(t,"Other")
        if cnt.get(s_,0)>=CAP: continue
        if pd.isna(fwd.get(t)): continue
        pick.append(t); cnt[s_]=cnt.get(s_,0)+1
        if len(pick)==N: break
    return pick

def roll_hs(D, names):
    r=RET.loc[:D].iloc[-WIN:][names]
    out={}
    for t in names:
        x=r[t].dropna()
        if len(x)<60: out[t]=np.nan; continue
        c=np.cov(x.values[1:],x.values[:-1])[0,1]
        out[t]=np.sqrt(-c) if c<0 else np.nan   # half-spread = sqrt(-cov)
    return pd.Series(out)

cache=[]
for D in months:
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    sc,sec=value_score(D)
    liq=DV.loc[:D].iloc[-60:].median().reindex(sc.index).fillna(0)
    nxt=PX.index[PX.index.get_indexer([D+pd.Timedelta(days=30)],method="nearest")[0]]
    if nxt<=D: continue
    fwd=PX.loc[nxt]/PX.loc[D]-1
    cache.append((D,sc,sec,liq,fwd))

res={}; hsall={}
for fl in FLOORS:
    rows=[]; prev=None; hs_list=[]
    for D,sc,sec,liq,fwd in cache:
        s=sc[liq>=fl]; s=s[fwd.reindex(s.index).notna()]
        pick=pick_from(s.sort_values(ascending=False).index,sec,fwd)
        if len(pick)<N//2: prev=None; continue
        w_t=pd.Series(1/len(pick),index=pick)
        if prev is None: w_d=pd.Series(dtype=float)
        else:
            pw,pr=prev; g=(1+pr.fillna(-1.0)); w_d=pw*g/(pw*g).sum()   # missing fwd = delisted, treated as -100%
        dw=w_t.reindex(w_t.index.union(w_d.index)).fillna(0)-w_d.reindex(w_t.index.union(w_d.index)).fillna(0)
        dw=dw.abs()
        names=list(dw.index)
        hs=roll_hs(D,names); hs_list.append(hs.reindex(pick))
        sig=RET.loc[:D].iloc[-60:][names].std()
        adv=DV.loc[:D].iloc[-60:][names].median()
        rows.append({"date":D,"ret":fwd.reindex(pick).mean(),"spy":fwd.get("SPY"),"first":prev is None,
                     "traded":dw.sum(),"hs":hs,"dw":dw,"sig":sig,"adv":adv,
                     "medliq":liq.reindex(pick).median()})
        prev=(w_t,fwd.reindex(pick))
    H_=pd.concat(hs_list); hsall[fl]=H_
    fill=H_.median()
    for r in rows:
        h=r["hs"].fillna(fill).clip(upper=0.05)
        r["c_spread"]=(r["dw"]*h).sum()
        for B in BOOKS:
            q=r["dw"]*B; r[f"c_imp{int(B/1e3)}k"]=(r["dw"]*Y*r["sig"]*np.sqrt(q/r["adv"].clip(lower=1))).fillna(0).sum()
        r["roll_defined"]=r["hs"].notna().mean()
        hb=pd.cut(r["adv"],[-1,2e6,1e7,5e7,np.inf],labels=False).map(dict(enumerate(BUCKET_HS))).fillna(BUCKET_HS[0])
        r["c_bucket"]=(r["dw"]*hb).sum()
    R=pd.DataFrame([{k:v for k,v in r.items() if k not in("hs","dw","sig","adv")} for r in rows])
    R=R[~R["first"]]                 # steady state: drop the initial build month
    R["ex"]=R.ret-R.spy
    res[fl]=R

def t(x): return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
print("Roll half-spread of held names (defined estimates only)")
for fl,H_ in hsall.items():
    d=H_.dropna()
    print(f"  ${fl/1e6:4.1f}m  defined {H_.notna().mean():4.0%}  median {d.median():.2%}  p75 {d.quantile(.75):.2%}  p90 {d.quantile(.9):.2%}  max {d.max():.2%}")
print(f"\n{'floor':>6s} {'mo':>3s} {'medliq$m':>8s} {'traded/mo':>9s} {'gross':>7s} {'spread':>7s} {'net':>7s} {'t':>5s} "
      f"{'imp100k':>7s} {'net100k':>8s} {'imp1m':>6s} {'net1m':>7s} {'t':>5s}")
summ={}
for fl,R in res.items():
    n0=R.ex-R.c_spread; n1=n0-R.c_imp100k; n2=n0-R.c_imp1000k
    summ[fl]=dict(gross=R.ex.mean(),spread=R.c_spread.mean(),net=n0.mean(),net100k=n1.mean(),net1m=n2.mean(),traded=R.traded.mean())
    print(f"{fl/1e6:6.1f} {len(R):3d} {R.medliq.median()/1e6:8.1f} {R.traded.mean():9.0%} {R.ex.mean():+7.2%} {R.c_spread.mean():7.2%} "
          f"{n0.mean():+7.2%} {t(n0):5.2f} {R.c_imp100k.mean():7.2%} {n1.mean():+8.2%} {R.c_imp1000k.mean():6.2%} {n2.mean():+7.2%} {t(n2):5.2f}")
print("\nADV-bucket spread model (primary; Roll fails sanity) -- base and 2x stress, + sqrt impact at $1m book")
for fl,R in res.items():
    nb=R.ex-R.c_bucket-R.c_imp1000k; ns=R.ex-2*R.c_bucket-R.c_imp1000k
    print(f"  ${fl/1e6:4.1f}m  gross {R.ex.mean():+.2%}  bucket cost {R.c_bucket.mean():.2%}  net {nb.mean():+.2%} (t {t(nb):.2f})  2x-stress net {ns.mean():+.2%} (t {t(ns):.2f})")
a,b=summ[5e5],summ[2e6]
# breakeven uniform half-spread h at $0.5m, with $2m names costing h/2:  gA - TA*h = gB - TB*h/2
h=(a["gross"]-b["gross"])/(a["traded"]-b["traded"]/2)
h0=a["gross"]/a["traded"]
print(f"\nBreakeven: uniform half-spread at $0.5m that equalises net with $2m (at half cost): {h:.2%}")
print(f"Breakeven: uniform half-spread at $0.5m that takes gross edge to zero: {h0:.2%}")
print("\nSanity: per-month cost extremes")
for fl,R in res.items():
    print(f"  ${fl/1e6:.1f}m  spread cost max {R.c_spread.max():.2%}  imp1m max {R.c_imp1000k.max():.2%}  traded range {R.traded.min():.0%}-{R.traded.max():.0%}  roll defined {R.roll_defined.mean():.0%}")
pd.concat({f"{int(k)}":v for k,v in res.items()}).to_parquet(ROOT/"research/out/thin_name_costs.parquet")
