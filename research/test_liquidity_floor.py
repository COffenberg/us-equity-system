"""Backlog #3: does the monthly value-book edge survive at real liquidity?

Same pipeline as value_monthly_backtest.py (23 chained months, 40 names, cap 8/sector),
re-run at liquidity floors $0.5m / $2m / $10m / $50m median 60-day dollar volume.
Two baselines per floor, because a floor changes the universe as well as the picks:
  - RANDOM: 200 random 40-name books from the SAME eligible universe, same sector cap.
    If the value book's edge over SPY is matched by random equal-weight books, the
    "edge" is a size / equal-weight effect, not value.
  - edge over random = value book minus mean random book, per month.
"""
import pandas as pd, numpy as np, pathlib, warnings, sys
warnings.filterwarnings("ignore")
ROOT=pathlib.Path(__file__).resolve().parent.parent
src=(ROOT/"research/value_monthly_backtest.py").read_text()
exec(src.split("months=pd.date_range")[0])   # load data + value_score, no run

months=pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
N,CAP=40,8
FLOORS=[5e5,2e6,1e7,5e7]
NRAND=200
rng=np.random.default_rng(7)

def pick_from(order, sec, fwd):
    pick,cnt=[],{}
    for t in order:
        s_=sec.get(t,"Other")
        if cnt.get(s_,0)>=CAP: continue
        if pd.isna(fwd.get(t)): continue
        pick.append(t); cnt[s_]=cnt.get(s_,0)+1
        if len(pick)==N: break
    return pick

cache=[]
for D in months:
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    sc,sec=value_score(D)
    liq=DV.loc[:D].iloc[-60:].median().reindex(sc.index).fillna(0)
    nxt=PX.index[PX.index.get_indexer([D+pd.Timedelta(days=30)],method="nearest")[0]]
    if nxt<=D: continue
    fwd=PX.loc[nxt]/PX.loc[D]-1
    cache.append((D,sc,sec,liq,fwd))

res={}
for fl in FLOORS:
    rows=[]
    for D,sc,sec,liq,fwd in cache:
        s=sc[liq>=fl]
        s=s[fwd.reindex(s.index).notna()]
        pick=pick_from(s.sort_values(ascending=False).index,sec,fwd)
        if len(pick)<N//2: continue
        r=fwd.reindex(pick)
        rnd=[]
        idx=s.index.values
        for _ in range(NRAND):
            p=pick_from(rng.permutation(idx),sec,fwd); rnd.append(fwd.reindex(p).mean())
        rows.append({"date":D,"elig":len(s),"n":len(pick),"ret":r.mean(),"spy":fwd.get("SPY"),
                     "rnd":np.mean(rnd),"rnd_sd":np.std(rnd),"max1":r.max(),"min1":r.min(),
                     "medliq":liq.reindex(pick).median()})
    R=pd.DataFrame(rows); R["ex"]=R.ret-R.spy; R["exr"]=R.ret-R.rnd; R["rex"]=R.rnd-R.spy
    res[fl]=R

def t(x): return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
print(f"{'floor':>6s} {'mo':>3s} {'elig':>5s} {'medliq$m':>8s} {'book-SPY':>9s} {'t':>5s} {'win':>4s} "
      f"{'rand-SPY':>9s} {'t':>5s} {'book-rand':>9s} {'t':>5s} {'win':>4s} {'ann book':>9s}")
for fl,R in res.items():
    yrs=len(R)/12; ab=(1+R.ret).prod()**(1/yrs)-1
    print(f"{fl/1e6:6.1f} {len(R):3d} {R.elig.median():5.0f} {R.medliq.median()/1e6:8.1f} "
          f"{R.ex.mean():+9.2%} {t(R.ex):5.2f} {(R.ex>0).mean():4.0%} "
          f"{R.rex.mean():+9.2%} {t(R.rex):5.2f} {R.exr.mean():+9.2%} {t(R.exr):5.2f} {(R.exr>0).mean():4.0%} {ab:+9.1%}")
print("\nSanity: extremes of single-name monthly returns in picked books")
for fl,R in res.items():
    print(f"  ${fl/1e6:.1f}m  max {R.max1.max():+.0%}  min {R.min1.min():+.0%}  n range {R.n.min()}-{R.n.max()}  elig min {R.elig.min()}")
# concentration of edge: drop best month
print("\nRobustness: edge without the single best month")
for fl,R in res.items():
    e=R.ex.drop(R.ex.idxmax()); x=R.exr.drop(R.exr.idxmax())
    print(f"  ${fl/1e6:.1f}m  book-SPY {e.mean():+.2%} (t {t(e):.2f})   book-rand {x.mean():+.2%} (t {t(x):.2f})")
pd.concat({f"{int(k)}":v.drop(columns=[]) for k,v in res.items()}).to_parquet(ROOT/"research/out/liquidity_floor.parquet")
