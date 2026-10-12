"""Proper monthly backtest of the value book — consecutive periods, chained.

The quarterly-snapshot tests measured 7 non-overlapping windows and then
annualised by compounding. That is an extrapolation, not a measurement: it
assumes 12 independent repeats a year that were never observed. This rebalances
every month over the whole available history and chains the actual returns.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT=pathlib.Path(__file__).resolve().parent.parent

F=pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
F["filed_date"]=pd.to_datetime(F.filed_date); F["period_end"]=pd.to_datetime(F.period_end)
F["dur"]=(F.period_end-pd.to_datetime(F.period_start)).dt.days
SH=pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
SH["filed_date"]=pd.to_datetime(SH.filed_date); SH["period_end"]=pd.to_datetime(SH.period_end)
SEC=pd.read_parquet(ROOT/"data/sec/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector
H=pd.read_parquet(ROOT/"data/prices/history_3y.parquet"); H["date"]=pd.to_datetime(H.date)
H=H[H.adj_close>=1.0]
PX=H.pivot_table(index="date",columns="ticker",values="adj_close",aggfunc="last").sort_index()
DV=H.assign(dv=H.close*H.volume).pivot_table(index="date",columns="ticker",values="dv",aggfunc="last")

def ttm(f,c,n=4):
    d=f[(f.concept==c)&(~f.instantaneous)&f.dur.between(60,115)].sort_values("period_end") \
       .drop_duplicates(["ticker","period_end"],keep="last").groupby("ticker").tail(n)
    g=d.groupby("ticker").val.agg(["sum","count"]); return g.loc[g["count"]==n,"sum"]
def inst(f,c):
    d=f[(f.concept==c)&f.instantaneous].sort_values("period_end"); return d.groupby("ticker").last().val

def value_score(D):
    f=F[F.filed_date<=D]; sh=SH[SH.filed_date<=D]
    px=PX.loc[:D].iloc[-1].dropna()
    s=inst(f,"EntityCommonStockSharesOutstanding")
    for c in ("CommonStockSharesOutstanding","WeightedAverageNumberOfDilutedSharesOutstanding"):
        d=sh[sh.concept==c].sort_values("period_end"); s=s.combine_first(d.groupby("ticker").last().val)
    mc=(px*s.reindex(px.index)).dropna(); mc=mc[mc>5e7]
    rev=ttm(f,"Revenues").combine_first(ttm(f,"RevenueFromContractWithCustomerExcludingAssessedTax"))
    ocf=ttm(f,"NetCashProvidedByUsedInOperatingActivities")
    eq=inst(f,"StockholdersEquity")
    out=pd.DataFrame({"SP":rev/mc,"cfp":ocf/mc,"bm":eq/mc})
    out=out.replace([np.inf,-np.inf],np.nan)
    for c in out.columns:
        lo,hi=out[c].quantile([.01,.99]); out[c]=out[c].clip(lo,hi)
    sec=SEC.reindex(out.index).fillna("Other")
    r=pd.DataFrame({c:out[c].groupby(sec).rank(pct=True) for c in out.columns})
    return r.mean(axis=1).dropna(), sec

months=pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
N,CAP,MINLIQ=40,8,5e5
rows=[]
for D in months:
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    sc,sec=value_score(D)
    liq=DV.loc[:D].iloc[-60:].median()
    sc=sc[(liq.reindex(sc.index).fillna(0)>=MINLIQ)]
    nxt=PX.index[PX.index.get_indexer([D+pd.Timedelta(days=30)],method="nearest")[0]]
    if nxt<=D: continue
    fwd=PX.loc[nxt]/PX.loc[D]-1
    pick,cnt=[],{}
    for t in sc.sort_values(ascending=False).index:
        s_=sec.get(t,"Other")
        if cnt.get(s_,0)>=CAP: continue
        if pd.isna(fwd.get(t)): continue
        pick.append(t); cnt[s_]=cnt.get(s_,0)+1
        if len(pick)==N: break
    if len(pick)<N//2: continue
    rows.append({"date":D,"n":len(pick),"ret":fwd.reindex(pick).mean(),
                 "spy":fwd.get("SPY",np.nan),"names":set(pick)})
R=pd.DataFrame(rows)
R["ex"]=R.ret-R.spy
prev=None; turn=[]
for s in R.names:
    turn.append(np.nan if prev is None else 1-len(s&prev)/len(s)); prev=s
R["turnover"]=turn
print(f"MONTHLY VALUE BOOK — {len(R)} consecutive months, {N} names, rebalanced every month\n")
print(f"{'month':12s} {'n':>3s} {'book':>8s} {'SPY':>8s} {'edge':>8s} {'turnover':>9s}")
for r in R.itertuples():
    print(f"{str(r.date)[:10]:12s} {r.n:3d} {r.ret:+8.2%} {r.spy:+8.2%} {r.ex:+8.2%} "
          f"{'' if r.turnover!=r.turnover else f'{r.turnover:8.0%}'}")
cum_b=(1+R.ret).prod()-1; cum_s=(1+R.spy).prod()-1
yrs=len(R)/12
print(f"\ncumulative  book {cum_b:+.1%}   SPY {cum_s:+.1%}   over {yrs:.1f} years")
print(f"annualised  book {((1+cum_b)**(1/yrs)-1):+.1%}   SPY {((1+cum_s)**(1/yrs)-1):+.1%}")
print(f"monthly edge {R.ex.mean():+.2%}  win {(R.ex>0).mean():.0%}  sd {R.ex.std():.2%}  "
      f"t {R.ex.mean()/(R.ex.std(ddof=1)/np.sqrt(len(R))):.2f}")
print(f"average turnover {R.turnover.mean():.0%}/month -> ~{R.turnover.mean()*N:.0f} trades a month")
