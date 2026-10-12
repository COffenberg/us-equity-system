"""Backlog item 2: which of SP, bm, cfp actually carries the edge?

Reuses the exact monthly-backtest pipeline in value_monthly_backtest.py (same
data, same universe, same rebalance, same chaining) but scores the book on
every non-empty subset of {SP, bm, cfp} instead of the mean of all three.
Screened on the real objective (book edge vs SPY), not spread.
"""
import pandas as pd, numpy as np, pathlib, warnings, itertools
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

def raw_components(D):
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
    return out, sec

months=pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
N,CAP,MINLIQ=40,8,5e5

# cache raw component ranks per month once, then re-combine per subset (cheap)
cache=[]
for D in months:
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    out,sec=raw_components(D)
    r=pd.DataFrame({c:out[c].groupby(sec).rank(pct=True) for c in out.columns})
    liq=DV.loc[:D].iloc[-60:].median()
    nxt=PX.index[PX.index.get_indexer([D+pd.Timedelta(days=30)],method="nearest")[0]]
    if nxt<=D: continue
    fwd=PX.loc[nxt]/PX.loc[D]-1
    cache.append(dict(D=D,r=r,sec=sec,liq=liq,fwd=fwd))

def run(components):
    rows=[]
    for c in cache:
        sc=c["r"][list(components)].mean(axis=1).dropna()
        sc=sc[(c["liq"].reindex(sc.index).fillna(0)>=MINLIQ)]
        pick,cnt=[],{}
        for t in sc.sort_values(ascending=False).index:
            s_=c["sec"].get(t,"Other")
            if cnt.get(s_,0)>=CAP: continue
            if pd.isna(c["fwd"].get(t)): continue
            pick.append(t); cnt[s_]=cnt.get(s_,0)+1
            if len(pick)==N: break
        if len(pick)<N//2: continue
        rows.append({"date":c["D"],"ret":c["fwd"].reindex(pick).mean(),"spy":c["fwd"].get("SPY",np.nan)})
    R=pd.DataFrame(rows); R["ex"]=R.ret-R.spy
    n=len(R); m=R.ex.mean(); sd=R.ex.std(ddof=1); t=m/(sd/np.sqrt(n)) if sd>0 else np.nan
    cum=(1+R.ret).prod()-1; cums=(1+R.spy).prod()-1
    return dict(n=n, monthly_edge=m, win=(R.ex>0).mean(), t=t, cum_book=cum, cum_spy=cums)

print(f"COMPONENT DECOMPOSITION — {len(cache)} months, same pipeline as value_monthly_backtest.py\n")
print(f"{'subset':12s} {'n':>3s} {'monthly edge':>13s} {'win':>6s} {'t':>6s} {'cum book':>10s} {'cum spy':>9s}")
results={}
for k in (1,2,3):
    for combo in itertools.combinations(["SP","bm","cfp"],k):
        res=run(combo)
        results[combo]=res
        print(f"{'+'.join(combo):12s} {res['n']:3d} {res['monthly_edge']:+13.2%} {res['win']:6.0%} "
              f"{res['t']:6.2f} {res['cum_book']:+10.1%} {res['cum_spy']:+9.1%}")

print("\nranked by monthly edge:")
for combo,res in sorted(results.items(), key=lambda kv: -kv[1]["monthly_edge"]):
    print(f"  {'+'.join(combo):12s} {res['monthly_edge']:+.2%}  t={res['t']:.2f}")
