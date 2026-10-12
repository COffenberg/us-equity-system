"""Load the five new signal modules, run them point-in-time, report coverage + sanity.

These were written by agents that died before verification ran, so nothing here is
trusted yet. This is the check, not a formality.
"""
import pandas as pd, numpy as np, pathlib, importlib.util, sys, warnings, traceback
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

F = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
F["filed_date"]=pd.to_datetime(F.filed_date); F["period_end"]=pd.to_datetime(F.period_end)
F["dur"]=(F.period_end-pd.to_datetime(F.period_start)).dt.days
SH = pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
SH["filed_date"]=pd.to_datetime(SH.filed_date); SH["period_end"]=pd.to_datetime(SH.period_end)
H = pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
H["date"]=pd.to_datetime(H.date); H=H[H.adj_close>=1.0]
PX = H.pivot_table(index="date",columns="ticker",values="adj_close",aggfunc="last").sort_index()

def load(name):
    p=ROOT/f"signals/pit_{name}.py"
    spec=importlib.util.spec_from_file_location(f"pit_{name}",p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

D=pd.Timestamp("2026-01-30")
D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
f=F[F.filed_date<=D]; sh=SH[SH.filed_date<=D]
hist=PX.loc[:D]; px=hist.iloc[-1].dropna()

for name in ["growth","quality","distress","redflags","fscore"]:
    print(f"\n{'='*62}\n{name.upper()}")
    try:
        m=load(name)
        out=m.compute(f, sh, px, hist, D)
        if not isinstance(out,pd.DataFrame): print(f"  BAD: returned {type(out)}"); continue
        print(f"  {out.shape[1]} signals, {len(out)} tickers")
        print(f"  {'signal':20s} {'cover':>6s} {'p10':>11s} {'median':>11s} {'p90':>11s}")
        for c in out.columns:
            s=pd.to_numeric(out[c],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
            if not len(s): print(f"  {c:20s} {0:6d}   ALL NULL"); continue
            q=s.quantile([.1,.5,.9])
            print(f"  {c:20s} {len(s):6d} {q.iloc[0]:11.4g} {q.iloc[1]:11.4g} {q.iloc[2]:11.4g}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
