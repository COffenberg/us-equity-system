"""PIT panel including the published survivors we had never built."""
import pandas as pd, numpy as np, pathlib, importlib.util, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
exec(open(ROOT/"research/pit_test.py").read().split("DATES=")[0].split('print(f"{')[0])
SH = pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
SH["filed_date"]=pd.to_datetime(SH.filed_date); SH["period_end"]=pd.to_datetime(SH.period_end)
spec=importlib.util.spec_from_file_location("pp",ROOT/"signals/pit_published.py")
PP=importlib.util.module_from_spec(spec); spec.loader.exec_module(PP)
DROP={"MomVol","FirmAgeMom"}     # duplicate of Mom6m; coverage 219

OLD = pd.read_parquet(ROOT/"research/out/pit_panel2.parquet")
DATES = sorted(OLD.date.unique())
rows=[]
for D in DATES:
    D=pd.Timestamp(D)
    ff=pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
    ff["filed_date"]=pd.to_datetime(ff.filed_date); ff["period_end"]=pd.to_datetime(ff.period_end)
    ff["dur"]=(ff.period_end-pd.to_datetime(ff.period_start)).dt.days
    ff=ff[ff.filed_date<=D]
    hist=PX.loc[:D]; pxl=hist.iloc[-1].dropna()
    try:
        new=PP.compute(ff, SH[SH.filed_date<=D], pxl, hist, D)
    except Exception as e:
        print(f"  {D.date()}: {type(e).__name__}: {e}"); continue
    new=new[[c for c in new.columns if c not in DROP]]
    sub=OLD[OLD.date==D].set_index("ticker")
    sec=sub.sector
    for c in new.columns:
        sub[c]=pd.to_numeric(new[c],errors="coerce").reindex(sub.index).groupby(sec).rank(pct=True)
    rows.append(sub.reset_index())
    print(f"  {D.date()}: +{len(new.columns)} signals, {len(sub)} names")
P=pd.concat(rows,ignore_index=True)
P.to_parquet(ROOT/"research/out/pit_panel3.parquet")
sigs=[c for c in P.columns if c not in ("fwd","dollar_vol","ret_3m_prior","sector","date","ticker")]
print(f"\n{len(P):,} name-dates, {len(sigs)} signals -> pit_panel3.parquet")
