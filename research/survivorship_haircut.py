"""How much of the value result is survivorship? Measure it, don't guess.

Yahoo deletes delisted symbols outright, so a company that failed is absent from
the price panel at EVERY date, not just after it failed. SEC filing records do
not delete anyone. A company filing 10-Ks in 2024 and never again either died,
was acquired, or went private -- and the price panel will not show it either way.

This finds names that were filing at each scoring date, disappeared afterwards,
and are missing from the panel, then asks what their absence is worth.
"""
import zipfile, glob, io, pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

subs=[]
for z in sorted(glob.glob(str(ROOT/"data/sec/dera/*.zip"))):
    try:
        with zipfile.ZipFile(z) as f:
            d=pd.read_csv(io.TextIOWrapper(f.open("sub.txt"),"latin-1"),sep="\t",
                          usecols=["cik","name","form","period","filed"],low_memory=False)
            subs.append(d)
    except Exception: continue
S=pd.concat(subs,ignore_index=True)
S=S[S.form.astype(str).str.startswith(("10-K","10-Q"))]
S["filed"]=pd.to_datetime(S.filed,format="%Y%m%d",errors="coerce")
S=S.dropna(subset=["filed"])
S["cik"]=S.cik.astype(int).astype(str).str.zfill(10)

uni=pd.read_csv(sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1],dtype={"cik":str})
c2t=dict(zip(uni.cik,uni.ticker))
S["ticker"]=S.cik.map(c2t)
S=S.dropna(subset=["ticker"])

last_filing=S.groupby("ticker").filed.max()
first_filing=S.groupby("ticker").filed.min()
PANEL=set(pd.read_parquet(ROOT/"research/out/pit_panel3.parquet").ticker)
H=pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
PRICED=set(H.ticker)

CUT=S.filed.max()-pd.Timedelta(days=200)      # allow for normal filing gaps
stopped=set(last_filing[last_filing<CUT].index) & set(first_filing[first_filing<CUT-pd.Timedelta(days=200)].index)
print(f"SEC filers mapped to tickers: {len(last_filing):,}")
print(f"  stopped filing >200d before the last DERA quarter: {len(stopped):,}")
print(f"  of those, absent from the price panel entirely:    {len(stopped-PRICED):,}")
print(f"  of those, present in the price panel:              {len(stopped&PRICED):,}")

# what do the ones we CAN see do before they vanish? that bounds the invisible ones
H["date"]=pd.to_datetime(H.date)
PX=H.pivot_table(index="date",columns="ticker",values="adj_close",aggfunc="last").sort_index()
vis=[t for t in (stopped&PRICED) if t in PX.columns]
rets=[]
for t in vis:
    s=PX[t].dropna()
    if len(s)<60: continue
    rets.append(s.iloc[-1]/s.iloc[max(0,len(s)-63)]-1)
rets=pd.Series(rets)
print(f"\nfinal-quarter return of visible stoppers (n={len(rets)}):")
print(f"  mean {rets.mean():+.1%}  median {rets.median():+.1%}  "
      f"p10 {rets.quantile(.1):+.1%}  p90 {rets.quantile(.9):+.1%}")
print(f"  share losing >50%: {(rets<-0.5).mean():.0%}")

# attrition rate per quarter, and the haircut it implies for a 40-name book
span_q=(S.filed.max()-S.filed.min()).days/91.25
rate=len(stopped)/max(1,len(last_filing))/span_q
print(f"\nquarterly attrition among filers: {rate:.2%}")
for loss in (-0.5,-0.8,-1.0):
    print(f"  if a vanished name realises {loss:+.0%}: drag on any book = {rate*loss*100:+.2f}pp/quarter")
print("\nNOTE: this is the drag on an EQUAL-WEIGHTED book drawn from the full filer")
print("population. A value screen selects distressed names, so its attrition is")
print("higher than average -- the drag below is a floor, not an estimate.")
