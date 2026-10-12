"""Build and cache the point-in-time signal panel, so every combination test
runs instantly instead of rebuilding signals from filings each time."""
import pandas as pd, numpy as np, pathlib, zipfile, glob, io, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
exec(open(ROOT/"research/pit_test.py").read().split("DATES=")[0].split('print(f"{')[0])

# insider panel, point-in-time
parts=[]
for z in sorted(glob.glob(str(ROOT/"data/sec/insider/*.zip"))):
    try:
        with zipfile.ZipFile(z) as f:
            rd=lambda n: pd.read_csv(io.TextIOWrapper(f.open(n),"latin-1"),sep="\t",low_memory=False)
            t,s,o=rd("NONDERIV_TRANS.tsv"),rd("SUBMISSION.tsv"),rd("REPORTINGOWNER.tsv")
    except Exception: continue
    t=t[t.TRANS_CODE.isin(["P","S"])]
    t=t.merge(s[["ACCESSION_NUMBER","FILING_DATE","ISSUERTRADINGSYMBOL"]],on="ACCESSION_NUMBER")
    parts.append(t)
I=pd.concat(parts,ignore_index=True)
I["ticker"]=I.ISSUERTRADINGSYMBOL.astype(str).str.upper().str.strip()
I["filed"]=pd.to_datetime(I.FILING_DATE,format="%d-%b-%Y",errors="coerce")
I["value"]=(pd.to_numeric(I.TRANS_SHARES,errors="coerce")*pd.to_numeric(I.TRANS_PRICEPERSHARE,errors="coerce")).where(lambda s:s.between(0,5e9))
I["signed"]=np.where(I.TRANS_CODE=="P",I.value,-I.value)
I=I[I.filed.notna()]

DATES=pd.to_datetime(["2024-10-31","2025-01-31","2025-04-30","2025-07-31",
                      "2025-10-31","2026-01-30","2026-04-30"])
HOLD=63
rows=[]
for D in DATES:
    if D>PX.index.max(): continue
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    S=signals_asof(D)
    sec=SEC.reindex(S.index).fillna("Other")
    R=pd.DataFrame({c: S[c].groupby(sec).rank(pct=True) for c in S.columns})   # sector-relative
    w=I[(I.filed<=D)&(I.filed>=D-pd.Timedelta(days=180))]
    ins=w.groupby("ticker").signed.sum()
    R["insider"]=ins.reindex(R.index).groupby(sec).rank(pct=True)
    fi=PX.index.get_indexer([D+pd.Timedelta(days=int(HOLD*1.45))],method="nearest")[0]
    R["fwd"]=(PX.iloc[fi]/PX.loc[D]-1).reindex(R.index)
    liq=DV.loc[:D].iloc[-60:].median()
    R["dollar_vol"]=liq.reindex(R.index)
    hist=PX.loc[:D]
    R["ret_3m_prior"]=(hist.iloc[-1]/hist.iloc[max(0,len(hist)-63)]-1).reindex(R.index)
    R["sector"]=sec; R["date"]=D; R["ticker"]=R.index
    rows.append(R.reset_index(drop=True))
P=pd.concat(rows,ignore_index=True).dropna(subset=["fwd"])
P.to_parquet(ROOT/"research/out/pit_panel.parquet")
print(f"cached {len(P):,} name-dates across {P.date.nunique()} dates -> research/out/pit_panel.parquet")
print("signals:", [c for c in P.columns if c not in ("fwd","dollar_vol","ret_3m_prior","sector","date","ticker")])
