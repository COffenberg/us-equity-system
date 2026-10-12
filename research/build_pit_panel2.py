"""Rebuild the PIT panel including the five new signal families.

Coverage floor of 1,500 tickers: below that a sector-relative rank is mostly
empty and the signal silently changes the universe rather than scoring it.
Dropped on that basis: altman_z (302), ohlson_o (455), earnings_vol (278),
ebit_to_liab (810), and the entire Piotroski f_score (68 -- it needs nine
components to all be present at once, and they rarely are in XBRL).
"""
import pandas as pd, numpy as np, pathlib, importlib.util, zipfile, glob, io, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
exec(open(ROOT/"research/pit_test.py").read().split("DATES=")[0].split('print(f"{')[0])

SH = pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
SH["filed_date"]=pd.to_datetime(SH.filed_date); SH["period_end"]=pd.to_datetime(SH.period_end)

KEEP = {
 "growth":   ["asset_growth","capex_intensity","noa","d_noa","sales_growth"],
 "quality":  ["gross_prof","cash_op_prof","roic","op_margin","d_op_margin","gp_stability"],
 "distress": ["cash_to_assets","equity_neg"],
 "redflags": ["accrual_ratio","d_accrual","tax_book_gap","d_tax_book_gap","cash_conv","d_nwc"],
}
MODS={}
for n in KEEP:
    p=ROOT/f"signals/pit_{n}.py"
    spec=importlib.util.spec_from_file_location(f"pit_{n}",p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); MODS[n]=m

# insider, as before
parts=[]
for z in sorted(glob.glob(str(ROOT/"data/sec/insider/*.zip"))):
    try:
        with zipfile.ZipFile(z) as zf:
            rd=lambda n: pd.read_csv(io.TextIOWrapper(zf.open(n),"latin-1"),sep="\t",low_memory=False)
            t,s=rd("NONDERIV_TRANS.tsv"),rd("SUBMISSION.tsv")
    except Exception: continue
    t=t[t.TRANS_CODE.isin(["P","S"])].merge(
        s[["ACCESSION_NUMBER","FILING_DATE","ISSUERTRADINGSYMBOL"]],on="ACCESSION_NUMBER")
    parts.append(t)
I=pd.concat(parts,ignore_index=True)
I["ticker"]=I.ISSUERTRADINGSYMBOL.astype(str).str.upper().str.strip()
I["filed"]=pd.to_datetime(I.FILING_DATE,format="%d-%b-%Y",errors="coerce")
I["value"]=(pd.to_numeric(I.TRANS_SHARES,errors="coerce")*pd.to_numeric(I.TRANS_PRICEPERSHARE,errors="coerce")).where(lambda s:s.between(0,5e9))
I["signed"]=np.where(I.TRANS_CODE=="P",I.value,-I.value); I=I[I.filed.notna()]

DATES=pd.to_datetime(["2024-10-31","2025-01-31","2025-04-30","2025-07-31",
                      "2025-10-31","2026-01-30","2026-04-30"])
HOLD=63
rows=[]
for D in DATES:
    if D>PX.index.max(): continue
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    S=signals_asof(D)
    f=F[F.filed_date<=D] if "F" in dir() else None
    ff=pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
    ff["filed_date"]=pd.to_datetime(ff.filed_date); ff["period_end"]=pd.to_datetime(ff.period_end)
    ff["dur"]=(ff.period_end-pd.to_datetime(ff.period_start)).dt.days
    ff=ff[ff.filed_date<=D]; sh=SH[SH.filed_date<=D]
    hist=PX.loc[:D]; px=hist.iloc[-1].dropna()
    for n,cols in KEEP.items():
        try:
            out=MODS[n].compute(ff,sh,px,hist,D)
            for c in cols:
                if c in out.columns:
                    S[c]=pd.to_numeric(out[c],errors="coerce").replace([np.inf,-np.inf],np.nan).reindex(S.index)
        except Exception as e:
            print(f"  {D.date()} {n}: {type(e).__name__}: {e}")
    sec=SEC.reindex(S.index).fillna("Other")
    R=pd.DataFrame({c:S[c].groupby(sec).rank(pct=True) for c in S.columns})
    w=I[(I.filed<=D)&(I.filed>=D-pd.Timedelta(days=180))]
    R["insider"]=w.groupby("ticker").signed.sum().reindex(R.index).groupby(sec).rank(pct=True)
    fi=PX.index.get_indexer([D+pd.Timedelta(days=int(HOLD*1.45))],method="nearest")[0]
    R["fwd"]=(PX.iloc[fi]/PX.loc[D]-1).reindex(R.index)
    R["dollar_vol"]=DV.loc[:D].iloc[-60:].median().reindex(R.index)
    R["sector"]=sec; R["date"]=D; R["ticker"]=R.index
    rows.append(R.reset_index(drop=True))
    print(f"  {D.date()}: {R.shape[1]-5} signals, {len(R)} names")
P=pd.concat(rows,ignore_index=True).dropna(subset=["fwd"])
P.to_parquet(ROOT/"research/out/pit_panel2.parquet")
sigs=[c for c in P.columns if c not in ("fwd","dollar_vol","sector","date","ticker")]
print(f"\ncached {len(P):,} name-dates, {len(sigs)} signals -> pit_panel2.parquet")
