"""Does insider buying add anything IN COMBINATION, even though it fails alone?

The univariate test in test_insider.py is the same pass/fail-per-signal gating
that killed an earlier project: it discards signals with IC 0.02-0.05 that are
worthless alone and valuable inside a blend. Three proper tests here:

  1. INTERACTION  -- does insider buying separate winners INSIDE composite quintiles?
  2. INCREMENTAL  -- does composite+insider beat composite alone?
  3. CONDITIONAL  -- does it work in subsets (small caps, after a fall, illiquid)?
"""
import zipfile, glob, io, pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
exec(open(ROOT/"research/pit_test.py").read().split("DATES=")[0].split('print(f"{')[0])  # reuse loaders

# ---- insider panel ----
parts=[]
for z in sorted(glob.glob(str(ROOT/"data/sec/insider/*.zip"))):
    try:
        with zipfile.ZipFile(z) as f:
            rd=lambda n: pd.read_csv(io.TextIOWrapper(f.open(n),"latin-1"),sep="\t",low_memory=False)
            t,s,o=rd("NONDERIV_TRANS.tsv"),rd("SUBMISSION.tsv"),rd("REPORTINGOWNER.tsv")
    except Exception: continue
    t=t[t.TRANS_CODE.isin(["P","S"])]
    t=t.merge(s[["ACCESSION_NUMBER","FILING_DATE","ISSUERTRADINGSYMBOL"]],on="ACCESSION_NUMBER")
    t=t.merge(o[["ACCESSION_NUMBER","RPTOWNERCIK"]].drop_duplicates("ACCESSION_NUMBER"),on="ACCESSION_NUMBER",how="left")
    parts.append(t)
I=pd.concat(parts,ignore_index=True)
I["ticker"]=I.ISSUERTRADINGSYMBOL.astype(str).str.upper().str.strip()
I["filed"]=pd.to_datetime(I.FILING_DATE,format="%d-%b-%Y",errors="coerce")
I["value"]=(pd.to_numeric(I.TRANS_SHARES,errors="coerce")*pd.to_numeric(I.TRANS_PRICEPERSHARE,errors="coerce")).where(lambda s:s.between(0,5e9))
I["signed"]=np.where(I.TRANS_CODE=="P",I.value,-I.value)
I=I[I.filed.notna()]

DATES=pd.to_datetime(["2025-01-31","2025-04-30","2025-07-31","2025-10-31","2026-01-30","2026-04-30"])
HOLD=63
# composite built from the signals that PASSED the PIT test, as production would
GOOD=["mom_12_1","bm","px_52w","SP"]
rows=[]
for D in DATES:
    if D>PX.index.max(): continue
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    S=signals_asof(D)
    sec=SEC.reindex(S.index).fillna("Other")
    comp=pd.concat([S[c].groupby(sec).rank(pct=True) for c in GOOD if c in S],axis=1).mean(axis=1)

    w=I[(I.filed<=D)&(I.filed>=D-pd.Timedelta(days=180))]
    g=w.groupby("ticker")
    ins=pd.DataFrame({"net":g.signed.sum(),
                      "breadth":g.apply(lambda d: d.loc[d.TRANS_CODE=="P","RPTOWNERCIK"].nunique()
                                                 -d.loc[d.TRANS_CODE=="S","RPTOWNERCIK"].nunique())})
    fi=PX.index.get_indexer([D+pd.Timedelta(days=int(HOLD*1.45))],method="nearest")[0]
    fwd=PX.iloc[fi]/PX.loc[D]-1
    d=pd.DataFrame({"comp":comp,"fwd":fwd.reindex(comp.index),"sec":sec,
                    "net":ins.net.reindex(comp.index),"breadth":ins.breadth.reindex(comp.index),
                    "mktcap":None}).dropna(subset=["comp","fwd"])
    d["ins_rank"]=d.net.groupby(d.sec).rank(pct=True)
    d["date"]=D
    rows.append(d)
A=pd.concat(rows)
print(f"{len(A):,} name-dates, {A.date.nunique()} dates, composite from {GOOD}\n")

# ---------- 1. INTERACTION ----------
print("1. INTERACTION — insider signal inside each composite quintile")
A["cq"]=A.groupby("date").comp.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))
sub=A[A.ins_rank.notna()].copy()
sub["iq"]=sub.groupby("date").ins_rank.transform(lambda s: pd.qcut(s,3,labels=False,duplicates="drop"))
tab=sub.pivot_table(index="cq",columns="iq",values="fwd",aggfunc="mean")
tab.columns=["insiders selling","neutral","insiders buying"]
tab.index=[f"composite Q{int(i)+1}" for i in tab.index]
print((tab*100).round(2).to_string())
print("\n  insider spread (buying - selling) within each composite quintile:")
for i,r in tab.iterrows():
    print(f"    {i:16s} {(r['insiders buying']-r['insiders selling'])*100:+6.2f} pp")

# ---------- 2. INCREMENTAL ----------
print("\n2. INCREMENTAL — does adding insider to the composite improve the spread?")
for w in [0.0,0.10,0.20,0.35]:
    A["blend"]=(1-w)*A.comp.fillna(A.comp.median())+w*A.ins_rank.fillna(0.5)
    q=A.groupby("date").blend.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))
    m=A.groupby([A.date,q]).fwd.mean().unstack()
    sp=(m[4]-m[0])
    print(f"   insider weight {w:4.0%}   mean Q5-Q1 {sp.mean():+6.2%}   win {(sp>0).mean():4.0%}")

# ---------- 3. CONDITIONAL ----------
print("\n3. CONDITIONAL — where insider buying might actually work")
# A carries duplicate tickers across dates, so build the columns positionally
A = A.reset_index().rename(columns={"index":"ticker"})
dv_col, pr_col = [], []
for D, gg in A.groupby("date", sort=False):
    liq  = DV.loc[:D].iloc[-60:].median()
    hist = PX.loc[:D]
    prior = hist.iloc[-1]/hist.iloc[max(0,len(hist)-63)] - 1
    dv_col.append(pd.Series(liq.reindex(gg.ticker).values, index=gg.index))
    pr_col.append(pd.Series(prior.reindex(gg.ticker).values, index=gg.index))
A["dv"] = pd.concat(dv_col).sort_index()
A["ret_3m_prior"] = pd.concat(pr_col).sort_index()

s2=A[A.ins_rank.notna()]
def spread(df):
    if len(df)<150: return np.nan,0
    q=pd.qcut(df.ins_rank,3,labels=False,duplicates="drop")
    m=df.groupby(q).fwd.mean()
    return (m.iloc[-1]-m.iloc[0]) if len(m)==3 else np.nan, len(df)
conds={
 "all names":            s2,
 "small (bottom-half $vol)": s2[s2.dv<=s2.dv.median()],
 "large (top-half $vol)":    s2[s2.dv> s2.dv.median()],
 "after a 3m fall":      s2[s2.ret_3m_prior<0],
 "after a 3m rise":      s2[s2.ret_3m_prior>=0],
 "weak composite (Q1-Q2)": s2[s2.cq<=1],
 "strong composite (Q4-Q5)": s2[s2.cq>=3],
}
print(f"   {'subset':26s} {'n':>7s} {'insider spread':>15s}")
for k,v in conds.items():
    sp,n=spread(v)
    print(f"   {k:26s} {n:7,d} {sp*100 if sp==sp else float('nan'):+15.2f} pp")
