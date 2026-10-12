"""Does insider buying predict forward returns in OUR data? Point-in-time."""
import zipfile, glob, io, pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

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
T=pd.concat(parts,ignore_index=True)
T["ticker"]=T.ISSUERTRADINGSYMBOL.astype(str).str.upper().str.strip()
T["filed"]=pd.to_datetime(T.FILING_DATE,format="%d-%b-%Y",errors="coerce")
T["value"]=(pd.to_numeric(T.TRANS_SHARES,errors="coerce")*pd.to_numeric(T.TRANS_PRICEPERSHARE,errors="coerce")).where(lambda s:s.between(0,5e9))
T["signed"]=np.where(T.TRANS_CODE=="P",T.value,-T.value)
T=T[T.filed.notna()]

H=pd.read_parquet(ROOT/"data/prices/history_3y.parquet").sort_values(["ticker","date"])
H["date"]=pd.to_datetime(H.date); H=H[H.adj_close>=1.0]
PX=H.pivot_table(index="date",columns="ticker",values="adj_close",aggfunc="last").sort_index()
DV=H.assign(dv=H.close*H.volume).pivot_table(index="date",columns="ticker",values="dv",aggfunc="last")
SEC=pd.read_parquet(ROOT/"data/sec/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector

DATES=[d for d in pd.to_datetime(["2025-01-31","2025-04-30","2025-07-31","2025-10-31","2026-01-30","2026-04-30"])
       if d <= PX.index.max()]
HOLD=63
res={}
for D in DATES:
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    w=T[(T.filed<=D)&(T.filed>=D-pd.Timedelta(days=180))]
    g=w.groupby("ticker")
    feat=pd.DataFrame({
        "net_value": g.signed.sum(),
        "breadth": g.apply(lambda d: d.loc[d.TRANS_CODE=="P","RPTOWNERCIK"].nunique()
                                    - d.loc[d.TRANS_CODE=="S","RPTOWNERCIK"].nunique()),
        "any_buy": g.apply(lambda d: (d.TRANS_CODE=="P").any().astype(int)),
    })
    liq=DV.loc[:D].iloc[-60:].median()
    uni=[t for t in PX.columns if liq.get(t,0)>=5e5 and pd.notna(PX.loc[D,t])]
    fi=PX.index.get_indexer([D+pd.Timedelta(days=int(HOLD*1.45))],method="nearest")[0]
    fwd=(PX.iloc[fi]/PX.loc[D]-1).reindex(uni)
    sec=SEC.reindex(uni).fillna("Other")
    # names with NO insider activity are the natural control group
    for name,series in [("net_value",feat.net_value),("breadth",feat.breadth)]:
        v=series.reindex(uni)
        d=pd.DataFrame({"v":v,"f":fwd,"s":sec}).dropna(subset=["f"])
        act=d[d.v.notna()&(d.v!=0)]
        if len(act)<200: continue
        r=act.groupby("s").v.rank(pct=True)
        q=pd.qcut(r,5,labels=False,duplicates="drop")
        m=act.groupby(q).f.mean()
        if len(m)==5:
            res.setdefault(name,[]).append(np.append(m.values, d[d.v.isna()|(d.v==0)].f.mean()))

print(f"{'signal':11s} {'n':>3s} {'Q1':>8s} {'Q3':>8s} {'Q5':>8s} {'Q5-Q1':>8s} {'mono':>5s} {'win':>5s} {'no-activity':>12s}")
for c,v in res.items():
    M=np.array(v); mean=M.mean(0); sp=M[:,4]-M[:,0]
    mono=sum(1 for i in range(4) if mean[i+1]>mean[i])
    print(f"{c:11s} {len(M):3d} {mean[0]:+8.2%} {mean[2]:+8.2%} {mean[4]:+8.2%} {sp.mean():+8.2%} "
          f"{mono:3d}/4 {(sp>0).mean():5.0%} {mean[5]:+12.2%}")
print(f"\nhorizon {HOLD} trading days, {len(DATES)} dates, sector-relative ranks among names with insider activity")
print("no-activity column = mean forward return of names with NO open-market insider trades")
