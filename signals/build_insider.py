"""Insider buying from SEC Form 4 bulk datasets.

Only codes P (open-market purchase) and S (open-market sale) are voluntary
decisions. Codes A (grants) and F (tax withholding) are the bulk of the file and
carry no information -- treating them as buys is the classic way to ruin this signal.
"""
import zipfile, glob, io, pandas as pd, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
TODAY = pd.Timestamp("2026-08-19")

parts = []
for z in sorted(glob.glob(str(ROOT/"data/sec/insider/*.zip"))):
    try:
        with zipfile.ZipFile(z) as f:
            rd = lambda n: pd.read_csv(io.TextIOWrapper(f.open(n), "latin-1"), sep="\t", low_memory=False)
            t, s, o = rd("NONDERIV_TRANS.tsv"), rd("SUBMISSION.tsv"), rd("REPORTINGOWNER.tsv")
    except (zipfile.BadZipFile, KeyError):
        print(f"  skipped {pathlib.Path(z).name}"); continue
    t = t[t.TRANS_CODE.isin(["P","S"])]
    t = t.merge(s[["ACCESSION_NUMBER","FILING_DATE","ISSUERTRADINGSYMBOL"]], on="ACCESSION_NUMBER")
    t = t.merge(o[["ACCESSION_NUMBER","RPTOWNERCIK","RPTOWNER_RELATIONSHIP"]].drop_duplicates("ACCESSION_NUMBER"),
                on="ACCESSION_NUMBER", how="left")
    parts.append(t)

T = pd.concat(parts, ignore_index=True)
T["ticker"] = T.ISSUERTRADINGSYMBOL.astype(str).str.upper().str.strip()
T["filed"] = pd.to_datetime(T.FILING_DATE, format="%d-%b-%Y", errors="coerce")
T = T[T.filed.notna() & (T.filed <= TODAY)]
T["shares"] = pd.to_numeric(T.TRANS_SHARES, errors="coerce")
T["price"]  = pd.to_numeric(T.get("TRANS_PRICEPERSHARE"), errors="coerce")
T["value"]  = (T.shares * T.price).where(lambda s: s.between(0, 5e9))
T["signed"] = np.where(T.TRANS_CODE == "P", T.value, -T.value)
T["is_exec"] = T.RPTOWNER_RELATIONSHIP.astype(str).str.contains("Officer|Director", case=False, na=False)
print(f"{len(T):,} open-market insider transactions, {T.ticker.nunique():,} tickers, "
      f"{T.filed.min().date()} to {T.filed.max().date()}")

W = T[T.filed >= TODAY - pd.Timedelta(days=180)]
g = W.groupby("ticker")
out = pd.DataFrame({
    "insider_net_value": g.signed.sum(),
    "insider_buy_value": g.apply(lambda d: d.loc[d.TRANS_CODE=="P","value"].sum()),
    "insider_n_buyers":  g.apply(lambda d: d.loc[d.TRANS_CODE=="P","RPTOWNERCIK"].nunique()),
    "insider_n_sellers": g.apply(lambda d: d.loc[d.TRANS_CODE=="S","RPTOWNERCIK"].nunique()),
    "insider_exec_net":  g.apply(lambda d: d.loc[d.is_exec,"signed"].sum()),
})
R = pd.read_parquet(ROOT/"signals/signal_ranks.parquet")
mc = R.mktcap
out["InsiderNetToCap"] = (out.insider_net_value/mc.reindex(out.index)).replace([np.inf,-np.inf], np.nan)
out["InsiderBuyerBreadth"] = out.insider_n_buyers - out.insider_n_sellers
out.to_parquet(ROOT/"signals/insider.parquet")

j = out.join(mc.rename("mktcap"), how="inner").dropna(subset=["InsiderNetToCap"])
print(f"\n{len(j):,} scored tickers have insider data in the last 180 days")
print(f"  net buyers (more buying than selling): {(j.InsiderNetToCap>0).sum():,}")
print(f"  net sellers                          : {(j.InsiderNetToCap<0).sum():,}")
print(f"  median net/cap {j.InsiderNetToCap.median():+.4%}   p10 {j.InsiderNetToCap.quantile(.1):+.3%}   p90 {j.InsiderNetToCap.quantile(.9):+.3%}")
print("\nlargest net insider buying, relative to market cap:")
for t, r in j.nlargest(10, "InsiderNetToCap").iterrows():
    print(f"  {t:6s} {r.InsiderNetToCap:+7.2%} of cap   ${r.insider_buy_value/1e6:8.1f}m bought   "
          f"{int(r.insider_n_buyers)} buyers / {int(r.insider_n_sellers)} sellers")
