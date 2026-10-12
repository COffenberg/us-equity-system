"""Institutional ownership breadth from SEC Form 13F.

13F is keyed by CUSIP and there is no free CUSIP->ticker map, so issuer names are
normalised and matched against the SEC roster instead. The hit rate is reported
rather than assumed -- an unmeasured join is how you get a signal for the wrong
companies.

Signal is BREADTH, not size: Chen/Hong/Stein (2002) found the number of distinct
holders, and its change, predicts returns -- not the dollar value held, which is
mostly a proxy for market cap.
"""
import zipfile, io, glob, re, pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

SUFFIX = re.compile(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LP|LLC|HOLDINGS?|"
                    r"GROUP|TRUST|THE|NEW|COM|CLASS\s*[A-C]|CL\s*[A-C]|SA|NV|AG|ADR|ADS)\b")
def norm(s):
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

uni = pd.read_csv(sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1], dtype={"cik": str})
uni["key"] = uni.title.map(norm)
uni = uni[uni.key.str.len() >= 3].drop_duplicates("key", keep="first")
name2tic = dict(zip(uni.key, uni.ticker))
print(f"roster keys: {len(name2tic):,}")

rows = []
for z in sorted(glob.glob(str(ROOT/"data/sec/f13/*.zip"))):
    period = pathlib.Path(z).stem
    try:
        with zipfile.ZipFile(z) as zf:
            info = pd.read_csv(io.TextIOWrapper(zf.open("INFOTABLE.tsv"), "latin-1"),
                               sep="\t", low_memory=False,
                               usecols=["ACCESSION_NUMBER","NAMEOFISSUER","CUSIP","VALUE","SSHPRNAMT","PUTCALL"])
            sub = pd.read_csv(io.TextIOWrapper(zf.open("SUBMISSION.tsv"), "latin-1"),
                              sep="\t", low_memory=False, usecols=["ACCESSION_NUMBER","CIK","PERIODOFREPORT"])
    except Exception as e:
        print(f"  skip {period}: {e}"); continue
    info = info[info.PUTCALL.isna()]                    # equity positions only, not options
    info = info.merge(sub, on="ACCESSION_NUMBER", how="left")
    info["key"] = info.NAMEOFISSUER.map(norm)
    info["ticker"] = info.key.map(name2tic)
    hit = info.ticker.notna().mean()
    g = (info.dropna(subset=["ticker"])
             .groupby(["ticker","PERIODOFREPORT"])
             .agg(holders=("CIK","nunique"), shares=("SSHPRNAMT","sum"), value=("VALUE","sum"))
             .reset_index())
    g["period"] = period
    rows.append(g)
    print(f"  {period}: {len(info):,} positions, name-match hit rate {hit:.1%}, "
          f"{g.ticker.nunique():,} tickers")

H = pd.concat(rows, ignore_index=True)
H["dt"] = pd.to_datetime(H.PERIODOFREPORT, format="%d-%b-%Y", errors="coerce")
H = H.dropna(subset=["dt"]).sort_values("dt")
H = H.groupby(["ticker","dt"], as_index=False).agg(holders=("holders","max"),
                                                   shares=("shares","max"), value=("value","max"))
H.to_parquet(ROOT/"data/sec/f13_holdings.parquet")
print(f"\n{len(H):,} ticker-quarters, {H.ticker.nunique():,} tickers, "
      f"{H.dt.min().date()} to {H.dt.max().date()}")
print(H.groupby(H.dt.dt.date).ticker.nunique().to_string())
