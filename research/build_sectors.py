"""cik -> SIC -> sector, from SEC's own filing metadata (sub.txt in the DERA sets)."""
import zipfile, glob, io, pandas as pd, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent

subs = []
for z in sorted(glob.glob(str(ROOT/"data/sec/dera/*.zip"))):
    try:                                     # skip archives still downloading
        with zipfile.ZipFile(z) as f, f.open("sub.txt") as fh:
            subs.append(pd.read_csv(io.TextIOWrapper(fh, "latin-1"), sep="\t",
                                    usecols=["cik","name","sic","period"], low_memory=False))
    except (zipfile.BadZipFile, KeyError):
        print(f"  skipped incomplete: {pathlib.Path(z).name}"); continue
sub = pd.concat(subs).dropna(subset=["sic"]).sort_values("period")
sub["cik"] = sub.cik.astype(int).astype(str).str.zfill(10)
sic = sub.groupby("cik").sic.last().astype(int)          # most recent SIC per filer

def sector(s):
    # specific rules first
    if 6020 <= s <= 6199: return "Banks"
    if 6300 <= s <= 6411: return "Insurance"
    if 6500 <= s <= 6599 or s == 6798: return "Real estate"
    if 6000 <= s <= 6999: return "Financial other"
    if s in (2833,2834,2835,2836) or s == 8731: return "Biotech & pharma"
    if 8000 <= s <= 8099: return "Healthcare services"
    if 3826 <= s <= 3851 or s == 3821: return "Medical devices"
    if 7370 <= s <= 7379 or 3570 <= s <= 3579 or 3670 <= s <= 3679 or s in (3674,5045,5734): return "Technology"
    if s in (7389,8742,8741,8744,8748,7363,7361,8711,8721,8734,7320,7322): return "Business services"
    if 4800 <= s <= 4899 or 2700 <= s <= 2799 or 7800 <= s <= 7999 or s in (7310,7311,7312): return "Media & telecom"
    if s in (1311,1381,1382,1389,2911,4922,4923,4924) or 1200 <= s <= 1299: return "Energy"
    if 4900 <= s <= 4991: return "Utilities"
    if 4000 <= s <= 4789: return "Transport"

    # complete fallback by SIC major group -- nothing should reach "Other"
    mg = s // 100
    if mg <= 9:                       return "Consumer staples"        # agriculture, fishing
    if mg in (10,12,14):              return "Materials"               # metal, coal, nonmetallic mining
    if mg == 13:                      return "Energy"
    if 15 <= mg <= 17:                return "Industrials"             # construction
    if mg in (20,21):                 return "Consumer staples"
    if mg in (22,23,25,27,30,31,39):  return "Consumer discretionary"  # apparel, furniture, print, plastics, leather
    if mg in (24,26,28,29,32,33):     return "Materials"               # lumber, paper, chemicals, stone, primary metal
    if 34 <= mg <= 38:                return "Industrials"             # fabricated metal through instruments
    if 50 <= mg <= 51:                return "Industrials"             # wholesale
    if 52 <= mg <= 59:                return "Consumer discretionary"  # retail
    if mg in (70,72,75,76,79,80,82,83,84,86,87,88,89): return "Business services"
    return "Other"

u = pd.read_csv(sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1], dtype={"cik": str})
u["sic"] = u.cik.map(sic)
u["sector"] = u.sic.dropna().astype(int).map(sector)
u = u.dropna(subset=["sector"])[["cik","ticker","sector","sic"]]
u.to_parquet(ROOT/"data/sec/sectors.parquet")

print(f"mapped {len(u):,} tickers / {u.cik.nunique():,} filers")
print(u.sector.value_counts().to_string())
print(f"\nunmapped tickers (no SIC in filings): {10395 - len(u):,}")
