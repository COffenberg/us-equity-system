"""Pull fallback share-count concepts. dei/EntityCommonStockSharesOutstanding alone
leaves 36% of liquid names with no market cap; these three cover most of the rest."""
import os
import json, time, urllib.request, urllib.error, pandas as pd, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT/"data/sec/frames"; CACHE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"]}

CONCEPTS = [("us-gaap","CommonStockSharesOutstanding",True),
            ("us-gaap","CommonStockSharesIssued",True),
            ("us-gaap","WeightedAverageNumberOfDilutedSharesOutstanding",False)]
PERIODS = [f"CY{y}Q{q}" for y in range(2023, 2027) for q in (1,2,3,4)]

rows, missing = [], 0
for tax, con, inst in CONCEPTS:
    for per in PERIODS:
        frame = per + ("I" if inst else "")
        f = CACHE/f"{con}_{frame}.json"
        if f.exists():
            txt = f.read_text()
        else:
            url = f"https://data.sec.gov/api/xbrl/frames/{tax}/{con}/shares/{frame}.json"
            try:
                txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
                f.write_text(txt)
            except urllib.error.HTTPError as e:
                if e.code == 404: missing += 1; time.sleep(0.12); continue
                raise
            time.sleep(0.12)
        d = json.loads(txt)
        for r in d.get("data", []):
            rows.append({"cik": str(r["cik"]).zfill(10), "concept": con, "val": r["val"],
                         "period_end": r.get("end"), "fy": r.get("fy"), "fp": r.get("fp"),
                         "form": r.get("form"), "accn": r.get("accn"), "instantaneous": inst})
    print(f"  {con}: {len(rows)} cumulative rows")

df = pd.DataFrame(rows)
u = pd.read_csv(ROOT/"data/universe/tickers_2026-08-18.csv", dtype={"cik": str})
df = df.merge(u[["cik","ticker"]], on="cik", how="inner")
df["period_end"] = pd.to_datetime(df.period_end)
df["filed_date"] = df.period_end + pd.Timedelta(days=45)   # conservative lag; no filed date in frames
df["period_start"] = pd.NaT
df.to_parquet(ROOT/"data/sec/shares_fallback.parquet")
print(f"\nwrote {len(df):,} rows | {df.ticker.nunique()} tickers | {missing} periods 404'd")
print(df.groupby("concept").ticker.nunique().to_string())
