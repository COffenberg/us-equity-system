#!/usr/bin/env python3
"""Assemble cached SEC frames responses into data/sec/fundamentals.parquet."""
import glob, gzip, json, os, pathlib
import pandas as pd

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
CACHE = os.path.join(ROOT, "data/sec/frames")
UNIV = sorted(glob.glob(os.path.join(ROOT, "data/universe/tickers_*.csv")))[-1]
ACCN = os.path.join(ROOT, "data/sec/accn_index.parquet")
OUT = os.path.join(ROOT, "data/sec/fundamentals.parquet")

recs = []
# the cache holds .json.gz (recorder/fetch_frames.py) and .json (research/bootstrap_all.py)
files = sorted(glob.glob(os.path.join(CACHE, "*.json.gz"))
               + glob.glob(os.path.join(CACHE, "*.json")))
if not files:
    raise SystemExit("no frames cached in %s -- run recorder/fetch_frames.py first" % CACHE)
for fp in files:
    gz = fp.endswith(".gz")
    base = os.path.basename(fp)[:-len(".json.gz" if gz else ".json")]
    fid = base.rsplit("_", 1)[1]                      # e.g. CY2023Q1I / CY2023Q1
    opener = gzip.open if gz else open
    with opener(fp, "rt", encoding="utf-8") as fh:
        js = json.load(fh)
    concept = js.get("tag")
    inst = fid.endswith("I")
    core = fid[2:-1] if inst else fid[2:]             # 2023Q1  (or 2023 for annual frames)
    fy = int(core[:4])
    fp_ = core[4:] if len(core) > 4 else "FY"         # Q1..Q4, or FY for 12-month annual frames
    for d in js.get("data", []):
        recs.append((d["cik"], concept, d.get("end"), d.get("start"), fy, fp_,
                     d.get("val"), d.get("accn"), inst))

df = pd.DataFrame(recs, columns=["cik_int", "concept", "period_end", "period_start",
                                 "fy", "fp", "val", "accn", "instantaneous"])
print("raw frame rows: %d  from %d files" % (len(df), len(files)))
print("unique concepts: %d  unique cik: %d" % (df.concept.nunique(), df.cik_int.nunique()))

dup = df.duplicated(["cik_int", "concept", "fy", "fp", "instantaneous"]).sum()
print("duplicate (cik,concept,frame) rows: %d" % dup)

df["cik"] = df.cik_int.astype(int).astype(str).str.zfill(10)

# ---- form type + REAL filing date from EDGAR quarterly index -------------------------
if os.path.exists(ACCN):
    ai = pd.read_parquet(ACCN)
    df = df.merge(ai, on="accn", how="left")
    print("form matched: %d/%d (%.1f%%)" % (df.form.notna().sum(), len(df),
                                            100 * df.form.notna().mean()))
else:
    raise SystemExit(
        "missing %s -- form and filed_date would both be null, which silently empties the\n"
        "domestic-filer filter in signals/build_all.py and removes the no-lookahead guard.\n"
        "Run recorder/fetch_filing_index.py first." % ACCN)

# ---- join tickers on zero-padded cik (many-to-many: share classes / units / warrants) --
u = pd.read_csv(UNIV, dtype={"cik": str})[["cik", "ticker"]]
n_before = len(df)
n_cik_before = df.cik.nunique()
out = df.merge(u, on="cik", how="inner")
print("rows after ticker join: %d (from %d)  cik matched: %d/%d"
      % (len(out), n_before, out.cik.nunique(), n_cik_before))

out["period_end"] = pd.to_datetime(out.period_end, errors="coerce")
out["period_start"] = pd.to_datetime(out.period_start, errors="coerce")
out["filed_date"] = pd.to_datetime(out.filed_date, errors="coerce")
out["val"] = pd.to_numeric(out.val, errors="coerce")

cols = ["cik", "ticker", "concept", "period_end", "fy", "fp", "form", "val", "accn",
        "period_start", "filed_date", "instantaneous"]
out = out[cols].sort_values(["ticker", "concept", "period_end"]).reset_index(drop=True)
out.to_parquet(OUT, index=False)
print("WROTE %s  rows=%d  tickers=%d  concepts=%d  size=%.1fMB"
      % (OUT, len(out), out.ticker.nunique(), out.concept.nunique(),
         os.path.getsize(OUT) / 1e6))
