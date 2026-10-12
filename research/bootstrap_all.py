"""Rebuild every derived artifact the pipeline needs, from a bare checkout.

GitHub runners are stateless. Three runs failed one file at a time — price panel,
then fundamentals — because derived data was gitignored as 're-downloadable'
without anything that actually re-derives it. This closes that for good: each
artifact is rebuilt only if absent, in dependency order.

An earlier version of this file claimed fundamentals.parquet had no build script.
That was wrong -- recorder/fetch_frames.py, recorder/fetch_filing_index.py and
recorder/build_fundamentals.py build it, and they carry the no-lookahead guarantee
the research rests on. This file now calls them instead of reinventing them badly:
its own first attempt left `form` null and guessed filed_date at period_end + 45
days, which emptied the domestic-filer filter and reintroduced lookahead.
"""
import os
import json, time, urllib.request, urllib.error, subprocess, sys, zipfile, glob
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"]}

def have(p):
    f = ROOT/p
    return f.exists() and f.stat().st_size > 1000

def run(script):
    print(f"  -> {script}")
    r = subprocess.run([sys.executable, str(ROOT/script)], capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
    for l in tail: print(f"     {l}")
    return r.returncode == 0

# ---------------------------------------------------------------- fundamentals
CONCEPTS_USD = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss",
    "OperatingIncomeLoss", "GrossProfit", "IncomeTaxExpenseBenefit", "Assets",
    "Liabilities", "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
    "LiabilitiesCurrent", "AssetsCurrent", "LongTermDebtNoncurrent",
    "RetainedEarningsAccumulatedDeficit", "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsOfDividendsCommonStock", "PaymentsForRepurchaseOfCommonStock",
    "ProceedsFromIssuanceOfCommonStock", "ProceedsFromIssuanceOfLongTermDebt",
    "RepaymentsOfLongTermDebt", "DepreciationDepletionAndAmortization",
]
INSTANT = {"Assets","Liabilities","StockholdersEquity","CashAndCashEquivalentsAtCarryingValue",
           "LiabilitiesCurrent","AssetsCurrent","LongTermDebtNoncurrent",
           "RetainedEarningsAccumulatedDeficit"}

def fetch_frame(tax, concept, unit, frame):
    cache = ROOT/f"data/sec/frames/{concept}_{frame}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://data.sec.gov/api/xbrl/frames/{tax}/{concept}/{unit}/{frame}.json"
    try:
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404: return None
        raise
    cache.write_text(txt); time.sleep(0.12)
    return json.loads(txt)

def build_fundamentals():
    print("building data/sec/fundamentals.parquet (SEC XBRL frames)")
    uni = pd.read_csv(sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1], dtype={"cik": str})
    cik2t = dict(zip(uni.cik, uni.ticker))
    _yrs = range(2020, pd.Timestamp.today().year + 1)
    periods = [f"CY{y}Q{q}" for y in _yrs for q in (1, 2, 3, 4)]
    # Annual frames too. build_all.py selects on dur.between(60,115) so the live book is
    # quarterly-only and does not need these -- but build_valprof.py (FY revenue, FY
    # operating cash flow), build_payout.py, build_events.py and the PIT research scripts
    # all filter on fp == "FY" or dur near 365. Fetching only quarterly left the Mac with
    # 246,666 annual rows the cloud did not have, so those scripts would quietly compute
    # on nothing here. Flow concepts only: an instant concept has no annual frame.
    annual = [f"CY{y}" for y in _yrs]
    rows, missing = [], 0
    for c in CONCEPTS_USD:
        inst = c in INSTANT
        for p in (periods if inst else periods + annual):
            frame = p + ("I" if inst else "")
            d = fetch_frame("us-gaap", c, "USD", frame)
            if d is None:
                missing += 1; continue
            for r in d.get("data", []):
                rows.append({"cik": str(r["cik"]).zfill(10), "concept": c, "val": r["val"],
                             "period_end": r.get("end"), "period_start": r.get("start"),
                             "fy": r.get("fy"), "fp": r.get("fp"), "form": r.get("form"),
                             "accn": r.get("accn"), "instantaneous": inst})
        print(f"  {c}: {len(rows):,} cumulative")
    for p in [f"CY{y}Q{q}I" for y in range(2020, 2027) for q in (1, 2, 3, 4)]:
        d = fetch_frame("dei", "EntityCommonStockSharesOutstanding", "shares", p)
        if d is None: continue
        for r in d.get("data", []):
            rows.append({"cik": str(r["cik"]).zfill(10), "concept": "EntityCommonStockSharesOutstanding",
                         "val": r["val"], "period_end": r.get("end"), "period_start": None,
                         "fy": r.get("fy"), "fp": r.get("fp"), "form": r.get("form"),
                         "accn": r.get("accn"), "instantaneous": True})
    print(f"  {len(rows):,} frame rows cached, {missing} frames 404'd")

    # The frames API returns only accn/cik/entityName/loc/start/end/val -- no `form` and no
    # filing date. An earlier version of this function invented filed_date as period_end + 45
    # days and left form null. Both were wrong: a null form silently empties the
    # domestic-filer filter in signals/build_all.py (the one that caught Fannie Mae preferred
    # at $36,480bn), and a 45-day guess is lookahead for every filer that reports slower.
    # recorder/fetch_filing_index.py recovers the real form and the real filing date from
    # EDGAR's quarterly master.idx, and recorder/build_fundamentals.py does the join. Use them.
    if not run("recorder/fetch_filing_index.py"):
        raise SystemExit("accession index failed -- without it form and filed_date are null")
    if not run("recorder/build_fundamentals.py"):
        raise SystemExit("fundamentals build failed")

    F = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
    pit = F.filed_date.notna().mean()
    dom = F.form.astype("string").str.startswith(("10-K", "10-Q")).fillna(False).sum()
    fy = int((F.fp == "FY").sum())
    print(f"  wrote {len(F):,} rows, {F.ticker.nunique():,} tickers, "
          f"{pit:.1%} with a real filing date, {dom:,} domestic-filer rows, {fy:,} annual")
    if dom == 0:
        raise SystemExit("no 10-K/10-Q rows -- the accession-index join failed; universe "
                         "resolution downstream would be empty")

# ---------------------------------------------------------------- bulk archives
def fetch_zip(url, dest):
    dest = ROOT/dest
    if dest.exists() and dest.stat().st_size > 1_000_000: return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dest,"wb") as f:
            f.write(r.read())
        return zipfile.is_zipfile(dest)
    except Exception as e:
        print(f"     {dest.name}: {type(e).__name__}"); dest.unlink(missing_ok=True); return False

if __name__ == "__main__":
    # Presence is not enough. A cached copy built before the accession-index join exists but
    # has a null `form` on every row, which empties the domestic-filer filter downstream.
    # Validate and rebuild, or the bad file survives in the Actions cache indefinitely.
    if not have("data/sec/fundamentals.parquet"):
        build_fundamentals()
    else:
        _f = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet", columns=["form","fp"])
        _dom = int(_f.form.astype("string").str.startswith(("10-K","10-Q")).fillna(False).sum())
        _fy = int((_f.fp == "FY").sum())
        if _dom == 0:
            print(f"fundamentals.parquet present but has 0 domestic-filer rows of {len(_f):,} "
                  "-- built without the accession-index join; rebuilding")
            build_fundamentals()
        elif _fy == 0:
            print(f"fundamentals.parquet present with {_dom:,} domestic-filer rows but 0 annual "
                  "(fp=FY) rows -- built before annual frames were fetched; rebuilding")
            build_fundamentals()
        else:
            print(f"fundamentals.parquet present ({_dom:,} domestic-filer, {_fy:,} annual rows)")

    if not have("data/sec/shares_fallback.parquet"): run("research/pull_shares.py")
    else: print("shares_fallback.parquet present")

    if not have("data/sec/sectors.parquet"):
        print("fetching DERA archives for SIC codes")
        for q in ("2026q2","2026q1","2025q4","2025q3"):
            fetch_zip(f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip",
                      f"data/sec/dera/{q}.zip")
        run("research/build_sectors.py")
    else: print("sectors.parquet present")

    if not have("data/sec/f13_holdings.parquet"):
        print("fetching 13F archives")
        for p in ("01mar2026-31may2026","01dec2025-28feb2026","01sep2025-30nov2025"):
            fetch_zip(f"https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{p}_form13f.zip",
                      f"data/sec/f13/{p}.zip")
        run("research/build_13f.py")
    else: print("f13_holdings.parquet present")

    print("\nbootstrap complete")
    for p in ("data/prices/history_3y.parquet","data/sec/fundamentals.parquet",
              "data/sec/shares_fallback.parquet","data/sec/sectors.parquet",
              "data/sec/f13_holdings.parquet"):
        print(f"  {'ok ' if have(p) else 'MISSING'}  {p}")
