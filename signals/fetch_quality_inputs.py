#!/usr/bin/env python3
"""Fetch the extra inputs the distress / accounting-quality signals need.

Two things the existing panel does not have:

1. Extra XBRL *annual duration* frames -- cash-flow subtotals (fincf/ivncf), the
   tax-footnote splits (txfed/txfo/txdi) and better `ib` proxies. Cached into the
   same `data/sec/frames/` store the recorder uses, so `build_fundamentals.py`
   would pick them up too.
2. SIC codes for every filer. Ohlson/Dichev's OScore is an industrials model and
   the OSAP definition carries an explicit SIC exclusion. Rather than 10k
   `submissions/CIK*.json` calls (multi-GB), this pulls DERA's quarterly
   Financial Statement Data Sets and keeps only `sub.txt`, which carries
   (adsh, cik, name, sic) for every filer in the quarter -- one request per
   quarter instead of one per company.

Idempotent: anything already on disk is skipped.
"""
import pathlib
import gzip, io, json, os, sys, time, zipfile
import requests

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
CACHE = os.path.join(ROOT, "data/sec/frames")
SICDIR = os.path.join(ROOT, "data/sec/dera")
HEADERS = {
    "User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"],
    "Accept-Encoding": "gzip, deflate",
}

# Annual (12-month duration) concepts. Balance-sheet items and quarterly tax are
# already cached by recorder/fetch_frames.py.
CONCEPTS = [
    # --- PctTotAcc: cash-flow subtotals + equity distribution lines -------------
    "NetCashProvidedByUsedInFinancingActivities",                       # fincf
    "NetCashProvidedByUsedInInvestingActivities",                       # ivncf
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",   # oancf fallback
    "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",   # fincf fallback
    "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",   # ivncf fallback
    "PaymentsOfDividends",                                              # dvt (total)
    "PaymentsOfDividendsPreferredStockAndPreferenceStock",              # dvt preferred leg
    "PaymentsForRepurchaseOfEquity",                                    # prstkcc (broad)
    "PaymentsForRepurchaseOfPreferredStockAndPreferenceStock",          # prstkcc preferred leg
    "ProceedsFromIssuanceOrSaleOfEquity",                               # sstk (broad)
    # --- Tax: the tax-footnote split ------------------------------------------
    "CurrentFederalTaxExpenseBenefit",                                  # txfed
    "CurrentForeignTaxExpenseBenefit",                                  # txfo
    "CurrentIncomeTaxExpenseBenefit",                                   # txc = txt - txdi
    "DeferredIncomeTaxExpenseBenefit",                                  # txdi
    # --- OScore: better `ib` (income before extraordinary items) proxies -------
    "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
    "ProfitLoss",
]
YEARS = list(range(2022, 2027))          # CY2022 .. CY2026
DERA_QUARTERS = ["2025q3", "2025q4", "2026q1", "2026q2"]


def get(sess, url, timeout=120):
    """One retry on 5xx / connection error, hard backoff on 429."""
    for attempt in (0, 1, 2):
        try:
            r = sess.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(3.0)
                continue
            return ("ERR:%s" % type(e).__name__, None)
        if r.status_code == 200:
            return (200, r)
        if r.status_code == 404:
            return (404, None)
        if r.status_code == 429:
            time.sleep(15.0)
            continue
        if 500 <= r.status_code < 600 and attempt < 2:
            time.sleep(3.0)
            continue
        return (r.status_code, None)
    return ("ERR", None)


def fetch_frames(sess):
    os.makedirs(CACHE, exist_ok=True)
    jobs = [(c, y) for c in CONCEPTS for y in YEARS]
    n_ok = n_cached = n_404 = n_err = 0
    for i, (concept, year) in enumerate(jobs, 1):
        fid = "CY%d" % year
        path = os.path.join(CACHE, "us-gaap_%s_USD_%s.json.gz" % (concept, fid))
        if os.path.exists(path) and os.path.getsize(path) > 0:
            n_cached += 1
            continue
        url = "https://data.sec.gov/api/xbrl/frames/us-gaap/%s/USD/%s.json" % (concept, fid)
        status, r = get(sess, url)
        time.sleep(0.15)
        if status == 200:
            js = r.json()
            tmp = path + ".tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(js, fh)
            os.replace(tmp, path)
            n_ok += 1
            print("  ok   %-75s n=%d" % ("%s/%s" % (concept, fid), len(js.get("data", []))), flush=True)
        elif status == 404:
            n_404 += 1
            print("  404  %s/%s" % (concept, fid), flush=True)
        else:
            n_err += 1
            print("  ERR  %s/%s -> %s" % (concept, fid, status), flush=True)
    print("frames: ok=%d cached=%d 404=%d err=%d  (of %d jobs)"
          % (n_ok, n_cached, n_404, n_err, len(jobs)), flush=True)


def fetch_sic(sess):
    """Keep only sub.txt out of each DERA quarterly zip (85 MB in -> ~4 MB kept)."""
    os.makedirs(SICDIR, exist_ok=True)
    for q in DERA_QUARTERS:
        out = os.path.join(SICDIR, "sub_%s.tsv.gz" % q)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print("  cached %s" % os.path.basename(out), flush=True)
            continue
        url = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/%s.zip" % q
        status, r = get(sess, url, timeout=300)
        time.sleep(0.5)
        if status != 200:
            print("  ERR  %s -> %s" % (q, status), flush=True)
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open("sub.txt") as fh:
                raw = fh.read()
        tmp = out + ".tmp"
        with gzip.open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, out)
        print("  ok   %s  (%.1f MB zip -> %.1f MB sub.txt gz)"
              % (q, len(r.content) / 1e6, os.path.getsize(out) / 1e6), flush=True)


if __name__ == "__main__":
    sess = requests.Session()
    print("--- XBRL annual frames ---", flush=True)
    fetch_frames(sess)
    print("--- DERA sub.txt (SIC codes) ---", flush=True)
    fetch_sic(sess)
    print("DONE", flush=True)
