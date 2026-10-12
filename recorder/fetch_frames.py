#!/usr/bin/env python3
"""Pull SEC XBRL 'frames' responses for the fundamentals panel and cache them on disk.

One request per (concept, period). Responses cached gzipped so this never re-runs.
Polite: 0.12s between requests, one retry on 5xx / connection error, backoff on 429.
"""
import pathlib
import gzip, json, os, sys, time
import requests

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
CACHE = os.path.join(ROOT, "data/sec/frames")
LOGF = os.path.join(ROOT, "data/sec/fetch_log%s.json" % ("_annual" if os.environ.get("ANNUAL") else ""))
HEADERS = {
    "User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"],
    "Accept-Encoding": "gzip, deflate",
}

# (taxonomy, concept, unit, kind)  kind: "I" = instantaneous (point-in-time), "D" = duration (flow)
CONCEPTS = [
    # ---- Income statement (duration) ----
    ("us-gaap", "Revenues", "USD", "D"),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", "D"),
    ("us-gaap", "NetIncomeLoss", "USD", "D"),
    ("us-gaap", "OperatingIncomeLoss", "USD", "D"),
    ("us-gaap", "GrossProfit", "USD", "D"),
    ("us-gaap", "IncomeTaxExpenseBenefit", "USD", "D"),
    # ---- Balance sheet (instantaneous) ----
    ("us-gaap", "Assets", "USD", "I"),
    ("us-gaap", "Liabilities", "USD", "I"),
    ("us-gaap", "StockholdersEquity", "USD", "I"),
    ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD", "I"),
    ("us-gaap", "LiabilitiesCurrent", "USD", "I"),
    ("us-gaap", "AssetsCurrent", "USD", "I"),
    ("us-gaap", "LongTermDebtNoncurrent", "USD", "I"),
    ("us-gaap", "RetainedEarningsAccumulatedDeficit", "USD", "I"),
    # ---- Cash flow (duration) ----
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD", "D"),
    ("us-gaap", "PaymentsOfDividendsCommonStock", "USD", "D"),
    ("us-gaap", "PaymentsForRepurchaseOfCommonStock", "USD", "D"),
    ("us-gaap", "ProceedsFromIssuanceOfCommonStock", "USD", "D"),
    ("us-gaap", "ProceedsFromIssuanceOfLongTermDebt", "USD", "D"),
    ("us-gaap", "RepaymentsOfLongTermDebt", "USD", "D"),
    ("us-gaap", "DepreciationDepletionAndAmortization", "USD", "D"),
    # ---- Shares (instantaneous, unit = shares) ----
    ("dei", "EntityCommonStockSharesOutstanding", "shares", "I"),
]

PERIODS = [(y, q) for y in range(2020, 2027) for q in (1, 2, 3, 4)]
PERIODS = [p for p in PERIODS if not (p[0] == 2026 and p[1] > 2)]  # CY2020Q1 .. CY2026Q2
if os.environ.get("ANNUAL"):
    # Annual (12-month) duration frames. Required because quarterly frames only accept ~3-month
    # durations, and cash-flow items are filed year-to-date -> they only ever land in Q1.
    PERIODS = [(y, 0) for y in range(2020, 2027)]
    CONCEPTS = [c for c in CONCEPTS if c[3] == "D"]


def frame_id(y, q, kind):
    if q == 0:
        return "CY%d" % y            # annual duration frame
    return "CY%dQ%d%s" % (y, q, "I" if kind == "I" else "")


def cache_path(tax, concept, unit, fid):
    return os.path.join(CACHE, "%s_%s_%s_%s.json.gz" % (tax, concept, unit, fid))


def fetch(sess, url):
    """Return (status_code, parsed_json_or_None). One retry on 5xx/conn error, backoff on 429."""
    for attempt in (0, 1):
        try:
            r = sess.get(url, headers=HEADERS, timeout=90)
        except requests.RequestException as e:
            if attempt == 0:
                time.sleep(3.0)
                continue
            return ("ERR:%s" % type(e).__name__, None)
        if r.status_code == 200:
            try:
                return (200, r.json())
            except ValueError:
                if attempt == 0:
                    time.sleep(3.0)
                    continue
                return ("BADJSON", None)
        if r.status_code == 404:
            return (404, None)
        if r.status_code == 429:      # rate limited: back off hard, then retry
            time.sleep(15.0)
            continue
        if 500 <= r.status_code < 600 and attempt == 0:
            time.sleep(3.0)
            continue
        return (r.status_code, None)
    return (r.status_code if 'r' in dir() else "ERR", None)


def main():
    os.makedirs(CACHE, exist_ok=True)
    sess = requests.Session()
    jobs = [(t, c, u, k, y, q) for (t, c, u, k) in CONCEPTS for (y, q) in PERIODS]
    log = {"ok": [], "missing_404": [], "empty": [], "other_error": [], "cached": []}
    t0 = time.time()
    for i, (tax, concept, unit, kind, y, q) in enumerate(jobs, 1):
        fid = frame_id(y, q, kind)
        key = "%s/%s/%s" % (tax, concept, fid)
        path = cache_path(tax, concept, unit, fid)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            log["cached"].append(key)
            continue
        url = "https://data.sec.gov/api/xbrl/frames/%s/%s/%s/%s.json" % (tax, concept, unit, fid)
        status, js = fetch(sess, url)
        time.sleep(0.12)
        if status == 200 and js is not None:
            n = len(js.get("data", []))
            if n == 0:
                log["empty"].append(key)
            else:
                log["ok"].append([key, n])
            tmp = path + ".tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(js, fh)
            os.replace(tmp, path)
        elif status == 404:
            log["missing_404"].append(key)
        else:
            log["other_error"].append([key, str(status)])
        if i % 25 == 0 or i == len(jobs):
            print("[%4d/%d] %.0fs  ok=%d 404=%d empty=%d err=%d  last=%s(%s)"
                  % (i, len(jobs), time.time() - t0, len(log["ok"]), len(log["missing_404"]),
                     len(log["empty"]), len(log["other_error"]), concept, fid), flush=True)
            with open(LOGF, "w") as fh:
                json.dump(log, fh, indent=1)
    with open(LOGF, "w") as fh:
        json.dump(log, fh, indent=1)
    print("DONE in %.0fs  ok=%d cached=%d 404=%d empty=%d err=%d"
          % (time.time() - t0, len(log["ok"]), len(log["cached"]), len(log["missing_404"]),
             len(log["empty"]), len(log["other_error"])), flush=True)


if __name__ == "__main__":
    main()
