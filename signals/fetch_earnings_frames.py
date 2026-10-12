#!/usr/bin/env python3
"""Pull the extra SEC XBRL frames the earnings-dynamics signals need.

The existing panel (data/sec/fundamentals.parquet) has NetIncomeLoss/Revenues from
CY2020Q1 onward but no per-share concepts. The OSAP definitions for
EarningsSurprise/SUE, RevenueSurprise and EarningsConsistency are all *per share*
(epspxq, revtq/cshprq), so EPS and weighted share counts are required, and the
history has to reach further back (EarningsConsistency needs 6 annual EPS points).

Cache dir is deliberately SEPARATE from data/sec/frames/ so that rebuilding the
shared fundamentals panel is unaffected. Idempotent: existing files are skipped.
"""
import pathlib
import gzip, json, os, sys, time
import requests

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
CACHE = os.path.join(ROOT, "data/sec/frames_earn")
LOGF = os.path.join(ROOT, "data/sec/fetch_log_earnings.json")
HEADERS = {
    "User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"],
    "Accept-Encoding": "gzip, deflate",
}

# (taxonomy, concept, unit, first_year_quarterly, first_year_annual)
# quarterly frames run <first>Q1 .. 2026Q2 ; annual frames run <first> .. 2026
NEW = [
    ("us-gaap", "EarningsPerShareDiluted", "USD-per-shares", 2019, 2018),
    ("us-gaap", "EarningsPerShareBasic", "USD-per-shares", 2019, 2018),
    ("us-gaap", "IncomeLossFromContinuingOperationsPerDilutedShare", "USD-per-shares", 2019, 2018),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", 2019, 2018),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares", 2019, 2018),
    ("us-gaap", "ProfitLoss", "USD", 2019, 2018),
    ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax", "USD", 2019, 2018),
    # back-years only for concepts already cached from 2020 in data/sec/frames/
    ("us-gaap", "NetIncomeLoss", "USD", 2019, 2018),
    ("us-gaap", "Revenues", "USD", 2019, 2018),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", 2019, 2018),
]
LAST_Y, LAST_Q = 2026, 2
EXISTING = os.path.join(ROOT, "data/sec/frames")   # skip anything already cached there


def jobs():
    out = []
    for tax, c, u, fy_q, fy_a in NEW:
        for y in range(fy_q, LAST_Y + 1):
            for q in (1, 2, 3, 4):
                if y == LAST_Y and q > LAST_Q:
                    continue
                out.append((tax, c, u, "CY%dQ%d" % (y, q)))
        for y in range(fy_a, LAST_Y + 1):
            out.append((tax, c, u, "CY%d" % y))
    return out


def fetch(sess, url):
    for attempt in (0, 1, 2):
        try:
            r = sess.get(url, headers=HEADERS, timeout=90)
        except requests.RequestException as e:
            time.sleep(3.0)
            if attempt == 2:
                return ("ERR:%s" % type(e).__name__, None)
            continue
        if r.status_code == 200:
            try:
                return (200, r.json())
            except ValueError:
                time.sleep(3.0)
                continue
        if r.status_code == 404:
            return (404, None)
        if r.status_code == 429:
            time.sleep(15.0)
            continue
        if 500 <= r.status_code < 600:
            time.sleep(3.0)
            continue
        return (r.status_code, None)
    return ("ERR", None)


def main():
    os.makedirs(CACHE, exist_ok=True)
    sess = requests.Session()
    J = jobs()
    log = {"ok": [], "missing_404": [], "empty": [], "other_error": [], "cached": []}
    t0 = time.time()
    for i, (tax, c, u, fid) in enumerate(J, 1):
        name = "%s_%s_%s_%s.json.gz" % (tax, c, u, fid)
        path = os.path.join(CACHE, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            log["cached"].append(fid + "/" + c); continue
        if os.path.exists(os.path.join(EXISTING, name)):
            log["cached"].append("shared:" + fid + "/" + c); continue
        url = "https://data.sec.gov/api/xbrl/frames/%s/%s/%s/%s.json" % (tax, c, u, fid)
        status, js = fetch(sess, url)
        time.sleep(0.15)
        key = "%s/%s" % (c, fid)
        if status == 200 and js is not None:
            n = len(js.get("data", []))
            (log["empty"] if n == 0 else log["ok"]).append([key, n])
            tmp = path + ".tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(js, fh)
            os.replace(tmp, path)
        elif status == 404:
            log["missing_404"].append(key)
        else:
            log["other_error"].append([key, str(status)])
        if i % 25 == 0 or i == len(J):
            print("[%4d/%d] %.0fs ok=%d cached=%d 404=%d empty=%d err=%d last=%s"
                  % (i, len(J), time.time() - t0, len(log["ok"]), len(log["cached"]),
                     len(log["missing_404"]), len(log["empty"]), len(log["other_error"]), key),
                  flush=True)
    with open(LOGF, "w") as fh:
        json.dump(log, fh, indent=1)
    print("DONE %.0fs ok=%d cached=%d 404=%d empty=%d err=%d"
          % (time.time() - t0, len(log["ok"]), len(log["cached"]), len(log["missing_404"]),
             len(log["empty"]), len(log["other_error"])), flush=True)


if __name__ == "__main__":
    main()
