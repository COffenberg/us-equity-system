#!/usr/bin/env python3
"""Fetch the extra SEC XBRL frames the payout / external-financing signals need.

Same cache + politeness contract as recorder/fetch_frames.py: one request per
(concept, frame), gzipped to data/sec/frames/, never re-fetched.
"""
import pathlib
import gzip, json, os, sys, time
import requests

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
CACHE = os.path.join(ROOT, "data/sec/frames")
HEADERS = {"User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"],
           "Accept-Encoding": "gzip, deflate"}

# --- annual duration frames: the cash-flow legs of dvc / prstkc / sstk / dltis / dltr / dv ---
DURATION = [
    # dividends (dv, dvc)
    "PaymentsOfDividends", "PaymentsOfDividendsPreferredStockAndPreferenceStock",
    "PaymentsOfDividendsMinorityInterest",
    # repurchases (prstkc)
    "PaymentsForRepurchaseOfEquity", "PaymentsForRepurchaseOfPreferredStockAndPreferenceStock",
    # equity issuance (sstk)
    "ProceedsFromIssuanceOrSaleOfEquity", "ProceedsFromIssuanceOfPrivatePlacement",
    "ProceedsFromIssuanceOfPreferredStockAndPreferenceStock", "ProceedsFromStockOptionsExercised",
    "ProceedsFromIssuanceOfWarrants",
    # long-term debt issuance (dltis)
    "ProceedsFromNotesPayable", "ProceedsFromLinesOfCredit", "ProceedsFromLongTermLinesOfCredit",
    "ProceedsFromIssuanceOfSeniorLongTermDebt", "ProceedsFromIssuanceOfUnsecuredDebt",
    "ProceedsFromIssuanceOfSecuredDebt", "ProceedsFromConvertibleDebt",
    "ProceedsFromRelatedPartyDebt", "ProceedsFromIssuanceOfDebt",
    # long-term debt repayment (dltr)
    "RepaymentsOfDebt", "RepaymentsOfNotesPayable", "RepaymentsOfLinesOfCredit",
    "RepaymentsOfLongTermLinesOfCredit", "RepaymentsOfSeniorDebt", "RepaymentsOfConvertibleDebt",
    "RepaymentsOfRelatedPartyDebt", "RepaymentsOfSecuredDebt", "RepaymentsOfUnsecuredDebt",
]
DURATION_FRAMES = ["CY2024", "CY2025"]

# --- instantaneous frames: dltt + dlc now and five years ago, plus a shares fallback ---
INSTANT = [
    "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent", "ShortTermBorrowings",
    "OtherShortTermBorrowings", "NotesPayableCurrent", "DebtCurrent",
    "DebtLongtermAndShorttermCombinedAmount", "LongTermNotesPayable", "LinesOfCreditCurrent",
    "ConvertibleDebtNoncurrent", "SecuredDebt", "UnsecuredDebt",
]
INSTANT_SHARES = ["CommonStockSharesOutstanding"]   # unit = shares, fallback for the dei tag
INSTANT_FRAMES = (["CY2020Q4I"] + ["CY2021Q%dI" % q for q in (1, 2, 3, 4)]
                  + ["CY2025Q%dI" % q for q in (1, 2, 3, 4)] + ["CY2026Q1I", "CY2026Q2I"])


def fetch(sess, url):
    for attempt in (0, 1):
        try:
            r = sess.get(url, headers=HEADERS, timeout=90)
        except requests.RequestException:
            if attempt == 0:
                time.sleep(3.0); continue
            return ("ERR", None)
        if r.status_code == 200:
            try:
                return (200, r.json())
            except ValueError:
                if attempt == 0:
                    time.sleep(3.0); continue
                return ("BADJSON", None)
        if r.status_code == 404:
            return (404, None)
        if r.status_code == 429:
            time.sleep(15.0); continue
        if 500 <= r.status_code < 600 and attempt == 0:
            time.sleep(3.0); continue
        return (r.status_code, None)
    return ("ERR", None)


def main():
    os.makedirs(CACHE, exist_ok=True)
    jobs = ([("us-gaap", c, "USD", f) for c in DURATION for f in DURATION_FRAMES]
            + [("us-gaap", c, "USD", f) for c in INSTANT for f in INSTANT_FRAMES]
            + [("us-gaap", c, "shares", f) for c in INSTANT_SHARES for f in INSTANT_FRAMES])
    sess, t0 = requests.Session(), time.time()
    n_ok = n_cached = n_404 = n_err = 0
    for i, (tax, concept, unit, fid) in enumerate(jobs, 1):
        path = os.path.join(CACHE, "%s_%s_%s_%s.json.gz" % (tax, concept, unit, fid))
        if os.path.exists(path) and os.path.getsize(path) > 0:
            n_cached += 1; continue
        status, js = fetch(sess, "https://data.sec.gov/api/xbrl/frames/%s/%s/%s/%s.json"
                                 % (tax, concept, unit, fid))
        time.sleep(0.13)
        if status == 200 and js is not None:
            with gzip.open(path + ".tmp", "wt", encoding="utf-8") as fh:
                json.dump(js, fh)
            os.replace(path + ".tmp", path); n_ok += 1
        elif status == 404:
            n_404 += 1
        else:
            n_err += 1
            print("  ERR %s %s -> %s" % (concept, fid, status), flush=True)
        if i % 40 == 0:
            print("[%4d/%d] %.0fs ok=%d cached=%d 404=%d err=%d"
                  % (i, len(jobs), time.time() - t0, n_ok, n_cached, n_404, n_err), flush=True)
    print("DONE %.0fs  ok=%d cached=%d 404=%d err=%d" % (time.time() - t0, n_ok, n_cached, n_404, n_err))


if __name__ == "__main__":
    main()
