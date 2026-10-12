#!/usr/bin/env python3
"""Fetch the handful of extra SEC XBRL frames the valuation/profitability group needs
that data/sec/fundamentals.parquet does not already carry.

Two gaps in the existing panel:
  1. Revenue.  Only `Revenues` and `RevenueFromContractWithCustomerExcludingAssessedTax`
     were pulled.  Filers legitimately use several other top-line tags.
  2. Shares outstanding for multi-class issuers.  `dei:EntityCommonStockSharesOutstanding`
     is tagged with a class axis by Alphabet, Berkshire, Fox, News Corp, ... and the
     frames API only returns *undimensioned* facts, so those issuers are absent entirely.
     `WeightedAverageNumberOfDilutedSharesOutstanding` is normally undimensioned and
     covers the whole company, so it works as a fallback total-share-count.

Responses are cached gzipped next to the existing ones in data/sec/frames/.
"""
import gzip, json, os, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "sec", "frames")
UA = {"User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"]}
SLEEP = 0.18  # < 10 req/s, deliberately conservative

# (taxonomy, concept, unit, frame)
JOBS = []
for c in ["RevenueFromContractWithCustomerIncludingAssessedTax",
          "SalesRevenueNet",
          "RevenuesNetOfInterestExpense",
          "InterestAndDividendIncomeOperating"]:
    for fr in ["CY2024", "CY2025"]:
        JOBS.append(("us-gaap", c, "USD", fr))
for fr in ["CY2025", "CY2026Q1", "CY2026Q2"]:
    JOBS.append(("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", fr))
    JOBS.append(("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares", fr))


def path_for(tax, concept, unit, frame):
    return os.path.join(CACHE, f"{tax}_{concept}_{unit}_{frame}.json.gz")


def main():
    os.makedirs(CACHE, exist_ok=True)
    for tax, concept, unit, frame in JOBS:
        p = path_for(tax, concept, unit, frame)
        if os.path.exists(p):
            print(f"cached   {concept} {frame}")
            continue
        url = f"https://data.sec.gov/api/xbrl/frames/{tax}/{concept}/{unit}/{frame}.json"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}  {concept} {frame}  -> skipped")
            time.sleep(SLEEP)
            continue
        except Exception as e:
            print(f"ERROR    {concept} {frame}: {e}")
            time.sleep(SLEEP)
            continue
        n = len(payload.get("data", []))
        with gzip.open(p, "wt") as fh:
            json.dump(payload, fh)
        print(f"ok       {concept} {frame}  n={n}")
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
