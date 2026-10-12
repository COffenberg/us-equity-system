#!/usr/bin/env python3
"""Build an accession -> (form_type, filed_date) map from EDGAR quarterly master.idx files.

Frames responses carry 'accn' but NOT 'form', 'fy' or 'fp'. This recovers the real form type
and — more importantly — the ACTUAL filing date, so downstream code can enforce no-lookahead
against the true date instead of an assumed 45-day lag.
"""
import io, os, pathlib, time
import pandas as pd
import requests

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
OUT = os.path.join(ROOT, "data/sec/accn_index.parquet")
CACHE = os.path.join(ROOT, "data/sec/full_index")
HEADERS = {"User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"],
           "Accept-Encoding": "gzip, deflate"}

# forms that can carry XBRL financial-statement facts (amendments picked up via startswith)
KEEP_PREFIX = ("10-K", "10-Q", "20-F", "40-F", "6-K", "8-K", "S-1", "S-4", "S-11",
               "424B", "POS AM", "10-KT", "10-QT", "N-CSR")

_today = pd.Timestamp.today()
PERIODS = [(y, q) for y in range(2020, _today.year + 1) for q in (1, 2, 3, 4)]
# a quarter's master.idx only exists once that quarter has begun
PERIODS = [p for p in PERIODS
           if not (p[0] == _today.year and p[1] > (_today.month - 1) // 3 + 1)]


def get(url, tries=2):
    for a in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
        except requests.RequestException:
            time.sleep(3.0); continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        time.sleep(5.0)
    return None


def main():
    os.makedirs(CACHE, exist_ok=True)
    frames = []
    for (y, q) in PERIODS:
        cp = os.path.join(CACHE, "master_%dQ%d.parquet" % (y, q))
        if os.path.exists(cp):
            frames.append(pd.read_parquet(cp)); print("cached %dQ%d" % (y, q), flush=True); continue
        url = "https://www.sec.gov/Archives/edgar/full-index/%d/QTR%d/master.idx" % (y, q)
        r = get(url)
        time.sleep(0.12)
        if r is None:
            print("MISS %dQ%d" % (y, q), flush=True); continue
        txt = r.content.decode("latin-1")
        body = txt.split("CIK|Company Name|Form Type|Date Filed|Filename", 1)
        if len(body) < 2:
            print("BADFMT %dQ%d" % (y, q), flush=True); continue
        rows = []
        for line in body[1].splitlines():
            if line.count("|") != 4:
                continue
            cik, _name, form, filed, fn = line.split("|")
            if not form.startswith(KEEP_PREFIX):
                continue
            accn = fn.rsplit("/", 1)[-1].replace(".txt", "")
            rows.append((accn, form, filed))
        df = pd.DataFrame(rows, columns=["accn", "form", "filed_date"]).drop_duplicates("accn")
        df.to_parquet(cp, index=False)
        frames.append(df)
        print("%dQ%d rows=%d" % (y, q, len(df)), flush=True)
    idx = pd.concat(frames, ignore_index=True).drop_duplicates("accn")
    idx.to_parquet(OUT, index=False)
    print("accn index rows=%d -> %s" % (len(idx), OUT), flush=True)


if __name__ == "__main__":
    main()
