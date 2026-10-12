# fundamentals.parquet — SEC XBRL frames panel

Built 2026-08-18 by `recorder/fetch_frames.py` + `recorder/fetch_filing_index.py` + `recorder/build_fundamentals.py`.

## What it is
One row per (cik/ticker, us-gaap concept, calendar frame). 1,799,910 rows | 8,434 tickers | 6,463 CIKs | 22 concepts.

Source: `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{Concept}/{unit}/CY{YYYY}Q{Q}[I].json`
Frames pulled: CY2020Q1..CY2026Q2 quarterly (26) **plus** CY2020..CY2026 annual (7) for the 13 duration concepts.
663 requests, 0 x 404, 0 empty, 0 errors. Raw responses cached gzipped in `data/sec/frames/*.json.gz` (49 MB).

## Columns
| col | notes |
|---|---|
| `cik` | 10-digit zero-padded |
| `ticker` | joined from `data/universe/tickers_2026-08-18.csv` on cik; **many-to-many** |
| `concept` | XBRL tag (`tag` field of the frames response) |
| `period_end` | `end` — balance-sheet date (instantaneous) or period end (duration) |
| `fy`, `fp` | **CALENDAR frame**, not the filer's fiscal year. `fp` ∈ Q1..Q4, `FY` = 12-month annual frame |
| `form` | recovered from EDGAR `full-index/*/master.idx` (99.6% matched) |
| `val` | USD, except `EntityCommonStockSharesOutstanding` = shares |
| `accn` | accession of the filing the value was taken from |
| `period_start` | *extra* — `start` (duration only); lets you verify the duration length |
| `filed_date` | *extra* — **actual** filing date from master.idx. Use this as the point-in-time gate |
| `instantaneous` | *extra* — True = balance-sheet/point-in-time frame |

## Judgement calls
1. **`fy`/`fp`/`form` are not in frames responses.** The API returns only `accn, cik, entityName, loc, start, end, val`. `fy`/`fp` are derived from the requested calendar frame; `form` + `filed_date` were recovered by downloading 27 EDGAR quarterly `master.idx` files (1.48 M accessions, cached in `data/sec/full_index/`).
2. **Annual frames added.** Quarterly frames only accept ~3-month durations. Cash-flow statements are filed year-to-date, so *every* cash-flow concept only ever lands in Q1 (OCF: 5,837 tickers in Q1 vs ~600 in Q2/Q3), and calendar-Q4 income is thin (10-Ks tag the annual figure, not the Q4 stub). Annual `CY{YYYY}` frames were pulled so TTM/annual flow signals are buildable. Use `fp == 'FY'` rows for any cash-flow or full-year quantity.
3. **Ticker join is many-to-many** (1.20x row multiplication). 1,449 CIKs map to >1 ticker — share classes but also SPAC units/warrants/rights and ETNs (BMO's CIK carries 32 tickers, mostly leveraged ETNs). Those non-common-equity tickers inherit their issuer's fundamentals. Filter downstream.
4. Cached as `.json.gz` rather than plain `.json` (49 MB vs ~460 MB). `gzip.open(p,'rt')` to read.

## Point-in-time warning
Frames returns, for each (concept, period), the value from the **most recently filed** accession containing that fact — usually a later filing's comparative column, and a *restated* value if one exists. Median lag from `period_end` to `filed_date` is 120 days; **45% of rows come from a filing more than a year after the period ended** (e.g. AAPL FY2020 revenue carries `filed_date` 2022-10-28, from the FY2022 10-K). The stated ">=45 day lag" assumption holds for only 45% of rows — gate on `filed_date`, not on an assumed lag.

This is harmless for the intended use (scoring today's cross-section: the most recent quarter has a median 37-day lag and 0% of rows filed >180d late). It makes this panel **unsafe for historical backtesting** — that would need per-company `companyconcept` histories to get original-as-filed values.
