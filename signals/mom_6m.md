# mom_6m

**6-month price momentum skipping the most recent month, per the OSAP definition (return between t-6 and t-1).**

- OSAP signal: `Mom6m` — Momentum (6 month), Jegadeesh and Titman 1993, Detailed Definition: "Stock return between months t-6 and t-1". OSAP's note: "OP doesn't skip the current month in the signal ... We follow McLean and Pontiff, Green-Hand-Zhang, and others in skipping the current month instead."
- Sign in OSAP: **+1** — high value = high expected return, use `rank_pct` directly.
- Coverage: **3,537** tickers of 3,616 in the investable universe (97.8%)
- Built 2026-08-18 from the price panel anchored on session 2026-08-17.

## Exact formula

```
mom_6m = P[t-21] / P[t-126] - 1

computed from derived.parquet as the exact algebraic identity
  (1 + ret_6m) / (1 + ret_1m) - 1
  = (P[t]/P[t-126]) / (P[t]/P[t-21]) - 1
  = P[t-21]/P[t-126] - 1
```

**Inputs:** `ret_6m` and `ret_1m` from data/prices/derived.parquet.
No XBRL / SEC concepts are used by this signal group — it is price and volume only.

## Distribution (raw, pre-winsorisation)

```
count    3537.000000
mean        0.109056
std         0.468850
min        -0.996553
1%         -0.593044
5%         -0.368789
25%        -0.107955
50%         0.053592
75%         0.225829
95%         0.824471
99%         1.485772
max        13.386704
```

Winsorisation: values clipped to the 1st/99th percentile of the investable universe
(`[-0.5930, 1.4858]`) **before** ranking, per the build rules. `value` in the parquet is
the winsorised number; `rank_pct` is the ascending percentile rank of it. Consequence worth
knowing: the clipped 1% tails become ties, so the ~35 most extreme
names on each end share a rank. Ordering is untouched everywhere else.

## Top 10 (highest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| AGL | 13.3867 | 1.4858 | 0.9951 | $96.86 | $31.6M | agilon health, inc. |
| RXT | 8.8810 | 1.4858 | 0.9951 | $3.95 | $71.6M | Rackspace Technology, Inc. |
| BAND | 4.5409 | 1.4858 | 0.9951 | $52.79 | $61.4M | Bandwidth Inc. |
| QTTB | 2.8590 | 1.4858 | 0.9951 | $15.95 | $6.9M | Q32 Bio Inc. |
| EVC | 2.7558 | 1.4858 | 0.9951 | $8.89 | $19.3M | ENTRAVISION COMMUNICATIONS CORP |
| MXL | 2.6784 | 1.4858 | 0.9951 | $84.76 | $294.6M | MAXLINEAR, INC |
| SLS | 2.6741 | 1.4858 | 0.9951 | $13.00 | $70.7M | SELLAS Life Sciences Group, Inc. |
| DELL | 2.3838 | 1.4858 | 0.9951 | $479.81 | $2,984.3M | Dell Technologies Inc. |
| CUE | 2.3833 | 1.4858 | 0.9951 | $27.64 | $3.4M | Cue Biopharma, Inc. |
| ABSI | 2.2529 | 1.4858 | 0.9951 | $9.24 | $30.4M | Absci Corp |

## Bottom 10 (lowest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| YYGH | -0.9966 | -0.5930 | 0.0052 | $1.44 | $1.3M | YY Group Holding Ltd. |
| ZCMD | -0.9898 | -0.5930 | 0.0052 | $1.06 | $1.3M | Zhongchao Inc. |
| VIVK | -0.9690 | -0.5930 | 0.0052 | $1.22 | $1.6M | Vivakor, Inc. |
| MF | -0.8912 | -0.5930 | 0.0052 | $11.60 | $1.5M | MindForge Inc. |
| DBGI | -0.8672 | -0.5930 | 0.0052 | $13.77 | $1.3M | Digital Brands Group, Inc. |
| RGC | -0.8392 | -0.5930 | 0.0052 | $5.35 | $1.8M | Regencell Bioscience Holdings Ltd |
| BIYA | -0.8364 | -0.5930 | 0.0052 | $2.63 | $1.4M | Baiya International Group Inc. |
| CRMT | -0.8224 | -0.5930 | 0.0052 | $3.14 | $1.2M | AMERICAS CARMART INC |
| FFAI | -0.7803 | -0.5930 | 0.0052 | $3.72 | $4.0M | FARADAY FUTURE INTELLIGENT ELECTRIC IN |
| VRRM | -0.7757 | -0.5930 | 0.0052 | $4.60 | $26.6M | VERRA MOBILITY Corp |

## Judgement calls

1. **This signal SKIPS the most recent month; the raw `ret_6m` column in derived.parquet
   does not.** The catalog is explicit that Mom6m is the t-6..t-1 return, and OSAP's own
   note says they deliberately skip the current month. Using the unskipped `ret_6m` would
   blend momentum with one month of short-term reversal (OSAP `STreversal`, sign -1) and
   partially cancel the signal. Spearman(mom_6m, raw ret_6m) is printed in the build log.
2. The skip is recovered by *algebra on two existing columns*, not by re-deriving prices,
   so it inherits exactly the same trading-calendar alignment and repairs as the rest of
   the panel. The identity is exact, not an approximation.
3. Requires both `ret_6m` and `ret_1m` present, which is a slightly stricter data
   requirement than `ret_6m` alone.

## Universe

Investable universe = **3,616 names**, one line per issuer CIK: SEC-roster common
equity with median 60d dollar volume >= $1,000,000, last close >= $1.00,
traded on >= 30 of the last 60 sessions, and adj_close not quarantined or
unadjustable upstream. Closed-end funds, commodity/crypto trusts, ETFs, ETNs, royalty
trusts, preferred series, warrants, rights and units are excluded
(`_excluded_funds.csv`), as are secondary share classes and baby bonds
(`_excluded_secondary_lines.csv`). Ranks are computed **within** this universe, so
`rank_pct` is comparable only to other signals ranked on the same universe.
