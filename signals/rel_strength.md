# rel_strength

**Relative strength: the stock's 12-month total return minus the equal-weighted median 12-month return of the investable universe. Positive = outperformed the typical US stock over the last year.**

- OSAP signal: `(none - not an OSAP signal)` — This is a cross-sectional recentring of a 12-month return, not a catalogued predictor. The nearest catalog entries are `Mom12m` (t-12..t-1) and `IntMom` (t-12..t-6). No Detailed Definition exists to follow, so the formula is exactly as specified in the task.
- Sign in OSAP: **+1** — high value = high expected return, use `rank_pct` directly.
- Coverage: **3,470** tickers of 3,616 in the investable universe (96.0%)
- Built 2026-08-18 from the price panel anchored on session 2026-08-17.

## Exact formula

```
rel_strength = ret_12m - median(ret_12m over the investable universe)
  ret_12m = P[t]/P[t-252] - 1  (does NOT skip the recent month)
  universe median on 2026-08-18 = +0.180622  (+18.06%)
```

**Inputs:** `ret_12m` from data/prices/derived.parquet, recentred on the equal-weighted universe median.
No XBRL / SEC concepts are used by this signal group — it is price and volume only.

## Distribution (raw, pre-winsorisation)

```
count    3470.000000
mean        0.212300
std         1.336164
min        -1.180255
1%         -0.969768
5%         -0.674701
25%        -0.274914
50%         0.000000
75%         0.355114
95%         1.703852
99%         4.266077
max        43.249933
```

Winsorisation: values clipped to the 1st/99th percentile of the investable universe
(`[-0.9698, 4.2661]`) **before** ranking, per the build rules. `value` in the parquet is
the winsorised number; `rank_pct` is the ascending percentile rank of it. Consequence worth
knowing: the clipped 1% tails become ties, so the ~35 most extreme
names on each end share a rank. Ordering is untouched everywhere else.

## Top 10 (highest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| AXTI | 43.2499 | 4.2661 | 0.9951 | $95.97 | $775.4M | AXT INC |
| SNDK | 37.0981 | 4.2661 | 0.9951 | $1,786.85 | $21,483.3M | Sandisk Corp |
| NUAI | 11.4291 | 4.2661 | 0.9951 | $5.46 | $34.1M | New ERA Energy & Digital, Inc. |
| ERAS | 10.4627 | 4.2661 | 0.9951 | $18.28 | $68.4M | Erasca, Inc. |
| ANRO | 9.5886 | 4.2661 | 0.9951 | $33.60 | $9.4M | Alto Neuroscience, Inc. |
| CLYM | 8.9589 | 4.2661 | 0.9951 | $17.44 | $13.7M | Climb Bio, Inc. |
| STTK | 7.4124 | 4.2661 | 0.9951 | $7.39 | $6.1M | Shattuck Labs, Inc. |
| QTTB | 7.3034 | 4.2661 | 0.9951 | $15.95 | $6.9M | Q32 Bio Inc. |
| LITE | 7.2725 | 4.2661 | 0.9951 | $968.90 | $3,963.0M | Lumentum Holdings Inc. |
| SLS | 7.2065 | 4.2661 | 0.9951 | $13.00 | $70.7M | SELLAS Life Sciences Group, Inc. |

## Bottom 10 (lowest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| VIVK | -1.1803 | -0.9698 | 0.0052 | $1.22 | $1.6M | Vivakor, Inc. |
| YYGH | -1.1802 | -0.9698 | 0.0052 | $1.44 | $1.3M | YY Group Holding Ltd. |
| ZCMD | -1.1792 | -0.9698 | 0.0052 | $1.06 | $1.3M | Zhongchao Inc. |
| FFAI | -1.1678 | -0.9698 | 0.0052 | $3.72 | $4.0M | FARADAY FUTURE INTELLIGENT ELECTRIC IN |
| BIYA | -1.1620 | -0.9698 | 0.0052 | $2.63 | $1.4M | Baiya International Group Inc. |
| DGNX | -1.1586 | -0.9698 | 0.0052 | $1.21 | $1.0M | Diginex Ltd |
| DBGI | -1.1441 | -0.9698 | 0.0052 | $13.77 | $1.3M | Digital Brands Group, Inc. |
| SDOT | -1.1238 | -0.9698 | 0.0052 | $10.57 | $7.3M | Sadot Group Inc. |
| NCT | -1.1211 | -0.9698 | 0.0052 | $3.87 | $1.5M | Intercont (Cayman) Ltd |
| ABTC | -1.1133 | -0.9698 | 0.0052 | $7.12 | $9.0M | American Bitcoin Corp. |

## Judgement calls

> [!warning] **This signal carries no cross-sectional information beyond `ret_12m`.**
> Subtracting a single constant from every stock is a monotone transform, so `rank_pct`
> here is **identical** to the rank of the raw 12-month return. Its value is the *level*
> (is this stock beating the median stock, yes/no, and by how much), which is useful as a
> filter or a reporting number, not as an extra column in a rank composite.

1. **It is close to, but not the same as, `mom_12_1`.** `rel_strength` uses the full
   t-0..t-12 window; `mom_12_1` skips the most recent month. Spearman between the two is
   **0.947** on this cross-section. Loading both into a composite is close to
   double-weighting momentum. If only one 12-month momentum column survives, it should be
   `mom_12_1` - the skipped month is what removes short-term reversal contamination, and
   that is the version with the documented premium.
2. **The median is over the investable universe** (3,616 names after the liquidity
   floor), not over all 9,687 tickers in the price panel. The unfiltered median is
   dominated by illiquid microcaps and is not a benchmark anyone could hold. Both numbers
   are printed in the build log.
3. **Median, not mean**, as specified. The mean 12m return of a US microcap-heavy
   cross-section is pulled hard by right-tail names and is not a robust centre.
4. Equal-weighted, as specified - so the benchmark is "the typical stock", not "the
   market". A cap-weighted benchmark would be a different (and in 2025-26, much harder)
   bar to clear.

## Universe

Investable universe = **3,616 names**, one line per issuer CIK: SEC-roster common
equity with median 60d dollar volume >= $1,000,000, last close >= $1.00,
traded on >= 30 of the last 60 sessions, and adj_close not quarantined or
unadjustable upstream. Closed-end funds, commodity/crypto trusts, ETFs, ETNs, royalty
trusts, preferred series, warrants, rights and units are excluded
(`_excluded_funds.csv`), as are secondary share classes and baby bonds
(`_excluded_secondary_lines.csv`). Ranks are computed **within** this universe, so
`rank_pct` is comparable only to other signals ranked on the same universe.
