# px_to_52w_high

**Proximity to the 52-week high. George & Hwang (2004): stocks trading near their 52-week high subsequently outperform, and the ratio dominates raw momentum in horse races. 1.00 = at the high.**

- OSAP signal: `High52` — 52 week high, George and Hwang 2004, Detailed Definition: "Price (prc/cfacshr) divided by the maximum price over the previous 12 months." OSAP tests it as a 30/70 split (LS Quantile 0.3), not deciles, held 6 months.
- Sign in OSAP: **+1** — high value = high expected return, use `rank_pct` directly.
- Coverage: **3,616** tickers of 3,616 in the investable universe (100.0%)
- Built 2026-08-18 from the price panel anchored on session 2026-08-17.

## Exact formula

```
px_to_52w_high = P[t] / max(P[t-251 .. t])

P = adjusted close. `prc/cfacshr` in the OSAP definition is CRSP's
split-adjusted price, which is what adj_close is here.
```

**Inputs:** `px_to_52w_high` from data/prices/derived.parquet.
No XBRL / SEC concepts are used by this signal group — it is price and volume only.

## Distribution (raw, pre-winsorisation)

```
count    3616.000000
mean        0.782760
std         0.204672
min         0.000375
1%          0.158881
5%          0.362466
25%         0.674769
50%         0.846686
75%         0.945482
95%         0.995418
99%         1.000000
max         1.000000
```

Winsorisation: values clipped to the 1st/99th percentile of the investable universe
(`[0.1589, 1.0000]`) **before** ranking, per the build rules. `value` in the parquet is
the winsorised number; `rank_pct` is the ascending percentile rank of it. Consequence worth
knowing: the clipped 1% tails become ties, so the ~36 most extreme
names on each end share a rank. Ordering is untouched everywhere else.

## Top 10 (highest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| MAUTF | 1.0000 | 1.0000 | 0.9844 | $14.22 | $1.0M | Montage Gold Corp. |
| TVTX | 1.0000 | 1.0000 | 0.9844 | $63.53 | $85.2M | Travere Therapeutics, Inc. |
| ZIM | 1.0000 | 1.0000 | 0.9844 | $28.81 | $25.0M | ZIM Integrated Shipping Services Ltd. |
| TNK | 1.0000 | 1.0000 | 0.9844 | $89.11 | $24.0M | TEEKAY TANKERS LTD. |
| BBT | 1.0000 | 1.0000 | 0.9844 | $32.61 | $20.8M | Beacon Financial Corp |
| NAT | 1.0000 | 1.0000 | 0.9844 | $6.82 | $19.0M | NORDIC AMERICAN TANKERS Ltd |
| SHG | 1.0000 | 1.0000 | 0.9844 | $75.92 | $16.7M | SHINHAN FINANCIAL GROUP CO LTD |
| DAC | 1.0000 | 1.0000 | 0.9844 | $146.28 | $12.7M | Danaos Corp |
| SXC | 1.0000 | 1.0000 | 0.9844 | $9.51 | $12.0M | SunCoke Energy, Inc. |
| NMM | 1.0000 | 1.0000 | 0.9844 | $82.77 | $7.9M | Navios Maritime Partners L.P. |

## Bottom 10 (lowest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| VIVK | 0.0004 | 0.1589 | 0.0053 | $1.22 | $1.6M | Vivakor, Inc. |
| YYGH | 0.0004 | 0.1589 | 0.0053 | $1.44 | $1.3M | YY Group Holding Ltd. |
| ZCMD | 0.0011 | 0.1589 | 0.0053 | $1.06 | $1.3M | Zhongchao Inc. |
| DGNX | 0.0048 | 0.1589 | 0.0053 | $1.21 | $1.0M | Diginex Ltd |
| FFAI | 0.0139 | 0.1589 | 0.0053 | $3.72 | $4.0M | FARADAY FUTURE INTELLIGENT ELECTRIC IN |
| BIYA | 0.0156 | 0.1589 | 0.0053 | $2.63 | $1.4M | Baiya International Group Inc. |
| DBGI | 0.0182 | 0.1589 | 0.0053 | $13.77 | $1.3M | Digital Brands Group, Inc. |
| SDOT | 0.0489 | 0.1589 | 0.0053 | $10.57 | $7.3M | Sadot Group Inc. |
| ABTC | 0.0510 | 0.1589 | 0.0053 | $7.12 | $9.0M | American Bitcoin Corp. |
| BATL | 0.0513 | 0.1589 | 0.0053 | $1.42 | $9.3M | BATTALION OIL CORP |

## Judgement calls

1. **Close-to-close, not close-to-intraday-high.** The panel carries no adjusted high, so
   the denominator is the max *adjusted close* over 252 sessions. Values are therefore
   very slightly higher than the true close-to-intraday-high ratio and cap at exactly
   1.0000 for a name that closed at its 52-week high today. OSAP's definition uses
   `prc`, i.e. closing price, so this matches the catalog rather than deviating from it.
2. **This is a bounded ratio in (0, 1], which is why it behaves so differently from
   momentum.** The 1%/99% winsorisation barely bites at the top (many names sit at 1.00)
   and clips only the deepest-drawdown tail at the bottom.
3. The top of this ranking is a large tie-block of names sitting exactly at their high.
   Winsorisation does not create that tie - the data does. Downstream, treat rank_pct
   near the top as "at the high", not as a fine ordering.

## Universe

Investable universe = **3,616 names**, one line per issuer CIK: SEC-roster common
equity with median 60d dollar volume >= $1,000,000, last close >= $1.00,
traded on >= 30 of the last 60 sessions, and adj_close not quarantined or
unadjustable upstream. Closed-end funds, commodity/crypto trusts, ETFs, ETNs, royalty
trusts, preferred series, warrants, rights and units are excluded
(`_excluded_funds.csv`), as are secondary share classes and baby bonds
(`_excluded_secondary_lines.csv`). Ranks are computed **within** this universe, so
`rank_pct` is comparable only to other signals ranked on the same universe.
