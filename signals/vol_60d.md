# vol_60d

**Annualised realised volatility of daily returns over the last 60 sessions. Low-volatility names have historically earned higher risk-adjusted returns; the sign is NEGATIVE.**

- OSAP signal: `RealizedVol` — Realized (Total) Volatility, Ang et al. 2006, Detailed Definition: "Standard deviation of residuals from CAPM regressions using the past month of daily data. Value weighted"
- Sign in OSAP: **-1** — high value = *low* expected return, the composite must use `1 - rank_pct`.
- Coverage: **3,615** tickers of 3,616 in the investable universe (100.0%)
- Built 2026-08-18 from the price panel anchored on session 2026-08-17.

## Exact formula

```
vol_60d = stdev(r_d over the last 60 sessions) * sqrt(252)
  where r_d = adj_close[d]/adj_close[d-1] - 1, kept only on sessions with a
  real print (a non-trading day contributes no fake 0% return),
  and at least 30 real prints are required in the window.
```

**Inputs:** `vol_60d` from data/prices/derived.parquet.
No XBRL / SEC concepts are used by this signal group — it is price and volume only.

## Distribution (raw, pre-winsorisation)

```
count    3615.000000
mean        0.552292
std         0.402168
min         0.030311
1%          0.161257
5%          0.205639
25%         0.314226
50%         0.460044
75%         0.686567
95%         1.172549
99%         1.750401
max         7.681633
```

Winsorisation: values clipped to the 1st/99th percentile of the investable universe
(`[0.1613, 1.7504]`) **before** ranking, per the build rules. `value` in the parquet is
the winsorised number; `rank_pct` is the ascending percentile rank of it. Consequence worth
knowing: the clipped 1% tails become ties, so the ~36 most extreme
names on each end share a rank. Ordering is untouched everywhere else.

## Top 10 (highest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| SDOT | 7.6816 | 1.7504 | 0.9950 | $10.57 | $7.3M | Sadot Group Inc. |
| STI | 7.5818 | 1.7504 | 0.9950 | $7.79 | $1.0M | Solidion Technology Inc. |
| JLHL | 7.3633 | 1.7504 | 0.9950 | $7.50 | $1.4M | Julong Holding Ltd |
| ZCMD | 5.4729 | 1.7504 | 0.9950 | $1.06 | $1.3M | Zhongchao Inc. |
| BIYA | 4.6668 | 1.7504 | 0.9950 | $2.63 | $1.4M | Baiya International Group Inc. |
| LASE | 3.6283 | 1.7504 | 0.9950 | $1.45 | $2.7M | Laser Photonics Corp |
| REPL | 3.0652 | 1.7504 | 0.9950 | $14.63 | $33.9M | Replimune Group, Inc. |
| DBGI | 2.9469 | 1.7504 | 0.9950 | $13.77 | $1.3M | Digital Brands Group, Inc. |
| MNTS | 2.8535 | 1.7504 | 0.9950 | $4.69 | $10.7M | Momentus Inc. |
| SKYQ | 2.8137 | 1.7504 | 0.9950 | $2.73 | $9.9M | Sky Quarry Inc. |

## Bottom 10 (lowest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| DBRG | 0.0303 | 0.1613 | 0.0053 | $15.91 | $38.7M | DigitalBridge Group, Inc. |
| AES | 0.0326 | 0.1613 | 0.0053 | $14.76 | $109.5M | AES CORP |
| OGN | 0.0335 | 0.1613 | 0.0053 | $13.70 | $31.5M | Organon & Co. |
| SLAB | 0.0459 | 0.1613 | 0.0053 | $219.29 | $62.3M | SILICON LABORATORIES INC. |
| RAMP | 0.0465 | 0.1613 | 0.0053 | $37.68 | $37.6M | LiveRamp Holdings, Inc. |
| GBTG | 0.0468 | 0.1613 | 0.0053 | $9.44 | $16.4M | Global Business Travel Group, Inc. |
| FXE | 0.0483 | 0.1613 | 0.0053 | $106.86 | $7.6M | Invesco CurrencyShares Euro Trust |
| TWO | 0.0554 | 0.1613 | 0.0053 | $12.01 | $15.4M | TWO HARBORS INVESTMENT CORP. |
| FXB | 0.0575 | 0.1613 | 0.0053 | $130.20 | $1.6M | Invesco CurrencyShares British Pound S |
| FXF | 0.0710 | 0.1613 | 0.0053 | $108.63 | $3.4M | Invesco CurrencyShares Swiss Franc Tru |

## Judgement calls

1. **This is TOTAL volatility, not the CAPM-residual volatility OSAP specifies, and the
   window is 60 sessions rather than 21.** Two deviations, both deliberate:
   - Residualising needs a market factor and a per-ticker beta regression, which the
     current panel does not carry. Total and idiosyncratic vol correlate ~0.95
     cross-sectionally in US equities, and OSAP itself carries both variants
     (`RealizedVol` total, `IdioVol3F` / `IdioVolCAPM` residual) with the same sign.
   - 60 sessions is a materially less noisy estimate than 21. The cost is slower
     response to a genuine vol regime change.
   If the residual version is wanted later, it needs a daily market-return series and a
   rolling regression - not available from derived.parquet, which is ticker-level.
2. **Sign is -1.** High vol = low expected return. A composite that adds `rank_pct`
   raw will get this backwards; it must use `1 - rank_pct`.
3. Vol is computed on forward-filled prices but only kept on real-print sessions, so a
   halted microcap does not get an artificially calm vol estimate from a run of zeros.

## Universe

Investable universe = **3,616 names**, one line per issuer CIK: SEC-roster common
equity with median 60d dollar volume >= $1,000,000, last close >= $1.00,
traded on >= 30 of the last 60 sessions, and adj_close not quarantined or
unadjustable upstream. Closed-end funds, commodity/crypto trusts, ETFs, ETNs, royalty
trusts, preferred series, warrants, rights and units are excluded
(`_excluded_funds.csv`), as are secondary share classes and baby bonds
(`_excluded_secondary_lines.csv`). Ranks are computed **within** this universe, so
`rank_pct` is comparable only to other signals ranked on the same universe.
