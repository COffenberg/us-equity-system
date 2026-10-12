# volume_trend

**Trend in trading volume: the slope of monthly dollar volume regressed on a linear time trend, scaled by average monthly volume. Haugen & Baker (1996) find rising volume predicts LOWER returns; sign is NEGATIVE.**

- OSAP signal: `VolumeTrend` — Volume Trend, Haugen and Baker 1996, Detailed Definition: "Rolling coefficient from regressing monthly trading volume on a linear time trend over a window of 60 months (require that at least 30 exist). Scale coefficient by 60-month average of trading volume." Rebalanced annually (Portfolio Period 12) in OSAP - this is a slow signal.
- Sign in OSAP: **-1** — high value = *low* expected return, the composite must use `1 - rank_pct`.
- Coverage: **3,366** tickers of 3,616 in the investable universe (93.1%)
- Built 2026-08-18 from the price panel anchored on session 2026-08-17.

## Exact formula

```
V_m  = sum of (close x volume) over calendar month m      [monthly dollar volume]
m    = 2023-09 .. 2026-07 (the 35 complete months in the 3y panel)
b    = OLS slope of V_m on the calendar month index t_m
volume_trend = b / mean(V_m)

Requires >= 24 months present with non-zero volume and requires the final
month (2026-07) to be present, so the trend is not measured on a stale name.
Units: fraction of average monthly volume added per month.
```

**Inputs:** `data/prices/history_3y.parquet` (date, ticker, close, volume) - the raw recorder output, aggregated to monthly dollar volume.
No XBRL / SEC concepts are used by this signal group — it is price and volume only.

## Distribution (raw, pre-winsorisation)

```
count    3366.000000
mean        0.029044
std         0.033047
min        -0.106995
1%         -0.042828
5%         -0.019830
25%         0.009523
50%         0.023614
75%         0.045726
95%         0.092341
99%         0.121792
max         0.152656
```

Winsorisation: values clipped to the 1st/99th percentile of the investable universe
(`[-0.0428, 0.1218]`) **before** ranking, per the build rules. `value` in the parquet is
the winsorised number; `rank_pct` is the ascending percentile rank of it. Consequence worth
knowing: the clipped 1% tails become ties, so the ~34 most extreme
names on each end share a rank. Ordering is untouched everywhere else.

## Top 10 (highest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| SDOT | 0.1527 | 0.1218 | 0.9951 | $10.57 | $7.3M | Sadot Group Inc. |
| NESR | 0.1484 | 0.1218 | 0.9951 | $35.78 | $49.6M | National Energy Services Reunited Corp |
| SHMD | 0.1432 | 0.1218 | 0.9951 | $5.68 | $3.3M | SCHMID Group N.V. |
| AAUC | 0.1404 | 0.1218 | 0.9951 | $22.37 | $11.6M | Allied Gold Corp |
| AXTI | 0.1400 | 0.1218 | 0.9951 | $95.97 | $775.4M | AXT INC |
| GALDY | 0.1392 | 0.1218 | 0.9951 | $41.66 | $4.1M | Galderma Group AG/ADR |
| TRT | 0.1379 | 0.1218 | 0.9951 | $11.60 | $5.0M | TRIO-TECH INTERNATIONAL |
| MRAM | 0.1353 | 0.1218 | 0.9951 | $19.85 | $29.7M | EVERSPIN TECHNOLOGIES INC. |
| EPRX | 0.1348 | 0.1218 | 0.9951 | $6.82 | $1.7M | EUPRAXIA PHARMACEUTICALS INC. |
| ALM | 0.1333 | 0.1218 | 0.9951 | $16.36 | $81.5M | Almonty Industries Inc. |

## Bottom 10 (lowest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| VFS | -0.1070 | -0.0428 | 0.0052 | $3.12 | $1.3M | VinFast Auto Ltd. |
| CHPT | -0.0778 | -0.0428 | 0.0052 | $5.88 | $3.3M | ChargePoint Holdings, Inc. |
| ALXO | -0.0720 | -0.0428 | 0.0052 | $1.97 | $1.6M | ALX ONCOLOGY HOLDINGS INC |
| NKTX | -0.0706 | -0.0428 | 0.0052 | $2.43 | $1.4M | Nkarta, Inc. |
| CHRS | -0.0699 | -0.0428 | 0.0052 | $1.15 | $1.1M | Coherus Oncology, Inc. |
| MED | -0.0682 | -0.0428 | 0.0052 | $11.44 | $1.8M | MEDIFAST INC |
| JRVR | -0.0634 | -0.0428 | 0.0052 | $4.02 | $1.3M | James River Group Holdings, Inc. |
| CCCC | -0.0627 | -0.0428 | 0.0052 | $4.26 | $9.2M | C4 Therapeutics, Inc. |
| CXAI | -0.0591 | -0.0428 | 0.0052 | $4.45 | $1.2M | CXApp Inc. |
| MGNX | -0.0564 | -0.0428 | 0.0052 | $4.06 | $2.8M | MACROGENICS INC |

## Judgement calls

1. **The window is 35 months, not OSAP's 60 - the panel is only 3 years deep.** The
   minimum-observation requirement was scaled proportionally (24 of 35 = 69%, vs OSAP's
   30 of 60 = 50%, so this is if anything stricter). A 35-month slope is a noisier estimate
   of a "volume trend" than a 60-month one and this is the single largest deviation from
   the catalog in this signal group. It cannot be fixed without more history; the recorder
   has been live since 2026-08-17, so it fixes itself over time.
2. **Dollar volume, not share volume.** OSAP uses CRSP `vol` (shares). Yahoo's raw share
   volume is not split-invariant and `history_3y.parquet` is the *unrepaired* record - 447
   unapplied splits were detected upstream. At an unapplied split, share volume steps by
   ~1/k and would be read as a volume trend that is pure corporate action. `close x volume`
   is invariant to that. This is a correctness fix, not a preference.
3. **Not the `volume_trend` column in derived.parquet.** That column is
   `dollar_vol_60d / median(close x volume, last 250)` - a 60d-vs-250d ratio, a fast
   volume-surge measure, not the catalog's regression slope. Spearman between the two
   over the 3,366 names holding both is **0.339**, i.e. they are related but
   are not the same signal. The catalog definition was implemented because the task
   requires the catalog definition; the ratio remains available in derived.parquet if a
   fast volume-surge variant is wanted.
4. **Partial months at both edges are dropped** (2023-08 starts on the 10th, 2026-08 ends
   on the 17th). Including them would put a large artificial down-tick at the end of every
   single ticker's series and bias the whole cross-section negative.
5. **Coverage is structurally lower than the other price signals** because 24 months of
   history is required: recent IPOs cannot have this signal at all. That is missing data,
   not a zero - it is left absent rather than imputed.
6. Sign is **-1**: rising volume predicts lower returns. The composite must use
   `1 - rank_pct`.

## Universe

Investable universe = **3,616 names**, one line per issuer CIK: SEC-roster common
equity with median 60d dollar volume >= $1,000,000, last close >= $1.00,
traded on >= 30 of the last 60 sessions, and adj_close not quarantined or
unadjustable upstream. Closed-end funds, commodity/crypto trusts, ETFs, ETNs, royalty
trusts, preferred series, warrants, rights and units are excluded
(`_excluded_funds.csv`), as are secondary share classes and baby bonds
(`_excluded_secondary_lines.csv`). Ranks are computed **within** this universe, so
`rank_pct` is comparable only to other signals ranked on the same universe.
