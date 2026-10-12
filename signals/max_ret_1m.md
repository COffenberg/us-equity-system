# max_ret_1m

**LOTTERY MARKER - NOT A BUY SIGNAL. The single largest daily return in the last 21 sessions. Bali, Cakici & Whitelaw (2011) document that high-MAX stocks EARN LESS on average: investors overpay for lottery-like right-tail payoffs. The top of this ranking is the sell/avoid end.**

- OSAP signal: `MaxRet` — Maximum return over month, Bali, Cakici, and Whitelaw 2011, Detailed Definition: "Maximum of daily returns (ret) over the previous month". OSAP notes: "Doesn't work as well EW. Very nonlinear return vs decile" - the effect is concentrated in the extreme decile, and needs value weighting.
- Sign in OSAP: **-1** — high value = *low* expected return, the composite must use `1 - rank_pct`.
- Coverage: **3,616** tickers of 3,616 in the investable universe (100.0%)
- Built 2026-08-18 from the price panel anchored on session 2026-08-17.

## Exact formula

```
max_ret_1m = max(r_d over the last 21 sessions)
  where r_d = adj_close[d]/adj_close[d-1] - 1 on real-print sessions only.
```

**Inputs:** `max_ret_1m` from data/prices/derived.parquet.
No XBRL / SEC concepts are used by this signal group — it is price and volume only.

## Distribution (raw, pre-winsorisation)

```
count    3616.000000
mean        0.079901
std         0.080277
min         0.001967
1%          0.010798
5%          0.018445
25%         0.036442
50%         0.058648
75%         0.096962
95%         0.210941
99%         0.363033
max         1.918367
```

Winsorisation: values clipped to the 1st/99th percentile of the investable universe
(`[0.0108, 0.3630]`) **before** ranking, per the build rules. `value` in the parquet is
the winsorised number; `rank_pct` is the ascending percentile rank of it. Consequence worth
knowing: the clipped 1% tails become ties, so the ~36 most extreme
names on each end share a rank. Ordering is untouched everywhere else.

## Top 10 (highest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| ZCMD | 1.9184 | 0.3630 | 0.9950 | $1.06 | $1.3M | Zhongchao Inc. |
| BIYA | 1.1606 | 0.3630 | 0.9950 | $2.63 | $1.4M | Baiya International Group Inc. |
| REPL | 1.0702 | 0.3630 | 0.9950 | $14.63 | $33.9M | Replimune Group, Inc. |
| UTZ | 0.8872 | 0.3630 | 0.9950 | $14.15 | $14.2M | Utz Brands, Inc. |
| JLHL | 0.7251 | 0.3630 | 0.9950 | $7.50 | $1.4M | Julong Holding Ltd |
| QMCO | 0.6469 | 0.3630 | 0.9950 | $24.92 | $12.5M | QUANTUM CORP /DE/ |
| RCEL | 0.6358 | 0.3630 | 0.9950 | $7.90 | $1.0M | AVITA Medical, Inc. |
| VATE | 0.6343 | 0.3630 | 0.9950 | $7.56 | $1.2M | INNOVATE Corp. |
| NLST | 0.5820 | 0.3630 | 0.9950 | $6.03 | $3.2M | NETLIST INC |
| CAPR | 0.5796 | 0.3630 | 0.9950 | $7.45 | $26.3M | CAPRICOR THERAPEUTICS, INC. |

## Bottom 10 (lowest value)

| ticker | raw | winsorised | rank_pct | last close | 60d $vol | name |
|---|---|---|---|---|---|---|
| BBCQ | 0.0020 | 0.0108 | 0.0053 | $10.16 | $1.5M | Bleichroeder Acquisition Corp. II |
| AES | 0.0020 | 0.0108 | 0.0053 | $14.76 | $109.5M | AES CORP |
| APGE | 0.0026 | 0.0108 | 0.0053 | $134.59 | $130.6M | Apogee Therapeutics, Inc. |
| TBPH | 0.0030 | 0.0108 | 0.0053 | $16.92 | $7.1M | Theravance Biopharma, Inc. |
| ACA | 0.0030 | 0.0108 | 0.0053 | $145.01 | $77.4M | Arcosa, Inc. |
| CRNX | 0.0031 | 0.0108 | 0.0053 | $84.75 | $100.4M | Crinetics Pharmaceuticals, Inc. |
| GBTG | 0.0032 | 0.0108 | 0.0053 | $9.44 | $16.4M | Global Business Travel Group, Inc. |
| DBRG | 0.0038 | 0.0108 | 0.0053 | $15.91 | $38.7M | DigitalBridge Group, Inc. |
| RAMP | 0.0040 | 0.0108 | 0.0053 | $37.68 | $37.6M | LiveRamp Holdings, Inc. |
| CCO | 0.0043 | 0.0108 | 0.0053 | $2.32 | $13.4M | Clear Channel Outdoor Holdings, Inc. |

## Judgement calls

> [!warning] **Sample-definition variable, not a buy signal.**
> This is built and ranked because the composite needs it as a *negative* screen and
> because it defines the lottery-stock sample. Reading a high `rank_pct` here as
> "attractive" inverts the documented effect. Sign is **-1**: the composite must use
> `1 - rank_pct`, and a defensible alternative use is a hard exclusion of the top decile
> rather than a linear tilt.

1. **The effect is very nonlinear (OSAP's own note) and works value-weighted, not
   equal-weighted.** A linear rank tilt across the whole cross-section is the *weakest*
   way to use it. Treat the top decile as the live part of the signal.
2. **It is mechanically entangled with `vol_60d`.** A single-day max return is largely a
   draw from the same distribution that produces high realised vol; the Spearman
   correlation between the two is printed in the build log and is high. Loading both into
   a composite double-counts volatility. Cluster analysis (`research/out/pool_clusters.csv`)
   should decide which one survives.
3. **21 sessions, not a calendar month.** Same convention as everything else here.
4. Upstream split repair matters more for this signal than any other: one unapplied split
   is a single +10,000% "daily" return, which would put a clean stock at rank 1.0000 on a
   lottery screen. 447 unapplied splits were repaired and 523 tickers quarantined before
   this was computed.

## Universe

Investable universe = **3,616 names**, one line per issuer CIK: SEC-roster common
equity with median 60d dollar volume >= $1,000,000, last close >= $1.00,
traded on >= 30 of the last 60 sessions, and adj_close not quarantined or
unadjustable upstream. Closed-end funds, commodity/crypto trusts, ETFs, ETNs, royalty
trusts, preferred series, warrants, rights and units are excluded
(`_excluded_funds.csv`), as are secondary share classes and baby bonds
(`_excluded_secondary_lines.csv`). Ranks are computed **within** this universe, so
`rank_pct` is comparable only to other signals ranked on the same universe.
