# lab smoke | 2026-10-12 | variants ever tested M = 3 (3 new in this batch) | bar: d t >= 1.93
bar rule: Bonferroni, family-wise alpha 0.1 one-sided: t of d (variant - control, monthly) >= quantile 1-0.1/M (Student-t df 23); M = distinct param hashes in research/lab_ledger.csv + this batch
hypothesis: Lab self-test, not a research claim: the batch runner must reproduce the registered results of BACKLOG item 14 (MONTHLY control, BUFFER80) and item 6 (QUARTERLY, mean of 3 phases) through its own spec path.
registered: LAB-CORRECT if, within rounding, MONTHLY = +0.25%/mo t 0.21 after-tax +8.5% traded 49%/rebal AND BUFFER80 = +0.55%/mo t 0.50 after-tax +11.6% traded 30%, d +0.30%/mo t 1.09, D +3.07pp, null 73%/79% (AR1) AND QUARTERLY = +0.48%/mo after-tax +11.1% (mean of phases), d +0.23%/mo t 0.80, D +2.54pp, null 38%/28% (iid). Any mismatch = LAB-WRONG: find and fix before use.
data: 24 months 2024-09-30..2026-08-31 (returns to 2026-09-30) | pool 2381-2931 | tickers_2026-10-10.csv | prices to 2026-10-09 | max mktcap 5,452bn | bad returns zeroed/mo max 117 | harness a78b18961f

| variant | vs SPY %/mo | t | after-tax ann | d %/mo (t) | D pp/yr | null% d/D | null% lvl | traded/reb | bar |
|---|---|---|---|---|---|---|---|---|---|
| MONTHLY (control) | +0.25 | 0.21 | +8.5% | - | - | - | 94% | 49% | - |
| QUARTERLY [rebalance=quarterly] | +0.48 | 0.43 | +11.0% | +0.23 (0.80) | +2.54 | 38%/28% iid | 99% | 82% | no |
| BUFFER80 [buffer=80] | +0.55 | 0.50 | +11.6% | +0.30 (1.09) | +3.07 | 73%/79% ar1 | 98% | 30% | no |

cov MONTHLY: 24mo pool 2381-2931 scored>=100% names 40-40 maxsec 8/8 worstrank 40 kept 32.8 held-flat 0 medADV $4.4m reb 24 OK
cov QUARTERLY: 24mo pool 2381-2931 scored>=100% names 40-40 maxsec 8/8 worstrank 40 kept 32.8 held-flat 0 medADV $4.4m reb 8/9/9 OK
cov BUFFER80: 24mo pool 2381-2931 scored>=100% names 40-40 maxsec 8/8 worstrank 80 kept 36.9 held-flat 0 medADV $4.6m reb 24 OK
legend: after cost (item 3b, notional $1m book), tax 42% (item 6); quarterly = mean of 3 phases; d = variant-control pre-tax; D = after-tax liquidated annualised; null% = share of 200 noise books <= this (d/D: harness null, after cost, same noise through variant and control construction, iid = item 6, ar1 = item 14; lvl: gross vs SPY vs AR(1) noise books of the same construction, costs off both sides); runtime per arm includes its construction's nulls
runtime: harness load 21s, noise draws 5s, per arm (run + its null): MONTHLY 13s, QUARTERLY 19s, BUFFER80 8s; total 68s
