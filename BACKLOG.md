# Research backlog — agent-owned

Ranked by expected value. Next up (2026-10-12): 9 (frozen fundamentals — before Q3 10-Qs land
late Oct), share-unit sanity in live build_all.py (see 12; BH in live book on a dual-class error), then 3c, 4. Item 11 mostly answered by 12.
**Awaiting the owner:** 7b (live cfp is not TTM; correct version changes 15/40 names); 12 (skew overlay refuted in backtest; BH share-count error in live book); 6 (quarterly rebalance supported after cost+tax); 13 (sector-relative ranking: registered verdict BELIEF BROKEN, fragile); 14 (rank-80 exit buffer supported after cost+tax — same lever as 6, choose one).
 The agent picks the top actionable item each run, does it,
records the result here, and re-ranks. Add new items freely; never delete a finished
one — refutations are findings.

Status: `open` | `running` | `done` | `refuted` | `blocked`

---

## 1. Is the +20.6pp edge selection bias? `open`
**Why it matters most:** everything else is downstream of whether the headline result
is real. 41 signals were screened on 23 months; two false positives were expected and
exactly two appeared.
**Test:** compare the live forward record (frozen books, `forward/prediction_*.json`)
against the backtest expectation of +1.47%/month. Registered threshold, set before the
data exists: **if the live edge over the first 6 months is below +0.4%/month, the
backtest is overstated by more than half and the book should be rebuilt on the
published 4-5pp premium instead.**
**Blocked until:** 2027-03 (6 monthly observations).

## 2. Which of SP, bm, cfp actually carries the edge? `done` 2026-09-30
Tested all 7 non-empty subsets of {SP,bm,cfp} on the MONTHLY backtest (23 chained
months, same pipeline as `value_monthly_backtest.py`, screened on real objective).
`research/test_component_decomp.py`.

| subset | monthly edge | t | win |
|---|---|---|---|
| SP | +2.14% | 1.64 | 65% |
| SP+bm+cfp (current book) | +1.46% | 1.67 | 57% |
| SP+bm | +0.93% | 1.01 | 61% |
| bm+cfp | +0.88% | 1.13 | 57% |
| bm | +0.81% | 1.10 | 57% |
| SP+cfp | +0.36% | 0.33 | 43% |
| cfp | +0.19% | 0.24 | 52% |

Result agrees with the quarterly test on ranking: **cfp is the weakest leg both
times** (this run: +0.19%/mo solo, near zero; quarterly run: weakest of +4.75/
+4.19/+1.97). Two independent measurements now agree cfp adds least.

Surprise: solo SP (+2.14%/mo) beats the full 3-factor book (+1.46%/mo) on this
sample, and SP+bm+cfp has the second-highest t only because of lower variance
(n=23, one horse — SP dominated by a handful of momentum-adjacent months —
carries meaningfully more single-factor risk than a blend even when its point
estimate is higher). Do not read "SP alone wins" as an instruction to concentrate
the book: at n=23 months the gap between every subset above bm+cfp is well inside
noise (finding #2 in AGENTS.md: the screen certifies 10% of pure noise on samples
this size).

**Not actioned** — dropping cfp from the live book would be a change to
`signals/rank.py` CLUSTERS, which is off-limits per AGENTS.md hard limits
without forward confirmation at a pre-registered threshold. Recommendation for
whoever reviews: if item 1's forward record confirms the edge is real, re-test
this decomposition on forward data (not backtest) before touching the live
weights — cfp's near-zero solo edge is consistent across two backtest vintages,
but "drop it" is a live-book change and needs the same discipline as everything
else on this list.

## 3. Does the edge survive at real liquidity? `done` 2026-10-01 — mostly NO
`research/test_liquidity_floor.py`, output `research/out/liquidity_floor.parquet`.
Same monthly pipeline (23 chained months, 40 names, cap 8). $0.5m row reproduces the
headline exactly (+1.46%/mo, t 1.67, +36.9%/yr), so the harness is sound. Added a
baseline the original never had: 200 random 40-name books per month from the SAME
eligible universe with the same sector cap.

| floor | eligible | book-SPY | t | random-SPY | book-random | t | ann. book |
|---|---|---|---|---|---|---|---|
| $0.5m | 3260 | +1.46% | 1.67 | +0.23% | +1.23% | 1.35 | +36.9% |
| $2m | 2673 | +1.14% | 1.18 | +0.03% | +1.11% | 1.46 | +31.3% |
| $10m | 1940 | **+0.06%** | 0.07 | -0.07% | +0.13% | 0.22 | +16.0% |
| $50m | 1063 | +0.57% | 0.82 | -0.18% | +0.74% | 1.25 | +24.2% |

Drop the best month (Oct 2024, small-cap election rally, +13.6% at $0.5m vs +4.5% at
$10m): $0.5m falls to +0.91% (t 1.27), $10m to -0.39%. Sanity: every book n=40, eligible
universe never below 862, single-name extremes +143%/-50% (no +inf, no blowups).

**Meaning:** at liquidity where institutions trade, the value tilt is
indistinguishable from zero over this window. Not monotone ($50m > $10m), and monthly
edge series across floors correlate only 0.16 ($0.5m vs $50m) — the floors pick
different books and n=23 cannot separate them. Random equal-weight books earn ~0 vs
SPY at every floor, so the edge is not a plain size/equal-weight effect; it is a value
tilt *inside thin names*. Strengthens finding #5: the excess over 4-5pp is most likely
thin-name luck/illiquidity premium, not a durable value edge.
**Does not change the book:** live floor is $500k (`signals/rank.py` line 88) and
a book of the modelled size can trade it. Cost of thin names (spread, impact) is NOT modelled — see
new item 3b.

**Registered threshold (2026-10-01, before any forward data at this floor exists):**
when item 1 is evaluated (2027-03), also rebuild each frozen month's book at a $10m floor
from the same snapshot date and measure it forward. If the $10m variant's 6-month
forward edge is **≥ +0.4%/month**, the value edge is real at institutional liquidity;
if **< 0**, the live edge is a thin-name effect and should be sized as such.

## 3b. What do thin-name trading costs eat? `done` 2026-10-05 — not enough to matter
`research/test_thin_name_costs.py`, output `research/out/thin_name_costs.parquet`.
Same pipeline as item 3, 22 steady-state months (initial build month dropped, so gross
is +1.60% not +1.46%). Traded weight = sum |target - drifted| incl. equal-weight resets:
**~49%/month two-sided** (~24% one-way; item text above guessed ~20% one-way — right).

| floor | traded/mo | gross | bucket spread | impact $1m | net | t | 2x-stress net |
|---|---|---|---|---|---|---|---|
| $0.5m | 49% | +1.60% | 0.13% | 0.13% | **+1.33%** | 1.48 | +1.20% |
| $2m | 46% | +1.32% | 0.08% | 0.08% | +1.17% | 1.18 | +1.09% |
| $10m | 45% | +0.32% | 0.04% | 0.04% | +0.25% | 0.32 | +0.21% |
| $50m | 42% | +0.86% | 0.01% | 0.01% | +0.83% | 1.25 | +0.82% |

Spread model is an ASSUMPTION (no quotes, no high/low): half-spread by 60d median ADV
<$2m 50bp, $2-10m 20bp, $10-50m 8bp, >$50m 3bp; impact = 0.7·σ_d·√(Q/ADV) per side.
Model-free check: a uniform half-spread of **1.06%** at $0.5m (with $2m names at half)
is needed before $2m beats $0.5m net; **3.25%** to take the $0.5m edge to zero.
Realistic thin-name half-spreads are 30-100bp, so neither is close.

**Roll estimator failed sanity and was discarded:** median half-spread 0.80% / 0.84% /
0.86% / 0.72% at $0.5m/$2m/$10m/$50m — flat across a 25x liquidity range, defined for
only 53-61% of names. Real $100m-ADV names trade at ~3-5bp. It is picking up short-
horizon reversal, not bid-ask bounce. Even using it (worst case) the $0.5m book nets
+1.00%/mo vs +0.80% at $2m — the ordering survives.

**Meaning:** costs do not explain the thin-name edge and do not argue for raising the
floor. The live $0.5m floor stands. What remains suspicious about the thin-name edge is
luck/selection (item 1) and survivorship of failures in the thinnest names (the
illiquid-tilt finding), not trading friction. Bounded caveat: costs here assume
patient limit orders at mid±half-spread; market orders on $0.5-2m names in a stressed
tape could double them — the 2x column covers that.

New question raised → **3c** below.

## 3c. Is the $0.5m edge a survivorship artefact in the thinnest names? `open`
AGENTS.md finding #4 measured attrition at 0.27%/quarter on the whole universe. The
illiquid-tilt study showed a 5% failure rate at -80% wipes out the thinnest bucket's
long-only return. Those two are not in conflict only if attrition is low *inside the
book's own names*. Test: attrition rate of names that were in the $0.5m-floor value book
vs the $10m-floor book, from the SEC roster (delisting = filer stops filing). Not a
re-run of #4 — it is the same question restricted to the subset where the edge lives.
Checked 2026-10-05: `survivorship_haircut.py` does NOT split by liquidity, so this is not
a duplicate. Catch: vanished filers are absent from the price panel at every date, so
they have no dollar volume — split them by a filings-based size proxy (TTM revenue or
total assets at the scoring date) against the same proxy for the book's names. Guard
against re-litigating finding #4: the question is only the conditional rate in the
bottom size tercile, and its answer is a number to feed item 1, not a new verdict on #4.

## 4. Build ShortInterest `open`
OSAP survivor, recent t=3.10, decay 1.18, never built. Free: FINRA publishes
semi-monthly short interest; SEC publishes fails-to-deliver. Two-source leg one is
already satisfied. Screen on the real objective, not spread.

## 5. Build the 13F short-sale-constraint family `open`
RIO_Volatility has the highest recent t in the whole catalog (4.58), and
IO_ShortInterest t=2.80. Both need 13F institutional ownership, which we now have
(`data/sec/f13_holdings.parquet`), crossed with short interest from item 4.

## 6. Monthly vs quarterly rebalance, after tax `done` 2026-10-10 — QUARTERLY-SUPPORTED as registered (family 2)
The 21-day horizon showed the largest edge but 12x turnover and 42% tax on
every realisation. The monthly backtest assumes monthly. Test quarterly on the same
chained basis and compare after tax.
`research/test_rebalance_tax.py`, log `research/out/rebalance_tax.log`, output `rebalance_tax.parquet`.
Reuses the item-12 harness verbatim (exec of its loader). Value-only book, 40 names, cap 8, 24 chained
months (2024-09..2026-08). Picks need a price at D only (no fwd-return lookahead); missing later prices
held flat. Costs = item 3b model (notional $1m book). Tax = 42% on net realised gains per calendar year, average
cost, loss carry-forward, paid at year end; everything liquidated and taxed at the end.
**Registered in the script header before the first run:** control = MONTHLY; candidate = QUARTERLY at
all 3 phase offsets, number = mean of phases. QUARTERLY-SUPPORTED if after-tax annualised D >= +1.0pp
AND pre-tax after-cost Q-M >= -0.10%/mo; REFUTED if D <= 0; else INCONCLUSIVE. Null = 200 random
sector-rank signals through both arms (diagnostic, not a gate).

| arm | vs SPY /mo | t | pre-tax ann | after-tax ann (liquidated) | costs (% W, 2y) | tax paid in window |
|---|---|---|---|---|---|---|
| MONTHLY (control) | +0.25% | 0.21 | +17.2% | +8.5% | 7.4% | 15.4% |
| QUARTERLY ph0 | +0.88% | 0.75 | +26.1% | +13.9% | 4.2% | 17.3% |
| QUARTERLY ph1 | +0.17% | 0.16 | +16.7% | +9.1% | 4.3% | 13.8% |
| QUARTERLY ph2 | +0.39% | 0.33 | +19.0% | +10.2% | 4.6% | 15.3% |
| SPY | | | +16.5% | | | |

Q-M pre-tax after cost **+0.23%/mo** (t 0.80, win 46%). **D = +2.54pp/yr** after tax (per phase
+5.41 / +0.56 / +1.66; unliquidated +3.27pp). Traded per rebalance 49% monthly vs 78-87% quarterly.
Null: random books gain *more* from going quarterly (pre-tax +0.34%/mo, 5-95% [-0.37, +0.98]; D +4.25pp);
value book sits at the 38th / 28th percentile.
**Verdict as registered: QUARTERLY-SUPPORTED.**
**Meaning (post-hoc, labelled):** the gain is mechanical — ~0.13%/mo of cost saved plus tax deferral —
not better signal. Noise books gain as much or more, so value's signal does not *improve* with a longer
hold; it decays slightly (value sits ~0.1%/mo below the noise mean, inside the band). That is the robust
part: a cost/tax saving does not need signal luck. Fragile part: phase spread is 5pp/yr on 8-9 rebalances,
so the size of D is not trustworthy, only its sign (positive in all 3 phases). Agrees with the old
horizon test's "monthly-to-quarterly" (dashboard), now on the real objective and after cost + tax.
Also measured: on this harness the monthly value book nets only +0.25%/mo vs SPY after costs
(t 0.21) — after tax it trails SPY's pre-tax return by 8pp/yr; only the gross number beat SPY.
**Not actioned:** cadence is a live-book change; needs the owner + forward confirmation.
**Registered forward threshold (2026-10-10), frozen books dated 2026-10-01 or later only:** at the 2027-03
evaluation (item 1), build the quarterly-hold variant from the same frozen books (hold the first book of
each quarter for 3 months, drifting) and compare to the monthly chain, after the item-3b cost model.
If mean monthly Q-M is **>= -0.10%/mo**, recommend switching to quarterly (tax saving stands); if
**< -0.30%/mo**, monthly stays; in between, no change.

## 7. Calibrate the evaluator against known answers `done` 2026-10-08 — price PASS, fundamentals FAIL as registered
`research/test_calibrate_osap.py`, output `research/out/calibrate_osap.parquet`. Rebuilt OSAP
long-short series from our data over the only overlap (formations 2023-08..2024-11, 16 returns).
Thresholds registered in the script header before the first run. Null = rank of the matched
OSAP series among all 198 OSAP signals by |corr| with ours.

| signal | spec | corr | rank/198 | ours %/mo | OSAP %/mo |
|---|---|---|---|---|---|
| STreversal | primary | **0.82** | **1** | -0.90 | +0.92 |
| SP | primary (annual June) | 0.50 | 42 | +2.89 | +0.58 |
| BM | primary | **-0.20** | 76 | **+7.20** | +0.18 |
| cfp | primary | 0.68 | 26 | +0.73 | +2.15 |
| Size | primary | 0.65 | 16 | +0.62 | -0.80 |
| SP | +split fix | 0.54 | 26 | +0.22 | +0.58 |
| BM | +split fix | 0.06 | 172 | +0.57 | +0.18 |
| cfp | +split fix | **0.82** | **5** | +2.68 | +2.15 |
| Size | +split fix | **0.80** | **3** | -1.32 | -0.80 |

**Verdict as registered:** price pipeline calibrated (STreversal 0.82 >= 0.70). Fundamentals
pipeline NOT calibrated: BM corr -0.20 < 0.20 triggers "every fundamentals result is suspect";
SP 0.50 misses top-10% rank. Thresholds not moved.
**Diagnosis (post-hoc, labelled as such):** `close` AND `adj_close` in history_3y are split-
adjusted back from today; XBRL share counts are as-reported. Market cap at past date D =
close(D) x shares(D) is wrong by every split after D — later reverse splitters (future losers)
look 10-50x too expensive, later forward splitters look cheap. That manufactured the BM LS of
+7.2%/mo, positive 16/16 months. Correcting with `splits_yf.csv` brings SP/BM LS down to
OSAP's level and lifts cfp/Size to top-5 matches. BM still does not track (OSAP filters
NYSE/AMEX, uses tangible book at FYE ME; we have no exchange code). Residual: STreversal has a
constant -1.8%/mo offset despite 0.82 corr — unexplained, likely $1 filter / CRSP vs Yahoo
closes. Data hygiene needed: 98 stock-months with |ret| > +300% or < -95% (Yahoo mis-adjusted
reverse splits up to +225,000x) dropped as errors.

**Knock-on, measured** (`research/test_split_lookahead.py`, outputs `split_lookahead_*m.parquet`):
the live-book backtest (`value_monthly_backtest.py`) carries the split lookahead AND never had
the live universe filter (domestic 10-K/10-Q filer, alpha ticker, one line per CIK). Its books
held FNMAS, FMCKJ, PSA-PL, PSA-PP, ALL-PH, UHAL-B — preferreds priced at ~$20 against the
parent's share count. 24 months (Sep 2026 now included, -11.7% vs SPY):

| floor | as published | split fix | live universe | both | first-23 both |
|---|---|---|---|---|---|
| $0.5m | +0.91 (t 0.91) | +0.86 | +0.68 | **+0.69 (t 0.68)** | +1.28 (was +1.46) |
| $2m | +0.55 | +0.40 | +0.23 | +0.36 (t 0.32) | +0.94 |
| $10m | -0.36 | -0.57 | -0.35 | -0.46 (t -0.49) | -0.03 |
| $50m | +0.31 | +0.38 | +0.44 | +0.52 (t 0.62) | +0.83 |

Corrected headline: book +22.9%/yr vs SPY +15.6% over 24 months, t 0.68 (dashboard said +37.0%
vs +16.5%, t 1.68). Split lookahead alone is small for long-only (4% of picks, -0.05%/mo); the
universe leak is the bigger piece (-0.22%/mo at $0.5m). The live book has neither flaw — it
prices at today's close and applies the filter. Only the evidence behind it was flattered.
"Edge lives in thin names" (item 3) shrinks: $0.5m-vs-$50m gap 0.95 -> 0.17pp/mo (24 mo).
Item 1's registered threshold (+0.4%/mo forward) is unchanged; note its comparison figure
"backtest +1.47%/mo" is overstated — corrected backtest is +0.69%/mo.

## 7b. Live cfp is not trailing cash flow `open` — needs the owner (live-book change)
Found 2026-10-08 (`research/test_cfp_ttm_defect.py`). 10-Q cash-flow statements are year-to-
date, so frames hold 3-month OCF almost only for Q1. `build_all.ttm()` sums the last four
60-115-day rows per ticker: for OCF **99% of those span >400 days — the same quarter from four
different years** (WMT: four Q1s = $19bn vs real TTM ~$42bn). Revenue and NI are ~TTM (84-90%
span 290-400 days: Q4 missing, one quarter doubled — tolerable). Live cfp vs FY-OCF cfp:
Spearman 0.69, 16% sign flips, 47% off by >50%. In-memory rebuild of rank.py (matches
signals/orders.json 40/40) with cfp on latest 10-K OCF: **15 of 40 live names change**
(out ATNI BDL CIM CLW DK FORR GENC GRNT IPI LCUT MTW MX STRT SXC VYX; in BV BXC COOK ENOV FSUN GLP
HOV KHC MAGN MATV MRDN SHOE SNEX TITN VREX). Same defect hits PctTotAcc and the runway/burn
filter in build_all.py (both use quarterly OCF), and every backtest that used `ttm(OCF)`
(item 2's "cfp is the weakest leg" was measured on the broken construction).
Not fixed: changing build_all.py changes the live book (AGENTS.md hard limit). Options for
The owner: (a) cfp on latest FY OCF, (b) fetch YTD frames and difference to true TTM, (c) drop cfp.

## 9. Live fundamentals are frozen at 2026-08-19 `open`
`signals/build_all.py` hard-codes `TODAY = pd.Timestamp("2026-08-19")` as the no-lookahead cut,
and `data/sec/fundamentals.parquet` max filed_date is 2026-08-17. Either fundamentals are not
being refreshed by the daily job, or they are and the cut discards them. Harmless until Q3 10-Qs
land (late Oct-mid Nov); after that the book ranks on stale accounts while looking healthy.
Check the cron/Actions chain, not just the file. Plumbing, not research — but nobody else is
looking.

## 10. Re-run items 2, 3b on the corrected harness `open`
Everything built on `value_monthly_backtest.py` inherits the universe leak + split lookahead +
broken OCF TTM. Item 3 was re-run here; 2 (component decomposition) and 3b (costs) were not.
Low priority until 7b is decided, since cfp's construction changes the answer.

## 11. Backtest the construction that is actually live `open`
AGENTS.md says "ranked on value only, capped 8 per sector". Live `orders.py` builds COMBINED =
0.7 value + 0.3 skew (px_52w, volume_trend), capped 6, plus filters (runway, listing age,
2-of-3 value signals) the backtest never applied. The headline backtest measures a different
book from the one on the dashboard. Measure the live construction on the corrected harness.
**2026-10-09:** mostly answered by item 12 (live construction +0.45%/mo vs SPY, t 0.47, 24 months, runway filter not replicated).

## 12. Does the live 0.3 skew overlay earn its place? `refuted` 2026-10-09 (family 4, falsification)
`research/test_skew_overlay.py`, output `research/out/skew_overlay.parquet`. The live book (orders.py)
is COMBINED = 0.7 value + 0.3 skew (px_52w, volume_trend), cap 6, but every backtest measured value
only. Rebuilt the corrected harness from item 7 (those scripts are not on disk) and ran the live
construction monthly, 24 chained months (2024-09..2026-08), 40 names, live filters minus runway.
**Registered in the script header before the first run:** control = CORE (same everything, skew
weight 0). Null = 200 overlays with skew replaced by a random sector-relative rank at 0.3 weight.
KEEP-EVIDENCE if COMBINED-CORE >= +0.20%/mo AND p <= 0.10; REFUTED if <= 0; else INCONCLUSIVE.

| book | vs SPY /mo | t | win | ann. book |
|---|---|---|---|---|
| CORE (control) | +0.61% | 0.51 | 58% | +21.4% |
| COMBINED (live) | +0.45% | 0.47 | 54% | +20.5% |
| SPY | | | | +15.6% |
| COMBINED - CORE | **-0.16%** | -0.20 | 50% | overlap 13.2/40 |
| random 0.3 overlays - CORE | mean -0.30%, 5-95% [-1.05, +0.42] | | | p = 0.36 |

**Verdict as registered: REFUTED.** The skew overlay replaces ~27 of 40 names and buys nothing over
value alone; it is no better than a random overlay at the same weight (it dilutes slightly less than
noise, p 0.36). Interpretation: any 30% overlay costs value ~0.3%/mo on average, which is weak evidence
value carries some signal; skew is indistinguishable from noise. Answers most of item 11: the live
construction measures +0.45%/mo t 0.47, below the corrected value-only headline (+0.69%, cap 8).
**Sanity (printed extremes):** first run showed a 5,281,338bn market cap (HXL; also HL, CPK ~1000x,
XBRL share-unit errors) and SMCI 10x too cheap in Nov 2024 (10:1 split between the share-count date
and D). Added a split fix for that gap and a daily turnover bound [1e-5, 0.3] on $vol/mktcap; applies
to control and candidate alike, thresholds unchanged. The verdict was REFUTED both before (-0.17%/mo)
and after (-0.16%) the fix. After the fix: max mktcap 5,452bn (NVDA), n=40 every month, single names
+142%/-60%, 0 book picks lost to the >+300%/<-95% bad-return filter. **The same share-unit errors are
probably in the live build_all.py** (no turnover bound there) and are worth a check.
**Checked same day — confirmed in the live book:** BH (Biglari Holdings, #2 in COMBINED) is scored at
$0.076bn mktcap = $367.70 x ~207k shares, which is the Class A count only; Class B (~2.07m, 1/5 vote)
is missing, so real mktcap is ~$1.1bn and every value ratio is ~15x too cheap. $vol/mktcap 0.60/day
(next-highest in the book 0.03). The only one of the 40 that fails the bound. Not fixed (live-book change).
**Not actioned:** dropping skew changes the live book; needs the owner + forward confirmation.
**Registered forward threshold (2026-10-09), on frozen books dated 2026-10-01 or later only.** Books
frozen before that already have CORE and COMBINED results (08-19: COMBINED +0.53pp ahead; 09-21:
+0.95pp ahead), so they are excluded. At the 2027-03 evaluation (item 1): if the mean monthly forward
COMBINED - CORE is **< 0**, recommend dropping the skew term; if **>= +0.2%/mo**, skew stays; in
between, no change.

## 13. Does sector-relative ranking beat market-wide ranking? `done` 2026-10-11 — BELIEF BROKEN as registered, fragile (family 4)
`research/test_sector_relative.py`, log `research/out/sector_relative.log`, output `sector_relative.parquet`.
rank.py ranks SP/bm/cfp inside sectors ("P/E 12 is cheap for tech, ordinary for a bank") — argued, never
measured. Item-12 harness verbatim (exec of its loader), value-only book, 40 names, cap 8, 24 chained months
(2024-09..2026-08), pre-cost.
**Registered in the script header before the first run:** control = SECREL (sector-relative ranks, as rank.py);
candidate = MKT (each of SP, bm, cfp ranked across the whole eligible universe; cap 8 still applies).
d = mean monthly MKT - SECREL. BELIEF SURVIVES if d <= 0; BELIEF BROKEN if d >= +0.20%/mo AND sign-flip
randomisation p <= 0.10; WEAKENED if 0 < d otherwise; INCONCLUSIVE if < 15 months. Absolute baseline = 200
random cap-8 books per month.

| book | vs SPY /mo | t | win | ann. book |
|---|---|---|---|---|
| SECREL (control) | +0.69% | 0.57 | 58% | +22.5% |
| MKT (candidate) | +1.20% | 0.98 | 62% | +29.6% |
| random cap-8 books | -0.12% | -0.21 | 46% | +13.2% |
| SPY | | | | +15.6% |
| MKT - SECREL | **+0.52%** | 1.56 | 54% | overlap 29.5/40, p 0.065 |

**Verdict as registered: BELIEF BROKEN.** Control reproduces the corrected item-7 headline (+0.69%/mo), so
the harness is consistent.
**Sanity (printed extremes):** n=40 every month in all 4 arms; max mktcap 5,452bn (NVDA); single names +142%
(CRNC) / -60% (CABO); eligible 2,381-2,931; up to 112 bad fwd returns/month dropped (item-7 hygiene, both arms).
**Post-hoc robustness, labelled — cannot change the verdict** (`test_sector_relative_robust.py`, log
`sector_relative_robust.log`). Design flaw found after the run: SECREL ranks over the full pre-filter pool (as
rank.py does), registered MKT over the eligible pool only. Pool-matched: MKT_full - SECREL **+0.47%/mo t 1.23**;
MKT_elig - SECREL_elig **+0.55%/mo t 1.25**. Drop best month: +0.32 / +0.42 (t 0.88 / 0.95). Same sign and size,
but neither pool-matched version would clear p <= 0.10. Swap legs (~10.5 names/month): market-wide ranking
buys Media & telecom, Energy, Consumer discretionary, Real estate, Other (mean +2.2%/mo) and drops Technology,
Materials, Utilities, Biotech, Medical devices, Industrials (mean +0.3%/mo). Not one sector: leave-one-out of the
swap keeps the gap at +0.33..+0.55%/mo; largest single-sector dependence is Media & telecom (VSAT, ECHO, THRY).
**Meaning:** no evidence that ranking inside sectors helps the value book; weak evidence it costs ~0.5%/mo by
forcing "cheapest in an expensive sector" names (tech, utilities, biotech) into the book. Treat as a fragile
single result (finding #2: the screen certifies 10% of noise), not a proven edge.
**Second finding, robust:** the 8-per-sector cap never binds on the value-only sector-relative book (uncapped =
capped in 24/24 months; market-wide 23/24). Banks top out at 5-8. The cap is not load-bearing for this book —
its only live effect is in the COMBINED book at cap 6.
**Not actioned:** ranking method is a live-book change (rank.py); needs the owner + forward confirmation.
**Registered forward threshold (2026-10-11), frozen books dated 2026-10-11 or later only:** at the 2027-03
evaluation (item 1), rebuild each frozen month's value-only book with market-wide ranks from the same snapshot
(as item 3's $10m variant) and compare to the sector-relative book, cap 8, after the item-3b cost model. If mean
monthly forward MKT - SECREL is **>= +0.20%/mo**, recommend market-wide ranking; if **< 0**, sector-relative
stays; in between, no change.

## 14. Rank buffer on exits (sell only below rank 80) `done` 2026-10-12 — BUFFER-SUPPORTED as registered (family 2)
`research/test_rank_buffer.py` (committed 4161718 before the first run), log `research/out/rank_buffer.log`, output
`rank_buffer.parquet`. The book sells any name that drops to rank 41 — never chosen against an alternative. Item-6 harness
verbatim (exec of its loader; same cost + tax model), value-only book, 40 names, cap 8, 24 chained months.
**Registered in the script header before the first run:** control = MONTHLY (fresh top-40 every month, = item 6's
control); candidate = BUFFER80 (monthly; keep a held name while priced and ranked <= 80; fill vacancies from the top,
cap 8 counting kept names; equal-weight reset both arms). BUFFER-SUPPORTED if after-tax annualised D >= +1.0pp AND
pre-tax after-cost B-M >= -0.10%/mo; REFUTED if D <= 0; else INCONCLUSIVE. Diagnostics only: B60, B120, vs quarterly,
null = 200 persistent random signals (AR(1) rho 0.9 — a buffer is meaningless on noise redrawn monthly).

| arm | vs SPY /mo | t | pre-tax ann | after-tax ann (liquidated) | traded/rebal | costs (% W, 2y) |
|---|---|---|---|---|---|---|
| MONTHLY (control) | +0.25% | 0.21 | +17.2% | +8.5% | 49% | 7.4% |
| BUFFER60 (diag) | +0.52% | 0.46 | +21.2% | +11.2% | 33% | 5.2% |
| **BUFFER80** | **+0.55%** | 0.50 | +21.7% | **+11.6%** | 30% | 4.9% |
| BUFFER120 (diag) | +0.71% | 0.63 | +23.8% | +13.0% | 28% | 4.8% |
| QUARTERLY mean of 3 phases (item 6) | +0.48% | | +20.6% | +11.1% | 78-87% | 4.2-4.6% |
| SPY | | | +16.5% | | | |

B80-M pre-tax after cost **+0.30%/mo** (t 1.09, win 54%); **D = +3.07pp/yr** after tax (unliquidated +4.82pp). Drop
month 1: +0.31%/mo; drop best month: +0.18%/mo. B80 vs quarterly: +0.07%/mo, +0.52pp after tax.
Null (persistent noise): B-M mean +0.10%/mo, 5-95% [-0.38, +0.67]; after-tax D mean +1.24pp [-2.45, +5.55]; value book
at the 73rd / 79th percentile — inside the noise band.
**Sanity (printed extremes):** n=40 every month all arms; worst held rank 80 (B80), cap never exceeded (max 8); kept
from last month 36.9 (B80) vs 32.8 (monthly); held-name single-month returns +160% / -90%; control reproduces item 6
to the digit.
**Verdict as registered: BUFFER-SUPPORTED.**
**Meaning (post-hoc, labelled):** ~0.11%/mo is mechanical (costs 7.4% -> 4.9% of W over 2y) and the rest is tax deferral
plus names at ranks 41-80 not underperforming the fresh entrants they would have been swapped for — i.e. the month-to-
month churn at the edge of the top 40 is noise, consistent with item 6 (value does not improve with a fresh look).
Monotone in width (60 < 80 < 120), positive in every variant. Noise books gain too (+0.10%/mo), so like item 6 this is a
cost/tax result, not a signal result; the value book's extra +0.20%/mo over noise is not distinguishable from luck.
Practical point: a buffer captures quarterly's after-tax benefit (+0.5pp better, not significant) while still looking at
the ranking monthly and without quarterly's 5pp/yr phase lottery. Items 6 and 14 are the same lever — pick one, not both.
**Not actioned:** exit rule is a live-book change (orders.py); needs the owner + forward confirmation.
**Registered forward threshold (2026-10-12), frozen books dated 2026-10-12 or later only:** frozen books store only the
top 40, so at the 2027-03 evaluation (item 1) rebuild the value-only ranking from each freeze date's snapshot (as item
3's $10m variant), chain the sell-at-41 book and the BUFFER80 book across consecutive freezes, after the item-3b cost
model. If mean forward B80-M is **>= -0.05%/mo**, recommend the buffer (its cost/tax saving stands); if **< -0.25%/mo**,
sell-at-41 stays; in between, no change.

## 8. The 111 unbuilt drivers `open`
Build only those with a published survivor behind them (the two-source rule makes
the 10% noise rate tolerable). Roughly 36 qualify. Batch them, screen on the real
objective, register thresholds before looking.

---

## Findings (do not delete)
_Cumulative 2026-10-12: 12 candidates tried with a result (items 2, 3, 3b, 6, 7, 12, 13, 14; survivorship, spread screen, concentration, adding drivers) / 3 killed (survivorship, adding drivers, skew overlay) / 1 live (value book). Incumbent beliefs flagged broken: 1 (sector-relative ranking, item 13, fragile)._
- **Rank-80 exit buffer beats sell-at-41 after cost + tax** `done` 2026-10-12. D +3.07pp/yr after tax, pre-tax B80-M +0.30%/mo t 1.09; turnover 49% -> 30%/rebalance; noise books gain too (value at 73rd pct) — a cost/tax lever like item 6, not signal. Item 14, awaiting the owner.
- **Market-wide ranking beat sector-relative ranking** `done` 2026-10-11. MKT-SECREL +0.52%/mo t 1.56 p 0.065 (registered: BELIEF BROKEN); pool-matched +0.47%/mo t 1.23 — fragile. Cap 8 never binds on the value book (24/24). Item 13, awaiting the owner + forward threshold.
- **Quarterly rebalance beats monthly after cost + tax** `done` 2026-10-10. D +2.54pp/yr after tax (positive in all 3 phases), pre-tax Q-M +0.23%/mo t 0.80; gain is mechanical (costs, deferral), random books gain as much. Item 6, awaiting the owner.
- **Live skew overlay adds nothing over value alone** `refuted` 2026-10-09. COMBINED-CORE -0.16%/mo (t -0.20), p 0.36 vs random overlays; live book +0.45%/mo vs SPY, t 0.47. Item 12.
- **Fundamentals pipeline fails OSAP calibration as registered; price pipeline passes** `done`
  2026-10-08. STreversal corr 0.82 (rank 1/198); BM -0.20, SP 0.50. Cause: split-adjusted
  prices x as-reported shares = future-split lookahead in historical market cap. Item 7.
- **Headline backtest overstated** `done` 2026-10-08. Correct universe + no split lookahead,
  24 months: +0.69%/mo t 0.68 (+22.9%/yr vs +15.6%), not +1.46% t 1.67. Item 7.
- **Live cfp sums four Q1s from four years** `open` 2026-10-08. 15/40 names change if fixed.
  Item 7b, awaiting the owner.
- **Trading costs do not explain the thin-name edge** `done` 2026-10-05. Net +1.33%/mo
  at $0.5m vs +1.17% at $2m; breakeven half-spread 1.06%. Roll estimator unusable. Item 3b.
- **Edge lives in thin names** `done` 2026-10-01. Monthly edge +1.46% at $0.5m floor,
  +0.06% (t 0.07) at $10m, +0.57% at $50m. Item 3.
- **Survivorship is not the explanation** `refuted` 2026-09-30. Measured 0.27%/quarter
  attrition; maximum drag 0.27pp against a 20.6pp edge.
- **Quintile spread is the wrong screen** `done` 2026-09-22. Promoted signals that
  measured -4.91% and -5.30% on the long-only objective.
- **Concentration below 40 names hurts** `done` 2026-09-22. Consistent across all six
  score constructions tested.
- **Adding drivers did not help** `done` 2026-09-22. 39 constructions, none beat SPY
  before the objective was corrected.
