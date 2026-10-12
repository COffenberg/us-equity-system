"""Construction test (BACKLOG item 14): a rank buffer on exits, after costs and tax.

The book sells any name that drops out of the top 40, even to rank 41. That exit rule was never
chosen against an alternative. Item 3b measured ~49% of the book traded every month and item 6
found the quarterly cadence wins after cost + tax for purely mechanical reasons. A rank buffer is
the standard way to get the cost/tax saving while keeping a monthly look at the signal: keep a
held name until it falls out of the top B, buy only to fill vacancies.

REGISTERED BEFORE THE FIRST RUN (2026-10-12):
  Control   : MONTHLY -- value-only book (mean sector-relative rank of SP, bm, cfp), 40 names,
              equal weight at each rebalance, cap 8/sector, every month a fresh top-40.
              Identical to item 6's control (same harness, same cost and tax model).
  Candidate : BUFFER80 -- same ranking, universe, filters, n, cap, monthly rebalance; a held name
              is kept while it is priced at D and its value score ranks in the top 80 of the
              eligible pool; vacancies are filled from the top of the ranking, cap 8 counting
              the kept names. Equal-weight reset at every rebalance, as the control.
  Primary   : D = annualised after-cost, after-tax (terminal liquidation) wealth, BUFFER80 minus
              MONTHLY.
  Verdict   : BUFFER-SUPPORTED if D >= +1.0pp/yr AND pre-tax after-cost monthly diff (B - M)
              >= -0.10%/mo.
              REFUTED (sell-at-41 stays) if D <= 0.
              INCONCLUSIVE otherwise, or fewer than 15 months.
  Diagnostics (cannot change the verdict): B=60 and B=120 widths; BUFFER80 vs item 6's
              QUARTERLY (mean of 3 phases); noise null = 200 PERSISTENT random sector-relative
              signals (AR(1), rho 0.9 per month -- a buffer is meaningless on noise redrawn each
              month) through MONTHLY and BUFFER80, reporting the value book's percentile.
  The verdict never changes the live book on its own (AGENTS.md hard limit).
Simplifications as item 6: tax paid pro rata without further realisations; names unpriced after D
held flat; runway filter not replicated.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

# item 6 harness verbatim up to (not including) its value run: STEP, pick, simulate, arms
_src = (ROOT/"research/test_rebalance_tax.py").read_text().split("VALUE_PICKS = ")[0]
exec(compile(_src, "test_rebalance_tax_harness", "exec"))
B_PRIMARY, NULL_RHO = 80, 0.9

def buffer_picks(scores, B):
    """scores[k] = Series (eligible, priced at D). Sell only when a held name leaves the top B."""
    out, held = [], []
    for k in range(K):
        sc, sec = scores[k], STEP[k]["o"].sector
        rk = sc.rank(ascending=False, method="first")
        keep = [t for t in held if t in rk.index and rk[t] <= B]
        cnt = {}
        for t in keep: cnt[sec[t]] = cnt.get(sec[t], 0) + 1
        picked = list(keep)
        for t in sc.sort_values(ascending=False).index:
            if len(picked) >= N: break
            if t in picked or cnt.get(sec[t], 0) >= CAP: continue
            picked.append(t); cnt[sec[t]] = cnt.get(sec[t], 0) + 1
        out.append(picked); held = picked
    return out

VSC = [STEP[k]["o"].value for k in range(K)]
VALUE_PICKS = [pick(VSC[k], STEP[k]["o"].sector) for k in range(K)]
ALL = set(range(K))
M = simulate(VALUE_PICKS, ALL)
BP = {b: buffer_picks(VSC, b) for b in (60, B_PRIMARY, 120)}
BS = {b: simulate(p, ALL) for b, p in BP.items()}
Q = [simulate(VALUE_PICKS, {0} | {k for k in range(K) if k % 3 == ph}) for ph in range(3)]
spy = np.array([s["spy"] for s in STEP])
tstat = lambda x: x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))

print("\nSANITY")
print(f"  months {K}, eligible {min(len(s['o']) for s in STEP)}-{max(len(s['o']) for s in STEP)}")
for b, p in BP.items():
    nn = [len(x) for x in p]
    kept = [len(set(p[k]) & set(p[k-1])) for k in range(1, K)]
    worst = max(STEP[k]["o"].value.rank(ascending=False, method="first").reindex(p[k]).max() for k in range(K))
    sc = max(pd.Series([STEP[k]["o"].sector[t] for t in p[k]]).value_counts().max() for k in range(K))
    print(f"  BUFFER{b:<3d} names/month {min(nn)}-{max(nn)}  kept from last month mean {np.mean(kept):.1f}  "
          f"worst held rank {worst:.0f}  max per sector {sc}")
kept = [len(set(VALUE_PICKS[k]) & set(VALUE_PICKS[k-1])) for k in range(1, K)]
print(f"  MONTHLY    names/month {min(map(len, VALUE_PICKS))}-{max(map(len, VALUE_PICKS))}  kept from last month mean {np.mean(kept):.1f}")
held = set(sum(BP[B_PRIMARY], []))
rr = pd.concat([s["r"].reindex(list(held)) for s in STEP]).dropna()
print(f"  BUFFER80 held-name single-month return extremes {rr.max():+.0%} / {rr.min():+.0%}")
for nm, a in [("MONTHLY", M)] + [(f"BUFFER{b}", s) for b, s in BS.items()] + [(f"QUARTERLY ph{i}", q) for i, q in enumerate(Q)]:
    print(f"  {nm:13s} rebalances {a['nreb']:2d}  traded/rebal {a['turn']:.0%}  costs {a['costs']:.2%} of W  "
          f"tax paid in-window {a['taxpaid']:.2%}  terminal tax {a['termtax']:.2%}  "
          f"monthly r extremes {a['monthly'].max():+.1%}/{a['monthly'].min():+.1%}")

print("\nRESULT (chained months; after costs)")
print(f"  SPY ann {np.prod(1+spy)**(12/K)-1:+.1%}")
def line(nm, a):
    ex = a["monthly"] - spy
    print(f"  {nm:13s} vs SPY {ex.mean():+.2%}/mo t {tstat(ex):5.2f}  pre-tax ann {a['pre_ann']:+.1%}  "
          f"after-tax ann (liquidated) {a['post_ann']:+.1%}  (unliquidated {a['post_noliq_ann']:+.1%})")
line("MONTHLY", M)
for b, s in BS.items(): line(f"BUFFER{b}", s)
for i, q in enumerate(Q): line(f"QUARTERLY ph{i}", q)
B8 = BS[B_PRIMARY]
diff = B8["monthly"] - M["monthly"]
D_ = B8["post_ann"] - M["post_ann"]
print(f"  B80 - M pre-tax after-cost {diff.mean():+.2%}/mo  t {tstat(diff):5.2f}  win {(diff>0).mean():.0%}")
print(f"  B80 - M annualised: pre-tax {(B8['pre_ann']-M['pre_ann'])*100:+.2f}pp  after-tax liquidated D = {D_*100:+.2f}pp  "
      f"unliquidated {(B8['post_noliq_ann']-M['post_noliq_ann'])*100:+.2f}pp")
d1 = diff[1:]
print(f"  drop month 1 (identical build): {d1.mean():+.2%}/mo t {tstat(d1):5.2f};  drop best month: "
      f"{np.sort(diff)[:-1].mean():+.2%}/mo")
for b in (60, 120):
    print(f"  diag B{b} - M: pre-tax {(BS[b]['monthly']-M['monthly']).mean():+.2%}/mo  after-tax D {(BS[b]['post_ann']-M['post_ann'])*100:+.2f}pp")
qm = np.mean([q["monthly"] for q in Q], axis=0)
print(f"  diag B80 - QUARTERLY(mean ph): pre-tax {(B8['monthly']-qm).mean():+.2%}/mo  "
      f"after-tax {(B8['post_ann']-np.mean([q['post_ann'] for q in Q]))*100:+.2f}pp")

rng = np.random.default_rng(20261012)
nd, nD = [], []
for i in range(NNULL):
    z, sc = pd.Series(dtype=float), []
    for k in range(K):
        o = STEP[k]["o"]
        e = pd.Series(rng.normal(size=len(o)), index=o.index)
        prev = z.reindex(o.index)
        z = (NULL_RHO*prev + np.sqrt(1-NULL_RHO**2)*e).fillna(e)
        big = o.sector.value_counts()[lambda x: x >= MIN_SECTOR].index
        grp = o.sector.where(o.sector.isin(big), "Other")
        sc.append(z.groupby(grp).rank(pct=True))
    m = simulate([pick(sc[k], STEP[k]["o"].sector) for k in range(K)], ALL)
    b = simulate(buffer_picks(sc, B_PRIMARY), ALL)
    nd.append((b["monthly"] - m["monthly"]).mean()); nD.append(b["post_ann"] - m["post_ann"])
nd, nD = np.array(nd), np.array(nD)
print(f"\nNOISE NULL ({NNULL} persistent random signals, rho {NULL_RHO}, MONTHLY vs BUFFER80)")
print(f"  pre-tax B - M: mean {nd.mean():+.2%}/mo, 5-95% [{np.quantile(nd,.05):+.2%}, {np.quantile(nd,.95):+.2%}]  "
      f"value book percentile {(nd <= diff.mean()).mean():.0%}")
print(f"  after-tax D:   mean {nD.mean()*100:+.2f}pp, 5-95% [{np.quantile(nD,.05)*100:+.2f}, {np.quantile(nD,.95)*100:+.2f}]  "
      f"value book percentile {(nD <= D_).mean():.0%}")

if K < 15: verdict = "INCONCLUSIVE (n<15)"
elif D_ <= 0: verdict = "REFUTED (sell-at-41 stays)"
elif D_ >= 0.01 and diff.mean() >= -0.001: verdict = "BUFFER-SUPPORTED"
else: verdict = "INCONCLUSIVE"
print(f"\nVERDICT as registered: {verdict}")
pd.DataFrame({"date": G[:-1], "spy": spy, "monthly": M["monthly"], "quarterly_mean": qm,
              **{f"buffer{b}": s["monthly"] for b, s in BS.items()}}).to_parquet(ROOT/"research/out/rank_buffer.parquet")
