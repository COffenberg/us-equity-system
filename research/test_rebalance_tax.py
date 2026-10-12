"""Construction test (BACKLOG item 6): monthly vs quarterly rebalance, after costs and tax.

The book rebalances monthly. That was never chosen against an alternative. Every monthly
rebalance turns over ~49% of the book (item 3b) and every sale realises a gain taxed at up to
42% (on net realised gains, realisation basis, average-cost method). Quarterly holding cuts
turnover and defers tax, but holds a staler signal.

REGISTERED BEFORE THE FIRST RUN (2026-10-10):
  Control   : MONTHLY -- value-only book (mean sector-relative rank of SP, bm, cfp), 40 names,
              equal weight at each rebalance, cap 8/sector, rebalanced every month.
  Candidate : QUARTERLY -- identical ranking, universe, filters, n, cap; rebalanced every 3rd
              month, buy-and-hold drift in between. Run at all 3 phase offsets; the candidate's
              number is the mean of the 3 phases (no picking the lucky phase).
  Both arms : same corrected harness as item 12 (live universe, split-corrected mktcap, share-
              unit turnover bound, bad-return hygiene), same cost model as item 3b (half-spread
              by 60d median ADV bucket + impact 0.7*sigma_d*sqrt(Q/ADV), notional $1m book), same tax model:
              42% on net realised gains per calendar year, average cost basis, losses carried
              forward, tax paid at year end; terminal liquidation taxed at the end.
  Primary   : D = annualised after-cost, after-tax terminal wealth, QUARTERLY minus MONTHLY.
  Verdict   : QUARTERLY-SUPPORTED if D >= +1.0pp/yr AND pre-tax after-cost monthly diff
              (Q - M) >= -0.10%/mo.
              REFUTED (monthly stays) if D <= 0.
              INCONCLUSIVE otherwise, or fewer than 15 months.
  Null      : 200 random sector-relative signals through the same two arms. Reports where the
              value book's pre-tax Q - M sits among noise books: if inside the noise band, the
              cadence effect is mechanical (costs/drift), not signal decay. Diagnostic, not a gate.
  The verdict never changes the live book on its own (AGENTS.md hard limit).
Simplifications (both arms alike): tax paid pro rata from holdings without triggering further
realisations; names with no price after D are held flat at their last price until sold (unknown
delisting returns); runway filter not replicated (as item 12).
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

# reuse the item-12 harness verbatim (data loading, universe, month_inputs), not a re-typed copy
_src = (ROOT/"research/test_skew_overlay.py").read_text().split("SKIPPED = ")[0]
exec(compile(_src, "test_skew_overlay_harness", "exec"))
N, CAP, TAX, AUM, NNULL = 40, 8, 0.42, 1e6, 200

PXF = PX.ffill()                                          # hold flat after last price
RET = PX.pct_change(fill_method=None)

months = pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
G = sorted({PX.index[PX.index.get_indexer([d], method="nearest")[0]] for d in months})
G.append(PX.index[PX.index.get_indexer([G[-1]+pd.Timedelta(days=30)], method="nearest")[0]])
K = len(G) - 1

def halfspread(dv):
    return np.select([dv < 2e6, dv < 1e7, dv < 5e7], [0.005, 0.002, 0.0008], 0.0003)

STEP, NO_PX = [], 0
for k in range(K):
    D, nxt = G[k], G[k+1]
    out, raw, _ = month_inputs(D)
    v = out.value.dropna(); o = out.loc[v.index]
    o = o[PX.loc[D].reindex(o.index).notna()]              # pickable = priced at D (no lookahead)
    dv = DV.loc[:D].iloc[-60:].median()
    sig = RET.loc[:D].iloc[-60:].std()
    r = PXF.loc[nxt]/PXF.loc[D] - 1
    bad = r[(r > 3) | (r < -0.95)].index
    r[bad] = 0.0                                           # Yahoo mis-adjustments, both arms
    STEP.append({"D": D, "o": o, "r": r, "dv": dv, "sig": sig, "spy": PX.loc[nxt, "SPY"]/PX.loc[D, "SPY"]-1,
                 "nbad": len(bad)})
    print(f"{D.date()} eligible {len(o):5d} bad {len(bad)}", flush=True)

def pick(score, sec):
    picked, cnt = [], {}
    for t in score.sort_values(ascending=False).index:
        s_ = sec[t]
        if cnt.get(s_, 0) >= CAP: continue
        picked.append(t); cnt[s_] = cnt.get(s_, 0) + 1
        if len(picked) == N: break
    return picked

def simulate(picks, rebal):
    """picks[k] = list of names; rebal = set of k at which to rebalance. Returns dict."""
    val, basis = {}, {}                                   # ticker -> value, cost basis (wealth units)
    W, real, carry, taxpaid, costs, turn = 1.0, 0.0, 0.0, 0.0, 0.0, []
    monthly = []
    for k in range(K):
        st = STEP[k]
        W0 = sum(val.values()) if val else 1.0
        if k in rebal:
            tgt_names = picks[k]
            dv, sig = st["dv"], st["sig"]
            # cost on the trade list, then shrink targets by the cost
            tgt = {t: W0/len(tgt_names) for t in tgt_names}
            names = set(val) | set(tgt)
            delta = {t: tgt.get(t, 0.0) - val.get(t, 0.0) for t in names}
            c = 0.0
            for t, d in delta.items():
                if abs(d) < 1e-12: continue
                a = dv.get(t, np.nan); a = a if a == a and a > 0 else 1e5
                s_ = sig.get(t, np.nan); s_ = s_ if s_ == s_ else 0.05
                c += abs(d) * (float(halfspread(a)) + 0.7*s_*np.sqrt(abs(d)*AUM/a))
            turn.append(sum(abs(d) for d in delta.values())/W0); costs += c
            scale = (W0 - c)/W0
            for t in names:                               # realise gains on reductions (avg cost)
                cur, new = val.get(t, 0.0), tgt.get(t, 0.0)*scale
                if new < cur and cur > 0:
                    frac = (cur - new)/cur
                    real += (cur - new) - basis[t]*frac; basis[t] *= (1 - frac)
                elif new > cur:
                    basis[t] = basis.get(t, 0.0) + (new - cur)
                if new > 0: val[t] = new
                else: val.pop(t, None); basis.pop(t, None)
        Wb = sum(val.values())
        for t in list(val):
            rr = st["r"].get(t, np.nan); val[t] *= 1 + (0.0 if rr != rr else rr)
        W1 = sum(val.values())
        monthly.append(W1/W0 - 1)                         # after costs, pre-tax
        if G[k+1].year != G[k].year:                      # year-end tax settlement
            gain = real + carry
            if gain > 0:
                tax = TAX*gain; taxpaid += tax; carry = 0.0
                f = 1 - tax/W1
                for t in val: val[t] *= f; basis[t] *= f
            else: carry = gain
            real = 0.0
    W = sum(val.values())
    pre = np.prod([1+m for m in monthly])
    unreal = sum(val[t] - basis[t] for t in val)
    termtax = TAX*max(0.0, real + carry + unreal)
    yrs = K/12
    return {"monthly": np.array(monthly), "pre_ann": pre**(1/yrs)-1, "W": W,
            "post_ann": ((W - termtax))**(1/yrs)-1, "post_noliq_ann": W**(1/yrs)-1,
            "taxpaid": taxpaid, "termtax": termtax, "costs": costs,
            "turn": np.mean(turn), "nreb": len(turn)}

def arms(picks):
    m = simulate(picks, set(range(K)))
    q = [simulate(picks, {0} | {k for k in range(K) if k % 3 == ph}) for ph in range(3)]
    return m, q

VALUE_PICKS = [pick(STEP[k]["o"].value, STEP[k]["o"].sector) for k in range(K)]
M, Q = arms(VALUE_PICKS)
spy = np.array([s["spy"] for s in STEP])
tstat = lambda x: x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))

print("\nSANITY")
print(f"  months {K}, eligible {min(len(s['o']) for s in STEP)}-{max(len(s['o']) for s in STEP)}, "
      f"picks/month min {min(len(p) for p in VALUE_PICKS)}, bad returns zeroed/month max {max(s['nbad'] for s in STEP)}")
held = set(sum(VALUE_PICKS, []))
rr = pd.concat([s["r"].reindex(list(held)) for s in STEP]).dropna()
print(f"  held-name single-month return extremes {rr.max():+.0%} / {rr.min():+.0%}")
stale = sum(1 for k in range(1, K) for t in VALUE_PICKS[k-1] if pd.isna(PX.loc[G[k]].get(t)))
print(f"  monthly-arm names with no price at next date (held flat): {stale}")
for nm, a in [("MONTHLY", M)] + [(f"QUARTERLY ph{i}", q) for i, q in enumerate(Q)]:
    print(f"  {nm:13s} rebalances {a['nreb']:2d}  traded/rebal {a['turn']:.0%}  costs {a['costs']:.2%} of W  "
          f"tax paid in-window {a['taxpaid']:.2%}  terminal tax {a['termtax']:.2%}  "
          f"monthly r extremes {a['monthly'].max():+.1%}/{a['monthly'].min():+.1%}")

print("\nRESULT (24-ish chained months; after costs)")
print(f"  SPY ann {np.prod(1+spy)**(12/K)-1:+.1%}")
def line(nm, a):
    ex = a["monthly"] - spy
    print(f"  {nm:13s} vs SPY {ex.mean():+.2%}/mo t {tstat(ex):5.2f}  pre-tax ann {a['pre_ann']:+.1%}  "
          f"after-tax ann (liquidated) {a['post_ann']:+.1%}  (unliquidated {a['post_noliq_ann']:+.1%})")
line("MONTHLY", M)
for i, q in enumerate(Q): line(f"QUARTERLY ph{i}", q)
qm = np.mean([q["monthly"] for q in Q], axis=0)
diff = qm - M["monthly"]
D_ = np.mean([q["post_ann"] for q in Q]) - M["post_ann"]
D_noliq = np.mean([q["post_noliq_ann"] for q in Q]) - M["post_noliq_ann"]
Dpre = np.mean([q["pre_ann"] for q in Q]) - M["pre_ann"]
print(f"  Q - M pre-tax after-cost {diff.mean():+.2%}/mo  t {tstat(diff):5.2f}  win {(diff>0).mean():.0%}")
print(f"  Q - M annualised: pre-tax {Dpre*100:+.2f}pp  after-tax liquidated D = {D_*100:+.2f}pp  "
      f"unliquidated {D_noliq*100:+.2f}pp  (per phase D: "
      + ", ".join(f"{(q['post_ann']-M['post_ann'])*100:+.2f}" for q in Q) + ")")

rng = np.random.default_rng(20261010)
nd, nD = [], []
for i in range(NNULL):
    pk = []
    for k in range(K):
        o = STEP[k]["o"]
        big = o.sector.value_counts()[lambda x: x >= MIN_SECTOR].index
        grp = o.sector.where(o.sector.isin(big), "Other")
        z = pd.Series(rng.normal(size=len(o)), index=o.index).groupby(grp).rank(pct=True)
        pk.append(pick(z, o.sector))
    m, q = arms(pk)
    nd.append((np.mean([x["monthly"] for x in q], axis=0) - m["monthly"]).mean())
    nD.append(np.mean([x["post_ann"] for x in q]) - m["post_ann"])
nd, nD = np.array(nd), np.array(nD)
print(f"\nNOISE NULL ({NNULL} random signals, same two arms)")
print(f"  pre-tax Q - M: mean {nd.mean():+.2%}/mo, 5-95% [{np.quantile(nd,.05):+.2%}, {np.quantile(nd,.95):+.2%}]  "
      f"value book percentile {(nd <= diff.mean()).mean():.0%}")
print(f"  after-tax D:   mean {nD.mean()*100:+.2f}pp, 5-95% [{np.quantile(nD,.05)*100:+.2f}, {np.quantile(nD,.95)*100:+.2f}]  "
      f"value book percentile {(nD <= D_).mean():.0%}")

if K < 15: verdict = "INCONCLUSIVE (n<15)"
elif D_ <= 0: verdict = "REFUTED (monthly stays)"
elif D_ >= 0.01 and diff.mean() >= -0.001: verdict = "QUARTERLY-SUPPORTED"
else: verdict = "INCONCLUSIVE"
print(f"\nVERDICT as registered: {verdict}")
pd.DataFrame({"date": G[:-1], "spy": spy, "monthly": M["monthly"], "quarterly_mean": qm,
              **{f"q_ph{i}": q["monthly"] for i, q in enumerate(Q)}}).to_parquet(ROOT/"research/out/rebalance_tax.parquet")
