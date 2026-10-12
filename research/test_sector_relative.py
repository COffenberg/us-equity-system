"""Falsification (family 4): does sector-relative ranking beat ranking across the whole market?

signals/rank.py ranks every value signal inside its own sector ("a P/E of 12 is cheap for tech
and ordinary for a bank"). That is a load-bearing design choice that was argued, never measured:
no screen has run the same book on market-wide ranks. If market-wide ranks do as well or better,
the belief is wrong; the cap-8 sector constraint would still spread the book either way.

REGISTERED BEFORE THE FIRST RUN (2026-10-11):
  Control   : SECREL -- value-only book, score = mean of sector-relative percentile ranks of
              SP, bm, cfp (sectors with < 25 names pooled as Other, as rank.py), 40 names,
              equal weight, cap 8/sector, monthly. Item-12 harness verbatim (live universe,
              split-corrected mktcap, share-unit turnover bound, bad-return hygiene).
  Candidate : MKT -- identical in every respect except each of SP, bm, cfp is ranked across
              the whole eligible universe instead of inside its sector. Cap 8 still applies.
  Primary   : d = mean monthly (MKT - SECREL) return, pre-cost.
  Null      : sign-flip randomisation of the monthly d series (10,000 draws); p = share of
              draws with mean >= observed |d| on the tested side. Plus 200 random 40-name
              books per month (same universe, cap 8) for the absolute baseline.
  Verdict   : BELIEF SURVIVES   if d <= 0 (sector-relative at least as good).
              BELIEF BROKEN     if d >= +0.20%/mo AND p <= 0.10.
              WEAKENED          if 0 < d, but not BROKEN.
              INCONCLUSIVE      if fewer than 15 months.
  Diagnostic (not gates): both arms uncapped; sector count / max sector weight per arm;
              overlap; turnover per arm (cost differences).
  The verdict never changes the live book on its own (AGENTS.md hard limit).
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

# reuse the item-12 harness verbatim (data loading, universe, month_inputs), not a re-typed copy
_src = (ROOT/"research/test_skew_overlay.py").read_text().split("SKIPPED = ")[0]
exec(compile(_src, "test_skew_overlay_harness", "exec"))
N, CAP8, NRAND = 40, 8, 200

def pick(score, sec, fwd, cap):
    picked, cnt = [], {}
    for t in score.sort_values(ascending=False).index:
        s_ = sec[t]
        if cap and cnt.get(s_, 0) >= cap: continue
        if pd.isna(fwd.get(t)): continue
        picked.append(t); cnt[s_] = cnt.get(s_, 0) + 1
        if len(picked) == N: break
    return picked

months = pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
rng = np.random.default_rng(20261011)
rows, extremes, prev = [], [], {}
for D in months:
    D = PX.index[PX.index.get_indexer([D], method="nearest")[0]]
    nxt = PX.index[PX.index.get_indexer([D+pd.Timedelta(days=30)], method="nearest")[0]]
    if nxt <= D: continue
    out, raw, _ = month_inputs(D)
    fwd = PX.loc[nxt]/PX.loc[D] - 1
    bad = fwd[(fwd > 3) | (fwd < -0.95)].index
    fwd = fwd.drop(bad)
    spy = fwd["SPY"]
    o = out[out.value.notna()]
    rw = raw.loc[o.index, ["SP", "bm", "cfp"]]
    mkt = rw.rank(pct=True).mean(axis=1)                    # market-wide, same NaN handling as mean()
    books = {"SECREL": pick(o.value, o.sector, fwd, CAP8), "MKT": pick(mkt, o.sector, fwd, CAP8),
             "SECREL_nocap": pick(o.value, o.sector, fwd, 0), "MKT_nocap": pick(mkt, o.sector, fwd, 0)}
    row = {"date": D, "eligible": len(o), "spy": spy, "dropped_bad": len(bad), "mc_max_bn": o.mc.max()/1e9,
           "overlap": len(set(books["SECREL"]) & set(books["MKT"]))}
    for nm, bk in books.items():
        x = fwd.reindex(bk); row[nm] = x.mean(); row["n_"+nm] = len(bk)
        sc = o.sector.reindex(bk).value_counts()
        row["nsec_"+nm] = len(sc); row["topsec_"+nm] = f"{sc.index[0]}:{sc.iloc[0]}"
        row["mcmed_"+nm] = o.mc.reindex(bk).median()/1e9
        row["turn_"+nm] = 1 - len(set(bk) & prev.get(nm, set()))/N if nm in prev else np.nan
        prev[nm] = set(bk)
        extremes.append((str(D.date()), nm, x.idxmax(), x.max(), x.idxmin(), x.min()))
    rr = [fwd.reindex(pick(pd.Series(rng.random(len(o)), index=o.index), o.sector, fwd, CAP8)).mean()
          for _ in range(NRAND)]
    row["rand"] = np.mean(rr)
    rows.append(row)
    print(f"{D.date()} elig {len(o):5d} secrel {row['SECREL']-spy:+7.2%} mkt {row['MKT']-spy:+7.2%} "
          f"d {row['MKT']-row['SECREL']:+6.2%} overlap {row['overlap']:2d} "
          f"top {row['topsec_SECREL']} / {row['topsec_MKT']}", flush=True)

R = pd.DataFrame(rows)
tstat = lambda x: x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
ann = lambda r: (1+r).prod()**(12/len(r)) - 1
d = (R.MKT - R.SECREL).values
flips = rng.choice([-1, 1], size=(10000, len(d)))
nm_ = (flips * d).mean(axis=1)
p_mkt = (nm_ >= d.mean()).mean()                           # tested side: MKT better

print("\nSANITY")
print(f"  months {len(R)}, eligible {R.eligible.min()}-{R.eligible.max()}, "
      f"n min {[int(R['n_'+k].min()) for k in ('SECREL','MKT','SECREL_nocap','MKT_nocap')]}, "
      f"bad fwd dropped/month max {R.dropped_bad.max()}")
print(f"  largest market cap seen {R.mc_max_bn.max():,.0f}bn (must be < ~5,500bn)")
ex = pd.DataFrame(extremes, columns=["date", "book", "best", "best_r", "worst", "worst_r"])
print(f"  single-name best {ex.best_r.max():+.0%} ({ex.loc[ex.best_r.idxmax(), 'best']}), "
      f"worst {ex.worst_r.min():+.0%} ({ex.loc[ex.worst_r.idxmin(), 'worst']})")
for k in ("SECREL", "MKT", "SECREL_nocap", "MKT_nocap"):
    print(f"  {k:13s} sectors held median {R['nsec_'+k].median():.0f}  median mktcap {R['mcmed_'+k].median():.2f}bn  "
          f"turnover/mo {R['turn_'+k].mean():.0%}  top-sector by month: {R['topsec_'+k].value_counts().head(3).to_dict()}")
print("\nRESULT (monthly, vs SPY, pre-cost)")
for k in ("SECREL", "MKT", "SECREL_nocap", "MKT_nocap", "rand"):
    x = R[k] - R.spy
    tag = {"SECREL": " (control)", "MKT": " (candidate)", "rand": " (random cap-8 books)"}.get(k, " (diagnostic)")
    print(f"  {k+tag:30s} {x.mean():+.2%}/mo  t {tstat(x):5.2f}  win {(x>0).mean():.0%}  ann book {ann(R[k]):+.1%}")
print(f"  SPY ann {ann(R.spy):+.1%}")
print(f"  MKT - SECREL {d.mean():+.2%}/mo  t {tstat(pd.Series(d)):5.2f}  win {(d>0).mean():.0%}  "
      f"overlap {R.overlap.mean():.1f}/40  sign-flip p(MKT better) = {p_mkt:.3f}")
dn = R.MKT_nocap - R.SECREL_nocap
print(f"  uncapped MKT - SECREL {dn.mean():+.2%}/mo  t {tstat(dn):5.2f}")
for k in ("SECREL", "MKT"):
    c = R[k+"_nocap"] - R[k]
    print(f"  {k}: uncapped - cap8 {c.mean():+.2%}/mo  t {tstat(c):5.2f}")
if len(R) < 15: verdict = "INCONCLUSIVE (n<15)"
elif d.mean() <= 0: verdict = "BELIEF SURVIVES"
elif d.mean() >= 0.002 and p_mkt <= 0.10: verdict = "BELIEF BROKEN"
else: verdict = "WEAKENED"
print(f"\nVERDICT as registered: {verdict}")
R.to_parquet(ROOT/"research/out/sector_relative.parquet")
