"""Falsification: does the 0.3 skew overlay in the LIVE book earn its place?

The live order book (orders.py) is COMBINED = 0.7 * value + 0.3 * skew, where
skew = mean sector-relative rank of px_to_52w_high and volume_trend. AGENTS.md and the
dashboard describe the book as "value only"; every backtest so far measured value only.
The skew term has never been screened on the real objective. It is a live belief.

REGISTERED BEFORE THE FIRST RUN (2026-10-09):
  Candidate : COMBINED book, 0.7 value + 0.3 skew, 40 names, equal weight, cap 6/sector,
              monthly, live filters (see below).
  Control   : CORE book -- identical universe, filters, cap, n, dates; skew weight 0.
  Null      : 200 overlays where the skew term is replaced by a random sector-relative
              rank at the same 0.3 weight. p = share of noise overlays whose mean monthly
              (book - CORE) is >= the real skew overlay's.
  Verdict   : KEEP-EVIDENCE if mean(COMBINED - CORE) >= +0.20%/mo AND p <= 0.10.
              REFUTED       if mean(COMBINED - CORE) <= 0.
              INCONCLUSIVE  otherwise, or if fewer than 15 months.
  The verdict never changes the live book on its own (AGENTS.md hard limit).

Harness = corrected one from BACKLOG item 7 (rebuilt here; the 10-08 scripts are not on
disk): live universe (domestic 10-K/10-Q filer as of D, alpha ticker, most-traded line per
CIK) and split-corrected market cap (split-adjusted close x as-reported shares is wrong by
every split after D; multiply close by the product of later split ratios).
Value is built the way the live book builds it, including its TTM-OCF defect (item 7b),
because the point is to test the book that is live.
Not replicated: the runway filter (needs cash/burn history) -- noted in the result.
Harness hygiene added after the first run printed a 5,281,338bn market cap (applies to control
and candidate alike; registered thresholds unchanged): split-between-filing-and-D share fix,
and a daily-turnover bound [1e-5, 0.3] on dollar volume / market cap.
"""
import pandas as pd, numpy as np, pathlib, re, sys, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
N, CAP, MINLIQ, MIN_SECTOR, W_SKEW, NNULL = 40, 6, 5e5, 25, 0.3, 200

F = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
F = F[F.filed_date.notna()]
F["filed_date"] = pd.to_datetime(F.filed_date); F["period_end"] = pd.to_datetime(F.period_end)
F["dur"] = (F.period_end - pd.to_datetime(F.period_start)).dt.days
F["dom"] = F.form.astype("string").str.startswith(("10-K", "10-Q")).fillna(False)
SH = pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
SH["filed_date"] = pd.to_datetime(SH.filed_date); SH["period_end"] = pd.to_datetime(SH.period_end)
SEC = pd.read_parquet(ROOT/"data/sec/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector
U = pd.read_csv(sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1], dtype={"cik": str}) \
      .drop_duplicates("ticker")
U = U[~U.ticker.str.contains(r"[^A-Z]", na=True)]
CIK = U.set_index("ticker").cik
TITLE = U.set_index("ticker").title.fillna("").str.upper()
IS_FUND = TITLE.str.contains(r"\bFUND\b|\bETF\b|\bINDEX\b|\bTRUST\b|\bPORTFOLIO\b|SHARES ETF", regex=True)

H = pd.read_parquet(ROOT/"data/prices/history_3y.parquet"); H["date"] = pd.to_datetime(H.date)
PX = H.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
CL = H.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
DV = (H.assign(dv=H.close*H.volume)
       .pivot_table(index="date", columns="ticker", values="dv", aggfunc="last").sort_index())
FIRST = PX.apply(lambda s: s.first_valid_index())
SPL = pd.read_csv(ROOT/"data/prices/splits_yf.csv", parse_dates=["date"])

def later_split_factor(D):
    s = SPL[SPL.date > D]
    return s.groupby("ticker").ratio.prod()

def ttm(f, c, n=4):
    d = f[(f.concept == c) & (~f.instantaneous) & f.dur.between(60, 115)].sort_values("period_end") \
         .drop_duplicates(["ticker", "period_end"], keep="last").groupby("ticker").tail(n)
    g = d.groupby("ticker").val.agg(["sum", "count"]); return g.loc[g["count"] == n, "sum"]
def inst(f, c):
    d = f[(f.concept == c) & f.instantaneous].sort_values("period_end"); return d.groupby("ticker").last().val

def sector_rank(df, sec):
    big = sec.value_counts()[lambda x: x >= MIN_SECTOR].index
    grp = sec.where(sec.isin(big), "Other")
    return pd.DataFrame({c: df[c].groupby(grp).rank(pct=True).fillna(df[c].rank(pct=True))
                         for c in df.columns})

BADTO = []
def month_inputs(D):
    f = F[F.filed_date <= D]
    dv60 = DV.loc[:D].iloc[-60:].median()
    dom = set(f.loc[f.dom, "ticker"])
    u = pd.DataFrame({"cik": CIK, "dv": dv60.reindex(CIK.index)})
    primary = set(u.sort_values("dv", ascending=False).drop_duplicates("cik").index)
    keep = sorted(dom & primary)
    f = f[f.ticker.isin(keep)]; sh = SH[(SH.filed_date <= D) & SH.ticker.isin(keep)]
    close = CL.loc[:D].iloc[-1].reindex(keep)
    fac = later_split_factor(D).reindex(keep).fillna(1.0)
    d = f[(f.concept == "EntityCommonStockSharesOutstanding") & f.instantaneous].sort_values("period_end")
    s = d.groupby("ticker").last()[["val", "period_end"]]
    for c in ("CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding",
              "CommonStockSharesIssued"):
        d = sh[sh.concept == c].sort_values("period_end"); s = s.combine_first(d.groupby("ticker").last()[["val", "period_end"]])
    # splits between the share-count date and D: count is pre-split, price is post-split
    # (SMCI 10:1 on 2024-10-01 looked 10x too cheap in Nov 2024 without this)
    m = SPL[(SPL.date <= D)].merge(s.period_end.rename("pe"), left_on="ticker", right_index=True)
    mid = m[m.date > m.pe].groupby("ticker").ratio.prod()
    shares = (s.val * mid.reindex(s.index).fillna(1.0)).reindex(keep)
    mc = (close * fac * shares).dropna(); mc = mc[mc > 0]
    # unit errors in XBRL share counts (HXL 5,000,000bn, HL/CPK ~1000x, BH dual class):
    # daily $ volume / market cap outside [1e-5, 0.3] is not a real company
    to = dv60.reindex(mc.index) / mc
    BADTO.append(sorted(to[(to < 1e-5) | (to > 0.3)].index))
    mc = mc[(to >= 1e-5) & (to <= 0.3)]
    rev = ttm(f, "Revenues").combine_first(ttm(f, "RevenueFromContractWithCustomerExcludingAssessedTax"))
    ocf = ttm(f, "NetCashProvidedByUsedInOperatingActivities")
    eq = inst(f, "StockholdersEquity")
    raw = pd.DataFrame({"SP": rev/mc, "bm": eq/mc, "cfp": ocf/mc}).reindex(mc.index)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    # skew inputs, as recorder/derive_price_signals.py builds them
    p = PX.loc[:D]
    raw["px_52w"] = (p.iloc[-1] / p.iloc[-252:].max()).reindex(raw.index)
    raw["volume_trend"] = (dv60 / DV.loc[:D].iloc[-250:].median().replace(0, np.nan)).reindex(raw.index)
    sec = SEC.reindex(raw.index).fillna("Other")
    R = sector_rank(raw, sec)
    nval = raw[["SP", "bm", "cfp"]].notna().sum(axis=1)
    age = (D - FIRST.reindex(raw.index)).dt.days
    adj = PX.loc[:D].iloc[-1].reindex(raw.index)
    ok = ((dv60.reindex(raw.index).fillna(0) >= MINLIQ) & (adj >= 1) & (mc.reindex(raw.index) >= 50e6)
          & ((age >= 250) | (FIRST.reindex(raw.index) <= PX.index[0])) & (nval >= 2)
          & ~IS_FUND.reindex(raw.index).fillna(False))
    out = pd.DataFrame({"value": R[["SP", "bm", "cfp"]].mean(axis=1),
                        "skw": R[["px_52w", "volume_trend"]].mean(axis=1),
                        "sector": sec, "mc": mc.reindex(raw.index)})[ok]
    return out, raw[ok], sector_rank(pd.DataFrame(index=out.index), out.sector)

SKIPPED = {"CORE": [], "COMB": []}
def pick(score, sec, fwd, tag=None, bad=()):
    picked, cnt = [], {}
    for t in score.sort_values(ascending=False).index:
        s_ = sec[t]
        if cnt.get(s_, 0) >= CAP: continue
        if pd.isna(fwd.get(t)):
            if tag and t in bad: SKIPPED[tag].append(t)
            continue
        picked.append(t); cnt[s_] = cnt.get(s_, 0) + 1
        if len(picked) == N: break
    return picked

months = pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
rng = np.random.default_rng(20261009)
rows, nullrows, extremes = [], [], []
for D in months:
    D = PX.index[PX.index.get_indexer([D], method="nearest")[0]]
    nxt = PX.index[PX.index.get_indexer([D+pd.Timedelta(days=30)], method="nearest")[0]]
    if nxt <= D: continue
    out, raw, _ = month_inputs(D)
    fwd = PX.loc[nxt]/PX.loc[D] - 1
    bad = fwd[(fwd > 3) | (fwd < -0.95)].index           # Yahoo mis-adjustments (item 7 hygiene)
    fwd = fwd.drop(bad)
    spy = fwd["SPY"]
    v = out.value.dropna(); o = out.loc[v.index]
    core = pick(o.value, o.sector, fwd, "CORE", set(bad))
    comb = pick(0.7*o.value + 0.3*o.skw, o.sector, fwd, "COMB", set(bad))
    rc, rb = fwd.reindex(core).mean(), fwd.reindex(comb).mean()
    for nm, bk in (("CORE", core), ("COMB", comb)):
        x = fwd.reindex(bk); extremes.append((str(D.date()), nm, x.idxmax(), x.max(), x.idxmin(), x.min()))
    rows.append({"date": D, "eligible": len(o), "n_core": len(core), "n_comb": len(comb),
                 "overlap": len(set(core) & set(comb)), "core": rc, "comb": rb, "spy": spy,
                 "dropped_bad": len(bad), "mc_max_bn": o.mc.max()/1e9,
                 "core_mc_med_bn": o.mc.reindex(core).median()/1e9,
                 "comb_mc_med_bn": o.mc.reindex(comb).median()/1e9})
    # noise overlays: random sector-relative rank at the same weight
    big = o.sector.value_counts()[lambda x: x >= MIN_SECTOR].index
    grp = o.sector.where(o.sector.isin(big), "Other")
    nr = []
    for i in range(NNULL):
        z = pd.Series(rng.normal(size=len(o)), index=o.index).groupby(grp).rank(pct=True)
        nr.append(fwd.reindex(pick(0.7*o.value + 0.3*z, o.sector, fwd)).mean() - rc)
    nullrows.append(nr)
    print(f"{str(D.date())} elig {len(o):5d} core {rc-spy:+7.2%} comb {rb-spy:+7.2%} "
          f"diff {rb-rc:+6.2%} overlap {len(set(core)&set(comb)):2d} bad {len(bad)}", flush=True)

R = pd.DataFrame(rows); NUL = np.array(nullrows)            # months x NNULL
R["core_ex"] = R.core - R.spy; R["comb_ex"] = R.comb - R.spy; R["diff"] = R.comb - R.core
tstat = lambda x: x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
null_means = NUL.mean(axis=0)
p = (null_means >= R["diff"].mean()).mean()
ann = lambda r: (1+r).prod()**(12/len(r)) - 1
print("\nSANITY")
print(f"  months {len(R)}, eligible {R.eligible.min()}-{R.eligible.max()}, n_core min {R.n_core.min()}, "
      f"n_comb min {R.n_comb.min()}, bad fwd dropped/month max {R.dropped_bad.max()}")
print(f"  turnover-bound exclusions/month max {max(len(b) for b in BADTO)}: e.g. {sorted(set(sum(BADTO, [])))[:20]}")
print(f"  would-be picks skipped for bad fwd return: CORE {SKIPPED['CORE']}  COMB {SKIPPED['COMB']}")
print(f"  largest market cap seen {R.mc_max_bn.max():,.0f}bn (must be < ~5,000bn)")
print(f"  median book mktcap  core {R.core_mc_med_bn.median():.2f}bn  comb {R.comb_mc_med_bn.median():.2f}bn")
ex = pd.DataFrame(extremes, columns=["date", "book", "best", "best_r", "worst", "worst_r"])
print(f"  single-name best {ex.best_r.max():+.0%} ({ex.loc[ex.best_r.idxmax(), 'best']}), "
      f"worst {ex.worst_r.min():+.0%} ({ex.loc[ex.worst_r.idxmin(), 'worst']})")
print("\nRESULT (monthly, vs SPY)")
for nm, c in (("CORE (control)", "core_ex"), ("COMBINED (live)", "comb_ex")):
    x = R[c]; print(f"  {nm:16s} {x.mean():+.2%}/mo  t {tstat(x):5.2f}  win {(x>0).mean():.0%}  "
                    f"ann book {ann(R[c.split('_')[0]]):+.1%}")
print(f"  SPY ann {ann(R.spy):+.1%}")
d = R["diff"]
print(f"  COMBINED - CORE {d.mean():+.2%}/mo  t {tstat(d):5.2f}  win {(d>0).mean():.0%}  "
      f"mean overlap {R.overlap.mean():.1f}/40")
print(f"  noise overlays: mean {null_means.mean():+.2%}, sd {null_means.std():.2%}, "
      f"5-95% [{np.quantile(null_means,.05):+.2%}, {np.quantile(null_means,.95):+.2%}]  p = {p:.2f}")
if len(R) < 15: verdict = "INCONCLUSIVE (n<15)"
elif d.mean() <= 0: verdict = "REFUTED"
elif d.mean() >= 0.002 and p <= 0.10: verdict = "KEEP-EVIDENCE"
else: verdict = "INCONCLUSIVE"
print(f"\nVERDICT as registered: {verdict}")
R.drop(columns=[]).to_parquet(ROOT/"research/out/skew_overlay.parquet")
