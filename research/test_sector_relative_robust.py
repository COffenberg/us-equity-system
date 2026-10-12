"""POST-HOC robustness for test_sector_relative.py (written 2026-10-11 AFTER seeing its result).

Not a registered test; cannot change the registered verdict. Checks whether the result is an
artefact of a ranking-pool mismatch in the registered design: SECREL (as in rank.py) ranks inside
sectors over the FULL pre-filter pool; the registered MKT ranked over the ELIGIBLE pool only.
  MKT_full   : market-wide ranks over the same full pre-filter pool as SECREL (pool-matched).
  SECREL_elig: sector-relative ranks over the eligible pool only (pool-matched to registered MKT).
Also: which names drive the gap (swap legs), drop-best-month, leave-one-sector-out of the swap.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
_src = (ROOT/"research/test_skew_overlay.py").read_text().split("SKIPPED = ")[0]
_src = _src.replace("return out, raw[ok], sector_rank", "return out, raw, sector_rank")   # full pool
assert "return out, raw, sector_rank" in _src
exec(compile(_src, "test_skew_overlay_harness", "exec"))
N, CAP8 = 40, 8
def pick(score, sec, fwd, cap=CAP8):
    picked, cnt = [], {}
    for t in score.sort_values(ascending=False).index:
        s_ = sec[t]
        if cap and cnt.get(s_, 0) >= cap: continue
        if pd.isna(fwd.get(t)): continue
        picked.append(t); cnt[s_] = cnt.get(s_, 0) + 1
        if len(picked) == N: break
    return picked
months = pd.date_range(PX.index.min()+pd.Timedelta(days=400), PX.index.max()-pd.Timedelta(days=35), freq="ME")
rows, legs = [], []
for D in months:
    D = PX.index[PX.index.get_indexer([D], method="nearest")[0]]
    nxt = PX.index[PX.index.get_indexer([D+pd.Timedelta(days=30)], method="nearest")[0]]
    if nxt <= D: continue
    out, rawall, _ = month_inputs(D)
    fwd = PX.loc[nxt]/PX.loc[D] - 1
    bad = fwd[(fwd > 3) | (fwd < -0.95)].index; fwd = fwd.drop(bad)
    o = out[out.value.notna()]
    v3 = rawall[["SP", "bm", "cfp"]]
    mkt_full = v3.rank(pct=True).mean(axis=1).reindex(o.index)
    mkt_elig = v3.loc[o.index].rank(pct=True).mean(axis=1)
    sec_elig = sector_rank(v3.loc[o.index], o.sector).mean(axis=1)
    B = {"SECREL": pick(o.value, o.sector, fwd), "MKT_elig": pick(mkt_elig, o.sector, fwd),
         "MKT_full": pick(mkt_full, o.sector, fwd), "SECREL_elig": pick(sec_elig, o.sector, fwd)}
    r = {k: fwd.reindex(b).mean() for k, b in B.items()}; r["date"] = D; r["spy"] = fwd["SPY"]
    r["ov_full"] = len(set(B["SECREL"]) & set(B["MKT_full"]))
    rows.append(r)
    for t in set(B["MKT_elig"]) - set(B["SECREL"]): legs.append((D, "MKT_only", t, o.sector[t], fwd[t]))
    for t in set(B["SECREL"]) - set(B["MKT_elig"]): legs.append((D, "SEC_only", t, o.sector[t], fwd[t]))
R = pd.DataFrame(rows); L = pd.DataFrame(legs, columns=["date", "leg", "t", "sector", "r"])
ts = lambda x: x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
for k in ("SECREL", "MKT_elig", "MKT_full", "SECREL_elig"):
    x = R[k]-R.spy; print(f"{k:12s} {x.mean():+.2%}/mo t {ts(x):.2f}")
for a, b in (("MKT_elig", "SECREL"), ("MKT_full", "SECREL"), ("MKT_elig", "SECREL_elig")):
    d = R[a]-R[b]; db = d.drop(d.idxmax())
    print(f"{a} - {b}: {d.mean():+.2%}/mo t {ts(d):.2f} win {(d>0).mean():.0%} | drop best month {db.mean():+.2%} t {ts(db):.2f}")
print(f"overlap SECREL vs MKT_full {R.ov_full.mean():.1f}/40")
print("\nswap legs (MKT_elig vs SECREL): names/month and mean fwd return")
g = L.groupby("leg").r.agg(["count", "mean", "median"]); g["count"] /= len(R); print(g)
print("\nswap legs by sector (count over 24 months, mean r):")
print(L.pivot_table(index="sector", columns="leg", values="r", aggfunc=["count", "mean"]).round(3))
print("\ntop single-name contributions MKT_only:"); print(L[L.leg=="MKT_only"].nlargest(8, "r")[["date","t","sector","r"]])
# leave-one-sector-out of the swap: gap per month recomputed without one sector's swap names
print("\nleave-one-sector-out gap (approx: sum of leg returns /40, sector removed):")
for s in L.sector.value_counts().index[:6]:
    m = L[L.sector != s]
    gap = (m[m.leg=="MKT_only"].groupby("date").r.sum().reindex(R.date, fill_value=0) -
           m[m.leg=="SEC_only"].groupby("date").r.sum().reindex(R.date, fill_value=0))/40
    print(f"  without {s:22s} {gap.mean():+.2%}/mo t {ts(gap):.2f}")
