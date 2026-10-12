"""What portfolio actually beats SPY? Pre-specified constructions, not a search.

Two gaps this closes:
  1. Every PIT test so far used a 4-signal composite. Production uses 23 signals
     across 9 clusters. The thing being measured was never the thing being run.
  2. The ~40-name floor was argued from the payoff shape of a lottery book. The
     Combined book is not that -- median cap $2-6bn, quality/value/momentum. So
     concentration is an empirical question, and the owner has to manage the result.

Constructions are fixed in advance and ALL are reported, including the failures.
The best of fifteen is biased upward; the spread across them is the honest signal.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel2.parquet")
H = pd.read_parquet(ROOT/"data/prices/history_3y.parquet"); H["date"]=pd.to_datetime(H.date)
PX = H.pivot_table(index="date",columns="ticker",values="adj_close",aggfunc="last").sort_index()

# production-equivalent clusters, from what the PIT panel actually holds
CLUST = {
 "value":      ["SP","cfp","bm"],
 "profit":     ["roaq","PctTotAcc","leverage"],
 "financing":  ["buyback","issuance"],
 "investment": ["asset_growth","d_noa"],
 "momentum":   ["mom_12_1","px_52w"],
 "lowrisk":    ["vol_60d"],
 "ownership":  ["insider"],
 "quality2":   ["gross_prof","cash_op_prof","roic"],
 "redflag":    ["accrual_ratio","d_accrual","cash_conv"],
}
for k,v in CLUST.items():
    P[f"K_{k}"] = P[[c for c in v if c in P.columns]].mean(axis=1)

FOUR = ["mom_12_1","bm","px_52w","SP"]                       # the old baseline
PASSED = ["mom_12_1","bm","px_52w","SP","asset_growth","d_noa","gp_stability",
          "d_accrual","cash_conv","equity_neg","sales_growth"]              # cleared the 4-test screen
DROP_FAIL = [k for k in CLUST if k not in ("profit","financing","quality2")]  # drop the losers

SCORES = {
 "4-signal baseline":        P[FOUR].mean(axis=1),
 "production 9 clusters":    P[[f"K_{k}" for k in CLUST]].mean(axis=1),
 "clusters minus losers":    P[[f"K_{k}" for k in DROP_FAIL]].mean(axis=1),
 "signals that passed":      P[[c for c in PASSED if c in P.columns]].mean(axis=1),
 "momentum + value only":    P[["K_momentum","K_value"]].mean(axis=1),
 "momentum only":            P["K_momentum"],
}
HOLD = 63
def bench(D):
    fi = PX.index.get_indexer([D+pd.Timedelta(days=int(HOLD*1.45))],method="nearest")[0]
    try: return PX["SPY"].iloc[fi]/PX.loc[D,"SPY"]-1
    except Exception: return np.nan

def run(score, n, sector_cap=6, min_dv=5e5):
    out=[]
    for D,g in P.assign(s=score).groupby("date"):
        g = g[(g.dollar_vol>=min_dv) & g.s.notna()]
        if len(g)<100: continue
        picked, cnt = [], {}
        for t,r in g.sort_values("s",ascending=False).iterrows():
            sec=r.sector
            if sector_cap and cnt.get(sec,0)>=sector_cap: continue
            picked.append(r.fwd); cnt[sec]=cnt.get(sec,0)+1
            if len(picked)==n: break
        if len(picked)<n: continue
        out.append((np.mean(picked), bench(D)))
    if not out: return None
    a=np.array(out); ex=a[:,0]-a[:,1]
    return dict(n=len(a), ret=a[:,0].mean(), spy=a[:,1].mean(),
                edge=ex.mean(), win=(ex>0).mean(), sd=ex.std(ddof=1),
                t=ex.mean()/(ex.std(ddof=1)/np.sqrt(len(ex))) if len(ex)>1 else np.nan)

print(f"{'construction':26s} {'N':>3s} {'ret':>7s} {'SPY':>7s} {'edge':>7s} {'win':>5s} {'sd':>6s} {'t':>5s}")
res=[]
for label, sc in SCORES.items():
    for n in (10,15,20,30,40):
        r = run(sc, n)
        if not r: continue
        res.append((label,n,r))
        print(f"{label:26s} {n:3d} {r['ret']:+7.2%} {r['spy']:+7.2%} {r['edge']:+7.2%} "
              f"{r['win']:5.0%} {r['sd']:6.2%} {r['t']:5.2f}")
print(f"\n{len(res)} constructions tested over {P.date.nunique()} quarterly dates.")
e=[r['edge'] for _,_,r in res]
print(f"edge across all: mean {np.mean(e):+.2%}, median {np.median(e):+.2%}, "
      f"best {max(e):+.2%}, worst {min(e):+.2%}, positive {sum(1 for x in e if x>0)}/{len(e)}")
