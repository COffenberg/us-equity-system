"""Does a peer's earnings result predict YOUR pre-earnings drift?

The 46.3% positive rate is unconditional. The question really being asked:
conditioned on peers who already reported beating, does the window improve?
Nothing here uses information dated after the moment the trade would be placed.
"""
import pandas as pd, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent

f = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
f = f[f.form.fillna("").str.startswith(("10-Q","10-K")) & f.filed_date.notna()]
ev = f[["ticker","filed_date"]].drop_duplicates()
ev["filed_date"] = pd.to_datetime(ev.filed_date)

h = pd.read_parquet(ROOT/"data/prices/history_3y.parquet").sort_values(["ticker","date"])
h["date"] = pd.to_datetime(h.date)
# Sub-penny names produce garbage returns -- adj_close of 0.0004 gave a +5,475%
# day and one division by zero returned +inf. Restrict to what is actually
# tradable before measuring anything.
d = pd.read_parquet(ROOT/"data/prices/derived.parquet").set_index("ticker")
ok = set(d.index[(d.dollar_vol_60d >= 5e5)])
h = h[h.ticker.isin(ok) & (h.adj_close >= 1.0)]
h["ret"] = h.groupby("ticker").adj_close.pct_change()
h = h[h.ret.abs() < 0.75]                      # a >75% single day is a corporate action
print(f"price panel after tradability filter: {h.ticker.nunique():,} tickers")
idx = h.dropna(subset=["ret"]).set_index(["ticker","date"]).ret

sec = pd.read_parquet(ROOT/"data/sec/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector
ev["sector"] = ev.ticker.map(sec)
ev = ev.dropna(subset=["sector"])

rows = []
for t, g in ev.groupby("ticker"):
    try: s = idx.loc[t]
    except KeyError: continue
    d = s.index
    for fd in g.filed_date:
        pre  = s[(d >= fd - pd.Timedelta(days=17)) & (d <= fd - pd.Timedelta(days=2))]
        post = s[(d >= fd - pd.Timedelta(days=1)) & (d <= fd + pd.Timedelta(days=2))]
        if len(pre) >= 6 and len(post) >= 2:
            rows.append({"ticker": t, "sector": g.sector.iloc[0], "filed_date": fd,
                         "pre_ret": (1+pre).prod()-1, "ann_ret": (1+post).prod()-1})
E = pd.DataFrame(rows).sort_values("filed_date").reset_index(drop=True)
print(f"{len(E):,} events with a clean pre-window and announcement window\n")

# --- peer signal: same-sector names that filed in the 45 days BEFORE this event ---
peer = []
for sector, g in E.groupby("sector"):
    g = g.sort_values("filed_date")
    for i, r in g.iterrows():
        w = g[(g.filed_date < r.filed_date - pd.Timedelta(days=1)) &
              (g.filed_date >= r.filed_date - pd.Timedelta(days=45))]
        peer.append({"i": i, "peer_ann": w.ann_ret.mean() if len(w) >= 3 else np.nan,
                     "peer_n": len(w)})
P = pd.DataFrame(peer).set_index("i")
E = E.join(P)
V = E[E.peer_ann.notna()].copy()
print(f"{len(V):,} events with >=3 sector peers reporting first\n")

base_pre = (V.pre_ret > 0).mean(); base_ann = (V.ann_ret > 0).mean()
print(f"BASELINE   pre-window positive {base_pre:.1%}   announcement positive {base_ann:.1%}")
print(f"           mean pre {V.pre_ret.mean():+.2%}   median pre {V.pre_ret.median():+.2%}\n")

V["bucket"] = pd.qcut(V.peer_ann, 5, labels=["peers worst","2","3","4","peers best"])
g = V.groupby("bucket", observed=True).agg(
    n=("pre_ret","size"),
    pre_hit=("pre_ret", lambda s: (s>0).mean()),
    pre_mean=("pre_ret","mean"),
    pre_med=("pre_ret","median"),
    ann_hit=("ann_ret", lambda s: (s>0).mean()),
    ann_med=("ann_ret","median"))
print("CONDITIONED ON HOW SECTOR PEERS ALREADY REPORTED")
print(f"{'bucket':12s} {'n':>6s} {'pre hit':>8s} {'pre med':>9s} {'ann hit':>8s} {'ann med':>9s}")
for b, r in g.iterrows():
    print(f"{str(b):12s} {int(r.n):6d} {r.pre_hit:7.1%} {r.pre_med:+9.2%} {r.ann_hit:7.1%} {r.ann_med:+9.2%}")

top, bot = g.loc["peers best"], g.loc["peers worst"]
print(f"\nSPREAD best-minus-worst   pre-window hit {top.pre_hit-bot.pre_hit:+.1%} pts"
      f"   median pre-return {top.pre_med-bot.pre_med:+.2%}")
a = V[V.bucket=="peers best"].pre_ret; b = V[V.bucket=="peers worst"].pre_ret
t = (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b))
print(f"t-stat on the pre-window difference: {t:.2f}   (n={len(a)} vs {len(b)})")

# ---------------------------------------------------------------- FALSIFIER
# If a sector is simply rallying, peers report into strength AND your stock
# drifts up. That would make this sector momentum with extra steps, not
# read-across. Control for raw sector return over the same 45-day window.
print("\n" + "="*66)
print("FALSIFIER: is this just sector momentum?")

secpx = h.groupby([h.date, h.ticker.map(sec)]).ret.mean().rename("secret").reset_index()
secpx.columns = ["date","sector","secret"]
sp = secpx.pivot(index="date", columns="sector", values="secret").fillna(0)
cum = (1+sp).cumprod()

def sector_mom(sector, fd, days=45):
    if sector not in cum.columns: return np.nan
    s = cum[sector]; w = s[(s.index < fd) & (s.index >= fd - pd.Timedelta(days=days))]
    return w.iloc[-1]/w.iloc[0] - 1 if len(w) > 5 else np.nan

V["sec_mom"] = [sector_mom(r.sector, r.filed_date) for r in V.itertuples()]
W = V.dropna(subset=["sec_mom"]).copy()

# double sort: within each sector-momentum bucket, does the peer signal still work?
W["mom_b"] = pd.qcut(W.sec_mom, 3, labels=["sector down","sector flat","sector up"])
W["peer_b"] = pd.qcut(W.peer_ann, 3, labels=["peers bad","peers mid","peers good"])
tab = W.pivot_table(index="mom_b", columns="peer_b", values="pre_ret",
                    aggfunc=lambda s: (s>0).mean(), observed=True)
print("\npre-window hit rate, double-sorted (rows = sector momentum):")
print((tab*100).round(1).to_string())

print("\nspread from the PEER signal, holding sector momentum fixed:")
for m in tab.index:
    print(f"  {str(m):12s} {(tab.loc[m,'peers good']-tab.loc[m,'peers bad'])*100:+5.1f} pts")
print("\nspread from SECTOR MOMENTUM, holding the peer signal fixed:")
for pcol in tab.columns:
    print(f"  {str(pcol):12s} {(tab.loc['sector up',pcol]-tab.loc['sector down',pcol])*100:+5.1f} pts")

import scipy.stats as st
r, pv = st.spearmanr(W.peer_ann, W.sec_mom)
print(f"\ncorrelation between the two signals: {r:+.3f}")
print(f"events per sector-quarter (overlap inflates every t-stat above): "
      f"{len(W)/ (W.sector.nunique()*12):.0f}")
