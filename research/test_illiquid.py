"""Is the illiquid edge real, or is it survivorship bias wearing a costume?"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P = pd.read_parquet(ROOT/"research/out/pit_panel.parquet")
BASE = ["mom_12_1","bm","px_52w","SP"]
P["comp"] = P[BASE].mean(axis=1)
P = P[P.dollar_vol.notna() & (P.dollar_vol > 0)]

# ---- 1. is it monotonic, or an artefact of cutting in half? ----
P["lq"] = P.groupby("date").dollar_vol.transform(lambda s: pd.qcut(s,5,labels=False,duplicates="drop"))
LAB = ["Q1 thinnest","Q2","Q3","Q4","Q5 most liquid"]
print("1. COMPOSITE SPREAD BY LIQUIDITY QUINTILE")
print(f"   {'bucket':16s} {'median $vol':>13s} {'n/date':>7s} {'baseline ret':>13s} {'Q5-Q1 spread':>13s} {'win':>5s}")
for q in range(5):
    s = P[P.lq==q]
    sp=[]
    for D,g in s.groupby("date"):
        v=g.comp.dropna()
        if len(v)<80: continue
        b=pd.qcut(v,5,labels=False,duplicates="drop")
        m=g.loc[v.index].groupby(b).fwd.mean()
        if len(m)==5: sp.append(m.iloc[-1]-m.iloc[0])
    print(f"   {LAB[q]:16s} {s.dollar_vol.median():13,.0f} {len(s)/P.date.nunique():7.0f} "
          f"{s.fwd.mean():+13.2%} {np.mean(sp):+13.2%} {np.mean([x>0 for x in sp]):5.0%}")

# ---- 2. survivorship exposure: what is missing from this panel entirely? ----
print("\n2. SURVIVORSHIP EXPOSURE")
uni = pd.read_csv(ROOT/"data/universe/tickers_2026-08-18.csv", dtype={"cik":str})
inpanel = set(P.ticker)
print(f"   SEC roster today                : {uni.ticker.nunique():,}")
print(f"   in our price panel at all       : {len(inpanel):,}")
print(f"   NEVER in the panel              : {uni.ticker.nunique()-len(inpanel):,}")
print("   Yahoo purges delisted symbols outright, so a company that failed during")
print("   2024-2026 is absent from EVERY date, not just the ones after it failed.")
print("   Failures concentrate in thin names, so Q1 is the bucket losing most of its losers.")

# ---- 3. how big is the hole? use the measured roster attrition ----
print("\n3. WHAT THE HOLE COULD BE WORTH")
print("   Measured earlier: 57.7% of the June-2021 SEC roster was gone by 2026,")
print("   and Yahoo retained history for only 18% of the vanished names.")
for rate in (0.05, 0.10, 0.20):
    for loss in (-0.60, -0.90):
        adj = P[P.lq==0].fwd.mean()*(1-rate) + loss*rate
        print(f"   if {rate:.0%} of thin names died at {loss:.0%}: Q1 baseline "
              f"{P[P.lq==0].fwd.mean():+.2%} -> {adj:+.2%}")
    break
for rate in (0.10, 0.20):
    adj = P[P.lq==0].fwd.mean()*(1-rate) + (-0.90)*rate
    print(f"   if {rate:.0%} died at -90%:                 Q1 baseline "
          f"{P[P.lq==0].fwd.mean():+.2%} -> {adj:+.2%}")

# ---- 4. can it actually be traded? ----
print("\n4. TRADABILITY — position size at 10% of median daily volume")
for q in range(5):
    dv = P[P.lq==q].dollar_vol.median()
    print(f"   {LAB[q]:16s} ${dv:12,.0f}/day  ->  ${dv*0.10:>10,.0f} per day to build or exit")
print("\n   A 40-name book at 2.5% each needs position sizes of:")
for cap in (200_000, 1_000_000):
    pos = cap/40
    print(f"     portfolio ${cap:,} -> ${pos:,.0f} per name", end="")
    ok = [LAB[q] for q in range(5) if P[P.lq==q].dollar_vol.median()*0.10 >= pos]
    print(f"   tradable in: {', '.join(ok) if ok else 'nothing'}")
