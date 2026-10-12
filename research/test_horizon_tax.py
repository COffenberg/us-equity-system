"""How long should we hold? Measure how the edge decays with horizon, then apply
realisation tax to find the after-tax optimum.

Holding period is meant to be an OUTPUT of the design, not an assumption. With
commission assumed zero and FX not modelled, the only thing arguing for
longer holds is tax deferral -- so this is now a two-variable question and both
variables are measurable.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
exec(open(ROOT/"research/pit_test.py").read().split("DATES=")[0].split('print(f"{')[0])

BASE=["mom_12_1","bm","px_52w","SP"]
DATES=pd.to_datetime(["2024-10-31","2025-01-31","2025-04-30","2025-07-31","2025-10-31","2026-01-30"])
HORIZONS=[21,63,126,252]

rows=[]
for D in DATES:
    if D>PX.index.max(): continue
    D=PX.index[PX.index.get_indexer([D],method="nearest")[0]]
    S=signals_asof(D); sec=SEC.reindex(S.index).fillna("Other")
    comp=pd.concat([S[c].groupby(sec).rank(pct=True) for c in BASE if c in S],axis=1).mean(axis=1)
    liq=DV.loc[:D].iloc[-60:].median()
    comp=comp[liq.reindex(comp.index).fillna(0)>=5e5]
    r={"date":D}
    for H in HORIZONS:
        tgt=D+pd.Timedelta(days=int(H*1.45))
        if tgt>PX.index.max(): r[H]=np.nan; continue
        fi=PX.index.get_indexer([tgt],method="nearest")[0]
        fwd=PX.iloc[fi]/PX.loc[D]-1
        d=pd.DataFrame({"c":comp,"f":fwd.reindex(comp.index)}).dropna()
        if len(d)<300: r[H]=np.nan; continue
        q=pd.qcut(d.c,5,labels=False,duplicates="drop")
        m=d.groupby(q).f.mean()
        r[H]=m.iloc[-1]; r[f"uni_{H}"]=d.f.mean(); r[f"sp_{H}"]=m.iloc[-1]-m.iloc[0]
    rows.append(r)
A=pd.DataFrame(rows)

print("EDGE BY HOLDING PERIOD — top composite quintile vs the universe average")
print(f"{'hold':>7s} {'n dates':>8s} {'top-Q ret':>10s} {'universe':>10s} {'edge':>8s} {'edge/yr':>9s} {'spread':>8s}")
ann={}
for H in HORIZONS:
    v=A[H].dropna(); u=A[f"uni_{H}"].dropna(); s=A[f"sp_{H}"].dropna()
    if not len(v): continue
    edge=v.mean()-u.mean()
    per_yr=252/H
    ann[H]=(v.mean(), edge, per_yr)
    print(f"{H:>5d}d {len(v):8d} {v.mean():+10.2%} {u.mean():+10.2%} {edge:+8.2%} {edge*per_yr:+9.2%} {s.mean():+8.2%}")

print("\nAFTER-TAX ANNUALISED, rebalancing at each frequency")
print("(Realisation tax modelled at a flat 42% on net gains.)")
T=0.42
print(f"\n{'hold':>7s} {'gross/yr':>10s} {'realised each cycle':>20s} {'held 5y then sold':>19s} {'deferral worth':>15s}")
for H,(ret,edge,per_yr) in sorted(ann.items()):
    g=(1+ret)**per_yr-1                                  # gross annual, compounding the cycle
    taxed_each=(1+ret*(1-T))**per_yr-1                   # pay tax every cycle
    five=(1+g)**5; held=(five-1)*(1-T)                   # defer, pay once at the end
    held_ann=(1+held)**(1/5)-1
    print(f"{H:>5d}d {g:+10.2%} {taxed_each:+20.2%} {held_ann:+19.2%} {held_ann-taxed_each:+15.2%}")

print("\nREAD-ACROSS, priced the same way")
print("  ~+2.10% per event, ~4 events/yr, held ~15 days each")
g=(1.021)**4-1; net=(1+0.021*(1-T))**4-1
print(f"  gross {g:+.2%}/yr   after tax each cycle {net:+.2%}/yr")
print(f"  hurdle it must clear (deferral forgone): ~1.5-2pp/yr -> {'CLEARS' if net>0.02 else 'FAILS'}")
