"""Point-in-time signal reconstruction + forward-return test.

Rebuilds signals as they would have looked on past dates (using only filings
already filed by then), then measures forward returns by quintile. This is the
first test of our OWN signals on our OWN data -- everything except read-across
has been inherited on the catalog's authority.

Honest limits, stated once: 3 years is one regime; the price panel holds only
surviving names, so failures are missing from the bottom quintile and the
measured spread is OPTIMISTIC; and quarterly dates 3 months apart give ~11
overlapping observations, nowhere near enough for significance.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

F = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
F["filed_date"] = pd.to_datetime(F.filed_date); F["period_end"] = pd.to_datetime(F.period_end)
F["dur"] = (F.period_end - pd.to_datetime(F.period_start)).dt.days
SH = pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
SH["filed_date"] = pd.to_datetime(SH.filed_date); SH["period_end"] = pd.to_datetime(SH.period_end)
SEC = pd.read_parquet(ROOT/"data/sec/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector

H = pd.read_parquet(ROOT/"data/prices/history_3y.parquet").sort_values(["ticker","date"])
H["date"] = pd.to_datetime(H.date)
H = H[H.adj_close >= 1.0]
PX = H.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
DV = H.assign(dv=H.close*H.volume).pivot_table(index="date", columns="ticker", values="dv", aggfunc="last")

def signals_asof(D):
    f = F[F.filed_date <= D]; sh = SH[SH.filed_date <= D]
    def inst(c):
        d = f[(f.concept == c) & f.instantaneous].sort_values("period_end")
        return d.groupby("ticker").last().val
    def ttm(c):
        d = f[(f.concept == c) & (~f.instantaneous) & f.dur.between(60,115)] \
              .sort_values("period_end").drop_duplicates(["ticker","period_end"],keep="last") \
              .groupby("ticker").tail(4)
        g = d.groupby("ticker").val.agg(["sum","count"]); return g.loc[g["count"]==4,"sum"]
    px = PX.loc[:D].iloc[-1].dropna()
    s = inst("EntityCommonStockSharesOutstanding")
    for c in ["CommonStockSharesOutstanding","WeightedAverageNumberOfDilutedSharesOutstanding"]:
        d = sh[sh.concept == c].sort_values("period_end")
        s = s.combine_first(d.groupby("ticker").last().val)
    mc = (px*s).dropna(); mc = mc[mc > 5e7]

    rev = ttm("Revenues").combine_first(ttm("RevenueFromContractWithCustomerExcludingAssessedTax"))
    ocf = ttm("NetCashProvidedByUsedInOperatingActivities"); ni = ttm("NetIncomeLoss")
    A = inst("Assets"); E = inst("StockholdersEquity")
    q_ni = f[(f.concept=="NetIncomeLoss") & (~f.instantaneous) & f.dur.between(60,115)] \
             .sort_values("period_end").groupby("ticker").last().val
    hist = PX.loc[:D]
    out = pd.DataFrame({
        "SP": rev/mc, "cfp": ocf/mc, "bm": E/mc, "roaq": q_ni/A,
        "PctTotAcc": -((ni-ocf)/ni.abs()),
        "leverage": -(inst("Liabilities")/A),
        "buyback": ttm("PaymentsForRepurchaseOfCommonStock")/mc,
        "issuance": -(ttm("ProceedsFromIssuanceOfCommonStock")/mc),
        "mom_12_1": hist.iloc[-1]/hist.iloc[max(0,len(hist)-252)] - 1
                    - (hist.iloc[-1]/hist.iloc[max(0,len(hist)-21)] - 1),
        "px_52w":  hist.iloc[-1]/hist.iloc[max(0,len(hist)-252):].max(),
        "vol_60d": -hist.pct_change().iloc[-60:].std()*np.sqrt(252),
    })
    liq = DV.loc[:D].iloc[-60:].median()
    keep = out.index[(liq.reindex(out.index) >= 5e5) & out.index.isin(mc.index)]
    return out.loc[keep]

DATES = pd.to_datetime(["2023-12-29","2024-03-28","2024-06-28","2024-09-30",
                        "2024-12-31","2025-03-31","2025-06-30","2025-09-30",
                        "2025-12-31","2026-03-31"])
HOLD = 63
res = {}
for D in DATES:
    if D not in PX.index: D = PX.index[PX.index.get_indexer([D], method="nearest")[0]]
    S = signals_asof(D)
    fwd_i = PX.index.get_indexer([D + pd.Timedelta(days=int(HOLD*1.45))], method="nearest")[0]
    fwd = PX.iloc[fwd_i]/PX.loc[D] - 1
    sec = SEC.reindex(S.index).fillna("Other")
    for c in S.columns:
        r = S[c].groupby(sec).rank(pct=True)                     # sector-relative, as in production
        d = pd.DataFrame({"r": r, "f": fwd.reindex(S.index)}).dropna()
        if len(d) < 300: continue
        q = pd.qcut(d.r, 5, labels=False, duplicates="drop")
        m = d.groupby(q).f.mean()
        if len(m) == 5: res.setdefault(c, []).append(m.values)

print(f"{'signal':11s} {'n':>3s} {'Q1(worst)':>10s} {'Q3':>8s} {'Q5(best)':>9s} {'Q5-Q1':>8s} {'mono':>5s} {'win':>5s}")
rows=[]
for c, v in res.items():
    M = np.array(v); mean = M.mean(0); spread = M[:,4]-M[:,0]
    mono = sum(1 for i in range(4) if mean[i+1] > mean[i])
    rows.append((c, len(M), mean[0], mean[2], mean[4], spread.mean(), mono, (spread>0).mean()))
for c,n,q1,q3,q5,sp,mono,win in sorted(rows, key=lambda x:-x[5]):
    flag = "  <<" if mono>=3 and win>=0.6 else ""
    print(f"{c:11s} {n:3d} {q1:+10.2%} {q3:+8.2%} {q5:+9.2%} {sp:+8.2%} {mono:3d}/4 {win:5.0%}{flag}")
print(f"\nhorizon {HOLD} trading days, {len(DATES)} rebalance dates, sector-relative ranks")
print("mono = how many of 4 quintile steps go the right way; win = share of dates with a positive spread")
