"""Mark every frozen prediction to market. Runs daily; a scoreboard, not a reveal.

The edge measured in this system lives at 21-63 days and is gone by 252. Waiting
a year to look is not caution, it is measuring after the signal has decayed.
So: every frozen list, every day, versus SPY and QQQ, from its own entry date.
"""
import pandas as pd, numpy as np, pathlib, json, glob, warnings, datetime as dt
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent

H = pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
H["date"] = pd.to_datetime(H.date)
PX = H.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
last_date = PX.index.max()
latest = PX.iloc[-1]

# benchmarks: from the panel if present, else a two-ticker yfinance call
bench = {}
for b in ("SPY", "QQQ"):
    if b in PX.columns and pd.notna(latest.get(b)): bench[b] = float(latest[b])
if len(bench) < 2:
    try:
        import yfinance as yf
        d = yf.download(["SPY","QQQ"], period="5d", progress=False, auto_adjust=False, group_by="ticker")
        for b in ("SPY","QQQ"):
            if b not in bench: bench[b] = float(d[b]["Close"].dropna().iloc[-1])
    except Exception as e:
        print(f"benchmark fetch failed: {e}")

rows, detail = [], []
for f in sorted(glob.glob(str(ROOT/"forward/prediction_*.json"))):
    rec = json.load(open(f))
    d0 = pd.Timestamp(rec["date"]); days = (last_date - d0).days
    b0 = rec.get("benchmark_prices", {})
    spy_r = bench["SPY"]/b0["SPY"]-1 if b0.get("SPY") and bench.get("SPY") else np.nan
    qqq_r = bench["QQQ"]/b0["QQQ"]-1 if b0.get("QQQ") and bench.get("QQQ") else np.nan
    for lst, names in rec["lists"].items():
        rets = []
        for n in names:
            t, p0 = n["ticker"], n.get("adj_close") or n.get("price")
            p1 = latest.get(t)
            if p0 and pd.notna(p1) and p0 > 0:
                r = p1/p0-1; rets.append(r)
                detail.append({"date": rec["date"], "list": lst, "ticker": t, "ret": r})
        if not rets: continue
        rets = np.array(rets)
        rows.append({"frozen": rec["date"], "list": lst, "days": days, "n": len(rets),
                     "priced": f"{len(rets)}/{len(names)}",
                     "ret": rets.mean(), "median": np.median(rets),
                     "spy": spy_r, "qqq": qqq_r,
                     "vs_spy": rets.mean()-spy_r if spy_r==spy_r else np.nan,
                     "hit_vs_spy": (rets > spy_r).mean() if spy_r==spy_r else np.nan,
                     "best": rets.max(), "worst": rets.min()})

S = pd.DataFrame(rows)
S.to_parquet(ROOT/"forward/scoreboard.parquet")
pd.DataFrame(detail).to_parquet(ROOT/"forward/scoreboard_detail.parquet")
json.dump({"as_of": str(last_date.date()), "bench": bench,
           "rows": S.to_dict("records")}, open(ROOT/"forward/scoreboard.json","w"), indent=1, default=str)

print(f"SCOREBOARD as of {last_date.date()}   SPY ${bench.get('SPY',0):.2f}  QQQ ${bench.get('QQQ',0):.2f}")
print(f"{'frozen':11s} {'list':11s} {'days':>5s} {'priced':>7s} {'return':>8s} {'SPY':>8s} {'vs SPY':>8s} {'hit':>5s} {'best':>8s} {'worst':>8s}")
for r in S.itertuples():
    print(f"{r.frozen:11s} {r.list:11s} {r.days:5d} {r.priced:>7s} {r.ret:+8.2%} {r.spy:+8.2%} "
          f"{r.vs_spy:+8.2%} {r.hit_vs_spy:5.0%} {r.best:+8.1%} {r.worst:+8.1%}")
