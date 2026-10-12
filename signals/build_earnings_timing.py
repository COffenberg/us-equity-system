"""Earnings-event signals. AnnouncementReturn survived triage at t=3.67 -- the
second-strongest signal in the whole pool -- and was never built. Filing dates
come free from the fundamentals panel; no paid earnings calendar needed."""
import pandas as pd, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
TODAY = pd.Timestamp("2026-08-19")

f = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
f = f[f.form.fillna("").str.startswith(("10-Q","10-K")) & f.filed_date.notna()]
f["filed_date"] = pd.to_datetime(f.filed_date)
ev = f[["ticker","filed_date"]].drop_duplicates().sort_values(["ticker","filed_date"])

h = pd.read_parquet(ROOT/"data/prices/history_3y.parquet").sort_values(["ticker","date"])
h["date"] = pd.to_datetime(h.date)
h["ret"] = h.groupby("ticker").adj_close.pct_change()

# --- 3-day return around each filing, then the most recent one per ticker ---
px = h[["ticker","date","ret"]].dropna()
idx = px.set_index(["ticker","date"]).ret
rows = []
for t, g in ev.groupby("ticker"):
    try: s = idx.loc[t]
    except KeyError: continue
    d = s.index
    for fd in g.filed_date:
        w = s[(d >= fd - pd.Timedelta(days=2)) & (d <= fd + pd.Timedelta(days=2))]
        if len(w) >= 2: rows.append({"ticker": t, "filed_date": fd, "ann_ret": (1+w).prod()-1})
ar = pd.DataFrame(rows).sort_values(["ticker","filed_date"])

latest = ar.groupby("ticker").last()
out = pd.DataFrame(index=latest.index)
out["AnnouncementReturn"] = latest.ann_ret                       # drift after a good print
out["days_since_earnings"] = (TODAY - latest.filed_date).dt.days

# --- when is the next report? median gap between this company's own filings ---
gap = ev.groupby("ticker").filed_date.apply(lambda s: s.diff().dt.days.median())
out["expected_gap_days"] = gap.reindex(out.index)
out["days_to_next_earnings"] = out.expected_gap_days - out.days_since_earnings
out["in_pre_earnings_window"] = out.days_to_next_earnings.between(0, 21)

# --- consistency of the market's reaction: does this name usually pop? ---
hist = ar.groupby("ticker").ann_ret
out["ann_ret_mean"] = hist.mean()
out["ann_ret_hit_rate"] = hist.apply(lambda s: (s > 0).mean())
out["ann_ret_n"] = hist.size()

out.to_parquet(ROOT/"signals/earnings_timing.parquet")
v = out[out.ann_ret_n >= 6]
print(f"built for {len(out)} tickers ({len(v)} with 6+ observed announcements)")
print(f"\nAnnouncementReturn  median {out.AnnouncementReturn.median():+.2%}  "
      f"p10 {out.AnnouncementReturn.quantile(.1):+.1%}  p90 {out.AnnouncementReturn.quantile(.9):+.1%}")
print(f"median days since last filing: {out.days_since_earnings.median():.0f}")
print(f"names reporting within 21 days: {int(out.in_pre_earnings_window.sum())}")
print(f"\nmean announcement return, all names, all events: {ar.ann_ret.mean():+.3%}")
print(f"  positive announcement windows: {(ar.ann_ret>0).mean():.1%} of {len(ar):,} events")
# Everything below is diagnostics against the PREVIOUS cycle's ranks -- the real output
# was written above. ranked.parquet is produced by rank.py, which runs after this script,
# so on a cold checkout (every GitHub run: signals/*.parquet is gitignored and uncached)
# it does not exist yet. Crashing here exits 1 with the artifact already on disk, which
# makes a cosmetic failure indistinguishable from a real one under the workflow's
# `|| echo "non-fatal"` guard. Skip instead.
_ranked = ROOT/"signals/ranked.parquet"
if not _ranked.exists():
    print("\nno ranked.parquet yet (rank.py runs after this) -- skipping cross-check")
    raise SystemExit(0)
o = pd.read_parquet(_ranked)
j = o.drop(columns=[c for c in out.columns if c in o.columns]).join(out, how="inner")
print(f"\nof {len(o)} scored names, {len(j)} have earnings timing; "
      f"{int(j.in_pre_earnings_window.sum())} report within 21 days")
for t in ["CRWV","NBIS"]:
    if t in out.index:
        r = out.loc[t]
        print(f"  {t}: last filing {r.days_since_earnings:.0f}d ago, "
              f"announcement return {r.AnnouncementReturn:+.1%}, hit rate {r.ann_ret_hit_rate:.0%} of {r.ann_ret_n:.0f}")
