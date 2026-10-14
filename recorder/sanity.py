"""Did the rebuild actually produce a sane book? Answers in one line.

health.py checks the snapshot. This checks what was built from it. Runs as the
last step of the daily workflow, after the commit, so a failure turns the run red
without ever costing the day's snapshot. The derived parquets are gitignored, so
this is the only place they can be checked against the run that made them.

Each check exists because the failure it catches has happened:
  - the price panel not advancing while the run exited 0 (19 days stale once)
  - named mega-caps silently missing from the ranking (three times)
  - a book with the wrong number of names
"""
import pandas as pd, pathlib, json, glob, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
MEGA = ["MSFT", "GOOGL", "META", "NVDA", "JPM"]

problems = []

snap = pathlib.Path(sorted(glob.glob(str(ROOT/"data/snapshots/ohlcv_*.parquet")))[-1])
snap_date = pd.to_datetime(pd.read_parquet(snap, columns=["date"]).date).max()
panel_date = pd.to_datetime(pd.read_parquet(ROOT/"data/prices/history_3y.parquet", columns=["date"]).date).max()
if panel_date < snap_date:
    problems.append(f"price panel at {panel_date.date()} but snapshot has {snap_date.date()}")

r = pd.read_parquet(ROOT/"signals/ranked.parquet")
if len(r) < 2000:
    problems.append(f"only {len(r):,} names ranked (expected ~2,500)")
missing = [t for t in MEGA if t not in set(r.index)]
if missing:
    problems.append(f"mega-caps missing from ranking: {', '.join(missing)}")

o = json.loads((ROOT/"signals/orders.json").read_text())
if len(o.get("holdings", [])) != o.get("n"):
    problems.append(f"book has {len(o.get('holdings', []))} names, expected {o.get('n')}")

if problems:
    print("FAIL  " + "; ".join(problems)); sys.exit(1)
print(f"OK    panel={panel_date.date()} ranked={len(r):,} book={o['n']} mega-caps present")
