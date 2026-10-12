"""Is the recorder actually working? Answers in one line.

Three silent failures have now happened -- a partial smoke-test snapshot, a
network-dead run, and eight days of Yahoo throttling -- and none of them raised
anything a human would see. This is the check that would have caught all three.
"""
import pandas as pd, pathlib, datetime as dt, sys, glob
ROOT = pathlib.Path(__file__).resolve().parent.parent
snaps = sorted(glob.glob(str(ROOT/"data/snapshots/ohlcv_*.parquet")))
today = dt.date.today()

if not snaps:
    print("FAIL  no snapshots at all"); sys.exit(1)

last = pathlib.Path(snaps[-1]); last_date = dt.date.fromisoformat(last.stem.split("_")[1])
age = (today - last_date).days
d = pd.read_parquet(last)
n = d.ticker.nunique()

status_f = ROOT/f"data/snapshots/status_{last_date}.csv"
thr = 0
if status_f.exists():
    s = pd.read_csv(status_f)
    thr = int((s.iloc[:, -1].astype(str) == "throttled").sum())

# business days missed since the last snapshot
missed = sum(1 for i in range(1, age+1)
             if (last_date + dt.timedelta(days=i)).weekday() < 5)

problems = []
# A run that produced nothing passed this check on 2026-09-08 because the
# previous file was only two days old. On a scheduled day, today's file must exist.
if today.weekday() in (1,2,3,4,5) and not (ROOT/f"data/snapshots/ohlcv_{today}.parquet").exists():
    problems.append(f"no snapshot for today ({today})")
if age > 4:      problems.append(f"{age}d stale ({missed} business days missing)")
if n < 4000:     problems.append(f"only {n:,} tickers (expected ~5,000 weekdays, ~9,600 Saturdays)")
if thr > 500:    problems.append(f"{thr:,} tickers throttled")

if problems:
    print(f"FAIL  last={last_date} tickers={n:,} — " + "; ".join(problems)); sys.exit(1)
print(f"OK    last={last_date} ({age}d ago) tickers={n:,} throttled={thr}")
