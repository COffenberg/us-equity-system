#!/bin/bash
# Daily cross-section snapshot. Runs Tue-Sat local, covering Mon-Fri US closes.
#
# Two failure modes seen in the wild, both handled here:
#   1. launchd defers the morning fire until the machine wakes, and Wi-Fi is not up yet
#      -> "No route to host" on the very first SEC call. (Killed 2026-08-20.)
#   2. Yahoo starts serving empty results under throttling
#      -> chunks marked throttled; the script resumes and refetches only those.
set -u
cd "$(dirname "$0")/.."
[ -z "${SEC_CONTACT_EMAIL:-}" ] && [ -f .sec_contact ] && export SEC_CONTACT_EMAIL="$(cat .sec_contact)"
: "${SEC_CONTACT_EMAIL:?SEC_CONTACT_EMAIL unset - create .sec_contact in the repo root}"

STAMP=$(date +%Y-%m-%d)
echo "=== $(date '+%F %T') starting snapshot $STAMP ==="

# --- wait for the network before touching anything ---
for i in $(seq 1 40); do
  if curl -s --max-time 10 -o /dev/null \
       -A "Offenberg US-Equity-System $SEC_CONTACT_EMAIL" \
       https://www.sec.gov/files/company_tickers.json; then
    [ "$i" -gt 1 ] && echo "$(date '+%F %T') network up after $((i*30))s"
    break
  fi
  if [ "$i" -eq 40 ]; then
    echo "$(date '+%F %T') ABORT: no network after 20 minutes"; exit 1
  fi
  sleep 30
done

# --- run, retrying the whole thing on hard failure ---
rc=1
for attempt in 1 2 3; do
  /usr/bin/python3 recorder/snapshot.py
  rc=$?
  [ "$rc" -eq 0 ] && break
  echo "$(date '+%F %T') attempt $attempt failed rc=$rc — retrying in 5 min"
  sleep 300
done

# --- if Yahoo throttled, give it time and resume ---
STATUS="data/snapshots/status_${STAMP}.csv"
if [ "$rc" -eq 0 ] && [ -f "$STATUS" ]; then
  THROTTLED=$(awk -F, 'NR>1 && $NF ~ /throttled/ {n++} END {print n+0}' "$STATUS")
  if [ "$THROTTLED" -gt 0 ]; then
    echo "$(date '+%F %T') $THROTTLED tickers throttled — resuming in 30 min"
    sleep 1800
    /usr/bin/python3 recorder/snapshot.py
    rc=$?
  fi
fi

# --- refresh the whole pipeline off the new snapshot ---
# The price panel went 19 days stale once because nothing downstream of the
# recorder ever re-ran. The snapshot is the update mechanism; use it.
if [ "$rc" -eq 0 ]; then
  echo "--- refreshing pipeline ---"
  /usr/bin/python3 research/refresh_prices.py     || echo "  refresh_prices FAILED"
  /usr/bin/python3 signals/build_all.py           >/dev/null 2>&1 || echo "  build_all FAILED"
  /usr/bin/python3 signals/build_earnings_timing.py >/dev/null 2>&1 || echo "  earnings_timing FAILED"
  /usr/bin/python3 signals/build_ownership.py     >/dev/null 2>&1 || echo "  ownership FAILED"
  /usr/bin/python3 signals/rank.py                >/dev/null 2>&1 || echo "  rank FAILED"
  /usr/bin/python3 signals/build_readacross.py    >/dev/null 2>&1 || echo "  readacross FAILED"
  # freeze a new forward record once the last one is 28+ days old. The edge lives
  # at 21-63 days and is gone by 252, so monthly is the honest cadence.
  LAST=$(ls forward/prediction_*.json 2>/dev/null | tail -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')
  if [ -n "$LAST" ]; then
    AGE=$(( ( $(date +%s) - $(date -j -f %Y-%m-%d "$LAST" +%s) ) / 86400 ))
    if [ "$AGE" -ge 28 ]; then
      /usr/bin/python3 forward/freeze.py "$STAMP" && echo "  froze forward record $STAMP (last was ${AGE}d ago)"
    fi
  fi
  /usr/bin/python3 orders.py                      || echo "  orders FAILED"
  /usr/bin/python3 forward/score.py               || echo "  scoreboard FAILED"
  /usr/bin/python3 dashboard.py                   || echo "  dashboard FAILED"
fi

# --- health check: the run "succeeding" is not the same as the data being good ---
echo "--- health ---"
/usr/bin/python3 recorder/health.py
hrc=$?
if [ "$hrc" -ne 0 ]; then
  echo "$(date '+%F %T') HEALTH CHECK FAILED — see above"
  # leave a breadcrumb somewhere a human will actually trip over it
  /usr/bin/python3 recorder/health.py > "$HOME/Desktop/RECORDER-PROBLEM.txt" 2>&1
else
  rm -f "$HOME/Desktop/RECORDER-PROBLEM.txt"
fi

echo "=== $(date '+%F %T') finished rc=$rc health=$hrc ==="
exit $rc
