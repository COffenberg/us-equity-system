# US Equity Selection System

Research system for an active, long-only US equity book: daily point-in-time data
capture, signal construction, a forward record of frozen predictions, and an
autonomous research loop that tests ideas against SPY. Research results and the
tried-list live in `BACKLOG.md`; operating rules for agents in `AGENTS.md`.

This README covers only what runs.

---

## Phase 0 — cross-section recorder (live since 2026-08-17)

The system is validated **forward**, not by backtest, because no free source
provides survivorship-bias-free US prices — verified exhaustively in
an earlier project. Forward validation only works if
point-in-time history actually accumulates, and the only free way to get it is
to record it as it happens.

### Scoping rule: record only what cannot be reconstructed later

| | Source | Recorded here? |
|---|---|---|
| Fundamentals, shares outstanding, insider transactions, filing text | SEC EDGAR — permanent, timestamped, free, pullable at any future date exactly as it stood | **No** |
| Prices and volume of names that later delist | Yahoo purges delisted symbols entirely — gone the day they delist | **Yes** |
| Universe membership over time | Nothing free archives it | **Yes** |

That rule is what keeps the recorder small: OHLCV plus the roster, nothing else.

### Run

```bash
python3 recorder/snapshot.py            # today's snapshot (no-op if already written)
python3 recorder/snapshot.py --limit 50 # smoke test
python3 recorder/snapshot.py --force    # overwrite today
```

Idempotent — safe to run repeatedly. Full universe is ~10,400 SEC-listed tickers
in 400-name chunks; expect roughly 15–30 minutes.

### Output

```
data/universe/tickers_YYYY-MM-DD.csv     roster verbatim as SEC reported it
data/snapshots/ohlcv_YYYY-MM-DD.parquet  one row per (ticker, date)
data/snapshots/status_YYYY-MM-DD.csv     per-ticker outcome; no_data == the delisting signal
data/recorder.log                        append-only run log
```

The roster is written **unfiltered**. Membership change is the perishable thing
being captured, so filtering happens at read time, never at write time.
`status_*.csv` rows flipping to `no_data` are how a delisting becomes visible —
that column is data, not an error log.

### Scheduling

Runs Tue–Sat on GitHub Actions (`.github/workflows/daily.yml`), with a backup slot
because GitHub drops scheduled runs under load.

### Not yet done

- **Gap backfill.** A missed day is permanently lost for delisted names, but
  survivors can be backfilled from Yahoo. Worth a `--backfill` flag.
- **Corporate actions.** `adj_close` carries Yahoo's own adjustment. Split dates
  are not recorded separately yet.

---

## Phase 1+

Signals, ranking, construction and the forward record are built — see `signals/`,
`orders.py`, `forward/` and `BACKLOG.md`.
