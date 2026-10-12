"""Build cross-sectional signals from the SEC fundamentals panel + price panel."""
import pandas as pd, numpy as np, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "signals"
TODAY = pd.Timestamp("2026-08-19")

f = pd.read_parquet(ROOT/"data/sec/fundamentals.parquet")
f = f[f.filed_date.notna() & (pd.to_datetime(f.filed_date) <= TODAY)]      # no lookahead

# --- universe resolution -------------------------------------------------
# Two traps, both caught by the market-cap sanity check on the first run:
#  (a) 1,449 CIKs map to several tickers (preferreds, units, share classes).
#      Joining on CIK gives the preferred line the PARENT's whole share count.
#  (b) foreign private issuers file 20-F/40-F; their share counts are in the
#      home-market line, not the ADR, so price x shares is nonsense.
# na=False: a cached frames response carries no `form`, so a bootstrap that skips the
# accession-index join leaves it null -- and a NaN mask either raises or, worse, quietly
# resolves to an empty universe. Fail loud below instead.
_dom = f.form.astype("string").str.startswith(("10-K","10-Q")).fillna(False)
DOMESTIC = set(f.loc[_dom, "ticker"])
if len(DOMESTIC) < 1000:
    raise SystemExit(
        f"only {len(DOMESTIC)} domestic filers resolved from {len(f):,} fundamental rows "
        f"({f.form.isna().mean():.1%} of rows have no form). The accession-index join in "
        "recorder/build_fundamentals.py has not run, so the preferred-line and ADR filters "
        "are not doing anything. Refusing to score an unfiltered universe.")
_d = pd.read_parquet(ROOT/"data/prices/derived.parquet")[["ticker","dollar_vol_60d"]]
_u = pd.read_csv(sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1], dtype={"cik": str})
_u = _u.merge(_d, on="ticker", how="left")
_u = _u[~_u.ticker.str.contains(r"[^A-Z]", na=True)]          # drop -UN, -PL, .WS ...
PRIMARY = set(_u.sort_values("dollar_vol_60d", ascending=False)
                .drop_duplicates("cik", keep="first").ticker)  # most-traded line per CIK
KEEP = DOMESTIC & PRIMARY
f = f[f.ticker.isin(KEEP)]
print(f"universe: {len(KEEP)} tickers (domestic filers, one primary line per CIK)")
f["period_end"] = pd.to_datetime(f.period_end)
f["dur"] = (f.period_end - pd.to_datetime(f.period_start)).dt.days

def latest_inst(concept, n=1):
    """nth-latest instantaneous value per ticker (n=1 latest, n=5 ~1yr ago)."""
    d = f[(f.concept == concept) & f.instantaneous].sort_values("period_end")
    g = d.groupby("ticker").tail(n).groupby("ticker")
    return g.first().val if n > 1 else d.groupby("ticker").last().val

def qseries(concept):
    """quarterly duration observations, one row per ticker/period."""
    d = f[(f.concept == concept) & (~f.instantaneous) & f.dur.between(60, 115)]
    return d.sort_values("period_end").drop_duplicates(["ticker","period_end"], keep="last")

def ttm(concept):
    """trailing-4-quarter sum; require all 4 quarters present."""
    d = qseries(concept).groupby("ticker").tail(4)
    g = d.groupby("ticker").val.agg(["sum","count"])
    return g.loc[g["count"] == 4, "sum"]

# ---------- market cap ----------
_h = pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
_h = _h.sort_values("date").groupby("ticker").last()
px = _h["close"]
_snap = pd.read_parquet(ROOT/"data/snapshots/ohlcv_2026-08-18.parquet")
px = px.combine_first(_snap.sort_values("date").groupby("ticker").last()["close"])
# Share-count fallback chain. The dei cover-page tag alone left 36% of liquid
# names without a market cap -- Dell, Ralph Lauren, Planet Fitness were being
# dropped for a missing tag, not for failing any test.
_fb = pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
_fb = _fb[_fb.ticker.isin(KEEP) & (pd.to_datetime(_fb.filed_date) <= TODAY)]
def _latest(frame, concept):
    d = frame[frame.concept == concept].sort_values("period_end")
    return d.groupby("ticker").last().val
sh = latest_inst("EntityCommonStockSharesOutstanding")
_before = sh.notna().sum()
for _c in ["CommonStockSharesOutstanding",
           "WeightedAverageNumberOfDilutedSharesOutstanding",
           "CommonStockSharesIssued"]:                    # issued last: includes treasury
    sh = sh.combine_first(_latest(_fb, _c))
print(f"share counts: {_before} from dei cover page -> {sh.notna().sum()} after fallback chain")
mc = (px * sh).dropna()
mc = mc[mc > 0]
_dv = pd.read_parquet(ROOT/"data/prices/derived.parquet").set_index("ticker").dollar_vol_60d
_impossible = mc.index[(mc > 1e10) & (_dv.reindex(mc.index).fillna(0) < 1e6)]
if len(_impossible):
    print(f"dropped {len(_impossible)} implausible market caps (>$10bn on <$1m/day): "
          f"{', '.join(_impossible[:8])}")
    mc = mc.drop(_impossible)
print(f"market cap: {len(mc)} tickers")
print("  20 largest — SANITY CHECK:")
for t, v in mc.nlargest(20).items(): print(f"    {t:6s} ${v/1e9:8,.0f}bn")

sig = {}
def add(name, s, higher_better=True):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if not higher_better: s = -s
    sig[name] = s

# ---------- valuation / profitability ----------
rev = ttm("Revenues").combine_first(ttm("RevenueFromContractWithCustomerExcludingAssessedTax"))
ocf = ttm("NetCashProvidedByUsedInOperatingActivities")
ni  = ttm("NetIncomeLoss")
assets = latest_inst("Assets"); equity = latest_inst("StockholdersEquity")
cash = latest_inst("CashAndCashEquivalentsAtCarryingValue")

add("SP",  rev/mc)                       # sales-to-price: cheap = high
add("cfp", ocf/mc)                       # cash-flow-to-price
add("bm",  equity/mc)                    # book-to-market
q_ni = qseries("NetIncomeLoss").groupby("ticker").last().val
add("roaq", q_ni/assets)                 # quarterly return on assets
add("PctTotAcc", (ni-ocf)/ni.abs(), higher_better=False)   # accruals: low = clean

# ---------- payout & external financing ----------
div = ttm("PaymentsOfDividendsCommonStock")
buyback = ttm("PaymentsForRepurchaseOfCommonStock")
issue = ttm("ProceedsFromIssuanceOfCommonStock")
npy = (div.reindex(mc.index).fillna(0) + buyback.reindex(mc.index).fillna(0)
       - issue.reindex(mc.index).fillna(0)) / mc
add("NetPayoutYield", npy[div.reindex(mc.index).notna() | buyback.reindex(mc.index).notna()
                          | issue.reindex(mc.index).notna()])

sh_now = latest_inst("EntityCommonStockSharesOutstanding")
sh_old = latest_inst("EntityCommonStockSharesOutstanding", n=5)
add("ShareIss1Y", (sh_now/sh_old - 1), higher_better=False)   # dilution is bad

netdebt = (ttm("ProceedsFromIssuanceOfLongTermDebt").reindex(assets.index).fillna(0)
           - ttm("RepaymentsOfLongTermDebt").reindex(assets.index).fillna(0))
neteq = issue.reindex(assets.index).fillna(0) - buyback.reindex(assets.index).fillna(0)
xfin = (netdebt + neteq)/assets
add("XFIN", xfin[assets.notna() & (issue.reindex(assets.index).notna()
      | ttm("ProceedsFromIssuanceOfLongTermDebt").reindex(assets.index).notna())],
    higher_better=False)                                       # raising money is bad

# ---------- earnings dynamics ----------
# A calendar-quarter pivot silently misaligns companies with off-calendar fiscal
# years, which is why the first run returned 553 names and all-zero streaks.
# Compare each quarter against the observation ~365d earlier in that ticker's OWN series.
q = qseries("NetIncomeLoss")[["ticker","period_end","val"]].sort_values("period_end")
prev = q.copy(); prev["join_end"] = prev.period_end + pd.Timedelta(days=365)
prev = prev.sort_values("join_end")
m = pd.merge_asof(q, prev[["ticker","join_end","val"]].rename(columns={"val":"val_ly"}),
                  left_on="period_end", right_on="join_end", by="ticker",
                  direction="nearest", tolerance=pd.Timedelta(days=45))
m["up"] = m.val > m.val_ly
m = m[m.val_ly.notna()]
recent = m.groupby("ticker").tail(8)
cnt = recent.groupby("ticker").size()
add("NumEarnIncrease", recent.groupby("ticker").up.sum().where(cnt >= 5))
def _streak(g):
    n = 0
    for v in g.up.values[::-1]:
        if not v: break
        n += 1
    return n
add("EarningsStreak", recent.groupby("ticker").apply(_streak).where(cnt >= 5))

# ---------- distress / runway ----------
q_ocf = qseries("NetCashProvidedByUsedInOperatingActivities").groupby("ticker").tail(4)
burn = q_ocf.groupby("ticker").val.sum()/12.0                 # monthly
runway = (cash/(-burn)).where(burn < 0)                        # months, only if burning
lev = (latest_inst("Liabilities")/assets)
add("leverage", lev, higher_better=False)

# ---------- price / technical ----------
d = pd.read_parquet(ROOT/"data/prices/derived.parquet").set_index("ticker")
add("mom_12_1", d.ret_12m_ex1m)
add("mom_6m", d.ret_6m)
add("px_to_52w_high", d.px_to_52w_high)
add("vol_60d", d.vol_60d, higher_better=False)
add("max_ret_1m", d.max_ret_1m, higher_better=False)
add("volume_trend", d.volume_trend)
add("rel_strength", d.ret_12m - d.ret_12m.median())

# ---------- investment / asset growth ----------
# Companies that expand their asset base fastest go on to underperform
# (Cooper/Gulen/Schill 2008; Hirshleifer et al 2004 for net operating assets).
# Added 2026-09-06 -- the only two of nineteen new candidates meeting the standard
# set for momentum, book-to-market and the 52-week high: documented in the
# published record AND passing the four-test screen on our own data.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("pit_growth", ROOT/"signals/pit_growth.py")
_g = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_g)
try:
    _sh_all = pd.read_parquet(ROOT/"data/sec/shares_fallback.parquet")
    _sh_all["filed_date"] = pd.to_datetime(_sh_all.filed_date)
    _hist = _h["close"].to_frame().T if False else None
    _px_hist = pd.read_parquet(ROOT/"data/prices/history_3y.parquet")
    _px_hist["date"] = pd.to_datetime(_px_hist.date)
    _px_hist = _px_hist[_px_hist.adj_close >= 1.0]
    _PX = _px_hist.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    _out = _g.compute(f, _sh_all[_sh_all.filed_date <= TODAY], px, _PX.loc[:TODAY], TODAY)
    for _c in ("asset_growth", "d_noa"):
        if _c in _out.columns:
            add(_c, pd.to_numeric(_out[_c], errors="coerce"))
    print(f"investment signals: asset_growth {sig.get('asset_growth', pd.Series(dtype=float)).notna().sum()}, "
          f"d_noa {sig.get('d_noa', pd.Series(dtype=float)).notna().sum()}")
except Exception as _e:
    print(f"investment signals FAILED: {type(_e).__name__}: {_e}")

# ---------- assemble ----------
df = pd.DataFrame(sig)
for c in df.columns:                                            # winsorise then rank
    lo, hi = df[c].quantile([.01, .99])
    df[c] = df[c].clip(lo, hi)
rank = df.rank(pct=True)
rank["mktcap"] = mc; rank["runway_months"] = runway
rank.to_parquet(OUT/"signal_ranks.parquet")
df.to_parquet(OUT/"signal_raw.parquet")

print(f"\n{'signal':18s} {'cover':>6s}  {'p10':>12s} {'median':>12s} {'p90':>12s}")
for c in df.columns:
    s = df[c].dropna(); q1,q5,q9 = s.quantile([.1,.5,.9])
    print(f"{c:18s} {len(s):6d}  {q1:12.4g} {q5:12.4g} {q9:12.4g}")
print(f"\nwrote {OUT/'signal_ranks.parquet'}  ({len(rank)} tickers x {len(df.columns)} signals)")
print(f"cash-burners with runway < 12mo: {(runway < 12).sum()}")
