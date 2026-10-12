#!/usr/bin/env python3
"""Earnings-dynamics signal group for the US equity selection system.

Builds, from SEC XBRL frames only (no paid analyst data):

  earnings_streak            OSAP EarningsStreak, re-specified on a time-series
                             surprise because the catalog definition needs IBES.
  num_earn_increase          OSAP NumEarnIncrease
  revenue_surprise           OSAP RevenueSurprise
  earnings_consistency       OSAP EarningsConsistency
  earnings_growth_stability  inverse volatility of YoY earnings growth (extra)
  sue                        OSAP EarningsSurprise == Foster/Olsen/Shevlin SUE

Run:  python3 signals/build_earnings.py
Prereq: python3 signals/fetch_earnings_frames.py   (idempotent, cached on disk)
"""
import pathlib
import glob, gzip, json, os, sys
import numpy as np
import pandas as pd

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
OUT = os.path.join(ROOT, "signals")
ASOF = pd.Timestamp("2026-08-18")

# calendar-frame quarter grid: CY2019Q1 .. CY2026Q2  -> columns 0..29
Y0, Y1, LASTQ = 2019, 2026, 2
NQ = (Y1 - Y0) * 4 + LASTQ                      # 30
# annual grid CY2018..CY2026 -> columns 0..8
AY0, AY1 = 2018, 2026
NA = AY1 - AY0 + 1

# a signal must be anchored no earlier than this column (CY2025Q4) or the filer is stale
MIN_ANCHOR_COL = (2025 - Y0) * 4 + 3            # 27

# investable-universe screen (OSAP's own abs(prc)>5 filter + a liquidity floor)
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 1_000_000.0

CONCEPTS = [
    "EarningsPerShareDiluted", "EarningsPerShareBasic",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "NetIncomeLoss", "ProfitLoss",
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
# per-family waterfall: ONE source is chosen per ticker (never mixed within a series)
EPS_SRC = ["EarningsPerShareDiluted", "EarningsPerShareBasic",
           "IncomeLossFromContinuingOperationsPerDilutedShare"]
IB_SRC = ["NetIncomeLoss", "ProfitLoss"]
REV_SRC = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
           "RevenueFromContractWithCustomerIncludingAssessedTax"]
SH_SRC = ["WeightedAverageNumberOfSharesOutstandingBasic",
          "WeightedAverageNumberOfDilutedSharesOutstanding"]
# annual WASO is an AVERAGE over the year, not a sum -> Q4 = 4*FY - Q1 - Q2 - Q3
AVG_CONCEPTS = set(SH_SRC)

NONCOMMON = ("-WT", "-RT", "-UN", "-RI", "-P", "W", "R", "U")   # suffix screen, see notes


# ----------------------------------------------------------------------------- load
def load_frames():
    recs = []
    for d in ("data/sec/frames", "data/sec/frames_earn"):
        for fp in glob.glob(os.path.join(ROOT, d, "*.json.gz")):
            base = os.path.basename(fp)[: -len(".json.gz")]
            parts = base.split("_")
            fid, concept = parts[-1], "_".join(parts[1:-2])
            if concept not in CONCEPTS or fid.endswith("I"):
                continue                       # instantaneous frames are not flows
            with gzip.open(fp, "rt") as fh:
                js = json.load(fh)
            core = fid[2:]
            fy = int(core[:4])
            fq = core[4:] or "FY"
            for x in js.get("data", []):
                recs.append((x["cik"], concept, fy, fq, x.get("end"),
                             x.get("val"), x.get("accn")))
    df = pd.DataFrame(recs, columns=["cik_int", "concept", "fy", "fq",
                                     "period_end", "val", "accn"])
    df["cik"] = df.cik_int.astype(int).astype(str).str.zfill(10)
    df["val"] = pd.to_numeric(df.val, errors="coerce")
    df["period_end"] = pd.to_datetime(df.period_end, errors="coerce")
    return df.drop(columns=["cik_int"])


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    df = load_frames()
    print("frame rows loaded: %d  concepts=%d  ciks=%d"
          % (len(df), df.concept.nunique(), df.cik.nunique()))

    # ---- point-in-time gate: only facts that were actually filed on or before ASOF
    ai = pd.read_parquet(os.path.join(ROOT, "data/sec/accn_index.parquet"))
    df = df.merge(ai, on="accn", how="left")
    df["filed_date"] = pd.to_datetime(df.filed_date, errors="coerce")
    n_nofile = df.filed_date.isna().sum()
    future = (df.filed_date > ASOF).sum()
    df = df[df.filed_date.notna() & (df.filed_date <= ASOF)]
    print("filed_date: unmatched %d rows dropped, %d rows filed after ASOF dropped -> %d rows"
          % (n_nofile, future, len(df)))

    # ---- ticker map (many-to-many: share classes and, unavoidably, some non-common lines)
    u = pd.read_csv(os.path.join(ROOT, "data/universe/tickers_2026-08-18.csv"),
                    dtype={"cik": str})[["cik", "ticker"]]
    df = df.merge(u, on="cik", how="inner")
    df = df[df.ticker.notna() & (df.ticker.astype(str) != "nan")]
    df["ticker"] = df.ticker.astype(str)
    print("after ticker join: %d rows, %d tickers" % (len(df), df.ticker.nunique()))

    # ---- grid coordinates
    qmap = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    isq = df.fq.isin(qmap)
    df["col"] = np.where(isq, (df.fy - Y0) * 4 + df.fq.map(qmap).fillna(-1), -1).astype(int)
    df["acol"] = np.where(~isq, df.fy - AY0, -1).astype(int)
    q = df[isq & (df.col >= 0) & (df.col < NQ)]
    a = df[(~isq) & (df.acol >= 0) & (df.acol < NA)]
    print("quarterly rows %d | annual rows %d" % (len(q), len(a)))

    tickers = np.array(sorted(df.ticker.unique()))
    tidx = {t: i for i, t in enumerate(tickers)}
    N = len(tickers)

    def to_grid(sub, ncol, colname):
        """(ticker, concept) -> dense [N, ncol] array per concept."""
        out = {}
        for c, g in sub.groupby("concept"):
            M = np.full((N, ncol), np.nan)
            ri = g.ticker.map(tidx).to_numpy()
            M[ri, g[colname].to_numpy()] = g.val.to_numpy()
            out[c] = M
        return out

    Q = to_grid(q, NQ, "col")
    A = to_grid(a, NA, "acol")
    # period_end per quarterly column, for the staleness check
    PE = np.full((N, NQ), np.datetime64("NaT"), dtype="datetime64[ns]")
    PE[q.ticker.map(tidx).to_numpy(), q.col.to_numpy()] = q.period_end.to_numpy()

    # ---- derive the missing calendar-Q4 stub -----------------------------------
    # 10-Ks tag the 12-month figure, not the Q4 3-month column, so the CY..Q4 frame
    # holds only ~1,200 filers vs ~5,500 for Q1-Q3. Q4 = FY - Q1 - Q2 - Q3.
    derived_counts = {}
    for c in CONCEPTS:
        if c not in Q or c not in A:
            continue
        M, AM = Q[c], A[c]
        mult = 4.0 if c in AVG_CONCEPTS else 1.0
        for y in range(Y0, Y1):
            ac = y - AY0
            c1, c2, c3, c4 = [(y - Y0) * 4 + k for k in range(4)]
            if ac < 0 or ac >= NA or c4 >= NQ:
                continue
            parts = M[:, c1] + M[:, c2] + M[:, c3]
            cand = mult * AM[:, ac] - parts
            fill = np.isnan(M[:, c4]) & np.isfinite(cand)
            M[fill, c4] = cand[fill]
            derived_counts[c] = derived_counts.get(c, 0) + int(fill.sum())
            # period_end for a derived Q4 = calendar quarter end (only used for staleness)
            pe = np.datetime64("%d-12-31" % y)
            need = fill & np.isnat(PE[:, c4])
            PE[need, c4] = pe
    print("Q4 stubs derived: " + ", ".join("%s=%d" % (k, v) for k, v in
                                           sorted(derived_counts.items())))

    # ---- Q4 derivation accuracy check (where a real Q4 frame also exists) -------
    chk = []
    for c in ("EarningsPerShareDiluted", "NetIncomeLoss"):
        Mr = to_grid(q[q.concept == c], NQ, "col").get(c)
        if Mr is None:
            continue
        for y in range(Y0, Y1):
            ac, c4 = y - AY0, (y - Y0) * 4 + 3
            cols = [(y - Y0) * 4 + k for k in range(4)]
            if c not in A:
                continue
            mult = 4.0 if c in AVG_CONCEPTS else 1.0
            cand = mult * A[c][:, ac] - (Mr[:, cols[0]] + Mr[:, cols[1]] + Mr[:, cols[2]])
            both = np.isfinite(cand) & np.isfinite(Mr[:, c4])
            if both.sum() > 20:
                err = np.abs(cand[both] - Mr[both, c4])
                scale = np.maximum(np.abs(Mr[both, c4]), 1e-9)
                chk.append((c, y, int(both.sum()),
                            float(np.median(err / scale)),
                            float(np.mean(err / scale < 0.02))))
    for c, y, n, med, frac in chk:
        print("  Q4 check %-24s CY%d n=%4d  median rel.err=%.4f  within2%%=%.2f"
              % (c, y, n, med, frac))

    # ---- one source per ticker per family --------------------------------------
    def pick(srcs, label, window=16):
        """Choose, per ticker, the single concept with the most observations in the
        last `window` quarters. Never mix concepts within one ticker's series: a
        switch from Diluted to Basic EPS is a level shift that fakes a surprise."""
        mats = [Q.get(s, np.full((N, NQ), np.nan)) for s in srcs]
        counts = np.stack([np.isfinite(m[:, NQ - window:]).sum(1) for m in mats])
        best = counts.argmax(0)
        best[counts.max(0) == 0] = -1
        M = np.full((N, NQ), np.nan)
        for i, s in enumerate(srcs):
            sel = best == i
            M[sel] = mats[i][sel]
        used = pd.Series(np.where(best < 0, "none", np.array(srcs + ["none"])[best]))
        print("%s source: %s" % (label, used.value_counts().to_dict()))
        return M

    def pick_annual(srcs, label):
        mats = [A.get(s, np.full((N, NA), np.nan)) for s in srcs]
        counts = np.stack([np.isfinite(m).sum(1) for m in mats])
        best = counts.argmax(0)
        best[counts.max(0) == 0] = -1
        M = np.full((N, NA), np.nan)
        for i, s in enumerate(srcs):
            sel = best == i
            M[sel] = mats[i][sel]
        used = pd.Series(np.where(best < 0, "none", np.array(srcs + ["none"])[best]))
        print("%s(annual) source: %s" % (label, used.value_counts().to_dict()))
        return M

    EPS = pick(EPS_SRC, "eps_q")
    IB = pick(IB_SRC, "ib_q")
    REV = pick(REV_SRC, "rev_q")
    SH = pick(SH_SRC, "shares_q")
    EPSA = pick_annual(EPS_SRC, "eps")

    # ---------------------------------------------------------------- primitives
    def yoy(M):
        """M[:, t] - M[:, t-4], NaN where either leg missing (no forward fill)."""
        out = np.full_like(M, np.nan)
        out[:, 4:] = M[:, 4:] - M[:, :-4]
        return out

    def roll_mean_sd(G, w=8, minobs=6):
        """Mean and sample SD of G over the w quarters STRICTLY BEFORE t."""
        mu = np.full_like(G, np.nan)
        sd = np.full_like(G, np.nan)
        for t in range(NQ):
            lo = t - w
            if lo < 0:
                continue
            win = G[:, lo:t]
            n = np.isfinite(win).sum(1)
            ok = n >= minobs
            with np.errstate(invalid="ignore"):
                m = np.nanmean(win, axis=1)
                s = np.nanstd(win, axis=1, ddof=1)
            mu[ok, t] = m[ok]
            sd[ok, t] = s[ok]
        return mu, sd

    def standardised_surprise(M):
        G = yoy(M)
        mu, sd = roll_mean_sd(G)
        with np.errstate(invalid="ignore", divide="ignore"):
            S = (G - mu) / sd
        S[~np.isfinite(S)] = np.nan
        S[sd <= 0] = np.nan
        return G, S

    stale_ok = np.zeros((N, NQ), dtype=bool)
    stale_ok[:, MIN_ANCHOR_COL:] = True

    def last_valid(S, require_base=None):
        """Value at each ticker's most recent quarter where S is finite and the
        anchor is CY2025Q4 or later. Returns (value, anchor_col)."""
        ok = np.isfinite(S) & stale_ok
        if require_base is not None:
            ok &= np.isfinite(require_base)
        has = ok.any(1)
        col = np.where(has, NQ - 1 - ok[:, ::-1].argmax(1), -1)
        val = np.full(N, np.nan)
        val[has] = S[has, col[has]]
        return val, col

    # ---------------------------------------------------------------- signals
    G_eps, SURP = standardised_surprise(EPS)

    # SUE / OSAP EarningsSurprise
    sue, sue_col = last_valid(SURP)

    # EarningsStreak: OSAP keeps a firm only when the current surprise has the same
    # sign as the previous one; signal = the surprise. Non-streak firms get 0 so the
    # cross-section stays complete and they sit neutral in a composite.
    prev = np.full_like(SURP, np.nan)
    prev[:, 1:] = SURP[:, :-1]
    same = np.isfinite(SURP) & np.isfinite(prev) & (np.sign(SURP) == np.sign(prev)) & (SURP != 0)
    STREAK = np.where(np.isfinite(SURP) & np.isfinite(prev), np.where(same, SURP, 0.0), np.nan)
    streak, streak_col = last_valid(STREAK)
    n_on_streak = np.isfinite(streak) & (streak != 0)

    # NumEarnIncrease: length of the current run of consecutive quarters with
    # ib[t] > ib[t-4], capped at 8.
    G_ib = yoy(IB)
    INC = np.where(np.isfinite(G_ib), (G_ib > 0).astype(float), np.nan)
    RUN = np.full_like(INC, np.nan)
    for t in range(NQ):
        if t < 4:
            continue
        run = np.zeros(N)
        alive = np.isfinite(INC[:, t])
        for k in range(8):
            c = t - k
            if c < 4:
                break
            step = alive & np.isfinite(INC[:, c]) & (INC[:, c] == 1)
            run = np.where(step, run + 1, run)
            alive = alive & step
        RUN[np.isfinite(INC[:, t]), t] = run[np.isfinite(INC[:, t])]
    num_inc, num_col = last_valid(RUN, require_base=INC)
    # variant the brief described (count within last 8, not consecutive) - reported only
    CNT8 = np.full_like(INC, np.nan)
    for t in range(11, NQ):
        win = INC[:, max(0, t - 7):t + 1]
        n = np.isfinite(win).sum(1)
        CNT8[n >= 6, t] = np.nansum(win, axis=1)[n >= 6]
    cnt8, _ = last_valid(CNT8)

    # RevenueSurprise: revenue per share, same seasonal-random-walk-with-drift form
    with np.errstate(invalid="ignore", divide="ignore"):
        RPS = np.where((SH > 0) & np.isfinite(REV), REV / SH, np.nan)
    G_rps, RSURP = standardised_surprise(RPS)
    revsurp, rev_col = last_valid(RSURP)

    # EarningsConsistency: average annual earnings growth over the previous 48 months
    EG = np.full((N, NA), np.nan)
    for y in range(2, NA):
        num = EPSA[:, y] - EPSA[:, y - 1]
        den = 0.5 * (np.abs(EPSA[:, y - 1]) + np.abs(EPSA[:, y - 2]))
        with np.errstate(invalid="ignore", divide="ignore"):
            g = num / den
        g[~np.isfinite(g)] = np.nan
        EG[:, y] = g
    EG[np.abs(EG) > 6.0] = np.nan                       # |growth| > 600% excluded
    lastA = np.where(np.isfinite(EG).any(1), NA - 1 - np.isfinite(EG)[:, ::-1].argmax(1), -1)
    econs = np.full(N, np.nan)
    egstab = np.full(N, np.nan)
    for i in range(N):
        e = lastA[i]
        if e < 3:
            continue
        win = EG[i, e - 3:e + 1]
        if np.isfinite(win).sum() < 3:
            continue
        g0, g1 = EG[i, e], EG[i, e - 1]
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue
        if np.sign(g0) != np.sign(g1):                  # sign-flip exclusion
            continue
        econs[i] = np.nanmean(win)
    # extra: inverse volatility of YoY earnings growth (the brief's wording)
    RG = np.full((N, NQ), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = 0.5 * (np.abs(EPS[:, :-4]) + np.abs(np.roll(EPS, 4, axis=1)[:, :-4]))
    scale = np.full((N, NQ), np.nan)
    scale[:, 4:] = 0.5 * (np.abs(EPS[:, :-4]) + np.abs(EPS[:, 4:]))
    with np.errstate(invalid="ignore", divide="ignore"):
        RG = G_eps / scale
    RG[~np.isfinite(RG)] = np.nan
    RG[np.abs(RG) > 6.0] = np.nan
    for i in range(N):
        w = RG[i, max(0, NQ - 12):]
        w = w[np.isfinite(w)]
        if len(w) >= 8:
            s = np.std(w, ddof=1)
            if s > 0:
                egstab[i] = 1.0 / s
    # anchor staleness for the two annual/long-window signals: reuse the eps anchor
    eps_anchor_ok = np.isfinite(EPS[:, MIN_ANCHOR_COL:]).any(1)
    econs[~eps_anchor_ok] = np.nan
    egstab[~eps_anchor_ok] = np.nan

    # ---------------------------------------------------------------- universe
    d = pd.read_parquet(os.path.join(ROOT, "data/prices/derived.parquet"))
    g = pd.read_parquet(os.path.join(ROOT, "data/prices/derived_diagnostics.parquet"))
    px = d[["ticker", "dollar_vol_60d"]].merge(
        g[["ticker", "last_close", "quarantined", "unadjustable"]], on="ticker")
    px = px.set_index("ticker")
    lc = px.last_close.reindex(tickers).to_numpy(dtype=float)
    dv = px.dollar_vol_60d.reindex(tickers).to_numpy(dtype=float)
    qn = px.quarantined.reindex(tickers).fillna(True).to_numpy(dtype=bool)
    un = px.unadjustable.reindex(tickers).fillna(True).to_numpy(dtype=bool)
    suffix_bad = np.array([t.endswith(("-WT", "-RT", "-UN", "-RI")) or
                           ("-P" in t) for t in tickers])
    inuniv = (np.isfinite(lc) & (lc >= MIN_PRICE) & np.isfinite(dv) &
              (dv >= MIN_DOLLAR_VOL) & ~qn & ~un & ~suffix_bad)
    print("\ninvestable universe: %d of %d SEC-mapped tickers "
          "(last_close>=$%.0f, median 60d $vol>=$%.0fm, not quarantined/unadjustable)"
          % (inuniv.sum(), N, MIN_PRICE, MIN_DOLLAR_VOL / 1e6))
    pd.DataFrame({"ticker": tickers, "investable": inuniv, "last_close": lc,
                  "dollar_vol_60d": dv}).to_parquet(
        os.path.join(OUT, "_universe_screen.parquet"), index=False)

    # ---------------------------------------------------------------- write
    results = {}

    def emit(name, raw, formula, concepts_used, extra_notes, higher_is_better=True):
        v = raw.copy()
        v[~inuniv] = np.nan
        s = pd.Series(v, index=tickers).dropna()
        n_all = int(np.isfinite(raw).sum())
        if len(s) == 0:
            print("!! %s produced nothing" % name)
            return
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        w = s.clip(lo, hi)                                    # winsorise 1% / 99%
        rp = w.rank(pct=True, method="average")
        out = pd.DataFrame({"ticker": w.index, "value": w.to_numpy(),
                            "rank_pct": rp.to_numpy()}).sort_values("ticker")
        out.to_parquet(os.path.join(OUT, name + ".parquet"), index=False)
        results[name] = dict(cov=len(out), cov_all=n_all, lo=float(lo), hi=float(hi),
                             series=s, w=w, formula=formula, concepts=concepts_used,
                             notes=extra_notes)
        print("\n--- %-26s coverage %d (investable) / %d (all SEC-mapped)"
              % (name, len(out), n_all))
        top = w.sort_values(ascending=False).head(10)
        bot = w.sort_values().head(10)
        print("    top10: " + ", ".join("%s %.2f" % (k, x) for k, x in top.items()))
        print("    bot10: " + ", ".join("%s %.2f" % (k, x) for k, x in bot.items()))
        print("    median $vol top10 = $%.1fm | universe median $%.1fm"
              % (np.nanmedian(px.dollar_vol_60d.reindex(top.index)) / 1e6,
                 np.nanmedian(dv[inuniv]) / 1e6))

    emit("earnings_streak", streak,
         "surp_t if sign(surp_t)==sign(surp_{t-1}) else 0, where "
         "surp_t = (dEPS_t - mean(dEPS_{t-8..t-1})) / sd(dEPS_{t-8..t-1}), dEPS_t = EPS_t - EPS_{t-4}",
         EPS_SRC, "n with a live streak: %d of %d" % (int((n_on_streak & inuniv).sum()),
                                                      int(np.isfinite(streak[inuniv]).sum())))
    emit("num_earn_increase", num_inc,
         "length of the current run of consecutive quarters with ib_t > ib_{t-4}, capped at 8",
         IB_SRC, "")
    emit("revenue_surprise", revsurp,
         "(dRPS_t - mean(dRPS_{t-8..t-1})) / sd(dRPS_{t-8..t-1}); "
         "RPS = quarterly revenue / weighted basic shares; dRPS_t = RPS_t - RPS_{t-4}",
         REV_SRC + SH_SRC, "")
    emit("earnings_consistency", econs,
         "mean of the last 4 annual values of g_y = (EPS_y - EPS_{y-1}) / "
         "(0.5*(|EPS_{y-1}|+|EPS_{y-2}|)); drop |g|>6; require sign(g_y)==sign(g_{y-1})",
         EPS_SRC, "")
    emit("earnings_growth_stability", egstab,
         "1 / sd(YoY quarterly earnings growth over the last 12 quarters), "
         "growth = (EPS_t - EPS_{t-4}) / (0.5*(|EPS_t|+|EPS_{t-4}|)), >=8 obs",
         EPS_SRC, "")
    emit("sue", sue,
         "(dEPS_t - mean(dEPS_{t-8..t-1})) / sd(dEPS_{t-8..t-1}), dEPS_t = EPS_t - EPS_{t-4}",
         EPS_SRC, "")

    # ------------------------------------------------------------ diagnostics
    print("\n" + "=" * 78)
    inv = pd.Series(inuniv, index=tickers)
    dfd = pd.DataFrame({k: v["series"] for k, v in results.items()})
    print("pairwise rank correlation (Spearman, investable universe):")
    print(dfd.corr(method="spearman").round(2).to_string())
    both = np.isfinite(num_inc) & np.isfinite(cnt8) & inuniv
    if both.sum() > 50:
        print("\nNumEarnIncrease (consecutive, catalog) vs count-in-last-8 (brief's wording): "
              "spearman %.3f over %d names"
              % (pd.Series(num_inc[both]).corr(pd.Series(cnt8[both]), method="spearman"),
                 both.sum()))
        print("  consecutive distribution:", pd.Series(num_inc[inuniv]).value_counts()
              .sort_index().to_dict())
    anch = pd.Series(sue_col[np.isfinite(sue) & inuniv]).value_counts().sort_index()
    print("\nSUE anchor quarter (grid col; 27=CY2025Q4, 28=CY2026Q1, 29=CY2026Q2):",
          anch.to_dict())

    with open(os.path.join(OUT, "_earnings_build_summary.json"), "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk in ("cov", "cov_all", "lo", "hi")} for k, v in results.items()},
                  fh, indent=1)
    return results, dfd, px, inuniv, tickers


if __name__ == "__main__":
    main()
