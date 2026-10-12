"""Per-ticker price / momentum / volatility features from history_3y.parquet.

ANCHOR: the last complete trading session. The fetch ran while the US market was
open, so the in-progress bar is excluded upstream and the anchor is 2026-08-17.

Positional lags are measured on a shared trading calendar (dates on which >=25%
of the cross-section printed), so "21 sessions ago" is the same date for every
ticker. Prices are reindexed onto it and forward-filled at most 5 sessions:
illiquid names genuinely do not print daily, and a stale quote for <=1 week is
standard. That ffill window doubles as the validity test -- if the ffilled price
is NaN at a lag, no real print existed within 5 sessions of it and the feature
stays NaN instead of being invented.

THREE DATA DEFECTS ARE REPAIRED HERE, because each one silently manufactures
returns that would land at the very top of a momentum ranking:

 1. Corrupt dividend adjustment. For thin foreign OTC lines Yahoo's Adj Close
    goes negative (observed down to -2.4e14) or absurd (up to 2.5e17), which
    produces returns below -100%. adj_close/close must lie in (0, 1.05]:
    backward adjustment can only pull past prices DOWN and equals 1 at the
    anchor. Rows outside that are dropped; a ticker more than 1% bad is
    discarded entirely, since its surviving fragments sit on inconsistent
    adjustment bases and splicing them would invent returns.

 2. Unapplied splits. Yahoo lags on split factors for microcaps: DFNS went
    $0.042 -> $4.26 in one session with adj_close never adjusted, giving a
    +73,686% ret_1m. A split shows an unmistakable signature -- price scales by
    k while volume scales by ~1/k on the same day. Detected splits are repaired
    by applying the cumulative factor backwards.

 3. Residual unexplained jumps. A >5x or <1/5 single-session move that the
    volume test cannot explain is not a return we can trust, so the ticker's
    adjusted-price features are nulled rather than reported. Most are sub-penny
    tape (bankruptcy shells printing $0.0002 -> $0.057) that the liquidity floor
    removes anyway, but nulling them means no fake 700x ever reaches a ranking.

Daily returns are computed on ffilled prices but kept only on sessions with a
real print, so a non-trading day contributes no fake 0% return to vol.

Volume is NOT forward-filled: a session with no print has zero dollar volume,
and calling it missing would flatter exactly the illiquid names a liquidity
floor exists to remove. Dollar volume uses raw close * volume -- the actual cash
traded, and split-invariant by construction, so it needs no repair.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "prices"
HIST = P / "history_3y.parquet"
OUT = P / "derived.parquet"

D_1M, D_6M, D_12M, D_60, D_250, D_252 = 21, 126, 252, 60, 250, 252
FFILL = 5
MIN_CAL_COVERAGE = 0.25
MIN_VOL_OBS = 30

ADJ_RATIO_MAX = 1.05      # adj_close/close above this is a broken adjustment
MAX_BAD_ADJ_FRAC = 0.01   # ticker-level: more than this and the whole series goes
SPLIT_HI, SPLIT_LO = 4.0, 0.25      # candidate single-day price jump
VOL_TOL_LO, VOL_TOL_HI = 0.1, 10.0  # (vol_before/vol_after)/price_ratio must be ~1
RESID_HI, RESID_LO = 5.0, 0.2       # unexplained jump this large -> quarantine

COLS = ["ticker", "ret_1m", "ret_6m", "ret_12m", "ret_12m_ex1m", "vol_60d",
        "max_ret_1m", "px_to_52w_high", "drawdown_from_peak", "dollar_vol_60d",
        "listing_age_days", "volume_trend"]


def main() -> int:
    df = pd.read_parquet(HIST)
    df["date"] = pd.to_datetime(df["date"])

    n_by_date = df.groupby("date")["ticker"].nunique()
    panel_n = df["ticker"].nunique()
    calendar = n_by_date[n_by_date >= MIN_CAL_COVERAGE * panel_n].index.sort_values()
    anchor = calendar[-1]
    print(f"panel {panel_n} tickers · {len(calendar)} sessions · anchor {anchor.date()}")
    if len(calendar) < D_12M + D_1M + 5:
        raise SystemExit(f"calendar too short: {len(calendar)} sessions")
    df = df[df["date"].isin(calendar)]

    # --- defect 1: corrupt dividend/split adjustment -------------------------
    ratio_ac = df["adj_close"] / df["close"].where(df["close"] > 0)
    row_bad = ~(ratio_ac > 0) | ~(ratio_ac <= ADJ_RATIO_MAX) | ~np.isfinite(ratio_ac)
    bad_frac = row_bad.groupby(df["ticker"]).mean()
    unadjustable = set(bad_frac.index[bad_frac > MAX_BAD_ADJ_FRAC])
    df.loc[row_bad | df["ticker"].isin(unadjustable), "adj_close"] = np.nan
    df.loc[df["close"] <= 0, ["close", "adj_close"]] = np.nan
    print(f"defect 1 adjustment : {int(row_bad.sum()):,} bad rows · "
          f"{len(unadjustable)} tickers unadjustable")
    pd.Series(sorted(unadjustable), name="ticker").to_csv(
        P / "bad_adjustment_tickers.csv", index=False)

    px = df.pivot(index="date", columns="ticker", values="adj_close").reindex(calendar)
    cl = df.pivot(index="date", columns="ticker", values="close").reindex(calendar)
    vo = df.pivot(index="date", columns="ticker", values="volume").reindex(calendar)
    obs = px.notna()

    # --- defect 2: unapplied splits -----------------------------------------
    # Compare each real print to the PREVIOUS real print, however long the gap.
    # A limited ffill would leave the ratio undefined across a long halt, and a
    # split straddling one would then go undetected and unrepaired.
    jump = px / px.ffill().shift(1)
    vm = vo.where(obs)
    v_before = vm.rolling(20, min_periods=3).median().shift(1)
    v_after = vm.iloc[::-1].rolling(20, min_periods=3).median().iloc[::-1]
    cand = (jump > SPLIT_HI) | (jump < SPLIT_LO)
    implied = (v_before / v_after) / jump
    conf = cand & (implied > VOL_TOL_LO) & (implied < VOL_TOL_HI) & (v_before > 0) & (v_after > 0)

    k = jump.where(conf, 1.0).fillna(1.0)
    cum = k.cumprod()
    factor = cum.iloc[-1] / cum          # product of split factors strictly after each date
    px = px * factor                      # backward-adjust the real observations
    print(f"defect 2 splits     : {int(conf.sum().sum())} repaired on "
          f"{int((conf.sum() > 0).sum())} tickers")

    splits = [{"ticker": t, "date": d.date(), "factor": round(jump.loc[d, t], 4)}
              for t in conf.columns[conf.sum() > 0] for d in conf.index[conf[t]]]
    pd.DataFrame(splits).sort_values(["ticker", "date"]).to_csv(P / "splits_repaired.csv", index=False)

    # --- defect 3: residual unexplained jumps -> quarantine ------------------
    resid = px / px.ffill().shift(1)
    bad_jump = (resid > RESID_HI) | (resid < RESID_LO)
    quarantine = bad_jump.columns[bad_jump.sum() > 0]
    px[quarantine] = np.nan
    pxf = px.ffill(limit=FFILL)
    obs_px = px.notna()
    print(f"defect 3 quarantine : {len(quarantine)} tickers with unexplained >5x moves")
    pd.Series(sorted(quarantine), name="ticker").to_csv(P / "quarantined_tickers.csv", index=False)

    # --- features ------------------------------------------------------------
    first_seen = cl.apply(lambda s: s.first_valid_index())   # listing date: raw close
    clf = cl.ffill(limit=FFILL)
    listed = pd.DataFrame(
        calendar.values[:, None] >= pd.to_datetime(first_seen).values[None, :],
        index=calendar, columns=cl.columns,
    )
    vof = vo.fillna(0.0).where(listed)

    def at(frame, lag):
        return frame.iloc[-1 - lag]

    last = at(pxf, 0)
    out = pd.DataFrame(index=cl.columns)
    out.index.name = "ticker"

    out["ret_1m"] = last / at(pxf, D_1M) - 1
    out["ret_6m"] = last / at(pxf, D_6M) - 1
    out["ret_12m"] = last / at(pxf, D_12M) - 1
    out["ret_12m_ex1m"] = at(pxf, D_1M) / at(pxf, D_12M) - 1

    rets = pxf.pct_change(fill_method=None).where(obs_px)
    win = rets.iloc[-D_60:]
    out["vol_60d"] = win.std().where(win.notna().sum() >= MIN_VOL_OBS) * np.sqrt(252)
    out["max_ret_1m"] = rets.iloc[-D_1M:].max()

    out["px_to_52w_high"] = last / pxf.iloc[-D_252:].max()
    out["drawdown_from_peak"] = last / pxf.cummax().iloc[-1] - 1

    dv = clf * vof
    out["dollar_vol_60d"] = dv.iloc[-D_60:].median()
    out["volume_trend"] = out["dollar_vol_60d"] / dv.iloc[-D_250:].median().replace(0, np.nan)
    out["listing_age_days"] = (pd.Timestamp(dt.date.today()) - pd.to_datetime(first_seen)).dt.days

    pd.DataFrame({
        "traded_days_3y": cl.notna().sum(),
        "traded_days_60": cl.iloc[-D_60:].notna().sum(),
        "first_seen": pd.to_datetime(first_seen),
        "last_close": at(clf, 0),
        "unadjustable": cl.columns.isin(list(unadjustable)),
        "quarantined": cl.columns.isin(list(quarantine)),
    }).reset_index().rename(columns={"index": "ticker"}).to_parquet(
        P / "derived_diagnostics.parquet", index=False)

    out = out.reset_index()[COLS]
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} · {len(out)} rows · anchor {anchor.date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
