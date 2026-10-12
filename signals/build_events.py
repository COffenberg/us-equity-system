#!/usr/bin/env python3
"""
Build the "Event & structural" signal group for the US equity selection system.

Signals produced
----------------
  age_ipo      listing age in days (OSAP `AgeIPO` / `FirmAge` stand-in), from the
               3y price panel, lower-bounded by historical SEC roster snapshots.
  exch_switch  NOT BUILDABLE from free data -> markdown note only, no parquet.
  div_omit     consistent quarterly dividend payer that paid zero last quarter,
               reconstructed from XBRL cash-flow dividend tags.
  cash_runway  cash / (trailing-4-quarter operating cash burn / 12) = months of
               runway. Exclusion-floor variable, burners only.

Correctness rules honoured
--------------------------
  * point-in-time: every XBRL fact is gated on its actual `filed` date, and where
    a fact was restated we keep the AS-FIRST-REPORTED value (min filed date).
  * no forward-fill of a fundamental across more than one quarter.
  * winsorise 1%/99% before ranking; rank cross-sectionally to percentile.
  * survivorship is not a concern (we score today's cross-section).

Usage
-----
  python3 signals/build_events.py            # fetch (cached) + build
  python3 signals/build_events.py --no-fetch # build from cache only
"""
from __future__ import annotations

import pathlib
import argparse
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- paths
ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
UNIVERSE = f"{ROOT}/data/universe/tickers_2026-08-18.csv"
DERIVED = f"{ROOT}/data/prices/derived.parquet"
FUNDA = f"{ROOT}/data/sec/fundamentals.parquet"
SIGDOC = f"{ROOT}/data/osap/SignalDoc.csv"
HISTROSTER = f"{ROOT}/data/hist"
OUT = f"{ROOT}/signals"
CC_CACHE = f"{ROOT}/data/sec/companyconcept"
FR_CACHE = f"{ROOT}/data/sec/frames"

TODAY = pd.Timestamp("2026-08-18")
UA = {
    "User-Agent": "Offenberg US-Equity-System " + os.environ["SEC_CONTACT_EMAIL"],
    "Accept-Encoding": "gzip",
}
RATE = 6.0  # requests/sec, well under SEC's 10/s

os.makedirs(OUT, exist_ok=True)
os.makedirs(CC_CACHE, exist_ok=True)

# ----------------------------------------------------------------------------- http
_lock = threading.Lock()
_next = [0.0]


def _throttle() -> None:
    with _lock:
        now = time.time()
        wait = _next[0] - now
        if wait > 0:
            time.sleep(wait)
        _next[0] = max(now, _next[0]) + 1.0 / RATE


def _get_json(url: str, tries: int = 4):
    """GET with throttle + retry. Returns dict, or None on a genuine 404."""
    for k in range(tries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.0 + 2.0 * k)
        except Exception:
            time.sleep(1.0 + 2.0 * k)
    return None


def frame_json(concept: str, unit: str, frame: str, taxonomy: str = "us-gaap"):
    """Cached SEC frames call."""
    path = f"{FR_CACHE}/{taxonomy}_{concept}_{unit}_{frame}.json.gz"
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    url = f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{frame}.json"
    d = _get_json(url)
    if d is None:
        d = {"data": []}
    with gzip.open(path, "wt") as fh:
        json.dump(d, fh)
    return d


def companyconcept(cik: str, concept: str, taxonomy: str = "us-gaap"):
    """Cached SEC companyconcept call. Returns the USD observation list."""
    d = os.path.join(CC_CACHE, concept)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{cik}.json.gz")
    if os.path.exists(path):
        try:
            with gzip.open(path, "rt") as fh:
                return json.load(fh)
        except Exception:
            os.remove(path)
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
    d2 = _get_json(url)
    obs = []
    if d2:
        for unit, rows in (d2.get("units") or {}).items():
            if unit in ("USD",):
                obs = rows
                break
    with gzip.open(path, "wt") as fh:
        json.dump(obs, fh)
    return obs


def bulk_companyconcept(pairs, label=""):
    """pairs: list of (cik, concept). Fetches in parallel with cache."""
    todo = [
        (c, k)
        for c, k in pairs
        if not os.path.exists(os.path.join(CC_CACHE, k, f"{c}.json.gz"))
    ]
    print(f"[fetch] {label}: {len(pairs)} pairs, {len(todo)} missing from cache", flush=True)
    if not todo:
        return
    t0 = time.time()
    done = [0]

    def one(p):
        companyconcept(p[0], p[1])
        done[0] += 1
        if done[0] % 250 == 0:
            el = time.time() - t0
            print(
                f"[fetch]   {done[0]}/{len(todo)}  {done[0]/el:.1f}/s  eta {(len(todo)-done[0])/max(done[0]/el,1e-9)/60:.1f}min",
                flush=True,
            )

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, todo))
    print(f"[fetch] {label}: done in {(time.time()-t0)/60:.1f} min", flush=True)


# ----------------------------------------------------------------------------- helpers
def winsorise(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    v = s.astype(float)
    if v.notna().sum() < 20:
        return v
    a, b = v.quantile(lo), v.quantile(hi)
    return v.clip(a, b)


def rank_pct(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in (0,1]. Ties share the average rank."""
    return s.rank(pct=True, method="average")


def write_signal(name: str, df: pd.DataFrame, extra_cols=()):
    cols = ["ticker", "value", "rank_pct"] + [c for c in extra_cols if c in df.columns]
    out = df[cols].sort_values("ticker").reset_index(drop=True)
    p = f"{OUT}/{name}.parquet"
    out.to_parquet(p, index=False)
    print(f"[write] {p}  rows={len(out)}", flush=True)
    return out


def show(name, df, col="value", n=10, extra=()):
    d = df.dropna(subset=[col]).sort_values(col, ascending=False)
    cols = ["ticker", col, "rank_pct"] + [c for c in extra if c in df.columns]
    print(f"\n--- {name}: TOP {n} (highest {col}) ---")
    print(d.head(n)[cols].to_string(index=False))
    print(f"--- {name}: BOTTOM {n} (lowest {col}) ---")
    print(d.tail(n)[cols].to_string(index=False))


def sigdoc_def(acronym: str) -> str:
    d = pd.read_csv(SIGDOC)
    r = d[d["Acronym"].astype(str) == acronym]
    if len(r) == 0:
        return "(not in SignalDoc)"
    r = r.iloc[0]
    return f"{r['LongDescription']} ({r['Authors']} {r['Year']}) | Sign={r['Sign']} | {r['Detailed Definition']}"


# ============================================================================= base
def load_base():
    """Universe + price-panel features + XBRL panel, restricted to tradable names."""
    uni = pd.read_csv(UNIVERSE, dtype={"cik": str})
    px = pd.read_parquet(DERIVED)
    fun = pd.read_parquet(FUNDA)
    fun["filed_date"] = pd.to_datetime(fun["filed_date"])
    fun["period_end"] = pd.to_datetime(fun["period_end"])
    fun["period_start"] = pd.to_datetime(fun["period_start"])
    # point-in-time gate + drop impossible future period ends (EDGAR date typos)
    fun = fun[(fun.filed_date <= TODAY) & (fun.period_end <= TODAY)]

    # a CIK carrying many tickers is usually a fund/ETN issuer whose share classes
    # would all inherit the same fundamentals -- keep it as a caveat column.
    ntk = uni.groupby("cik")["ticker"].nunique().rename("cik_tickers")
    uni = uni.join(ntk, on="cik")

    base = px[["ticker", "dollar_vol_60d", "listing_age_days"]].merge(
        uni[["ticker", "cik", "title", "cik_tickers"]].drop_duplicates("ticker"),
        on="ticker",
        how="left",
    )
    print(
        f"[base] price panel {len(px)} tickers | with cik {base.cik.notna().sum()} | "
        f"XBRL rows after PIT gate {len(fun)}",
        flush=True,
    )
    return base, fun, uni


def load_rosters():
    """Historical SEC company_tickers snapshots -> {ticker: earliest snapshot date}."""
    out = {}
    for fn, dt in [
        ("roster_20190601000000.json", pd.Timestamp("2019-06-01")),
        ("roster_20210601000000.json", pd.Timestamp("2021-06-01")),
        ("roster_20230601000000.json", pd.Timestamp("2023-06-01")),
    ]:
        p = os.path.join(HISTROSTER, fn)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            magic = fh.read(2)
        op = gzip.open if magic == b"\x1f\x8b" else open
        with op(p, "rt") as fh:
            d = json.load(fh)
        for v in d.values():
            t = v.get("ticker")
            if t and (t not in out or dt < out[t]):
                out[t] = dt
    return out


# ============================================================================= AgeIPO
def build_age(base: pd.DataFrame):
    print("\n" + "=" * 78 + "\n AgeIPO / FirmAge  (listing age)\n" + "=" * 78, flush=True)
    print("SignalDoc AgeIPO  :", sigdoc_def("AgeIPO"), flush=True)
    print("SignalDoc FirmAge :", sigdoc_def("FirmAge"), flush=True)
    print("SignalDoc IndIPO  :", sigdoc_def("IndIPO"), flush=True)

    d = base.copy()
    d["age_price_panel"] = d["listing_age_days"].astype(float)
    PANEL_CAP = float(np.nanmax(d["age_price_panel"]))  # right-censoring point

    rost = load_rosters()
    floor = d["ticker"].map(lambda t: (TODAY - rost[t]).days if t in rost else np.nan)
    d["age_roster_floor"] = floor.astype(float)

    # the panel age is exact below the cap and a lower bound at the cap; the roster
    # floor is always a lower bound. take the tighter (larger) of the two.
    d["value"] = np.fmax(d["age_price_panel"], d["age_roster_floor"].fillna(-1))
    d["is_lower_bound"] = d["age_price_panel"] >= PANEL_CAP - 1

    # OSAP IndIPO: 1 if IPO 6-36 months ago. Our censoring point is 36.3 months,
    # so this window is fully observable.
    d["ind_ipo"] = ((d["value"] >= 183) & (d["value"] <= 1095)).astype(int)
    d.loc[d["value"].isna(), "ind_ipo"] = np.nan

    d["value_w"] = winsorise(d["value"])
    d["rank_pct"] = rank_pct(d["value_w"])

    print(f"\n[age] panel right-censoring cap = {PANEL_CAP:.0f} days "
          f"({PANEL_CAP/365.25:.2f} yr); {int(d.is_lower_bound.sum())} tickers sit at it")
    print("[age] roster uncensoring buckets (days):")
    print(d.loc[d.is_lower_bound, "value"].value_counts().sort_index().to_string())
    print(f"[age] coverage = {int(d['value'].notna().sum())} tickers")
    print(f"[age] exact (uncensored) ages = {int((~d.is_lower_bound).sum())}")
    print(f"[age] IndIPO==1 (listed 6-36m ago) = {int(d['ind_ipo'].sum())}")
    print("[age] distribution:")
    print(d["value"].describe().to_string())
    return d


# ============================================================================= XBRL quarterisation
DIV_CONCEPTS = [
    "PaymentsOfDividendsCommonStock",   # primary, per the task spec
    "PaymentsOfDividends",              # fallback: total dividends paid (incl. pref/NCI)
    "PaymentsOfOrdinaryDividends",      # fallback: used by some REITs/foreign filers
]
OCF_CONCEPT = "NetCashProvidedByUsedInOperatingActivities"
OCF_FALLBACK = "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"


def quarterise(obs, as_first_reported=True):
    """XBRL duration facts -> non-overlapping ~3-month periods.

    US cash-flow statements are filed YEAR-TO-DATE, so a Q3 10-Q carries a
    9-month figure. We therefore (a) take any fact that is already ~3 months
    straight, and (b) difference consecutive YTD facts that share a `start`.

    Returns DataFrame [q_start, q_end, val, filed, src].
    """
    if not obs:
        return pd.DataFrame(columns=["q_start", "q_end", "val", "filed", "src"])
    df = pd.DataFrame(obs)
    if "start" not in df.columns:
        return pd.DataFrame(columns=["q_start", "q_end", "val", "filed", "src"])
    df = df.dropna(subset=["start", "end", "val"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df = df[df["filed"] <= TODAY]
    if df.empty:
        return pd.DataFrame(columns=["q_start", "q_end", "val", "filed", "src"])
    # as-first-reported: for each (start,end) keep the value from the EARLIEST
    # filing that carried it. Restatements are lookahead; originals are not.
    df = df.sort_values("filed")
    keep = "first" if as_first_reported else "last"
    df = df.drop_duplicates(subset=["start", "end"], keep=keep)
    df["days"] = (df["end"] - df["start"]).dt.days

    rows = []
    direct = df[(df["days"] >= 80) & (df["days"] <= 100)]
    for r in direct.itertuples():
        rows.append((r.start, r.end, float(r.val), r.filed, "direct"))

    for _s, g in df.groupby("start"):
        g = g.sort_values("end")
        prev_end, prev_val = None, 0.0
        for r in g.itertuples():
            if prev_end is None:
                qs, qv = r.start, float(r.val)
            else:
                qs, qv = prev_end + pd.Timedelta(days=1), float(r.val) - prev_val
            dur = (r.end - qs).days
            if 80 <= dur <= 100:
                rows.append((qs, r.end, qv, r.filed, "ytd_diff"))
            prev_end, prev_val = r.end, float(r.val)

    q = pd.DataFrame(rows, columns=["q_start", "q_end", "val", "filed", "src"])
    if q.empty:
        return q
    # prefer a directly-tagged 3-month fact over a differenced one
    q["_p"] = (q["src"] == "ytd_diff").astype(int)
    q = q.sort_values(["q_end", "_p", "filed"]).drop_duplicates("q_end", keep="first")
    return q.drop(columns="_p").sort_values("q_end").reset_index(drop=True)


def ttm_from_quarters(q: pd.DataFrame):
    """Trailing-4-quarter sum from quarterised facts. Returns (val, end, n_q, span)."""
    if q is None or q.empty:
        return (np.nan, pd.NaT, 0, np.nan)
    qq = q.sort_values("q_end").tail(4)
    if len(qq) < 4:
        return (np.nan, qq["q_end"].max() if len(qq) else pd.NaT, len(qq), np.nan)
    span = (qq["q_end"].iloc[-1] - qq["q_start"].iloc[0]).days
    # the four quarters must actually be contiguous (no missing quarter bridged)
    if not (330 <= span <= 400):
        return (np.nan, qq["q_end"].iloc[-1], len(qq), span)
    return (float(qq["val"].sum()), qq["q_end"].iloc[-1], 4, span)


def payer_cik_concepts():
    """Which CIKs tagged which dividend concept? One frames call per (concept, year)."""
    out = {}
    for c in DIV_CONCEPTS:
        s = set()
        for y in (2023, 2024, 2025, 2026):
            d = frame_json(c, "USD", f"CY{y}")
            for row in d.get("data", []):
                s.add(str(row["cik"]).zfill(10))
        out[c] = s
        print(f"[div] {c}: {len(s)} CIKs tagged an annual value 2023-2026", flush=True)
    allc = set().union(*out.values())
    print(f"[div] payer universe = {len(allc)} CIKs", flush=True)
    return out, allc


# ============================================================================= DivOmit
def build_div_omit(base: pd.DataFrame, fun: pd.DataFrame, do_fetch=True):
    print("\n" + "=" * 78 + "\n DivOmit  (dividend omission)\n" + "=" * 78, flush=True)
    print("SignalDoc DivOmit :", sigdoc_def("DivOmit"), flush=True)

    concept_ciks, payer_ciks = payer_cik_concepts()

    if do_fetch:
        pairs = [(c, k) for k, s in concept_ciks.items() for c in s]
        bulk_companyconcept(pairs, "dividend concepts")
        bulk_companyconcept([(c, OCF_CONCEPT) for c in sorted(payer_ciks)],
                            "OCF for payers (filing calendar)")

    cik2tk = (
        base.dropna(subset=["cik"]).groupby("cik")["ticker"].apply(list).to_dict()
    )

    recs = []
    for cik in sorted(payer_ciks):
        if cik not in cik2tk:
            continue  # payer not in the tradable price panel
        # --- dividends: union across concepts, max per quarter (tag switches are
        #     common; taking the max stops a tag change looking like an omission)
        dq = {}
        for c in DIV_CONCEPTS:
            if cik not in concept_ciks[c]:
                continue
            q = quarterise(companyconcept(cik, c))
            for r in q.itertuples():
                k = r.q_end.normalize()
                dq[k] = max(dq.get(k, 0.0), float(r.val))
        # --- filing calendar: quarters for which a cash-flow statement exists
        cfq = quarterise(companyconcept(cik, OCF_CONCEPT))
        if cfq.empty or len(cfq) < 5:
            recs.append(dict(cik=cik, status="no_cf_calendar", n_cf_q=len(cfq)))
            continue
        cfq = cfq.sort_values("q_end")
        ends = list(cfq["q_end"].dt.normalize())
        filed = list(cfq["filed"])

        def div_at(e):
            # tolerate a few days of fiscal-calendar drift between statements
            best = None
            for k in dq:
                if abs((k - e).days) <= 7:
                    best = max(best, dq[k]) if best is not None else dq[k]
            return 0.0 if best is None else best

        vals = [div_at(e) for e in ends]
        q0_end, q0_filed, q0_div = ends[-1], filed[-1], vals[-1]
        prior4 = vals[-5:-1]
        prior4_ends = ends[-5:-1]
        # the prior four quarters must be contiguous, else "paid every quarter"
        # is being asserted over a gap in the filing record
        span = (prior4_ends[-1] - prior4_ends[0]).days
        contiguous = 240 <= span <= 300 and 80 <= (q0_end - prior4_ends[-1]).days <= 100
        stale = (TODAY - q0_filed).days > 200

        n_paid = sum(1 for v in prior4 if v > 0)
        payer_type = {4: "quarterly", 2: "semiannual", 1: "annual", 0: "none"}.get(
            n_paid, "irregular"
        )
        consistent = n_paid == 4 and contiguous
        omit = int(consistent and q0_div <= 0 and not stale)
        recs.append(
            dict(
                cik=cik,
                status="ok",
                payer_type=payer_type,
                consistent_quarterly=int(consistent),
                contiguous=int(contiguous),
                stale_filer=int(stale),
                q0_end=q0_end,
                q0_filed=q0_filed,
                q0_div=q0_div,
                prior4_min=min(prior4) if prior4 else np.nan,
                prior4_sum=float(np.sum(prior4)),
                n_cf_q=len(cfq),
                div_omit=omit,
            )
        )
    R = pd.DataFrame(recs)
    print(f"\n[div] payer CIKs matched to tradable tickers: {len(R)}")
    print("[div] status:"); print(R["status"].value_counts().to_string())
    ok = R[R.status == "ok"]
    print("[div] payer_type among evaluable payers:")
    print(ok["payer_type"].value_counts().to_string())
    print(f"[div] consistent quarterly payers (at-risk population) = {int(ok['consistent_quarterly'].sum())}")
    print(f"[div] stale filers excluded                            = {int(ok['stale_filer'].sum())}")
    print(f"[div] OMISSIONS detected                               = {int(ok['div_omit'].sum())}")
    print("[div] latest filed quarter-end of at-risk payers:")
    print(ok.loc[ok.consistent_quarterly == 1, "q0_end"].dt.to_period("Q").value_counts().sort_index().to_string())

    # --- expand CIK -> tickers, then to the full evaluable cross-section
    rows = []
    for r in R.to_dict("records"):
        for t in cik2tk.get(r["cik"], []):
            rows.append({**r, "ticker": t})
    T = pd.DataFrame(rows)

    xbrl_tickers = set(fun["ticker"].dropna())
    d = base.copy()
    d = d.merge(
        T[["ticker", "payer_type", "consistent_quarterly", "q0_end", "q0_div",
           "prior4_sum", "stale_filer", "div_omit"]],
        on="ticker", how="left",
    )
    # every filer that is not a consistent payer is a genuine 0, not missing
    evaluable = d["ticker"].isin(xbrl_tickers) | d["div_omit"].notna()
    d["value"] = np.where(evaluable, d["div_omit"].fillna(0.0), np.nan)
    d["payer_type"] = d["payer_type"].fillna("non_payer")
    d["consistent_quarterly"] = d["consistent_quarterly"].fillna(0)
    d["value_w"] = winsorise(d["value"])   # no-op on a binary, applied for consistency
    d["rank_pct"] = rank_pct(d["value_w"])
    print(f"\n[div] evaluable tickers (0 or 1 assigned) = {int(d['value'].notna().sum())}")
    print(f"[div] flagged (value==1)                  = {int((d['value'] == 1).sum())}")
    return d, R


# ============================================================================= cash runway
def latest_per_ticker(df, val="val"):
    return (
        df.sort_values(["ticker", "period_end", "filed_date"])
        .groupby("ticker", as_index=False)
        .tail(1)
    )


def frames_panel(concept, frames, unit="USD", taxonomy="us-gaap"):
    """Pull instantaneous frames and attach filed dates via the local accn index."""
    accn = pd.read_parquet(f"{ROOT}/data/sec/accn_index.parquet")
    accn["filed_date"] = pd.to_datetime(accn["filed_date"])
    amap = dict(zip(accn["accn"], accn["filed_date"]))
    rows = []
    for fr in frames:
        d = frame_json(concept, unit, fr, taxonomy)
        for r in d.get("data", []):
            rows.append(
                (str(r["cik"]).zfill(10), pd.Timestamp(r["end"]), float(r["val"]), r.get("accn"))
            )
    if not rows:
        return pd.DataFrame(columns=["cik", "period_end", "val", "accn", "filed_date"])
    p = pd.DataFrame(rows, columns=["cik", "period_end", "val", "accn"])
    p["filed_date"] = p["accn"].map(amap)
    # no filed date recoverable -> assume the standard 45-day reporting lag
    p["filed_date"] = p["filed_date"].fillna(p["period_end"] + pd.Timedelta(days=45))
    return p[(p.filed_date <= TODAY) & (p.period_end <= TODAY)]


def build_cash_runway(base: pd.DataFrame, fun: pd.DataFrame, do_fetch=True):
    print("\n" + "=" * 78 + "\n cash_runway  (months of cash at the trailing-4Q burn rate)\n" + "=" * 78, flush=True)
    print("SignalDoc has no cash-runway signal; nearest catalogued relatives:")
    print("  Cash    :", sigdoc_def("Cash"))
    print("  cashdebt:", sigdoc_def("cashdebt"), flush=True)

    tk2cik = base.dropna(subset=["cik"]).set_index("ticker")["cik"].to_dict()

    # ---- cash: primary tag, then the restricted-cash-inclusive tag as fallback
    c1 = fun[fun.concept == "CashAndCashEquivalentsAtCarryingValue"]
    c1 = latest_per_ticker(c1)[["ticker", "period_end", "val"]].rename(
        columns={"period_end": "cash_end", "val": "cash"}
    )
    c1["cash_src"] = "CashAndCashEquivalentsAtCarryingValue"

    fb_frames = [f"CY{y}Q{q}I" for y in (2025, 2026) for q in (1, 2, 3, 4)]
    fb = frames_panel("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", fb_frames)
    cik2tk = base.dropna(subset=["cik"]).groupby("cik")["ticker"].apply(list).to_dict()
    fb_rows = []
    for r in fb.itertuples():
        for t in cik2tk.get(r.cik, []):
            fb_rows.append((t, r.period_end, r.val, r.filed_date))
    c2 = pd.DataFrame(fb_rows, columns=["ticker", "period_end", "val", "filed_date"])
    if len(c2):
        c2 = latest_per_ticker(c2)[["ticker", "period_end", "val"]].rename(
            columns={"period_end": "cash_end", "val": "cash"}
        )
        c2["cash_src"] = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    print(f"[cash] primary tag {len(c1)} tickers | fallback tag {len(c2)} tickers")
    cash = pd.concat([c1, c2[~c2.ticker.isin(c1.ticker)]], ignore_index=True)
    print(f"[cash] merged {len(cash)} tickers "
          f"({int((cash.cash_src != 'CashAndCashEquivalentsAtCarryingValue').sum())} from fallback)")

    # ---- OCF: panel annual 12-month figure (all names) ...
    o = fun[(fun.concept == OCF_CONCEPT) & (fun.fp == "FY")].copy()
    o["dur"] = (o.period_end - o.period_start).dt.days
    o = o[(o.dur >= 330) & (o.dur <= 400)]
    oa = latest_per_ticker(o)[["ticker", "period_end", "val"]].rename(
        columns={"period_end": "ocf_end", "val": "ocf_ttm"}
    )
    oa["ocf_src"] = "frames_annual"
    print(f"[ocf ] panel annual 12m OCF: {len(oa)} tickers; median age "
          f"{int((TODAY - oa.ocf_end).dt.days.median())}d; negative {int((oa.ocf_ttm < 0).sum())}")

    # ... refreshed to a true trailing-4-quarter figure for everything that could
    # plausibly be burning cash (a stale burn rate is worthless for a runway floor)
    m0 = base[["ticker"]].merge(cash, on="ticker", how="left").merge(oa, on="ticker", how="left")
    cand_t = set(m0.loc[(m0.ocf_ttm < 0) | (m0.ocf_ttm.isna() & m0.cash.notna()), "ticker"])
    cand_c = sorted({tk2cik[t] for t in cand_t if t in tk2cik})
    print(f"[ocf ] refresh candidates: {len(cand_t)} tickers / {len(cand_c)} CIKs")
    if do_fetch:
        bulk_companyconcept([(c, OCF_CONCEPT) for c in cand_c], "OCF for burn candidates")
        bulk_companyconcept([(c, OCF_FALLBACK) for c in cand_c], "OCF fallback tag")

    fresh = []
    for cik in cand_c:
        q = quarterise(companyconcept(cik, OCF_CONCEPT))
        v, e, n, span = ttm_from_quarters(q)
        src = OCF_CONCEPT
        if not np.isfinite(v):
            q2 = quarterise(companyconcept(cik, OCF_FALLBACK))
            v2, e2, n2, s2 = ttm_from_quarters(q2)
            if np.isfinite(v2):
                v, e, n, src = v2, e2, n2, OCF_FALLBACK
        if np.isfinite(v):
            fresh.append((cik, v, e, src))
    F = pd.DataFrame(fresh, columns=["cik", "ocf_ttm_f", "ocf_end_f", "ocf_src_f"])
    print(f"[ocf ] refreshed TTM built for {len(F)} CIKs")
    if len(F):
        print(f"[ocf ] refreshed TTM median age {int((TODAY - F.ocf_end_f).dt.days.median())}d")

    m = base.merge(cash, on="ticker", how="left").merge(oa, on="ticker", how="left")
    m["cik"] = m["ticker"].map(tk2cik)
    m = m.merge(F, on="cik", how="left")
    use_f = m["ocf_ttm_f"].notna()
    m["ocf_ttm"] = np.where(use_f, m["ocf_ttm_f"], m["ocf_ttm"])
    m["ocf_end"] = np.where(use_f, m["ocf_end_f"], m["ocf_end"])
    m["ocf_src"] = np.where(use_f, m["ocf_src_f"] + "_cc_ttm", m["ocf_src"])
    m["ocf_end"] = pd.to_datetime(m["ocf_end"])
    print(f"[ocf ] final OCF: {int(m.ocf_ttm.notna().sum())} tickers "
          f"({int(use_f.sum())} refreshed); median age {int((TODAY - m.ocf_end).dt.days.median())}d")

    m["burn_annual"] = -m["ocf_ttm"]
    m["is_burner"] = m["burn_annual"] > 0
    m["date_gap_days"] = (m["cash_end"] - m["ocf_end"]).dt.days
    m["value"] = np.where(
        m["is_burner"] & m["cash"].notna(),
        m["cash"].clip(lower=0) / (m["burn_annual"] / 12.0),
        np.nan,
    )

    n_all = len(m)
    n_data = int((m.cash.notna() & m.ocf_ttm.notna()).sum())
    n_burn = int((m.is_burner & m.cash.notna()).sum())
    print(f"\n[runway] tickers in price panel        : {n_all}")
    print(f"[runway] with both cash and TTM OCF    : {n_data}")
    print(f"[runway] of which cash burners (OCF<0) : {n_burn}")
    print(f"[runway] non-burners (no constraint)   : {n_data - n_burn}")
    print(f"[runway] no usable data                : {n_all - n_data}")
    print("[runway] cash_end - ocf_end gap (days):")
    print(m.loc[m.is_burner, "date_gap_days"].describe().to_string())

    r = m[m["value"].notna()].copy()
    r["value_w"] = winsorise(r["value"])
    r["rank_pct"] = rank_pct(r["value_w"])
    print("\n[runway] months of runway, distribution:")
    print(r["value"].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .99]).to_string())
    for thr in (3, 6, 12, 24):
        print(f"[runway]   under {thr:>2}m: {int((r['value'] < thr).sum())}")
    return r, m


# ============================================================================= reporting
LIQ_FLOOR = 1_000_000.0  # $1m median daily dollar volume


def md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (tabulate is not installed)."""
    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        if isinstance(v, (pd.Timestamp,)):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, (np.floating, float)):
            av = abs(v)
            if av >= 1e6:
                return f"{v:,.0f}"
            if av >= 100:
                return f"{v:,.1f}"
            return f"{v:,.4g}"
        if isinstance(v, (np.integer, int)):
            return f"{v:,d}"
        t = str(v)
        return t[:40].replace("|", "/")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def decile_report(name, d, extra=(), n=10):
    """Top/bottom n, unfiltered and above the liquidity floor. Returns md text."""
    out = []
    dd = d.dropna(subset=["value"]).copy()
    cols = ["ticker", "title", "value", "rank_pct", "dollar_vol_60d"] + list(extra)
    cols = [c for c in cols if c in dd.columns]
    for label, sub in [
        ("ALL", dd),
        (f"dollar_vol_60d >= ${LIQ_FLOOR:,.0f}/day", dd[dd.dollar_vol_60d >= LIQ_FLOOR]),
    ]:
        s = sub.sort_values("value", ascending=False)
        out.append(f"\n### {name} — {label}  (n={len(sub)})")
        out.append("\n**Top %d (highest value)**\n" % n)
        out.append(md_table(s.head(n)[cols]))
        out.append("\n**Bottom %d (lowest value)**\n" % n)
        out.append(md_table(s.tail(n)[cols]))
    txt = "\n".join(out)
    print(txt)
    return txt


def liquidity_check(name, d, top=True):
    """Is the extreme decile just illiquid shells?"""
    dd = d.dropna(subset=["value"]).copy()
    q = dd["value"].quantile(0.9 if top else 0.1)
    ext = dd[dd["value"] >= q] if top else dd[dd["value"] <= q]
    med = ext["dollar_vol_60d"].median()
    frac = (ext["dollar_vol_60d"] < LIQ_FLOOR).mean()
    side = "top" if top else "bottom"
    print(f"[liq ] {name} {side} decile: n={len(ext)} median $vol={med:,.0f} "
          f"| {frac:.1%} below the ${LIQ_FLOOR:,.0f} floor "
          f"(whole cross-section: {(dd.dollar_vol_60d < LIQ_FLOOR).mean():.1%})")
    return dict(n=len(ext), median_dv=med, frac_illiquid=frac,
                base_illiquid=(dd.dollar_vol_60d < LIQ_FLOOR).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    fetch = not a.no_fetch

    base, fun, uni = load_base()

    # ---------------------------------------------------------------- AgeIPO
    age = build_age(base)
    age_out = age.assign(value=age["value"])
    show("age_ipo", age_out, extra=("is_lower_bound",))
    liq_age_lo = liquidity_check("age_ipo", age_out, top=False)
    liq_age_hi = liquidity_check("age_ipo", age_out, top=True)
    write_signal("age_ipo", age_out,
                 ("age_price_panel", "age_roster_floor", "is_lower_bound",
                  "ind_ipo", "dollar_vol_60d", "cik_tickers", "title"))
    age_md = decile_report("age_ipo", age_out, extra=("is_lower_bound", "ind_ipo"))

    # ---------------------------------------------------------------- DivOmit
    dv, DR = build_div_omit(base, fun, do_fetch=fetch)
    show("div_omit", dv, extra=("payer_type", "q0_end", "q0_div"))
    liq_dv = liquidity_check("div_omit", dv, top=True)
    write_signal("div_omit", dv,
                 ("payer_type", "consistent_quarterly", "q0_end", "q0_div",
                  "prior4_sum", "dollar_vol_60d", "cik_tickers", "title"))
    flagged = dv[dv.value == 1].sort_values("dollar_vol_60d", ascending=False)
    print("\n--- div_omit: every flagged name ---")
    print(flagged[["ticker", "title", "q0_end", "q0_div", "prior4_sum",
                   "dollar_vol_60d"]].to_string(index=False))

    # ---------------------------------------------------------------- cash_runway
    cr, cr_all = build_cash_runway(base, fun, do_fetch=fetch)
    show("cash_runway", cr, extra=("cash", "ocf_ttm", "ocf_end", "cash_end"))
    liq_cr_lo = liquidity_check("cash_runway", cr, top=False)
    liq_cr_hi = liquidity_check("cash_runway", cr, top=True)
    write_signal("cash_runway", cr,
                 ("cash", "cash_end", "cash_src", "ocf_ttm", "ocf_end", "ocf_src",
                  "burn_annual", "date_gap_days", "dollar_vol_60d", "cik_tickers", "title"))
    cr_md = decile_report("cash_runway", cr, extra=("cash", "ocf_ttm"))

    # ---------------------------------------------------------------- notes
    write_notes(age_out, liq_age_lo, liq_age_hi,
                dv, DR, flagged, liq_dv,
                cr, cr_all, liq_cr_lo, liq_cr_hi, base)
    print("\nALL DONE", flush=True)



# ============================================================================= markdown notes
def top_bot(d, n=5, extra=()):
    dd = d.dropna(subset=["value"]).sort_values("value", ascending=False)
    cols = [c for c in ["ticker", "title", "value", "rank_pct", "dollar_vol_60d"] + list(extra)
            if c in dd.columns]
    return md_table(dd.head(n)[cols]), md_table(dd.tail(n)[cols])


def write_notes(age, liq_age_lo, liq_age_hi, dv, DR, flagged, liq_dv,
                cr, cr_all, liq_cr_lo, liq_cr_hi, base):
    n_px = len(base)
    cap = float(np.nanmax(age["age_price_panel"]))

    # ---------------------------------------------------------------- age_ipo.md
    t5, b5 = top_bot(age, 5, extra=("is_lower_bound", "ind_ipo"))
    buckets = age.loc[age.is_lower_bound, "value"].value_counts().sort_index()
    bucket_md = md_table(pd.DataFrame({"age_days_lower_bound": buckets.index,
                                       "tickers": buckets.values}))
    txt = f"""# age_ipo — listing age (OSAP `AgeIPO` / `FirmAge` stand-in)

`signals/age_ipo.parquet` · built {TODAY.date()} by `signals/build_events.py`

## Exact formula

```
age_price_panel  = trading days between today and the ticker's first real print
                   in data/prices/history_3y.parquet  (derived.parquet::listing_age_days)
age_roster_floor = today - earliest SEC company_tickers snapshot the ticker appears in
                   (snapshots at 2019-06-01, 2021-06-01, 2023-06-01)
value            = max(age_price_panel, age_roster_floor)          [days]
value_w          = winsorise(value, 1%, 99%)
rank_pct         = cross-sectional percentile rank of value_w, ties averaged
ind_ipo          = 1 if 183 <= value <= 1095   (OSAP IndIPO: IPO 6-36 months ago)
```

## What the catalogue actually specifies, and what I built instead

`SignalDoc.csv` defines the three age signals as:

| acronym | Detailed Definition | buildable here? |
|---|---|---|
| `AgeIPO` | "Age is (current year - founding year from **Jay Ritter's dataset**). Exclude if IndIPO == 0." | **no** — Ritter's hand-collected founding-year file is not in `data/`, and founding year is not in any free SEC feed |
| `FirmAge` | "Months since start of CRSP coverage." | **yes, as a proxy** — months since first print in our own price panel is the same construct on a different tape |
| `IndIPO` | "1 if IPO in the past 6-36 months. 0 otherwise. IPO dates are taken from Jay Ritter's IPO data." | **yes, as a proxy** — our right-censoring point is {cap:.0f} days = {cap/30.44:.1f} months, so the whole 6-36 month window is *inside* the observable range and the indicator is exactly reproducible from listing date rather than IPO date |

So `value` is `FirmAge` (listing age), and the parquet carries `ind_ipo` as a
bonus column that reproduces the `IndIPO` definition. **Neither is literally
`AgeIPO`** — `AgeIPO` needs a founding year we do not have. Do not label the
output `AgeIPO` downstream without this caveat.

Triage context (`research/out/osap_triage.csv`), which matters for how you sign this:

| signal | recent_ann | recent_t | decay | OSAP Sign |
|---|---|---|---|---|
| AgeIPO | 10.44 | 2.13 | 0.41 | +1 |
| IndIPO | 4.33 | 1.66 | 0.57 | -1 |
| FirmAge | -4.00 | -1.37 | 3.86 | -1 |

`FirmAge` — the thing this file actually measures — has a **negative** recent
t-stat and a decay ratio of 3.9, i.e. it has stopped working and then some. The
event-conditioned versions (`AgeIPO`, `RDIPO`) are the ones with live t-stats.
Treat raw listing age as a **condition input** (young listings behave
differently, size/quality interactions are unstable there), not as a weighted
alpha term.

## Data sources used

No XBRL. `data/prices/derived.parquet` (`listing_age_days`) plus the three
`data/hist/roster_*.json` SEC `company_tickers.json` snapshots.

## Coverage

* **{int(age['value'].notna().sum()):,} tickers** carry a value ({int(age['value'].notna().sum())/n_px:.1%} of the {n_px:,}-ticker price panel).
* **{int((~age.is_lower_bound).sum()):,}** are exact ages (younger than the panel).
* **{int(age.is_lower_bound.sum()):,}** are lower bounds (older than the panel start).
* **{int(age['ind_ipo'].sum()):,}** satisfy `ind_ipo == 1` (listed 6-36 months ago).

The price panel is right-censored at **{cap:.0f} days** ({cap/365.25:.2f} years) because
`history_3y.parquet` starts 2023-08-10. Without help, {int(age.is_lower_bound.sum()):,} of {n_px:,} tickers
({age.is_lower_bound.mean():.0%}) would tie at a single value and the percentile rank would be
degenerate over three quarters of the cross-section. The roster snapshots split
that block into four:

{bucket_md}

Presence in a snapshot *proves* the listing existed then; absence proves nothing
(the 2019 `company_tickers.json` file held only 6,259 names against 12,400 in
2021, so it under-covers). That asymmetry is why the combination is `max()` of two
lower bounds and never `min()`. Everything at or above {cap:.0f} days is still a lower
bound — flagged per-row by `is_lower_bound`.

## Top 5 / bottom 5

**Oldest (highest value)**

{t5}

**Youngest (lowest value)**

{b5}

## Liquidity check

Youngest decile: median 60d dollar volume **${liq_age_lo['median_dv']:,.0f}**, {liq_age_lo['frac_illiquid']:.1%} below the
${LIQ_FLOOR:,.0f}/day floor, against {liq_age_lo['base_illiquid']:.1%} for the whole cross-section.
Oldest decile: median **${liq_age_hi['median_dv']:,.0f}**, {liq_age_hi['frac_illiquid']:.1%} below the floor.
`dollar_vol_60d` is carried in the parquet so the composite stage can apply its
own floor; **no floor is applied inside this file.**

## Judgement calls

1. **Listing age, not firm age.** A spin-off or a post-bankruptcy relisting looks
   like a one-year-old company here. That is what CRSP-based `FirmAge` does too,
   so it matches the catalogue construct rather than economic reality.
2. **Ties are averaged, not broken.** `rank_pct` uses `method="average"`, so each
   censored block shares one percentile. That honestly encodes "we cannot order
   these", instead of inventing an order from row position.
3. **Winsorising is near-inert here** — age is bounded below by 0 and above by the
   censoring point. Applied anyway for consistency with the other signals.
4. **Yahoo first-trade dates were not fetched.** They would uncensor the old tail
   properly but cost ~7,300 extra requests against a source that already
   throttled this project badly (see `data/prices/derived.md`), to sharpen the
   part of the distribution the signal cares least about.
"""
    open(f"{OUT}/age_ipo.md", "w").write(txt)
    print(f"[write] {OUT}/age_ipo.md", flush=True)
