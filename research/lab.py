#!/usr/bin/env python3
"""Lab: batch experiment runner for the value book -- many variants per session from one JSON spec.

WHY. A nightly agent should test an idea by writing a short JSON batch, not a new script per idea,
and a lab number must be measured exactly like a registered experiment. So the lab does NOT re-type
the harness. It executes the corrected chain the way the chain executes itself (exec of source up to
a marker), and runs control + every variant through the identical pick -> simulate path:
  research/test_skew_overlay.py  (item 12) data, live universe, split-corrected mktcap, month_inputs
  research/test_rebalance_tax.py (item 6)  STEP frames, halfspread + impact cost (notional $1m), pick, simulate,
                                           arms (3 quarterly phases), 42% tax, iid noise null
  research/test_rank_buffer.py   (item 14) buffer_picks (exit buffer), persistent AR(1) noise null
Item 6 is executed in two pieces split at "STEP, NO_PX = " so two read-only taps can be installed
before its STEP loop runs. They copy the per-signal sector ranks that month_inputs already computes
(and then discards) onto the STEP frames, so the signal mix can be varied. Nothing the harness computes
is changed: at load the lab asserts mean(SP, bm, cfp ranks) == the harness value score, and smoke.json
reproduces items 6 and 14 to the digit. NOT research/value_monthly_backtest.py (universe leak, split
lookahead, broken TTM-OCF: BACKLOG items 7, 7b, 10).

USAGE (/usr/bin/python3, from anywhere; ROOT is derived from this file so it also runs in CI)
  python3 research/lab.py research/batches/<id>.json              run + ledger + scoreboard
  python3 research/lab.py research/batches/<id>.json --check      validate the spec only (fast)
  python3 research/lab.py research/batches/<id>.json --no-ledger  dev only; scoreboard says so

BATCH JSON (unknown keys anywhere are an error)
  {"batch_id": "buf_width_1",                       # [a-z0-9_-]+, used in output file names
   "hypothesis": "A wider exit buffer keeps more of the after-tax gain",
   "registered": "SUPPORTED if D >= +1.0pp/yr AND d >= -0.10%/mo; REFUTED if D <= 0; else INCONCLUSIVE",
   "control":  {"name": "MONTHLY"},                 # harness defaults below; name optional
   "variants": [{"name": "BUF120", "buffer": 120},
                {"name": "SPCFP_B80", "buffer": 80, "signal": {"SP": 1, "cfp": 1}},
                {"name": "LIQ2M", "min_adv": 2e6}],
   "nnull": 200}                                    # optional, noise books per null (0 = skip)
  The control is the harness defaults plus its own keys; a VARIANT IS THE CONTROL PLUS ITS OVERRIDES
  (change one thing at a time). "registered" is frozen: once a batch_id is in the ledger, re-running it
  with any changed text or params is refused -- use a new batch_id.

PARAMETERS (default = item 6/14 control: value book, 40 names, cap 8, monthly, sell at rank 41)
  n          int 5..400, default 40.  Names in the book, equal weight at every rebalance.
  cap        int 1..n or null, default 8.  Max names per sector (sector = data/sec/sectors.parquet;
             kept buffer names count toward it).  null = uncapped (cap >= n is the same and is stored
             as null).
  rebalance  "monthly" | "quarterly", default "monthly".  quarterly = item 6 arms(): buy-and-hold drift
             between rebalances, run at all 3 phase offsets, reported as the MEAN of the 3 (never pick a
             phase).
  buffer     int > n or null, default null.  Item 14 exit buffer: keep a held name while it is priced
             and ranks <= buffer in the (floored) pool; fill vacancies from the top. Monthly only (the
             harness defines the buffer on the monthly chain; quarterly + buffer is refused).
  signal     {component: weight > 0}, default {"value": 1}.  score = sum(w*rank)/sum(w) over the
             components a name has (NaN-skipping, as the harness builds value).  Components:
               value         harness value score = mean sector rank of SP, bm, cfp (the live score)
               skw           harness skew score = mean sector rank of px_52w, volume_trend (item 12)
               SP bm cfp     sales/price, book/market, op. cash flow/price (cfp carries the live TTM
                             defect, item 7b), each as month_inputs ranks it
               px_52w volume_trend   price/52w high, 60d/250d median $ volume
             Sector ranks are month_inputs' own: inside sectors (< 25 names pooled as Other), over the
             PRE-filter pool, exactly the numbers that build value.
  rank_scope "sector" | "market", default "sector".  market = item 13's MKT: each raw component ranked
             across the eligible pool (STEP pool, before any min_adv floor). value/skw are sector
             composites and are refused under market scope.
  min_adv    number >= 5e5, default 5e5.  Liquidity floor on 60d median $ volume, applied to the scored
             pool as item 3 did (ranks unchanged, names below the floor removed). Below 5e5 is refused:
             month_inputs already applies MINLIQ = 5e5.
  aum        number > 0, default 1e6.  Notional book size in $ (a model input, not real capital) fed to item 6's impact term
             0.7*sigma*sqrt(trade/ADV) (cost MODEL unchanged, only its book-size input).
  Fixed (not parameters): universe and live filters, equal weight, half-spread buckets, 42% tax on net
  realised gains per calendar year with loss carry-forward and terminal liquidation, months/window.

OUTPUT
  stdout and research/out/lab_<batch_id>.md: a ~20-line scoreboard, one row per variant:
    vs SPY %/mo (after cost, pre-tax) and t; after-tax annualised (liquidated); d = variant - control
    %/mo with paired t; D = after-tax annualised difference (pp/yr, the primary of items 6/14); null
    percentiles; traded per rebalance; bar.  Plus one coverage line per variant.
  research/out/lab_<batch_id>.parquet: one row per arm with every stat, params, monthly series, picks,
    and the full null distributions.
  research/out/lab_<batch_id>.log: harness stdout (the STEP loop's per-month lines).
  research/lab_ledger.csv: one row appended per arm tested (control included).

NOISE NULL (the harness's own nulls, same seeds and draw order, so smoke reproduces items 6/14)
  Construction = (n, cap, rebalance, buffer, min_adv, aum).  Noise = random sector-relative signal
  replacing the whole score.  iid redrawn every month (item 6, seed 20261010) unless a buffer is
  involved, then persistent AR(1) rho 0.9 (item 14, seed 20261012; a buffer is meaningless on noise
  redrawn monthly).  Every construction sees the same noise draws (common random numbers).
    null% d/D  THE HARNESS NULL. Share of noise books whose (variant construction - control
               construction) d and D are <= the variant's, same noise through both, after cost
               (items 6/14).  "-" when the construction equals the control's (signal-only variant:
               the difference is identically 0 on noise; read null% lvl instead).
    null% lvl  Lab addition: does this signal select better than random signals built the same way?
               Share of AR(1) noise books with the variant's construction whose GROSS (cost-free)
               vs-SPY %/mo is <= the variant's gross.  Cost-free on both sides and AR(1) always,
               because noise books churn far more than value (traded/rebal: iid 193%, AR(1) 101%,
               value 49%), so an after-cost or iid level null would flatter any persistent signal.
               Costs are switched off through the harness's own inputs (halfspread -> 0, AUM -> 0),
               only for this column.

MULTIPLE COMPARISONS
  M = number of DISTINCT param hashes ever tested = research/lab_ledger.csv plus this batch (a re-run of
  identical params on the same harness is not a new trial).  A variant CLEARS THE BAR only if its paired
  t of d (variant - control, monthly, after cost) >= t_bar(M) = Student-t quantile at 1 - 0.10/M,
  df = months - 1: Bonferroni at family-wise alpha 0.10 one-sided (alpha 0.10 = the repo's registered
  p <= 0.10 gates).  It tightens as M grows (24 months, df 23: M=3 -> 1.93, M=10 -> 2.50,
  M=30 -> 2.98, M=100 -> 3.48, M=300 -> 3.93, M=1000 -> 4.42).
  Registered experiments from BACKLOG items that this lab can express are listed in
  research/lab_prior.csv and counted in M as if ledgered; the rest (items 1-2, 7-11) are not, so M still
  understates the true count. The bar is a filter for noise-mining, not a verdict; the batch's registered
  rule is the verdict. It only rises: no budget keeps it reachable, which is why anything found here
  goes to a forward test registered in advance, never straight into the book.
  CANONICAL PARAMS: at sector scope an equal-weight SP+bm+cfp signal is stored as value (it is the same
  score), and cap >= n as null, so the same construction cannot hide under a second hash.
  REFERENCE ARMS: a variant whose param hash is already in the ledger or the prior list is a
  reference, shown as "ref" in the bar column, and never clears. So does a variant whose monthly
  returns are identical to the control's (nothing was tested). It may be re-included so an
  interaction is visible, but re-testing old params under a new rule, or on a later data state, cannot
  turn them into a discovery: the first run is the trial.

NIGHT MODE (research/.night_mode exists; the night-lab workflow writes it, nothing else should)
  JSON {"start": "<ISO UTC>", "budget": <int>}. While it exists the lab additionally refuses:
  --no-ledger (a run whose numbers were seen must be counted); re-running a batch_id already in the
  ledger (the verdict is the first run's); any arm with a non-default aum (a cost-model input, not a
  candidate); a control that is not already in the ledger or prior list; nnull other than 200; a
  batch_id over 36 or a variant name over 24 characters; a second lab run while one is in progress;
  and a batch that would take tonight's NEW param hashes (first ledgered at or after "start") above
  "budget". Each night widens M; the budget slows that, it does not stop it.

FAIL LOUD (exit code 2, nothing written to the ledger)
  missing data or harness files; harness chain markers changed; unknown or invalid params; duplicate
  variants or a variant equal to the control; batch_id re-run with a changed spec; implausible coverage:
  < 15 months, pool < 500 or max mktcap > 10,000bn, pool after floor < 2n, < n names in any month,
  fewer than 3n scored names, sector over cap, buffer rank breach, non-finite or |book month| > 50%.
"""
import argparse, contextlib, csv, datetime, hashlib, json, math, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT/"research", ROOT/"research"/"out"
LEDGER = RES/"lab_ledger.csv"
NIGHT_MARK = RES/".night_mode"
PRIOR = RES/"lab_prior.csv"
H12, H6, H14 = RES/"test_skew_overlay.py", RES/"test_rebalance_tax.py", RES/"test_rank_buffer.py"
ALPHA_FW = 0.10
SEED = {"iid": 20261010, "ar1": 20261012}         # item 6's and item 14's null seeds (asserted below)
DEFAULTS = {"n": 40, "cap": 8, "rebalance": "monthly", "buffer": None, "signal": {"value": 1.0},
            "rank_scope": "sector", "min_adv": 5e5, "aum": 1e6}
COMPONENTS = ("SP", "bm", "cfp", "px_52w", "volume_trend")
COMPOSITES = ("value", "skw")
CONS_KEYS = ("n", "cap", "rebalance", "buffer", "min_adv", "aum")
LEDGER_COLS = ["timestamp", "batch_id", "spec_hash", "variant", "role", "param_hash", "params",
               "window", "months", "vs_spy_mo", "t", "after_tax_ann", "d_mo", "d_t", "D_pp",
               "null_pct_d", "null_pct_D", "null_pct_lvl", "turnover", "M_after", "t_bar", "clears_bar",
               "harness_sha"]


class LabError(Exception):
    pass


def fail(msg):
    raise LabError(msg)


# ----------------------------------------------------------------------------------------- spec
def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        fail(f"{what}: must be a finite number, got {v!r}")
    return float(v)


def resolve(base, over, where):
    unknown = set(over) - set(DEFAULTS) - {"name"}
    if unknown:
        fail(f"{where}: unknown param(s) {sorted(unknown)}; allowed: name, {', '.join(DEFAULTS)}")
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in over.items():
        if k != "name": p[k] = v
    n = p["n"]
    if isinstance(n, bool) or not isinstance(n, int) or not 5 <= n <= 400:
        fail(f"{where}: n must be an int 5..400, got {n!r}")
    cap = p["cap"]
    if cap is not None:
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            fail(f"{where}: cap must be an int >= 1 or null, got {cap!r}")
        if cap >= n: cap = None
    p["cap"] = cap
    if p["rebalance"] not in ("monthly", "quarterly"):
        fail(f"{where}: rebalance must be 'monthly' or 'quarterly', got {p['rebalance']!r}")
    b = p["buffer"]
    if b is not None:
        if isinstance(b, bool) or not isinstance(b, int) or b <= n:
            fail(f"{where}: buffer must be an int > n ({n}) or null, got {b!r}")
        if p["rebalance"] != "monthly":
            fail(f"{where}: buffer needs rebalance 'monthly' (item 14 defines the buffer on the monthly "
                 f"chain; quarterly + buffer is not expressible by the harness)")
    if p["rank_scope"] not in ("sector", "market"):
        fail(f"{where}: rank_scope must be 'sector' or 'market', got {p['rank_scope']!r}")
    sig = p["signal"]
    if not isinstance(sig, dict) or not sig:
        fail(f"{where}: signal must be a non-empty object {{component: weight}}")
    allowed = COMPONENTS + (COMPOSITES if p["rank_scope"] == "sector" else ())
    bad = [c for c in sig if c not in allowed]
    if bad:
        fail(f"{where}: unknown signal component(s) {bad} for rank_scope {p['rank_scope']!r}; "
             f"allowed: {', '.join(allowed)}")
    sig = {c: _num(w, f"{where}: signal weight {c}") for c, w in sig.items()}
    if any(w <= 0 for w in sig.values()):
        fail(f"{where}: signal weights must be > 0 (got {sig})")
    if p["rank_scope"] == "sector" and set(sig) == {"SP", "bm", "cfp"} and len(set(sig.values())) == 1:
        sig = {"value": 1.0}                               # the harness value score, by definition
    p["signal"] = sig
    p["min_adv"] = _num(p["min_adv"], f"{where}: min_adv")
    if p["min_adv"] < 5e5:
        fail(f"{where}: min_adv must be >= 5e5 (month_inputs already applies MINLIQ = 5e5)")
    p["aum"] = _num(p["aum"], f"{where}: aum")
    if p["aum"] <= 0:
        fail(f"{where}: aum must be > 0")
    return p


def canon(p):
    s = sum(p["signal"].values())
    q = dict(p, signal={c: round(w/s, 6) for c, w in sorted(p["signal"].items())})
    return json.dumps(q, sort_keys=True, separators=(",", ":"))


def phash(p):
    return hashlib.sha1(canon(p).encode()).hexdigest()[:12]


def cons_key(p):
    return tuple(p[k] for k in CONS_KEYS)


def changes(p, ctl):
    d = []
    for k in DEFAULTS:
        if canon(dict(ctl, **{k: p[k]})) != canon(ctl):
            v = p[k]
            if k == "signal": v = "+".join(f"{w:g}{c}" if w != 1 else c for c, w in v.items())
            elif k in ("min_adv", "aum"): v = f"{v/1e6:g}m"
            d.append(f"{k}={v}")
    return ", ".join(d)


def load_spec(path):
    try:
        spec = json.loads(Path(path).read_text())
    except FileNotFoundError:
        fail(f"batch file not found: {path}")
    except json.JSONDecodeError as e:
        fail(f"batch file is not valid JSON: {path}: {e}")
    if not isinstance(spec, dict): fail("batch must be a JSON object")
    top = {"batch_id", "hypothesis", "registered", "control", "variants", "nnull"}
    unknown = set(spec) - top
    if unknown: fail(f"batch: unknown top-level key(s) {sorted(unknown)}; allowed: {sorted(top)}")
    for k in ("batch_id", "hypothesis", "registered", "control", "variants"):
        if k not in spec: fail(f"batch: missing required key {k!r}")
    bid = spec["batch_id"]
    if not isinstance(bid, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", bid):
        fail(f"batch_id must match [a-z0-9][a-z0-9_-]*, got {bid!r}")
    for k in ("hypothesis", "registered"):
        if not isinstance(spec[k], str) or len(spec[k].strip()) < 10:
            fail(f"batch: {k!r} must be written out (>= 10 chars) before running")
    nnull = spec.get("nnull", 200)
    if isinstance(nnull, bool) or not isinstance(nnull, int) or not 0 <= nnull <= 2000:
        fail(f"batch: nnull must be an int 0..2000, got {nnull!r}")
    c = spec["control"]
    if not isinstance(c, dict): fail("batch: control must be an object")
    ctl = resolve(DEFAULTS, c, "control")
    arms = [("control", str(c.get("name", "CONTROL")), ctl)]
    if not isinstance(spec["variants"], list) or not spec["variants"]:
        fail("batch: variants must be a non-empty list")
    for i, v in enumerate(spec["variants"]):
        if not isinstance(v, dict) or not v.get("name"):
            fail(f"variant #{i+1}: must be an object with a name")
        arms.append(("variant", str(v["name"]), resolve(ctl, v, f"variant {v['name']!r}")))
    names = [a[1] for a in arms]
    for nm in names:
        if not re.fullmatch(r"[A-Za-z0-9_.+=-]{1,40}", nm): fail(f"bad variant name {nm!r}")
    if len(set(names)) != len(names): fail(f"duplicate variant names in {names}")
    seen = {}
    for role, nm, p in arms:
        h = phash(p)
        if h in seen:
            fail(f"{nm!r} has the same params as {seen[h]!r} ({canon(p)}); nothing to test")
        seen[h] = nm
    spec_hash = hashlib.sha1(json.dumps({"h": spec["hypothesis"], "r": spec["registered"], "n": nnull,
                                         "a": [(r, nm, canon(p)) for r, nm, p in arms]},
                                        sort_keys=True).encode()).hexdigest()[:12]
    return {"batch_id": bid, "hypothesis": spec["hypothesis"].strip(), "registered": spec["registered"].strip(),
            "nnull": nnull, "arms": arms, "spec_hash": spec_hash}


# --------------------------------------------------------------------------------------- ledger
def read_ledger():
    if not LEDGER.exists(): return pd.DataFrame(columns=LEDGER_COLS)
    L = pd.read_csv(LEDGER, dtype=str, keep_default_na=False)
    miss = set(LEDGER_COLS) - set(L.columns)
    if miss: fail(f"{LEDGER} is missing columns {sorted(miss)}; refusing to append to a foreign file")
    return L


def check_rerun(L, B):
    prev = L[L.batch_id == B["batch_id"]]
    if len(prev) and set(prev.spec_hash) != {B["spec_hash"]}:
        fail(f"batch_id {B['batch_id']!r} was already run with a different spec (ledger spec_hash "
             f"{sorted(set(prev.spec_hash))} vs now {B['spec_hash']}). The registered rule and params are "
             f"frozen once run: use a new batch_id.")
    return len(prev.drop_duplicates("timestamp")) if len(prev) else 0


def read_night_mode():
    if not NIGHT_MARK.exists(): return None
    try:
        m = json.loads(NIGHT_MARK.read_text())
        return {"start": str(m["start"]), "budget": int(m["budget"])}
    except Exception as e:
        fail(f"{NIGHT_MARK} is unreadable ({e}); refusing to guess whether night rules apply")


def read_prior():
    if not PRIOR.exists(): return set()
    out = set()
    for r in csv.DictReader(open(PRIOR)):
        out.add(phash(resolve(DEFAULTS, json.loads(r["overrides"]), f"{PRIOR.name} {r['name']}")))
    return out


def night_checks(night, a, B, L, reruns, new_hashes, known):
    bid, arms = B["batch_id"], B["arms"]
    if len(bid) > 36 or any(len(nm) > 24 for _, nm, _ in arms):
        fail("night lab: batch_id at most 36 characters and variant names at most 24")
    if B["nnull"] != 200:
        fail("night lab: nnull is fixed at 200 (omit it)")
    if phash(arms[0][2]) not in known:
        fail("night lab: the control must be a parameter set already in the ledger or research/lab_prior.csv "
             "(the research control, the live book, or an earlier arm)")
    if a.no_ledger:
        fail("night lab: --no-ledger is refused. A run whose numbers were seen must be counted; "
             "use --check to validate a spec without running it.")
    if reruns:
        fail(f"night lab: batch_id {bid!r} is already in the ledger and its verdict is the first run's. "
             f"Register a new batch_id; earlier params re-included there show as reference arms.")
    bad = [nm for role, nm, p in arms if p["aum"] != DEFAULTS["aum"]]
    if bad:
        fail(f"night lab: aum is a cost-model input, not a candidate; no arm may change it ({bad})")
    first = L.groupby("param_hash").timestamp.min() if len(L) else pd.Series(dtype=str)
    used = int((first >= night["start"]).sum())
    if used + len(new_hashes) > night["budget"]:
        fail(f"night lab: tonight's budget is {night['budget']} new param sets; {used} used, this batch "
             f"adds {len(new_hashes)}. Stop for the night or drop variants.")
    return used


def bar(M, df):
    try:
        from scipy.stats import t as student_t
        return float(student_t.ppf(1 - ALPHA_FW/M, df)), f"Student-t df {df}"
    except ImportError:                                   # requirements.txt has scipy; fallback only
        from statistics import NormalDist
        return NormalDist().inv_cdf(1 - ALPHA_FW/M), "normal approx (scipy missing)"


# -------------------------------------------------------------------------------------- harness
EXEC6 = 'exec(compile(_src, "test_rebalance_tax_harness", "exec"))'


def harness_sources():
    for f in (H12, H6, H14):
        if not f.exists(): fail(f"harness file missing: {f}")
    s12, s6, s14 = H12.read_text(), H6.read_text(), H14.read_text()
    link6 = '(ROOT/"research/test_skew_overlay.py").read_text().split("SKIPPED = ")[0]'
    link14 = '(ROOT/"research/test_rebalance_tax.py").read_text().split("VALUE_PICKS = ")[0]'
    if link6 not in s6: fail("harness chain changed: item 6 no longer execs item 12 up to 'SKIPPED = '")
    if link14 not in s14: fail("harness chain changed: item 14 no longer execs item 6 up to 'VALUE_PICKS = '")
    h12 = s12.split("SKIPPED = ")[0]
    h6 = s6.split("VALUE_PICKS = ")[0]                     # exactly what item 14 execs
    if h6.count("STEP, NO_PX = ") != 1: fail("item 6 marker 'STEP, NO_PX = ' not found exactly once")
    h6a, h6b = h6.split("STEP, NO_PX = ")
    h14 = s14.split("\nVSC = ")[0]                         # item 14 up to its value run
    if h14.count(EXEC6) != 1: fail("item 14 no longer contains its exec of item 6 exactly once")
    h14_tail = h14.split(EXEC6)[1]                         # B_PRIMARY, NULL_RHO, buffer_picks
    if "def buffer_picks" not in h14_tail: fail("item 14 harness part has no buffer_picks")
    for nm, src in (("item 12", h12), ("item 6", h6), ("item 14", h14)):
        if "to_parquet" in src: fail(f"{nm} harness part writes files; the lab must not")
    if f"default_rng({SEED['iid']})" not in s6: fail("item 6 null seed changed; update SEED in lab.py")
    if f"default_rng({SEED['ar1']})" not in s14: fail("item 14 null seed changed; update SEED in lab.py")
    sha = hashlib.sha1((h12 + h6 + h14).encode()).hexdigest()[:10]
    return h12, h6, h6a, h6b, h14, h14_tail, sha


def check_data(texts):
    missing = []
    for txt in texts:
        for m in re.finditer(r'ROOT/"([^"]+)"(\)\.glob\("([^"]+)"\))?', txt):
            p = ROOT/m.group(1)
            if m.group(3):
                if not sorted(p.glob(m.group(3))): missing.append(f"{p}/{m.group(3)}")
            elif not p.exists():
                missing.append(str(p))
    if missing:
        fail("missing data files the harness reads (gitignored; run in the live repo or restore them):\n  "
             + "\n  ".join(sorted(set(missing))))


def load_harness(log):
    h12, h6, h6a, h6b, h14, h14_tail, sha = harness_sources()
    check_data([h12, h6, h14])
    ns = {"__name__": "lab_harness", "__file__": str(RES/"lab.py")}
    with contextlib.redirect_stdout(log):
        exec(compile(h6a, "test_rebalance_tax_harness", "exec"), ns)       # runs item 12's exec inside
        _install_taps(ns)
        exec(compile("STEP, NO_PX = " + h6b, "test_rebalance_tax_harness", "exec"), ns)
        exec(compile(h14_tail, "test_rank_buffer_harness", "exec"), ns)
    for k in ("STEP", "K", "G", "pick", "simulate", "arms", "buffer_picks", "NNULL", "MIN_SECTOR",
              "NULL_RHO", "MINLIQ", "AUM", "N", "CAP"):
        if k not in ns: fail(f"harness did not define {k!r}")
    # the taps must expose the very ranks that built value (else the signal mix is not the harness's)
    for k, st in enumerate(ns["STEP"]):
        o = st["o"]
        v = o[["r_SP", "r_bm", "r_cfp"]].mean(axis=1)
        if not np.allclose(v.values, o.value.values, rtol=0, atol=1e-12):
            fail(f"tap check failed at {st['D'].date()}: captured SP/bm/cfp ranks do not rebuild value")
    return ns, sha


def _install_taps(ns):
    """Read-only taps: month_inputs computes per-signal sector ranks R (over the pre-filter pool) and
    discards them. Copy R and the raw values onto `out` so STEP frames carry them. Return values of
    the harness functions are otherwise untouched."""
    sector_rank0, month_inputs0, last = ns["sector_rank"], ns["month_inputs"], {}

    def sector_rank(df, sec):
        R = sector_rank0(df, sec)
        if "SP" in df.columns: last["R"] = R
        return R

    def month_inputs(D):
        last.clear()
        out, raw, extra = month_inputs0(D)
        if "R" not in last: fail("tap: month_inputs no longer calls sector_rank on the signal frame")
        R = last["R"]
        add = {}
        for c in COMPONENTS:
            add["r_" + c] = R[c].reindex(out.index)
            add["x_" + c] = raw[c].reindex(out.index)
        return out.assign(**add), raw, extra

    ns["sector_rank"], ns["month_inputs"] = sector_rank, month_inputs


# ---------------------------------------------------------------------------------------- engine
class Lab:
    def __init__(self, ns):
        self.ns, self.STEP, self.K, self.G = ns, ns["STEP"], ns["K"], ns["G"]
        self.spy = np.array([s["spy"] for s in self.STEP])
        self.GRP = []
        for st in self.STEP:                            # item 6/14 null grouping, per month
            o = st["o"]
            big = o.sector.value_counts()[lambda x: x >= ns["MIN_SECTOR"]].index
            self.GRP.append(o.sector.where(o.sector.isin(big), "Other"))
        self._mask = {}

    def mask(self, min_adv):
        if min_adv not in self._mask:
            self._mask[min_adv] = [(st["dv"].reindex(st["o"].index).fillna(0) >= min_adv).values
                                   for st in self.STEP]
        return self._mask[min_adv]

    def scores(self, p):
        out, msk = [], self.mask(p["min_adv"])
        for k, st in enumerate(self.STEP):
            o = st["o"]
            num = pd.Series(0.0, index=o.index); den = pd.Series(0.0, index=o.index)
            for c, w in p["signal"].items():
                if c in COMPOSITES: r = o[c]
                elif p["rank_scope"] == "sector": r = o["r_" + c]
                else: r = o["x_" + c].rank(pct=True)          # item 13 MKT: across the eligible pool
                num = num + (w*r).fillna(0.0); den = den + w*r.notna()
            out.append((num/den.where(den > 0))[msk[k]])
        return out

    def picks(self, p, scores):
        ns = self.ns
        ns["N"], ns["CAP"] = p["n"], (p["cap"] if p["cap"] is not None else p["n"])
        if p["buffer"] is not None:
            return ns["buffer_picks"](scores, p["buffer"])
        return [ns["pick"](scores[k], self.STEP[k]["o"].sector) for k in range(self.K)]

    def book(self, p, picks, gross=False):
        """Item 6 simulate (monthly) or arms (quarterly, 3 phases). gross=True switches the harness's
        own cost inputs off (halfspread -> 0, AUM -> 0 kills impact) for the cost-free level null;
        restored afterwards, so every after-cost number uses the untouched cost model."""
        ns = self.ns
        hs0 = ns["halfspread"]
        ns["AUM"] = 0.0 if gross else p["aum"]
        if gross: ns["halfspread"] = lambda dv: 0.0
        try:
            if p["rebalance"] == "monthly":
                sims = [ns["simulate"](picks, set(range(self.K)))]
            else:
                _, sims = ns["arms"](picks)                   # item 6: 3 phase offsets
        finally:
            ns["halfspread"] = hs0
        b = {k: float(np.mean([s[k] for s in sims])) for k in
             ("pre_ann", "post_ann", "post_noliq_ann", "taxpaid", "termtax", "costs", "turn")}
        b["monthly"] = np.mean([s["monthly"] for s in sims], axis=0)
        b["phases"] = [s["monthly"] for s in sims] if len(sims) > 1 else None
        b["phase_post_ann"] = [s["post_ann"] for s in sims]
        b["nreb"] = [s["nreb"] for s in sims]
        return b

    def noise(self, kind, rng):
        """One noise draw: item 6 (iid) / item 14 (AR(1)) signal, same draw order and expressions."""
        out, z, rho = [], pd.Series(dtype=float), self.ns["NULL_RHO"]
        for k, st in enumerate(self.STEP):
            o = st["o"]
            e = pd.Series(rng.normal(size=len(o)), index=o.index)
            if kind == "iid":
                out.append(e.groupby(self.GRP[k]).rank(pct=True))
            else:
                prev = z.reindex(o.index)
                z = (rho*prev + np.sqrt(1-rho**2)*e).fillna(e)
                out.append(z.groupby(self.GRP[k]).rank(pct=True))
        return out

    def coverage(self, nm, p, scores, picks, b):
        K, msgs = self.K, []
        pool = [len(s) for s in scores]
        scored = [int(s.notna().sum()) for s in scores]
        nn = [len(x) for x in picks]
        cap = p["cap"] if p["cap"] is not None else p["n"]
        msec = max(self.STEP[k]["o"].sector.reindex(picks[k]).value_counts().max() for k in range(K))
        worst = max(scores[k].rank(ascending=False, method="first").reindex(picks[k]).max() for k in range(K))
        kept = np.mean([len(set(picks[k]) & set(picks[k-1])) for k in range(1, K)])
        flat = sum(1 for k in range(1, K) for t in picks[k-1] if pd.isna(self.ns["PX"].loc[self.G[k]].get(t)))
        adv = np.median([self.STEP[k]["dv"].reindex(picks[k]).median() for k in range(K)])
        if K < 15: msgs.append(f"only {K} months (< 15)")
        if min(pool) < 2*p["n"]: msgs.append(f"pool after floor {min(pool)} < 2n")
        if min(scored) < 3*p["n"]: msgs.append(f"only {min(scored)} scored names in a month (< 3n)")
        if min(nn) < p["n"]: msgs.append(f"{min(nn)} names in a month (< n {p['n']})")
        if msec > cap: msgs.append(f"sector count {msec} > cap {cap}")
        if p["buffer"] is not None and worst > p["buffer"]: msgs.append(f"held rank {worst:.0f} > buffer")
        m = b["monthly"]
        if not np.all(np.isfinite(m)) or np.abs(m).max() > 0.5:
            msgs.append(f"implausible book month {np.abs(m).max():.0%}")
        line = (f"cov {nm}: {K}mo pool {min(pool)}-{max(pool)} scored>={min(a/b for a, b in zip(scored, pool)):.0%} "
                f"names {min(nn)}-{max(nn)} maxsec {msec}/{cap} worstrank {worst:.0f} kept {kept:.1f} "
                f"held-flat {flat} medADV ${adv/1e6:.1f}m reb {'/'.join(map(str, b['nreb']))} "
                + ("OK" if not msgs else "FAIL: " + "; ".join(msgs)))
        return line, msgs, {"pool_min": min(pool), "pool_max": max(pool), "names_min": min(nn),
                            "max_per_sector": int(msec), "worst_rank": float(worst), "kept_mean": float(kept),
                            "held_flat": int(flat), "book_med_adv": float(adv)}


def tstat(x):
    return float(x.mean()/(x.std(ddof=1)/np.sqrt(len(x))))


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Batch experiment runner (see module docstring).")
    ap.add_argument("batch"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    B = load_spec(a.batch)
    L = read_ledger()
    reruns = check_rerun(L, B)
    arms = B["arms"]
    hashes = [phash(p) for _, _, p in arms]
    prior = set(L.param_hash) | read_prior()                   # tested before this batch: reference arms
    M = len(prior | set(hashes))
    new = len(set(hashes) - prior)
    night = read_night_mode()
    if night:
        import fcntl
        OUT.mkdir(parents=True, exist_ok=True)
        lockf = open(OUT/".lab.lock", "w")
        try: fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: fail("night lab: another lab run is in progress; run batches one at a time")
    night_used = night_checks(night, a, B, L, reruns, set(hashes) - prior, prior) if night else None
    if a.check:
        h = harness_sources()
        check_data([h[0], h[1], h[4]])
        print(f"OK {B['batch_id']} spec {B['spec_hash']}: {len(arms)} arms, {new} new param hashes, M would be {M}"
              + (f"; night budget {night_used + new}/{night['budget']}" if night else ""))
        for role, nm, p in arms:
            ref = "  (reference: already in the ledger, cannot clear)" if role == "variant" and phash(p) in prior else ""
            print(f"  {role:7s} {nm:16s} {phash(p)} {canon(p)}{ref}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    bid = B["batch_id"]
    logf = open(OUT/f"lab_{bid}.log", "w")
    t0 = time.time()
    ns, hsha = load_harness(logf)
    t_load = time.time() - t0
    lab, K, STEP, G = Lab(ns), ns["K"], ns["STEP"], ns["G"]
    pool = [len(s["o"]) for s in STEP]
    mcmax = max(s["o"].mc.max() for s in STEP)
    ufile = sorted((ROOT/"data/universe").glob("tickers_*.csv"))[-1].name
    glob_msgs = []
    if K < 15: glob_msgs.append(f"{K} months < 15")
    if min(pool) < 500: glob_msgs.append(f"eligible pool {min(pool)} < 500")
    if mcmax > 1e13: glob_msgs.append(f"max mktcap {mcmax/1e9:,.0f}bn > 10,000bn (share-unit error)")
    data_line = (f"data: {K} months {G[0].date()}..{G[K-1].date()} (returns to {G[K].date()}) | pool "
                 f"{min(pool)}-{max(pool)} | {ufile} | prices to {ns['PX'].index.max().date()} | max mktcap "
                 f"{mcmax/1e9:,.0f}bn | bad returns zeroed/mo max {max(s['nbad'] for s in STEP)} | harness {hsha}")
    if glob_msgs: fail(data_line + "\nFAIL: " + "; ".join(glob_msgs))

    # real runs: control and every variant through the identical path
    R, cov_lines, cov_fail = {}, [], []
    for role, nm, p in arms:
        t1 = time.time()
        sc = lab.scores(p); pk = lab.picks(p, sc); b = lab.book(p, pk)
        line, msgs, cov = lab.coverage(nm, p, sc, pk, b)
        g = lab.book(p, pk, gross=True)["monthly"]               # cost-free, for the level null only
        R[nm] = {"role": role, "p": p, "picks": pk, "b": b, "gross": g, "cov": cov, "t_real": time.time() - t1}
        cov_lines.append(line)
        if msgs: cov_fail.append(line)
    if cov_fail:
        print(data_line); print("\n".join(cov_lines))
        fail(f"implausible coverage in {len(cov_fail)} arm(s); nothing run further, ledger untouched")

    # noise nulls. d/D: the harness's own null (item 6 iid; item 14 AR(1) when a buffer is involved),
    # after cost, same noise through variant and control constructions. lvl: AR(1), cost-free, every
    # construction. One pass per noise kind; every construction sees the same draws.
    ctl_nm, ctl = arms[0][1], arms[0][2]
    dkind = lambda p: "ar1" if (p["buffer"] is not None or ctl["buffer"] is not None) else "iid"
    need_cost, need_gross, owner = {"iid": {}, "ar1": {}}, {}, {}
    for role, nm, p in arms:
        owner.setdefault(cons_key(p), nm)
        need_gross[cons_key(p)] = p
        if role == "variant" and cons_key(p) != cons_key(ctl):
            need_cost[dkind(p)][cons_key(p)] = p; need_cost[dkind(p)][cons_key(ctl)] = ctl
    NUL = {"iid": {}, "ar1": {}}; NULG = {}
    t_null, t_noise = {}, 0.0
    for kd in ("iid", "ar1"):
        todo = dict(need_cost[kd]); todo.update(need_gross if kd == "ar1" else {})
        if not todo or B["nnull"] == 0: continue
        rng = np.random.default_rng(SEED[kd])
        for i in range(B["nnull"]):
            t2 = time.time(); sc_all = lab.noise(kd, rng); t_noise += time.time() - t2
            for ck, p in todo.items():
                t3 = time.time()
                msk = lab.mask(p["min_adv"])
                pk = lab.picks(p, [sc_all[k][msk[k]] for k in range(K)])
                if ck in need_cost[kd]:
                    b = lab.book(p, pk); NUL[kd].setdefault(ck, []).append((b["monthly"], b["post_ann"]))
                if kd == "ar1" and ck in need_gross:
                    NULG.setdefault(ck, []).append(lab.book(p, pk, gross=True)["monthly"])
                t_null[ck] = t_null.get(ck, 0.0) + time.time() - t3

    # stats
    cm, cpost = R[ctl_nm]["b"]["monthly"], R[ctl_nm]["b"]["post_ann"]
    tbar, tbar_how = bar(M, K - 1)
    for role, nm, p in arms:
        r, b, ck = R[nm], R[nm]["b"], cons_key(p)
        ex = b["monthly"] - lab.spy
        r.update(vs_spy=float(ex.mean()), t=tstat(ex), changes=changes(p, ctl) if role == "variant" else "")
        r["gross_vs_spy"] = float((r["gross"] - lab.spy).mean())
        r["null_lvl"] = np.array([(g - lab.spy).mean() for g in NULG.get(ck, [])])
        r["null_pct_lvl"] = float((r["null_lvl"] <= r["gross_vs_spy"]).mean()) if len(r["null_lvl"]) else np.nan
        r["null_d"] = r["null_D"] = np.array([])
        r["null_pct_d"] = r["null_pct_D"] = np.nan
        r["null_kind_d"] = ""
        if role == "variant":
            d = b["monthly"] - cm
            r.update(d_mo=float(d.mean()), d_t=tstat(d), D=b["post_ann"] - cpost)
            r["ref"] = phash(p) in prior or bool(np.array_equal(b["monthly"], cm))   # seen before, or a no-op
            r["clears"] = False if r["ref"] else bool(r["d_t"] >= tbar)
            if ck != cons_key(ctl):
                kd = dkind(p); r["null_kind_d"] = kd
                nv, nc = NUL[kd].get(ck, []), NUL[kd].get(cons_key(ctl), [])
                if nv:
                    r["null_d"] = np.array([(mv - mc).mean() for (mv, _), (mc, _) in zip(nv, nc)])
                    r["null_D"] = np.array([pv - pc for (_, pv), (_, pc) in zip(nv, nc)])
                    r["null_pct_d"] = float((r["null_d"] <= r["d_mo"]).mean())
                    r["null_pct_D"] = float((r["null_D"] <= r["D"]).mean())
        else:
            r.update(d_mo=np.nan, d_t=np.nan, D=np.nan, clears=None, ref=False)
        r["runtime"] = r["t_real"] + (t_null.get(ck, 0.0) if owner[ck] == nm else 0.0)

    # scoreboard
    pc = lambda x: "-" if x != x else f"{x:.0%}"
    sg = lambda x, d=2: "-" if x != x else f"{x*100:+.{d}f}"
    today = datetime.date.today().isoformat()
    led = "NOT LEDGERED: --no-ledger dev run" if a.no_ledger else f"{new} new in this batch"
    S = [f"# lab {bid} | {today} | variants ever tested M = {M} ({led}) | bar: d t >= {tbar:.2f}",
         f"bar rule: Bonferroni, family-wise alpha {ALPHA_FW:g} one-sided: t of d (variant - control, monthly) "
         f">= quantile 1-{ALPHA_FW:g}/M ({tbar_how}); M = distinct param hashes in research/lab_ledger.csv + research/lab_prior.csv + this batch",
         f"hypothesis: {B['hypothesis']}",
         f"registered: {B['registered']}",
         data_line + (f" | re-run #{reruns + 1} of this batch_id" if reruns else "")
         + (f" | night lab: {night_used + new}/{night['budget']} new param sets tonight" if night else ""),
         "",
         "| variant | vs SPY %/mo | t | after-tax ann | d %/mo (t) | D pp/yr | null% d/D | null% lvl | traded/reb | bar |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for role, nm, p in arms:
        r = R[nm]
        label = f"{nm} (control)" if role == "control" else f"{nm} [{r['changes']}]"
        dcol = "-" if role == "control" else f"{sg(r['d_mo'])} ({r['d_t']:.2f})"
        ncol = "-" if r["null_pct_d"] != r["null_pct_d"] else \
            f"{r['null_pct_d']:.0%}/{r['null_pct_D']:.0%} {r['null_kind_d']}"
        lcol = "-" if r["null_pct_lvl"] != r["null_pct_lvl"] else f"{r['null_pct_lvl']:.0%}"
        bcol = "-" if role == "control" else ("ref" if r["ref"] else "CLEARS" if r["clears"] else "no")
        S.append(f"| {label} | {sg(r['vs_spy'])} | {r['t']:.2f} | {r['b']['post_ann']*100:+.1f}% | {dcol} | "
                 f"{sg(r['D'])} | {ncol} | {lcol} | {r['b']['turn']:.0%} | {bcol} |")
    S.append("")
    S += cov_lines
    S.append(f"legend: after cost (item 3b, notional ${ctl['aum']/1e6:g}m book), tax 42% (item 6); quarterly = mean of 3 "
             f"phases; d = variant-control pre-tax; D = after-tax liquidated annualised; null% = share of "
             f"{B['nnull']} noise books <= this (d/D: harness null, after cost, same noise through variant and "
             f"control construction, iid = item 6, ar1 = item 14; lvl: gross vs SPY vs AR(1) noise books of the "
             f"same construction, costs off both sides); runtime per arm includes its construction's nulls")
    S.append(f"runtime: harness load {t_load:.0f}s, noise draws {t_noise:.0f}s, per arm (run + its null): "
             + ", ".join(f"{nm} {R[nm]['runtime']:.0f}s" for _, nm, _ in arms)
             + f"; total {time.time() - t_start:.0f}s")
    text = "\n".join(S)
    print(text)
    (OUT/f"lab_{bid}.md").write_text(text + "\n")

    # full results
    rows = []
    for role, nm, p in arms:
        r, b = R[nm], R[nm]["b"]
        rows.append({"batch_id": bid, "spec_hash": B["spec_hash"], "variant": nm, "role": role,
                     "param_hash": phash(p), "params": canon(p), "changes": r["changes"], "months": K,
                     "window_start": str(G[0].date()), "window_end": str(G[K-1].date()),
                     "vs_spy_mo": r["vs_spy"], "t": r["t"], "pre_tax_ann": b["pre_ann"],
                     "after_tax_ann": b["post_ann"], "after_tax_unliq_ann": b["post_noliq_ann"],
                     "phase_after_tax_ann": b["phase_post_ann"], "costs_frac_W": b["costs"],
                     "tax_paid_frac": b["taxpaid"], "terminal_tax_frac": b["termtax"], "turnover": b["turn"],
                     "rebalances": b["nreb"], "d_mo": r["d_mo"], "d_t": r["d_t"], "D": r["D"],
                     "null_kind_d": r["null_kind_d"], "null_pct_d": r["null_pct_d"], "null_pct_D": r["null_pct_D"],
                     "gross_vs_spy_mo": r["gross_vs_spy"], "null_pct_lvl": r["null_pct_lvl"],
                     "null_d": r["null_d"].tolist(), "null_D": r["null_D"].tolist(),
                     "null_lvl": r["null_lvl"].tolist(), "M_after": M, "t_bar": tbar, "clears_bar": r["clears"],
                     **r["cov"], "runtime_s": r["runtime"], "dates": [str(g.date()) for g in G[:K]],
                     "spy": lab.spy.tolist(), "monthly": b["monthly"].tolist(), "gross_monthly": r["gross"].tolist(),
                     "phase_monthly": [x.tolist() for x in b["phases"]] if b["phases"] else None,
                     "picks": [list(x) for x in r["picks"]], "harness_sha": hsha,
                     "ledgered": not a.no_ledger, "reference": bool(r.get("ref", False))})
    pd.DataFrame(rows).to_parquet(OUT/f"lab_{bid}.parquet")

    if not a.no_ledger:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fresh = not LEDGER.exists()
        with open(LEDGER, "a", newline="") as fh:
            w = csv.writer(fh)
            if fresh: w.writerow(LEDGER_COLS)
            for role, nm, p in arms:
                r = R[nm]
                f = lambda x, d=6: "" if x is None or x != x else f"{x:.{d}f}"
                w.writerow([stamp, bid, B["spec_hash"], nm, role, phash(p), canon(p),
                            f"{G[0].date()}..{G[K-1].date()}", K, f(r["vs_spy"]), f(r["t"], 3),
                            f(r["b"]["post_ann"]), f(r["d_mo"]), f(r["d_t"], 3), f(r["D"]),
                            f(r["null_pct_d"], 3), f(r["null_pct_D"], 3), f(r["null_pct_lvl"], 3),
                            f(r["b"]["turn"], 4), M, f"{tbar:.3f}",
                            "" if r["clears"] is None else int(r["clears"]), hsha])
    logf.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LabError as e:
        print(f"LAB FAIL: {e}", file=sys.stderr)
        sys.exit(2)
