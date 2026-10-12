"""Regenerate the data-driven parts of tracker.html from live parquet.

Everything narrative in the tracker is hand-written and stays untouched. Only the
numbers that go stale are replaced: the header strip, the three ranked tables, and
the read-across watchlist. Run after the daily pipeline and the page is current
without a session.
"""
import pandas as pd, numpy as np, pathlib, html, re, warnings, datetime as dt
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent
T = ROOT/"tracker.html"

o = pd.read_parquet(ROOT/"signals/ranked.parquet")
CL = [c[2:] for c in o.columns if c.startswith("c_")]
try:    W = pd.read_parquet(ROOT/"signals/watchlist_readacross.parquet")
except Exception: W = pd.DataFrame()
snaps = sorted(ROOT.glob("data/snapshots/ohlcv_*.parquet"))
last_snap = snaps[-1].stem.split("_")[1] if snaps else "none"
n_days = len(snaps)
preds = sorted(ROOT.glob("forward/prediction_*.json"))
n_pred = sum(1 for _ in preds) * 145

def top(col, n=15, cap=6):
    picked, counts = [], {}
    for t, r in o.sort_values(col, ascending=False).iterrows():
        s = r.get("sector", "Other")
        if counts.get(s, 0) >= cap: continue
        picked.append(t); counts[s] = counts.get(s, 0) + 1
        if len(picked) == n: break
    return o.loc[picked]

def rows(col):
    out = []
    for i, (t, r) in enumerate(top(col).iterrows(), 1):
        d = r[[f"c_{c}" for c in CL]].astype(float).dropna().nlargest(2)
        drv = ", ".join(f"{k[2:]} {v:.2f}" for k, v in d.items())
        mc = "&mdash;" if pd.isna(r.get("mktcap_bn")) else f"${r.mktcap_bn:,.1f}bn"
        soon = ' <b style="color:var(--brass)">&bull;</b>' if r.get("in_pre_earnings_window") else ""
        out.append(f'<tr><td class="r">{i}</td><td class="tk">{t}{soon}</td>'
                   f'<td>{html.escape(str(r.get("name") or "")[:22])}</td>'
                   f'<td class="dv">{html.escape(str(r.get("sector","")))}</td>'
                   f'<td class="r">{mc}</td><td class="r sc">{r[col]:.3f}</td>'
                   f'<td class="dv">{drv}</td></tr>')
    return "\n".join(out)

def watch_rows():
    if not len(W): return None
    out = []
    for i, (t, r) in enumerate(W.sort_values("COMBINED", ascending=False).head(15).iterrows(), 1):
        mc = "" if pd.isna(r.get("mktcap_bn")) else ""
        out.append(f'<tr><td class="r">{i}</td><td class="tk">{t}</td>'
                   f'<td>{html.escape(str(r.get("name") or "")[:24])}</td>'
                   f'<td class="dv">{html.escape(str(r.get("sector","")))}</td>'
                   f'<td class="r">{r.days_to_next_earnings:.0f}d</td>'
                   f'<td class="r sc">{r.peer_pct:.2f}</td>'
                   f'<td class="r">{r.COMBINED:.3f}</td></tr>')
    return "\n".join(out)

def score_rows():
    f = ROOT/"forward/scoreboard.parquet"
    if not f.exists(): return None
    S = pd.read_parquet(f)
    S = S[S.days >= 0].sort_values(["frozen","list"])
    out = []
    for r in S.itertuples():
        tone = "var(--done)" if r.vs_spy > 0 else "var(--open)"
        out.append(f'<tr><td class="tk">{r.frozen}</td><td>{r.list}</td><td class="r">{r.days}</td>'
                   f'<td class="r">{r.ret:+.2%}</td><td class="r">{r.spy:+.2%}</td>'
                   f'<td class="r sc" style="color:{tone}">{r.vs_spy:+.2%}</td>'
                   f'<td class="r">{r.hit_vs_spy:.0%}</td><td class="r dv">{r.best:+.0%} / {r.worst:+.0%}</td></tr>')
    return "\n".join(out)

import order_panel
_ob = order_panel.build()

h = T.read_text()
def swap_tbody(h, header, body):
    if body is None or header not in h: return h
    i = h.index(header); a = h.index("<tbody>", i) + 7; b = h.index("</tbody>", a)
    return h[:a] + "\n" + body + "\n" + h[b:]

for col, key in [("CORE","Core &mdash; top 15 of 40"),
                 ("SKEWLIST","Skew &mdash; top 15 of 40"),
                 ("COMBINED","Combined &mdash; top 15 of 40")]:
    h = swap_tbody(h, key, rows(col))
h = swap_tbody(h, "Reporting soon, peers already beat", watch_rows())
h = swap_tbody(h, "Every frozen list, marked to market", score_rows())
if _ob:
    import re as _re
    h = _re.sub(r"<!--ORDERS-->.*?<!--/ORDERS-->",
                lambda m: "<!--ORDERS-->" + _ob + "<!--/ORDERS-->", h, flags=_re.S)

n_sig = len([c for c in o.columns if c.startswith("c_")])
today = dt.date.today()
subs = [
 (r'<h2>Ranked lists &mdash; [^<]*</h2>', f'<h2>Ranked lists &mdash; {today:%-d %b %Y}</h2>'),
 (r'<div data-tone="done"><b>[\d,]+</b><span>Names scored</span></div>',
  f'<div data-tone="done"><b>{len(o):,}</b><span>Names scored</span></div>'),
 (r'<div><b>\d+</b><span>Clusters</span></div>', f'<div><b>{n_sig}</b><span>Clusters</span></div>'),
 (r'<div><b>\d+</b><span>Days recorded</span></div>', f'<div><b>{n_days}</b><span>Days recorded</span></div>'),
 (r'<div data-tone="brass"><b>[\d,]+</b><span>Frozen predictions</span></div>',
  f'<div data-tone="brass"><b>{n_pred}</b><span>Frozen predictions</span></div>'),
 (r'<div(?: data-tone="brass")?><b>\d+</b><span>Names qualify now</span></div>',
  f'<div><b>{len(W)}</b><span>Names qualify now</span></div>'),
]
# Live header numbers were hand-written and went stale; derive them from data.
try:
    import json as _json
    _S = pd.read_parquet(ROOT/"forward/scoreboard.parquet")
    _live = _S[_S.list == "COMBINED"].sort_values("frozen").iloc[-1]
    _e = float(_live.vs_spy) * 100
    _tone = "done" if _e >= 0 else "open"
    _o = _json.load(open(ROOT/"signals/orders.json"))
    subs += [
     (r'<div data-tone="\w+"><b>[^<]*</b><span>Edge vs SPY, live</span></div>',
      f'<div data-tone="{_tone}"><b>{_e:+.1f}pp</b><span>Edge vs SPY, live</span></div>'),
     (r'<div data-tone="brass"><b>\d+</b><span>Trades this month</span></div>',
      f'<div data-tone="brass"><b>{len(_o["adds"]) + len(_o["drops"])}</b><span>Trades this month</span></div>'),
     (r'<div><b>\d+</b><span>Monthly records</span></div>',
      f'<div><b>{len(preds)}</b><span>Monthly records</span></div>'),
    ]
except Exception as _err:
    print(f"WARN live header not refreshed: {_err}")
for pat, rep in subs: h = re.sub(pat, rep, h)
T.write_text(h)
print(f"tracker.html regenerated — {len(o):,} names, {n_sig} clusters, "
      f"{n_days} snapshots, last {last_snap}, {len(W)} on watchlist")
