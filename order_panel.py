"""Render the order panel from signals/orders.json.

Fractional shares mean sizing is a percentage question only. Equal weight means
every line is the same number, so the page states it once instead of repeating
it forty times. What is left is the list, and what changed.
"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent

def build():
    f = ROOT / "signals/orders.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    tk = lambda ts: ", ".join(f'<b class="tk">{t}</b>' for t in ts) or "nothing"

    trades = ""
    if d.get("prev_date"):
        n_tr = len(d["adds"]) + len(d["drops"])
        trades = (
          '      <div class="trades">\n'
          f'        <h3>This month&rsquo;s trades &mdash; {n_tr} of them</h3>\n'
          f'        <p><span class="buy">Buy</span> {tk(d["adds"])} at {d["weight_pct"]}% each.</p>\n'
          f'        <p><span class="sell">Sell</span> {tk(d["drops"])} in full.</p>\n'
          f'        <p class="meta">The other {d["unchanged"]} positions are unchanged. '
          'Leave them alone &mdash; trading only the difference is what keeps turnover and tax down.</p>\n'
          '      </div>\n')

    lines = []
    for i, o in enumerate(d["holdings"], 1):
        soon = ' <b style="color:var(--brass)" title="reports within 21 days">&bull;</b>' if o["soon"] else ""
        lines.append(f'<tr><td class="r">{i}</td><td class="tk">{o["ticker"]}{soon}</td>'
                     f'<td>{o["name"]}</td><td class="dv">{o["sector"]}</td>'
                     f'<td class="dv">{o["why"]}</td></tr>')

    return (trades +
      '      <div class="listbox">\n'
      f'        <h3>The book &mdash; {d["n"]} names, {d["weight_pct"]}% each</h3>\n'
      f'        <p class="sub">Equal weight, capped at six per sector, priced {d["as_of"]}. '
      'Equal because nothing in the evidence justifies betting more on one name than another. '
      'A brass dot marks a company reporting within 21 days.</p>\n'
      '        <div class="tw"><table><thead><tr><th class="r">#</th><th>Ticker</th>'
      '<th>Company</th><th>Sector</th><th>Why it is here</th></tr></thead><tbody>\n'
      + "\n".join(lines) +
      '\n        </tbody></table></div>\n      </div>\n')
