"""Carry the cloud's fresh numbers into the published dashboard without touching its prose.

The published page carries hand-written notes that are not in this repo. The cloud run
regenerates tracker.html here from fresh data. Publishing either one alone loses
something: the repo copy drops the notes, a stale copy keeps old numbers. This takes the
regions dashboard.py owns from the cloud file and swaps them into the published HTML,
leaving every other byte of the published page as it was. No parquet needed.

Keep REGIONS in step with what dashboard.py writes.

  python refresh_live.py <published.html> <cloud tracker.html> <out.html>
"""
import re, sys, pathlib

TBODIES = ["Core &mdash; top 15 of 40", "Skew &mdash; top 15 of 40", "Combined &mdash; top 15 of 40",
           "Reporting soon, peers already beat", "Every frozen list, marked to market"]
BLOCKS = [r"<!--ORDERS-->.*?<!--/ORDERS-->"]
CELLS = [r'<h2>Ranked lists &mdash; [^<]*</h2>',
         r'<div data-tone="done"><b>[\d,]+</b><span>Names scored</span></div>',
         r'<div><b>\d+</b><span>Clusters</span></div>',
         r'<div><b>\d+</b><span>Days recorded</span></div>',
         r'<div data-tone="brass"><b>[\d,]+</b><span>Frozen predictions</span></div>',
         r'<div(?: data-tone="brass")?><b>\d+</b><span>Names qualify now</span></div>',
         r'<div data-tone="\w+"><b>[^<]*</b><span>Edge vs SPY, live</span></div>',
         r'<div data-tone="brass"><b>\d+</b><span>Trades this month</span></div>',
         r'<div><b>\d+</b><span>Monthly records</span></div>']


def tbody(h, key):
    i = h.index(key); a = h.index("<tbody>", i) + 7
    return a, h.index("</tbody>", a)


def main(live_p, cloud_p, out_p):
    live, cloud = pathlib.Path(live_p).read_text(), pathlib.Path(cloud_p).read_text()
    h, done, missing = live, [], []
    for key in TBODIES:
        if key not in h and key not in cloud: continue          # region not on either page
        if key not in h or key not in cloud: missing.append(key); continue
        ca, cb = tbody(cloud, key); a, b = tbody(h, key)
        h = h[:a] + cloud[ca:cb] + h[b:]; done.append(key)
    for pat in BLOCKS + CELLS:
        mc, ml = re.search(pat, cloud, re.S), re.search(pat, h, re.S)
        if not mc and not ml: continue
        if not (mc and ml): missing.append(pat); continue
        h = h[:ml.start()] + mc.group(0) + h[ml.end():]; done.append(pat)
    # Everything outside the swapped regions must be byte-identical to the published page.
    def strip(x):
        for key in TBODIES:
            if key in x: a, b = tbody(x, key); x = x[:a] + x[b:]
        for pat in BLOCKS + CELLS: x = re.sub(pat, "", x, flags=re.S)
        return x
    if strip(h) != strip(live):
        sys.exit("FAIL  prose outside the data regions changed; not writing")
    pathlib.Path(out_p).write_text(h)
    print(f"OK    {len(done)} regions refreshed from the cloud copy, prose untouched"
          + (f"; ON ONE PAGE ONLY (left as published): {missing}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
