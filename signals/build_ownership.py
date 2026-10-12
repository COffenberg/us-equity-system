"""Institutional ownership breadth, live. Chen/Hong/Stein (2002).

Worthless alone (+0.42% univariate) and strong in combination (+7.86% inside
composite quintiles, +9.92% in thin names). Survives a liquidity double-sort:
+4.78pp in thin names, +2.26pp in mid, -0.48pp in liquid. Institutions know most
where fewest of them are looking.
"""
import pandas as pd, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
TODAY = pd.Timestamp("2026-09-06")

H = pd.read_parquet(ROOT/"data/sec/f13_holdings.parquet")
H["avail"] = H.dt + pd.Timedelta(days=45)          # 13F filing deadline
pub = H[H.avail <= TODAY]
per = sorted(pub.dt.unique())
cur, prv = per[-1], per[-2]
c = pub[pub.dt == cur].set_index("ticker")
p = pub[pub.dt == prv].set_index("ticker")

out = pd.DataFrame(index=c.index.union(p.index))
out["inst_holders"]   = c.holders.reindex(out.index)
out["inst_d_holders"] = c.holders.reindex(out.index) - p.holders.reindex(out.index)
out["inst_d_shares"]  = (c.shares.reindex(out.index) /
                         p.shares.reindex(out.index).replace(0, np.nan)) - 1
out = out.replace([np.inf, -np.inf], np.nan)
out.to_parquet(ROOT/"signals/ownership.parquet")
print(f"13F as of {str(cur)[:10]} vs {str(prv)[:10]} (both past their filing deadline)")
print(f"  {len(out):,} tickers | holders median {out.inst_holders.median():.0f}, "
      f"p90 {out.inst_holders.quantile(.9):.0f}")
print(f"  net new holders: {int((out.inst_d_holders>0).sum()):,} gained, "
      f"{int((out.inst_d_holders<0).sum()):,} lost")
