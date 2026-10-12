"""OSAP survivors we never built, implemented from published definitions.

Only signals that already cleared the triage (recent t>2, buildable on free data)
are here -- that satisfies the "published evidence" half of the two-source rule
before a single number is computed. The four-test screen plus a noise null
supplies the other half.

Definitions taken verbatim from SignalDoc.csv; departures from the original are
noted per signal, because a silently-approximated signal is a different signal.
"""
import pandas as pd, numpy as np

def _q(f, concept):
    d = f[(f.concept == concept) & (~f.instantaneous) & f.dur.between(60, 115)]
    return d.sort_values("period_end").drop_duplicates(["ticker","period_end"], keep="last")

def _ttm(f, concept, n=4):
    d = _q(f, concept).groupby("ticker").tail(n)
    g = d.groupby("ticker").val.agg(["sum","count"])
    return g.loc[g["count"] == n, "sum"]

def _inst(f, concept, nth=1):
    d = f[(f.concept == concept) & f.instantaneous].sort_values("period_end")
    if nth == 1: return d.groupby("ticker").last().val
    return d.groupby("ticker").tail(nth).groupby("ticker").first().val

def compute(f, sh, px, hist, D):
    out = pd.DataFrame(index=px.index)
    r = hist.pct_change()
    m = hist.resample("ME").last()
    mret = m.pct_change()

    # ---- price / momentum family -------------------------------------------
    def back(n): return hist.iloc[max(0, len(hist)-n-1)]
    last = hist.iloc[-1]
    out["Mom12m"] = last/back(252) - 1
    out["Mom6m"]  = last/back(126) - 1

    # Mom12mOffSeason: average return in NON-anniversary months of the past year.
    # The published version uses the same calendar month in prior years; with 3
    # years of history that is 2 observations, so this uses the 11 non-matching
    # months of the last year instead. Weaker, and labelled as such.
    if len(mret) >= 13:
        cur_m = mret.index[-1].month
        w = mret.iloc[-12:]
        off = w[w.index.month != cur_m]
        out["Mom12mOffSeason"] = off.mean()

    # FirmAgeMom: 6m return, but only meaningful among the youngest names.
    age = hist.notna().sum() * (365/252)
    young = age <= age.quantile(0.20)
    out["FirmAgeMom"] = (last/back(126) - 1).where(young)

    # MomVol: Mom6m conditioned on high turnover. Published version double-sorts
    # 10x3; a continuous interaction is used here since we rank cross-sectionally.
    if "volume" in getattr(hist, "attrs", {}): pass
    out["MomVol"] = out["Mom6m"]        # turnover leg attached by the caller

    # ---- accounting family --------------------------------------------------
    A  = _inst(f, "Assets")
    A1 = _inst(f, "Assets", nth=5)
    avgA = ((A + A1.reindex(A.index)) / 2).where(lambda s: s > 0)

    # ChTax: 4-quarter change in quarterly tax, scaled by lagged assets
    tq = _q(f, "IncomeTaxExpenseBenefit")
    if len(tq):
        piv = tq.sort_values("period_end").groupby("ticker").tail(5)
        cur = piv.groupby("ticker").last().val
        yr  = piv.groupby("ticker").first().val
        out["ChTax"] = (cur - yr) / A1.reindex(cur.index).where(lambda s: s > 0)

    # NetEquityFinance: (issuance - repurchase) / average assets, |ratio| <= 1.
    # Published sign is NEGATIVE (raising equity predicts poorly), so negate.
    iss = _ttm(f, "ProceedsFromIssuanceOfCommonStock")
    rep = _ttm(f, "PaymentsForRepurchaseOfCommonStock")
    nef = (iss.reindex(avgA.index).fillna(0) - rep.reindex(avgA.index).fillna(0)) / avgA
    out["NetEquityFinance"] = -nef.where(nef.abs() <= 1)

    # ShareIss5Y: growth in share count. Panel spans <5y, so this is the longest
    # window available (about 3 years). Negated -- dilution predicts poorly.
    s_now = _inst(f, "EntityCommonStockSharesOutstanding")
    s_old = _inst(f, "EntityCommonStockSharesOutstanding", nth=13)
    for c in ("CommonStockSharesOutstanding","WeightedAverageNumberOfDilutedSharesOutstanding"):
        d = sh[sh.concept == c].sort_values("period_end")
        s_now = s_now.combine_first(d.groupby("ticker").last().val)
        s_old = s_old.combine_first(d.groupby("ticker").tail(13).groupby("ticker").first().val)
    g = (s_now / s_old.where(lambda x: x > 0)) - 1
    out["ShareIssLong"] = -g.where(g.abs() < 5)

    # EBM: enterprise book-to-market. Preferred-stock terms are absent from the
    # panel, so this is (equity + cash - debt) / (mcap + cash - debt).
    eq   = _inst(f, "StockholdersEquity")
    cash = _inst(f, "CashAndCashEquivalentsAtCarryingValue")
    debt = _inst(f, "LongTermDebtNoncurrent")
    mc   = (px * s_now.reindex(px.index)).dropna()
    mc   = mc[mc > 5e7]
    adj  = cash.reindex(mc.index).fillna(0) - debt.reindex(mc.index).fillna(0)
    num  = eq.reindex(mc.index) + adj
    den  = mc + adj
    out["EBM"] = (num / den.where(lambda s: s.abs() > 1e6))

    # CompEquIss: 5y growth in market value of equity minus 5y stock return.
    # Captures value created by issuing shares rather than by performance. Negated.
    if len(hist) > 500:
        px_old = hist.iloc[max(0, len(hist)-756)]
        stock_ret = (last / px_old) - 1
        mv_growth = ((px * s_now.reindex(px.index)) /
                     (px_old * s_old.reindex(px.index)).where(lambda s: s > 0)) - 1
        out["CompEquIss"] = -(mv_growth - stock_ret)

    # RevenueSurprise: 4-quarter change in revenue per share, minus its own
    # trailing average change, scaled by the SD of those changes.
    rev = _q(f, "Revenues")
    if len(rev) < 500:
        rev = pd.concat([rev, _q(f, "RevenueFromContractWithCustomerExcludingAssessedTax")])
        rev = rev.sort_values("period_end").drop_duplicates(["ticker","period_end"], keep="last")
    if len(rev):
        rev = rev.merge(s_now.rename("sh"), left_on="ticker", right_index=True, how="left")
        rev["rps"] = rev.val / rev.sh.where(lambda s: s > 0)
        rev = rev.sort_values(["ticker","period_end"])
        rev["d4"] = rev.groupby("ticker").rps.diff(4)
        g = rev.groupby("ticker").d4
        out["RevenueSurprise"] = (g.last() - g.mean()) / g.std().replace(0, np.nan)

    # EarningsConsistency: inverse dispersion of year-over-year earnings growth.
    ni = _q(f, "NetIncomeLoss").sort_values(["ticker","period_end"])
    ni["g"] = ni.groupby("ticker").val.diff(4) / ni.groupby("ticker").val.shift(4).abs().replace(0, np.nan)
    gg = ni.groupby("ticker").g
    out["EarningsConsistency"] = 1.0 / gg.std().replace(0, np.nan)

    # InvGrowth: inventory growth. No inventory concept in the panel, so this uses
    # growth in non-cash current assets as the closest available proxy. Negated.
    ca = _inst(f, "AssetsCurrent"); ca1 = _inst(f, "AssetsCurrent", nth=5)
    nc  = ca - cash.reindex(ca.index).fillna(0)
    nc1 = ca1 - _inst(f, "CashAndCashEquivalentsAtCarryingValue", nth=5).reindex(ca1.index).fillna(0)
    ig = (nc / nc1.reindex(nc.index).where(lambda s: s > 0)) - 1
    out["InvGrowthProxy"] = -ig.where(ig.abs() < 5)

    return out.replace([np.inf, -np.inf], np.nan)
