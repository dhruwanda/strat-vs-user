"""
Estimated capital-gains impact of the smallcase's realised trades.

Rules applied (Indian listed equity, STT paid, FY 2026-27):
  * holding period > 12 months -> long term (s.112A), else short term (s.111A)
  * STCG 20%, LTCG 12.5% on the excess over a Rs 1,25,000 annual exemption
  * 4% health & education cess on the tax
  * set-off: short-term losses relieve both short- and long-term gains;
    long-term losses relieve only long-term gains
  * unrealised gains are never taxed

This is an ESTIMATE. The exemption and the loss set-off are annual, taxpayer-level
quantities: the true liability depends on gains and losses elsewhere in the
investor's portfolio, surcharge thresholds, and carried-forward losses. Two views
are produced - marginal (this smallcase seen as an addition to an existing
position that already uses the exemption) and standalone (the exemption applied
in full here).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _fy(d, start_month: int) -> str:
    y = d.year if d.month >= start_month else d.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def asset_class(symbol: str, cfg) -> str:
    s = str(symbol).lower()
    if any(m in s for m in cfg.nonequity_etf_markers):
        return "non_equity_etf"
    return "equity"


def classify(realisations: pd.DataFrame, cfg) -> pd.DataFrame:
    if not len(realisations):
        return realisations.assign(term=[], financial_year=[], asset_class=[])
    r = realisations.copy()
    r["asset_class"] = r["symbol"].map(lambda x: asset_class(x, cfg))
    months = (r["sell_date"] - r["buy_date"]).dt.days / 30.436875
    thresh = np.where(r["asset_class"] == "equity", cfg.tax.long_term_months,
                      cfg.tax.nonequity_long_term_months)
    r["term"] = np.where(months > thresh, "long", "short")
    r.loc[r["buy_date"].isna(), "term"] = "unknown"
    r["financial_year"] = r["sell_date"].map(lambda d: _fy(d, cfg.tax.fy_start_month))
    return r


def base_summary(classified: pd.DataFrame) -> pd.DataFrame:
    """Deterministic tax base, no rates: realised gains and losses by financial
    year, asset class and holding-period term. This is what the engine asserts;
    treatment and rates belong to the interpretation layer."""
    if not len(classified):
        return pd.DataFrame(columns=["financial_year", "asset_class", "term",
                                     "realised_gain", "realised_loss", "net"])
    g = classified.copy()
    g["realised_gain"] = g["gain"].clip(lower=0)
    g["realised_loss"] = g["gain"].clip(upper=0)
    out = (g.groupby(["financial_year", "asset_class", "term"])
           .agg(realised_gain=("realised_gain", "sum"),
                realised_loss=("realised_loss", "sum"),
                net=("gain", "sum"), lots=("gain", "size"))
           .reset_index())
    return out


def estimate(realisations: pd.DataFrame, cfg) -> tuple:
    """Returns (per-FY estimate frame, notes list)."""
    t = cfg.tax
    r = classify(realisations, cfg)
    out, notes = [], []
    if not len(r):
        return pd.DataFrame(), ["no realised smallcase trades in the period"]

    for fy, g in r.groupby("financial_year"):
        st_eq = float(g.loc[(g.term == "short") & (g.asset_class == "equity"), "gain"].sum())
        st_ne = float(g.loc[(g.term == "short") & (g.asset_class != "equity"), "gain"].sum())
        lt_eq = float(g.loc[(g.term == "long") & (g.asset_class == "equity"), "gain"].sum())
        lt_ne = float(g.loc[(g.term == "long") & (g.asset_class != "equity"), "gain"].sum())
        unk = float(g.loc[g.term == "unknown", "gain"].sum(skipna=True))

        # set-off: by TERM, across asset classes (s.70). Short-term losses can
        # relieve any capital gain; long-term losses only long-term gains.
        st_net = st_eq + st_ne
        lt_net = lt_eq + lt_ne
        if st_net < 0 and lt_net > 0:
            use = min(-st_net, lt_net)
            lt_net -= use
            st_net += use
        # allocate the surviving nets back pro-rata to positive components so
        # each class is taxed at its own rate
        def _split(net, a, b):
            pos = max(0.0, a) + max(0.0, b)
            if net <= 0 or pos <= 0:
                return 0.0, 0.0
            return net * max(0.0, a) / pos, net * max(0.0, b) / pos
        st_eq_tax_base, st_ne_tax_base = _split(st_net, st_eq, st_ne)
        lt_eq_base, lt_ne_base = _split(lt_net, lt_eq, lt_ne)

        for label, exemption in (("marginal (exemption assumed used elsewhere)", 0.0),
                                 ("standalone (full exemption applied here)",
                                  t.ltcg_annual_exemption)):
            ex_used = min(exemption, lt_eq_base)   # s.112A exemption: equity only
            st_tax = st_eq_tax_base * t.stcg_rate + st_ne_tax_base * t.nonequity_stcg_rate
            lt_tax = max(0.0, lt_eq_base - ex_used) * t.ltcg_rate                      + lt_ne_base * t.nonequity_ltcg_rate
            tax = st_tax + lt_tax
            out.append(dict(
                financial_year=fy, view=label,
                st_gain_equity=st_eq, st_gain_nonequity_etf=st_ne,
                lt_gain_equity=lt_eq, lt_gain_nonequity_etf=lt_ne,
                unclassified_gain=unk,
                short_term_after_setoff=st_net, long_term_after_setoff=lt_net,
                ltcg_exemption_applied=ex_used,
                stcg_tax=st_tax, ltcg_tax=lt_tax, cess=tax * t.cess_rate,
                estimated_tax=tax * (1 + t.cess_rate),
                loss_carried_forward=min(0.0, st_net) + min(0.0, lt_net)))

    notes = [
        "Gold/silver ETFs are NOT equity-oriented: short-term gains are taxed at "
        f"the investor's slab rate (configured here as {t.nonequity_stcg_rate:.0%} "
        "- an assumption, adjust in TaxRules), long-term (>"
        f"{t.nonequity_long_term_months}m, listed) at "
        f"{t.nonequity_ltcg_rate:.1%} under s.112 with no s.112A exemption.",
        f"Equity: STCG {t.stcg_rate:.0%} (s.111A), LTCG {t.ltcg_rate:.1%} above "
        f"Rs {t.ltcg_annual_exemption:,.0f} (s.112A), cess {t.cess_rate:.0%}; "
        f"holding period threshold {t.long_term_months} months.",
        "Gains are computed FIFO across ALL units of each scrip, which is the "
        "statutory basis. Where the investor also bought a constituent outside the "
        "smallcase, a smallcase sale can consume those earlier personal lots.",
        "Losses reduce the estimate only against gains inside this smallcase. Real "
        "set-off happens across the whole portfolio and across carried-forward years.",
        "Unrealised gains carry no tax and are excluded.",
        "Surcharge is not modelled; it depends on total income.",
    ]
    return pd.DataFrame(out), notes
