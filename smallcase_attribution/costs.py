"""
Attribute transaction costs to the smallcase.

Account-level charge totals cannot simply be subtracted, because they also cover
unrelated trades. Instead every charge head is derived per trade from a rate card
and then CALIBRATED so the per-trade amounts sum exactly to the total the broker
reported. The calibration factor is published: a factor far from 1.0 means the
rate assumption is wrong and the reader can see it.

Directly identifiable charges bypass the rate card entirely:
  * smallcase transaction fees - named in the ledger, matched to an event date
  * DP charges - the ledger names the scrip and the sale date, so each one is
    matched to the sale that caused it
  * demat AMC - an account-level fee that exists whether or not the smallcase
    does; reported but NOT attributed
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd


def _is_etf(symbol: str, cfg) -> bool:
    s = symbol.lower()
    return any(m in s for m in cfg.rates.etf_symbol_markers)


def per_trade_charges(trades: pd.DataFrame, cfg) -> pd.DataFrame:
    r = cfg.rates
    t = trades.copy()
    etf = t["symbol"].map(lambda s: _is_etf(s, cfg))
    buy = t["trade_type"] == "buy"

    t["stt"] = np.where(
        etf,
        np.where(buy, r.stt_etf_buy, r.stt_etf_sell) * t["value"],
        np.where(buy, r.stt_delivery_buy, r.stt_delivery_sell) * t["value"])
    t["exchange_txn"] = t["value"] * t["exchange"].map(
        lambda e: r.exchange_txn.get(str(e).upper(), min(r.exchange_txn.values())))
    t["sebi_turnover"] = t["value"] * r.sebi_turnover
    t["ipft"] = t["value"] * r.ipft
    t["stamp_duty"] = np.where(buy, t["value"] * r.stamp_duty_buy, 0.0)
    t["brokerage"] = t["value"] * r.brokerage_delivery
    t["gst"] = r.gst_rate * (t["brokerage"] + t["exchange_txn"]
                             + t["sebi_turnover"] + t["ipft"])
    return t


_HEAD_MAP = {
    "stt": ["securities transaction tax"],
    "exchange_txn": ["exchange transaction charges"],
    "sebi_turnover": ["sebi turnover fees"],
    "ipft": ["ipft"],
    "stamp_duty": ["stamp duty"],
    "brokerage": ["brokerage"],
    "gst": ["integrated gst", "central gst", "state gst"],
    "clearing": ["clearing charges"],
}


def calibrate(trades_with_charges: pd.DataFrame, reported: dict, cfg) -> tuple:
    """Scale each derived head so it sums to the broker's reported total."""
    t = trades_with_charges.copy()
    rep = {k.strip().lower(): v for k, v in reported.items()}
    diag = []
    # GST is a function of the other heads, so calibrate it last and rebuild it
    # from the already-calibrated components rather than from the raw rate card.
    order = [h for h in _HEAD_MAP if h != "gst"] + ["gst"]
    for head in order:
        keys = _HEAD_MAP[head]
        if head == "gst":
            t["gst"] = cfg.rates.gst_rate * (t["brokerage"] + t["exchange_txn"]
                                             + t["sebi_turnover"] + t["ipft"])
        total_rep = sum(v for k, v in rep.items()
                        if any(k.startswith(x) for x in keys))
        if head not in t.columns:
            t[head] = 0.0
        derived = float(t[head].sum())
        if derived > 0 and total_rep > 0:
            factor = total_rep / derived
            method = "derived from rate card, calibrated to broker total"
        elif total_rep > 0:
            # nothing derivable (e.g. brokerage at a zero rate): allocate by turnover
            factor = np.nan
            t[head] = total_rep * t["value"] / t["value"].sum()
            derived = 0.0
            method = "not derivable per trade; allocated pro-rata by turnover"
        else:
            factor = np.nan
            method = "reported as nil"
        if np.isfinite(factor):
            t[head] = t[head] * factor
        diag.append(dict(charge_head=head, broker_reported_total=total_rep,
                         derived_before_calibration=derived,
                         calibration_factor=factor, method=method))
    t["total_charges"] = t[list(_HEAD_MAP)].sum(axis=1)
    return t, pd.DataFrame(diag)


def parse_other_debits(other: pd.DataFrame, cfg) -> pd.DataFrame:
    """Classify ledger entries and pull the scrip/date out of DP charge text."""
    if not len(other):
        return other.assign(category=[], symbol=[], event_date=[])
    o = other.copy()
    p = o["particulars"].str.lower()
    o["category"] = np.select(
        [p.str.contains("|".join(cfg.dp_charge_keywords)),
         p.str.contains("|".join(cfg.smallcase_fee_keywords)),
         p.str.contains("|".join(cfg.amc_keywords))],
        ["dp_charge", "smallcase_fee", "demat_amc"], default="other")
    ex = o["particulars"].str.extract(
        r"(?i)sale of\s+(?P<symbol>[A-Za-z0-9&\-\.]+)\s+on\s+(?P<d>\d{2}/\d{2}/\d{4})")
    o["symbol"] = ex["symbol"].str.upper()
    o["event_date"] = pd.to_datetime(ex["d"], format="%d/%m/%Y", errors="coerce")
    fee_d = o["particulars"].str.extract(r"(?i)fee for\s+(?P<d>\d{2}/\d{2}/\d{4})")["d"]
    o.loc[o["event_date"].isna(), "event_date"] = pd.to_datetime(
        fee_d, format="%d/%m/%Y", errors="coerce")
    o["event_date"] = o["event_date"].fillna(o["posting_date"])
    return o


def attribute_costs(attributed_trades: pd.DataFrame, other: pd.DataFrame,
                    cfg) -> tuple:
    """Returns (per-trade table, summary table, unattributed table)."""
    t = attributed_trades
    sc = t["attribution"] == "smallcase"
    heads = [h for h in _HEAD_MAP if h in t.columns]

    rows = []
    for h in heads:
        rows.append(dict(cost_head=h.replace("_", " "),
                         smallcase=float(t.loc[sc, h].sum()),
                         outside=float(t.loc[~sc, h].sum()),
                         total=float(t[h].sum()),
                         basis="per-trade, rate card calibrated to broker total"))

    o = other
    sc_sell_keys = set(zip(t.loc[sc & (t.trade_type == "sell"), "symbol"],
                           t.loc[sc & (t.trade_type == "sell"), "trade_date"]))
    any_sell_keys = set(zip(t.loc[t.trade_type == "sell", "symbol"],
                            t.loc[t.trade_type == "sell", "trade_date"]))
    dp = o[o["category"] == "dp_charge"].copy()
    if len(dp):
        dp["matched_smallcase"] = [(s, d) in sc_sell_keys
                                   for s, d in zip(dp["symbol"], dp["event_date"])]
        dp["matched_any"] = [(s, d) in any_sell_keys
                             for s, d in zip(dp["symbol"], dp["event_date"])]
        rows.append(dict(cost_head="dp charges",
                         smallcase=float(dp.loc[dp["matched_smallcase"], "debit"].sum()),
                         outside=float(dp.loc[~dp["matched_smallcase"], "debit"].sum()),
                         total=float(dp["debit"].sum()),
                         basis="ledger names the scrip and sale date; matched to the sale"))
    fee = o[o["category"] == "smallcase_fee"]
    if len(fee):
        rows.append(dict(cost_head="smallcase transaction fees",
                         smallcase=float(fee["debit"].sum()), outside=0.0,
                         total=float(fee["debit"].sum()),
                         basis="named in the ledger; charged on buy / invest-more / SIP only"))
    amc = o[o["category"] == "demat_amc"]
    unattr = []
    if len(amc):
        unattr.append(dict(cost_head="demat AMC", amount=float(amc["debit"].sum()),
                           reason="account-level fee, incurred whether or not the "
                                  "smallcase exists; not attributable"))
    misc = o[o["category"] == "other"]
    if len(misc) and misc["debit"].sum():
        unattr.append(dict(cost_head="other ledger entries",
                           amount=float(misc["debit"].sum()),
                           reason="not identifiable as smallcase-related"))
    if getattr(cfg, "subscription_fee", None):
        rows.append(dict(cost_head="subscription fee (configured)",
                         smallcase=float(cfg.subscription_fee), outside=0.0,
                         total=float(cfg.subscription_fee),
                         basis="CONFIGURED PLACEHOLDER, not from any file; replace "
                               "with actual manager invoices"))
    else:
        unattr.append(dict(cost_head="smallcase subscription fee", amount=np.nan,
                           reason="billed by the manager outside the broker "
                                  "account; not present in any supplied file"))
    summary = pd.DataFrame(rows)
    return t, summary, pd.DataFrame(unattr)
