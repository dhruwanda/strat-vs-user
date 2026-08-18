"""Orchestration for the implementation attribution and its validation."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import price_data as PD, model_book as MB, decomposition as DC
from . import strategy_replication as SR


def run_gap_decomposition(v1: dict, store: PD.PriceStore, cfg) -> dict:
    ev = v1["events"]
    val = v1["strategy_index"]["date"].iloc[-1]
    start = v1["strategy_index"]["date"].iloc[0]

    impl = DC.attribute_implementation(v1["recommended_lines"],
                                       v1["smallcase_trades"], ev, store, cfg)
    timeline = DC.rebalance_timeline(v1["index_values"], v1["constituents"],
                                     ev, store, cfg, start)
    ev_reb, ev_inv = DC.quantity_convention_evidence(ev, v1["mapping"], store, cfg)
    rep_daily, rep_summary = SR.compare_conventions(
        v1["index_values"], v1["constituents"], v1["mapping"], store,
        pd.Timestamp("2025-12-01"), val)

    # long-horizon P&L context (model book on the investor's cash flows)
    model = MB.simulate(ev, v1["mapping"], store, cfg, val)
    p_term = {s: store.bar(s, val, "close", "context terminal")
              for s in set(v1["smallcase_trades"]["symbol"])}
    apnl = _actual_pnl(v1["smallcase_trades"], p_term)

    imp = v1["implementation"]
    costs = float(v1["cost_summary"]["smallcase"].sum())
    taxdf = v1["tax_estimate"]
    tax = (float(taxdf.loc[taxdf["view"].str.startswith("marginal"),
                           "estimated_tax"].sum()) if len(taxdf) else 0.0)
    div = imp.get("dividends") or 0.0
    s = impl["summary"].set_index("measure")["amount"]
    overall = pd.DataFrame([
        ("Strategy return (official index, rebased to first investment)",
         v1["strategy_summary"]["absolute_return"], "pct"),
        ("Implementation price effect (model ref vs actual fills)",
         s["Implementation price effect - TOTAL"], "rs"),
        ("  of which buys", s["Implementation price effect - BUYS"], "rs"),
        ("  of which sells", s["Implementation price effect - SELLS"], "rs"),
        ("Quantity cash deviation (actual vs model quantities)",
         s["Quantity cash component - TOTAL"], "rs"),
        ("Context: model-book P&L on the investor's cash flows",
         model["pnl_total"], "rs"),
        ("Context: actual pre-cost P&L (same closes)", apnl, "rs"),
        ("Context: pre-cost P&L gap", apnl - model["pnl_total"], "rs"),
        ("Dividends (separate; configured until the Zerodha report is supplied)",
         div, "rs"),
        ("Attributable costs incl. configured subscription", -costs, "rs"),
        ("Tax: rates not applied by the engine (see Tax Base sheet)",
         -tax, "rs"),
        ("Net outcome after costs and estimated tax",
         imp["total_returns"] - costs - tax, "rs"),
    ], columns=["metric", "value", "unit"])

    daily = _daily_series(v1, model, store)
    return dict(implementation=impl, timeline=timeline, daily_series=daily,
                dividend_statement=v1.get("dividend_statement", pd.DataFrame()),
                tax_base=v1.get("tax_base", pd.DataFrame()),
                evidence_rebalance=ev_reb, evidence_invest=ev_inv,
                replication_daily=rep_daily, replication_summary=rep_summary,
                model=model, overall=overall,
                deferred_repairs=v1.get("deferred_repairs", pd.DataFrame()),
                missing_prices=store.missing_report(),
                splits_detected=store.splits_in_span())


def _daily_series(v1, model, store):
    """Day-by-day market value of the model book and the actual smallcase
    holdings, on the same closes. Days missing any price are left blank."""
    idx = v1["strategy_index"]
    days = list(idx["date"])
    def walk(trades, date_col, side_col, qty_col):
        t = trades.sort_values(date_col).to_dict("records")
        i, hold, out = 0, {}, []
        for d in days:
            while i < len(t) and pd.Timestamp(t[i][date_col]) <= d:
                x = t[i]
                sgn = 1 if x[side_col] in ("buy",) else -1
                hold[x["symbol"]] = hold.get(x["symbol"], 0.0) + sgn * x[qty_col]
                i += 1
            live = {s: q for s, q in hold.items() if q > 1e-9}
            vals = [q * store.bar(s, d, "close", "daily chart") for s, q in live.items()]
            out.append(np.nan if (not live or any(not np.isfinite(v) for v in vals))
                       else float(sum(vals)))
        return out
    at = v1["smallcase_trades"][["symbol", "trade_date", "trade_type", "quantity"]]
    mt = model["trades"].rename(columns={"model_date": "trade_date",
                                         "side": "trade_type", "qty": "quantity"})
    return pd.DataFrame({"date": days,
                         "model book": walk(mt, "trade_date", "trade_type", "quantity"),
                         "your smallcase": walk(at, "trade_date", "trade_type", "quantity")})


def _actual_pnl(sc_trades, p_term):
    total = 0.0
    for s, g in sc_trades.groupby("symbol"):
        buys = float(g.loc[g["trade_type"] == "buy", "value"].sum())
        sells = float(g.loc[g["trade_type"] == "sell", "value"].sum())
        q = float(g.loc[g["trade_type"] == "buy", "quantity"].sum()
                  - g.loc[g["trade_type"] == "sell", "quantity"].sum())
        pt = p_term.get(s, np.nan)
        total += sells - buys + (q * pt if q > 1e-9 else 0.0)
    return total


def load_store(prices_csv: str, actions_csv: str | None = None):
    if not os.path.exists(prices_csv):
        return None
    return PD.PriceStore.from_csv(
        prices_csv, actions_csv if actions_csv and os.path.exists(actions_csv) else None)
