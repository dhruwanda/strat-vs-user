"""Orchestration for the implementation attribution and its validation."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import price_data as PD, model_book as MB, decomposition as DC
from . import strategy_replication as SR


def run_gap_decomposition(v1: dict, store: PD.PriceStore, cfg,
                          validate: bool = False) -> dict:
    """validate=True additionally replays the official index and builds the
    daily model-vs-actual series. Both need a full daily price history and are
    methodology checks, not app inputs, so they are off by default."""
    ev = v1["events"]
    val = v1["strategy_index"]["date"].iloc[-1]
    start = v1["strategy_index"]["date"].iloc[0]
    calendar_report = store.restrict_to_calendar(v1["index_values"]["date"])

    impl = DC.attribute_implementation(
        v1["recommended_lines"], v1["smallcase_trades"], ev, store, cfg,
        calendar=list(v1["index_values"]["date"]))
    timeline = DC.rebalance_timeline(v1["index_values"], v1["constituents"],
                                     ev, store, cfg, start)
    if validate:
        ev_reb, ev_inv = DC.quantity_convention_evidence(ev, v1["mapping"],
                                                         store, cfg)
    else:
        ev_reb, ev_inv = pd.DataFrame(), pd.DataFrame()
    if validate:
        rep_daily, rep_summary = SR.compare_conventions(
            v1["index_values"], v1["constituents"], v1["mapping"], store,
            v1["strategy_index"]["date"].iloc[0], val)
        model = MB.simulate(ev, v1["mapping"], store, cfg, val)
    else:
        rep_daily, rep_summary = pd.DataFrame(), pd.DataFrame()
        model = dict(trades=pd.DataFrame(), positions=pd.DataFrame(),
                     pnl_by_symbol=pd.DataFrame(), pnl_total=float("nan"),
                     notes=pd.DataFrame(), terminal_prices={})
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

    daily = (_daily_series(v1, model, store) if validate and len(model["trades"])
             else pd.DataFrame())
    return dict(implementation=impl, timeline=timeline, daily_series=daily,
                calendar_report=calendar_report,
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


def gap_reconciliation(v1: dict, out: dict | None) -> dict:
    """
    Reconcile the model-on-your-cash-flows return (Layer A') with your actual
    return (Layer B), in rupees and percentage points.

    Nothing new is computed here: Layer A' is the engine's cashflow_model
    (each event's net cash grown by the official index to the valuation date),
    Layer B is the engine's implementation summary, and the contributors are
    the engine's dividend attribution and implementation attribution. The only
    arithmetic is differencing them and dividing by money put in.

    The leg-level price effect is measured at TRANSACTION prices; what that
    advantage is worth today has since been moved by the market. That is why a
    small unreconciled remainder is reported rather than hidden.
    """
    layers = v1["layers"].set_index("layer")
    a_dash = layers.loc["A'. Strategy, cash-flow matched"]
    b = layers.loc["B. Actual implementation"]
    denom = v1["implementation"]["money_put_in"]
    pp = lambda x: 100.0 * x / denom if denom else np.nan  # noqa: E731

    gap_rs = float(b["value_rs"]) - float(a_dash["value_rs"])
    div = float(v1["implementation"].get("dividends") or 0.0)
    if out is not None:
        s = out["implementation"]["summary"].set_index("measure")["amount"]
        price = float(s["Implementation price effect - TOTAL"])
        qty = float(s["Quantity cash component - TOTAL"])
    else:
        price = qty = np.nan

    rows = [dict(item="Dividends you received", rupees=div,
                 note="the strategy index is price-return, so it excludes them"),
            dict(item="Price differences", rupees=price,
                 note="your fills against the price the model transacted at"),
            dict(item="Quantity differences", rupees=qty,
                 note="shares you held against the model's quantities")]
    known = sum(r["rupees"] for r in rows if np.isfinite(r["rupees"]))
    rows.append(dict(item="Not reconciled", rupees=gap_rs - known,
                     note="leg-level effects are measured at transaction "
                          "prices; the market has moved since"))
    contrib = pd.DataFrame(rows)
    contrib["pp"] = contrib["rupees"].map(pp)

    return dict(
        model_return_pct=float(a_dash["return_pct"]) * 100,
        your_return_pct=float(b["return_pct"]) * 100,
        model_value_rs=float(a_dash["value_rs"]),
        your_value_rs=float(b["value_rs"]),
        gap_rs=gap_rs, gap_pp=pp(gap_rs),
        strategy_index_pct=float(layers.loc["A. Strategy", "return_pct"]) * 100,
        contributors=contrib, denominator=denom)


def event_gap_series(v1: dict, out: dict) -> pd.DataFrame:
    """Per-event and cumulative price-difference contribution, in pp.

    Event-level only: no daily price history, no implied daily portfolio value.
    """
    ev = v1["event_summary"][["event_id", "date", "kind"]]
    by = out["implementation"]["by_event"][
        ["event_id", "implementation_price_effect"]]
    d = ev.merge(by, on="event_id", how="left").fillna({"implementation_price_effect": 0.0})
    denom = v1["implementation"]["money_put_in"]
    d["pp"] = 100.0 * d["implementation_price_effect"] / denom
    d["cumulative_pp"] = d["pp"].cumsum()
    d["label"] = d["kind"].str.replace("rebalance", "Rebalance").str.replace(
        "invest", "Investment")
    return d


def load_store(prices_csv: str, actions_csv: str | None = None):
    if not os.path.exists(prices_csv):
        return None
    return PD.PriceStore.from_csv(
        prices_csv, actions_csv if actions_csv and os.path.exists(actions_csv) else None)
