"""Model vs you: builds the comparison and the reconciliation between them.

Takes the analysis dict produced by pipeline.run() plus a price store, and
returns the model share book, the leg-level implementation attribution, the
waterfall that bridges the model return to the actual return, and the
per-event series behind the chart."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import price_data as PD, model_book as MB, decomposition as DC
from . import strategy_replication as SR


def build(analysis: dict, store: PD.PriceStore, cfg,
                          validate: bool = False) -> dict:
    """validate=True additionally replays the official index and builds the
    daily model-vs-actual series. Both need a full daily price history and are
    methodology checks, not app inputs, so they are off by default."""
    ev = analysis["events"]
    val = analysis["strategy_index"]["date"].iloc[-1]
    start = analysis["strategy_index"]["date"].iloc[0]
    calendar_report = store.restrict_to_calendar(analysis["index_values"]["date"])

    impl = DC.attribute_implementation(
        analysis["recommended_lines"], analysis["smallcase_trades"], ev, store, cfg,
        calendar=list(analysis["index_values"]["date"]))
    timeline = DC.rebalance_timeline(analysis["index_values"], analysis["constituents"],
                                     ev, store, cfg, start)
    if validate:
        ev_reb, ev_inv = DC.quantity_convention_evidence(ev, analysis["mapping"],
                                                         store, cfg)
    else:
        ev_reb, ev_inv = pd.DataFrame(), pd.DataFrame()
    # the reconstructed model share book is needed for the reconciliation, so
    # it is always built; only the index-replication check is optional
    model = MB.simulate(ev, analysis["mapping"], store, cfg, val)
    if validate:
        rep_daily, rep_summary = SR.compare_conventions(
            analysis["index_values"], analysis["constituents"], analysis["mapping"], store,
            analysis["strategy_index"]["date"].iloc[0], val)
    else:
        rep_daily, rep_summary = pd.DataFrame(), pd.DataFrame()
    p_term = {s: store.bar(s, val, "close", "context terminal")
              for s in set(analysis["smallcase_trades"]["symbol"])}
    apnl_by_symbol = _actual_pnl_by_symbol(analysis["smallcase_trades"], p_term)
    apnl = float(apnl_by_symbol.sum())

    imp = analysis["implementation"]
    costs = float(analysis["cost_summary"]["smallcase"].sum())
    taxdf = analysis["tax_estimate"]
    tax = (float(taxdf.loc[taxdf["view"].str.startswith("marginal"),
                           "estimated_tax"].sum()) if len(taxdf) else 0.0)
    div = imp.get("dividends") or 0.0
    s = impl["summary"].set_index("measure")["amount"]
    overall = pd.DataFrame([
        ("Strategy return (official index, rebased to first investment)",
         analysis["strategy_summary"]["absolute_return"], "pct"),
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

    daily = (_daily_series(analysis, model, store) if validate and len(model["trades"])
             else pd.DataFrame())
    return dict(implementation=impl, timeline=timeline, daily_series=daily,
                calendar_report=calendar_report,
                actual_pnl_same_closes=apnl, actual_pnl_by_symbol=apnl_by_symbol,
                dividend_statement=analysis.get("dividend_statement", pd.DataFrame()),
                tax_base=analysis.get("tax_base", pd.DataFrame()),
                evidence_rebalance=ev_reb, evidence_invest=ev_inv,
                replication_daily=rep_daily, replication_summary=rep_summary,
                model=model, overall=overall,
                deferred_repairs=analysis.get("deferred_repairs", pd.DataFrame()),
                missing_prices=store.missing_report(),
                splits_detected=store.splits_in_span())


def _daily_series(analysis, model, store):
    """Day-by-day market value of the model book and the actual smallcase
    holdings, on the same closes. Days missing any price are left blank."""
    idx = analysis["strategy_index"]
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
    at = analysis["smallcase_trades"][["symbol", "trade_date", "trade_type", "quantity"]]
    mt = model["trades"].rename(columns={"model_date": "trade_date",
                                         "side": "trade_type", "qty": "quantity"})
    return pd.DataFrame({"date": days,
                         "model book": walk(mt, "trade_date", "trade_type", "quantity"),
                         "your smallcase": walk(at, "trade_date", "trade_type", "quantity")})


def _actual_pnl_by_symbol(sc_trades, p_term) -> pd.Series:
    """Price-return P&L per symbol on the given closes: sells - buys + the
    terminal value of whatever is still held."""
    out = {}
    for s, g in sc_trades.groupby("symbol"):
        buys = float(g.loc[g["trade_type"] == "buy", "value"].sum())
        sells = float(g.loc[g["trade_type"] == "sell", "value"].sum())
        q = float(g.loc[g["trade_type"] == "buy", "quantity"].sum()
                  - g.loc[g["trade_type"] == "sell", "quantity"].sum())
        pt = p_term.get(s, np.nan)
        out[s] = sells - buys + (q * pt if q > 1e-9 else 0.0)
    return pd.Series(out, name="pnl").rename_axis("symbol")


def reconciliation(analysis: dict, out: dict | None) -> dict:
    """
    Bridge the model-on-your-cash-flows return (Layer A') to your actual
    return (Layer B), in rupees and percentage points.

    The first three lines are the leg-level attribution, measured at the prices
    you traded. The remaining lines are the three reasons that measurement does
    not equal the gap on its own, each named and quantified, so the rows sum to
    the gap exactly instead of leaving an unexplained remainder.
    """
    layers = analysis["layers"].set_index("layer")
    a_dash = layers.loc["A'. Strategy, cash-flow matched"]
    b = layers.loc["B. Actual implementation"]
    c = layers.loc["C. Net outcome"]
    denom = analysis["implementation"]["money_put_in"]
    pp = lambda x: 100.0 * x / denom if denom else np.nan  # noqa: E731

    a2 = float(a_dash["value_rs"])
    b_rs = float(b["value_rs"])
    gap_rs = b_rs - a2
    div = float(analysis["implementation"].get("dividends") or 0.0)

    rows = [dict(item="Dividends you received", rupees=div,
                 note="the strategy index is price-return, so it excludes them")]
    if out is not None and np.isfinite(out["model"]["pnl_total"]):
        s_ = out["implementation"]["summary"].set_index("measure")["amount"]
        price = float(s_["Implementation price effect - TOTAL"])
        qty = float(s_["Quantity cash component - TOTAL"])
        mb = float(out["model"]["pnl_total"])
        apnl = float(out["actual_pnl_same_closes"])
        rows += [
            dict(item="Price differences", rupees=price,
                 note="your fills against the price the model transacted at"),
            dict(item="Quantity differences", rupees=qty,
                 note="shares you held against the model's quantities"),
            dict(item="Market moves since you traded",
                 rupees=(apnl - mb) - price - qty,
                 note="the two lines above are measured at transaction prices; "
                      "this is what the market did to them afterwards"),
            dict(item="Broker vs exchange closing price",
                 rupees=b_rs - div - apnl,
                 note="your statement and the exchange archive disagree on the "
                      "valuation-day close"),
            dict(item="Whole shares instead of index units", rupees=mb - a2,
                 note="the index holds fractional units; a real portfolio buys "
                      "whole shares and leaves small cash residuals"),
        ]
    known = sum(r["rupees"] for r in rows if np.isfinite(r["rupees"]))
    left = gap_rs - known
    if abs(left) > max(1.0, 0.0005 * abs(gap_rs)):
        rows.append(dict(item="Not reconciled", rupees=left,
                         note="legs the price data could not cover"))
    contrib = pd.DataFrame(rows)
    contrib["pp"] = contrib["rupees"].map(pp)

    return dict(
        model_return_pct=float(a_dash["return_pct"]) * 100,
        your_return_pct=float(b["return_pct"]) * 100,
        net_return_pct=float(c["return_pct"]) * 100,
        net_value_rs=float(c["value_rs"]),
        model_value_rs=a2, your_value_rs=b_rs,
        gap_rs=gap_rs, gap_pp=pp(gap_rs),
        strategy_index_pct=float(layers.loc["A. Strategy", "return_pct"]) * 100,
        contributors=contrib, denominator=denom)


def by_event(analysis: dict, out: dict) -> pd.DataFrame:
    """Per-event and cumulative price-difference contribution, in pp.

    Event-level only: no daily price history, no implied daily portfolio value.
    """
    ev = analysis["event_summary"][["event_id", "date", "kind", "lag_days"]]
    by = out["implementation"]["by_event"][
        ["event_id", "implementation_price_effect"]]
    d = ev.merge(by, on="event_id", how="left").fillna(
        {"implementation_price_effect": 0.0})
    denom = analysis["implementation"]["money_put_in"]
    d["pp"] = 100.0 * d["implementation_price_effect"] / denom
    d["cumulative_pp"] = d["pp"].cumsum()
    d["label"] = d["kind"].map({"rebalance": "Rebalance", "invest": "Investment",
                                "exit": "Exit"}).fillna(d["kind"])
    d["applied"] = [
        ("same day as the model" if (k == "rebalance" and (l == 0))
         else f"{int(l)} days after the model date"
         if (k == "rebalance" and pd.notna(l)) else "your own investment")
        for k, l in zip(d["kind"], d["lag_days"])]
    return d


def load_store(prices_csv: str, actions_csv: str | None = None):
    if not os.path.exists(prices_csv):
        return None
    return PD.PriceStore.from_csv(
        prices_csv, actions_csv if actions_csv and os.path.exists(actions_csv) else None)
