"""Orchestration. Produces a dict of frames; excel_report renders it."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import loaders, mapping as M, events as EV, reconstruct as RC
from . import ledger, costs as CO, tax as TX, strategy as ST, attribution as AT
from . import dividends as DV
from .config import Config


def run(timeline_path: str, tradebook_path: str, pnl_path: str | None = None,
        cfg: Config | None = None, reported: dict | None = None,
        dividends_path: str | None = None) -> dict:
    """
    reported: optional dict of the figures shown on the smallcase Investments
              page, used only for cross-checking. Never used in a calculation.
              Keys: money_put_in, current_value, current_investment,
                    current_returns, realized_returns, dividends, total_returns
    """
    cfg = cfg or Config()
    reported = reported or {}
    out: dict = {"assumptions": [], "limitations": []}

    # ---------- inputs -------------------------------------------------
    index_values = loaders.load_index_values(timeline_path)
    constituents = loaders.load_constituents(timeline_path)
    trades = loaders.load_tradebook(tradebook_path)
    pnl = loaders.load_pnl(pnl_path) if pnl_path else {
        "holdings": pd.DataFrame(), "charges": {}, "summary": {},
        "other": pd.DataFrame(columns=["particulars", "posting_date", "debit", "credit"]),
        "period": (None, None)}

    other = CO.parse_other_debits(pnl["other"], cfg)
    fees = other[other["category"] == "smallcase_fee"] if len(other) else other

    # ---------- events, mapping, attribution ---------------------------
    ev = EV.detect_events(trades, constituents, fees, index_values, cfg)
    if not ev:
        raise ValueError("no basket orders found; cannot identify smallcase trades")
    mp, map_diag = M.resolve(ev, constituents, cfg.mapping_overrides,
                             cfg.mapping_margin_warn)
    first_date = ev[0]["date"]

    # ---------- A: strategy --------------------------------------------
    norm = ST.normalise(index_values, first_date)
    strat = ST.summarise(norm)
    cal = ST.rebalance_calendar(index_values, constituents, first_date, ev)

    # ---------- recommended quantities ---------------------------------
    rec_lines, rec_summary = RC.reconstruct_all(ev, mp, cfg)

    # ---------- attribution, incl. deferred (circuit-limited) legs -----
    attributed = EV.attribute_trades(trades, ev, mp, constituents, cfg)
    attributed, deferred_repairs = EV.detect_deferred_legs(
        attributed, rec_lines, ev, cfg)
    sc_trades = attributed[attributed["attribution"] == "smallcase"].copy()

    # ---------- B: implementation --------------------------------------
    positions, realisations = ledger.average_cost_book(sc_trades)
    hold = pnl["holdings"]
    prices = (dict(zip(hold["symbol"], hold["close_price"]))
              if len(hold) and "close_price" in hold else {})
    positions["current_price"] = positions["symbol"].map(prices)
    positions["market_value"] = positions["qty"] * positions["current_price"]
    positions["unrealized_pnl"] = positions["market_value"] - positions["cost_basis"]
    positions["weight_actual"] = (positions["market_value"] /
                                  positions["market_value"].sum())
    last_ver = constituents[constituents["version_start"] ==
                            constituents["version_start"].max()]
    tgt = {mp[n]: w for n, w in zip(last_ver["name"], last_ver["weight"]) if n in mp}
    positions["weight_prescribed"] = positions["symbol"].map(tgt)
    positions["weight_deviation"] = (positions["weight_actual"]
                                     - positions["weight_prescribed"])
    missing_px = positions.loc[positions["current_price"].isna()
                               | (positions["current_price"] == 0), "symbol"].tolist()

    net_by_event = [e["buy_value"] - e["sell_value"] for e in ev]
    money_put_in = float(sum(c for c in net_by_event if c > 0))
    cur_inv = float(positions["cost_basis"].sum())
    cur_val = float(positions["market_value"].sum(skipna=True))
    realized = float(realisations["realized_pnl"].sum()) if len(realisations) else 0.0
    div_stmt = pd.DataFrame()
    if dividends_path:
        div_stmt = DV.attribute_dividends(DV.load_dividends(dividends_path),
                                          sc_trades)
        dividends = float(div_stmt["attributed_amount"].sum())
    else:
        dividends = reported.get("dividends")
        if dividends is None:
            dividends = cfg.dividends_override
    impl = {
        "money_put_in": money_put_in,
        "current_investment": cur_inv,
        "current_value": cur_val,
        "current_returns": cur_val - cur_inv,
        "realized_returns": realized,
        "dividends": dividends,
        "total_returns": (cur_val - cur_inv) + realized + (dividends or 0.0),
        "total_return_pct": (((cur_val - cur_inv) + realized + (dividends or 0.0))
                             / money_put_in if money_put_in else np.nan),
        "net_cash_deployed": float(sum(net_by_event)),
    }

    recon = pd.DataFrame([
        dict(metric=k.replace("_", " ").title(),
             smallcase_reported=reported.get(k),
             reconstructed=impl.get(k),
             difference=((impl.get(k) - reported[k])
                         if reported.get(k) is not None and impl.get(k) is not None
                         else np.nan),
             difference_pct=((impl.get(k) - reported[k]) / abs(reported[k])
                             if reported.get(k) else np.nan))
        for k in ("money_put_in", "current_investment", "current_value",
                  "current_returns", "realized_returns", "dividends", "total_returns")])
    recon["definition_used"] = [
        "sum of POSITIVE net cash across events; a partial exit does not reduce it",
        "sum of running-average cost x quantity still held",
        "sum of quantity held x broker closing price",
        "current value minus current investment",
        "average-cost basis: (sell price - running average cost) x quantity",
        "not present in any supplied file",
        "current returns + realized returns + dividends",
    ]
    recon["likely_explanation"] = ""
    recon["data_required"] = ""

    # ---------- costs ---------------------------------------------------
    with_ch = CO.per_trade_charges(attributed, cfg)
    with_ch, calib = CO.calibrate(with_ch, pnl["charges"], cfg)
    per_trade, cost_summary, unattributed = CO.attribute_costs(with_ch, other, cfg)
    sc_costs = float(cost_summary["smallcase"].sum())

    # ---------- tax -----------------------------------------------------
    sc_ts = set(sc_trades["exec_ts"])
    relevant = trades[trades["symbol"].isin(set(sc_trades["symbol"]))]
    open_lots, fifo_real, intraday = ledger.fifo_book(relevant, sc_ts)
    fifo_sc = (fifo_real[fifo_real["sale_from_smallcase"]]
               if len(fifo_real) else fifo_real)
    tax_base = TX.base_summary(TX.classify(fifo_sc, cfg))
    if getattr(cfg, "apply_tax_rates", False):
        tax_tbl, tax_notes = TX.estimate(fifo_sc, cfg)
        tax_marginal = float(tax_tbl.loc[tax_tbl["view"].str.startswith("marginal"),
                                         "estimated_tax"].sum()) if len(tax_tbl) else 0.0
    else:
        tax_tbl, tax_marginal = pd.DataFrame(), 0.0
        tax_notes = ["Rates deliberately NOT applied by the engine. It "
                     "establishes realised gains/losses, holding-period term and "
                     "asset class (the Tax Base table); the applicable treatment "
                     "and rates are for the interpretation layer or the user's "
                     "tax adviser. Set Config.apply_tax_rates=True with your own "
                     "TaxRules to produce an illustrative estimate."]

    # ---------- C: net outcome + gap ------------------------------------
    cfm = AT.cashflow_matched_model(ev, index_values, norm["date"].iloc[-1])
    model_value = float(cfm["model_value_today"].sum())
    cfm_prev = AT.cashflow_matched_model(ev, index_values, norm["date"].iloc[-1],
                                         basis="prev_close")
    model_value_prev = float(cfm_prev["model_value_today"].sum())
    lag_ev = AT.timing_lag_evidence(cal, index_values)
    drift = AT.weight_drift(positions)
    wdev = positions["weight_deviation"].abs().sum() if len(positions) else np.nan
    bridge = AT.gap_bridge(cur_val, model_value, model_value_prev,
                           impl["net_cash_deployed"], dividends, sc_costs,
                           -tax_marginal, lag_ev, wdev)

    stock = AT.stock_level(sc_trades, positions, realisations, prices,
                           per_trade, constituents, mp, first_date)

    layers = pd.DataFrame([
        dict(layer="A. Strategy", question="How did the smallcase strategy perform?",
             measure="Official index, rebased to 100 on the first investment date",
             value_rs=np.nan, return_pct=strat["absolute_return"],
             basis="No costs, no taxes, no cash-flow timing. Price return only."),
        dict(layer="A'. Strategy, cash-flow matched",
             question="What would the strategy have returned on MY cash flows?",
             measure="Each event's net cash grown by the index to the valuation date",
             value_rs=model_value - impl["net_cash_deployed"],
             return_pct=((model_value - impl["net_cash_deployed"]) / money_put_in
                         if money_put_in else np.nan),
             basis="Like-for-like with the investor's tranches. Still no costs or taxes."),
        dict(layer="B. Actual implementation",
             question="How did my actual smallcase investment perform?",
             measure="Realised + unrealised + dividends on reconstructed holdings",
             value_rs=impl["total_returns"], return_pct=impl["total_return_pct"],
             basis="Gross of transaction costs and taxes."),
        dict(layer="C. Net outcome",
             question="What did I keep after costs and estimated tax?",
             measure="Layer B minus attributable costs minus estimated tax",
             value_rs=impl["total_returns"] - sc_costs - tax_marginal,
             return_pct=((impl["total_returns"] - sc_costs - tax_marginal) / money_put_in
                         if money_put_in else np.nan),
             basis="Labelled Net Return After Costs & Taxes. Tax is an estimate."),
    ])

    # ---------- assumptions and limitations -----------------------------
    out["assumptions"] = _assumptions(cfg, ev, rec_summary, calib, mp, map_diag)
    out["limitations"] = _limitations(missing_px, pnl, other, reported, cal, rec_summary)

    out.update(dict(
        index_values=index_values, constituents=constituents, trades=trades,
        events=ev, mapping=mp, mapping_diagnostics=map_diag,
        attributed_trades=attributed, smallcase_trades=sc_trades,
        strategy_index=norm, strategy_summary=strat, rebalance_calendar=cal,
        recommended_lines=rec_lines, event_summary=rec_summary,
        positions=positions, realisations=realisations, implementation=impl,
        reconciliation=recon, per_trade_costs=per_trade, cost_summary=cost_summary,
        cost_calibration=calib, unattributed_costs=unattributed,
        fifo_realisations=TX.classify(fifo_sc, cfg), fifo_open_lots=open_lots,
        tax_estimate=tax_tbl, tax_notes=tax_notes, tax_base=tax_base,
        dividend_statement=div_stmt,
        deferred_repairs=deferred_repairs,
        cashflow_model=cfm, cashflow_model_prev_close=cfm_prev,
        model_value_same_close=model_value, model_value_prev_close=model_value_prev,
        gap_bridge=bridge, stock_attribution=stock, timing_lag=lag_ev,
        weight_drift=drift, intraday_legs=intraday,
        layers=layers, other_debits=other, broker_pnl=pnl,
    ))
    return out


def _assumptions(cfg, ev, rec_summary, calib, mp, map_diag):
    inv = [e for e in ev if e["kind"] == "invest"]
    fee_ok = sum(1 for e in inv if e["fee_on_date"])
    reb = [e for e in ev if e["kind"] == "rebalance"]
    reb_fee = sum(1 for e in reb if e["fee_on_date"])
    good = rec_summary[rec_summary["lines_reconstructible"] > 0]
    return pd.DataFrame([
        dict(area="Trade attribution",
             assumption=f"A trade belongs to the smallcase only if it executed inside a "
                        f"basket of at least {cfg.basket_min_symbols} distinct symbols "
                        f"sharing one execution timestamp.",
             evidence=f"{len(ev)} basket events found. Every leg of each carries an "
                      f"identical execution second, which a standalone order cannot."),
        dict(area="Event merging",
             assumption=f"Legs within {cfg.event_merge_seconds}s form one event.",
             evidence="Rebalance sell and buy legs land ~2s apart; separate orders on the "
                      "same day are minutes apart."),
        dict(area="Fresh money vs rebalance",
             assumption="A smallcase transaction fee marks a first buy, invest-more or SIP; "
                        "rebalances and exits are free.",
             evidence=f"{fee_ok} of {len(inv)} buy-only events carry a fee; "
                      f"{reb_fee} of {len(reb)} rebalances do. Matches smallcase's "
                      f"published fee policy."),
        dict(area="Recommended quantity, investments",
             assumption="qty = round((V_before + amount) x weight - held x price) / price",
             evidence="Top-up to prescribed weights. Median value deviation across "
                      f"investment events: "
                      f"{good.loc[good.kind=='invest','value_deviation_pct_of_event'].median():.3%}"
                      if len(good) else "n/a"),
        dict(area="Recommended quantity, rebalances",
             assumption="Each touched constituent is driven to weight x portfolio value; "
                        "dropped names go to zero.",
             evidence="Implied portfolio value from each traded leg agrees to a "
                      f"median CV of "
                      f"{good.loc[good.kind=='rebalance','probe_dispersion_cv'].median():.2%}"
                      if "probe_dispersion_cv" in good else "n/a"),
        dict(area="Reference price",
             assumption="Execution VWAP stands in for the live quote smallcase used to "
                        "size the order.",
             evidence="Unobservable in the tradebook. Causes sub-share quantity "
                      "differences, largest in low-priced scrips."),
        dict(area="Charge derivation",
             assumption="Per-trade charges from a rate card, then scaled so each head "
                        "sums to the broker's reported total.",
             evidence="Calibration factors: " + ", ".join(
                 f"{r.charge_head} {r.calibration_factor:.3f}"
                 for r in calib.itertuples() if np.isfinite(r.calibration_factor))),
        dict(area="Cost basis conventions",
             assumption="Average cost for the smallcase view; FIFO across all units for tax.",
             evidence="smallcase reports realised returns on average cost; Indian tax law "
                      "uses FIFO on a fungible demat holding."),
        dict(area="Name to symbol mapping",
             assumption="Resolved from basket membership and rebalance add/remove events, "
                        "with name similarity as a tiebreak.",
             evidence=f"{len(mp)} constituents mapped; "
                      f"{int((map_diag['confidence']=='review').sum())} flagged for review."),
    ])


def _limitations(missing_px, pnl, other, reported, cal, rec_summary):
    rows = [
        dict(limitation="Dividends are not in any supplied file",
             impact="Total Returns cannot be fully rebuilt from broker data alone.",
             data_required="Zerodha Console > Reports > Dividends, filtered to the "
                           "constituent symbols and the investment period."),
        dict(limitation="smallcase subscription fee is invisible here",
             impact="Layer C understates the true cost of running the strategy.",
             data_required="Manager subscription invoices, or the smallcase billing history."),
        dict(limitation="No end-of-day closing prices for constituents on event dates",
             impact="Execution slippage against the index's closing-price basis cannot be "
                    "separated from weight drift; both sit in one residual.",
             data_required="Daily OHLC for every constituent over the investment period."),
        dict(limitation="smallcase's order-time reference price is unobservable",
             impact="Reconstructed quantities differ by a share or so; value deviation is "
                    "reported per event instead of claiming exactness.",
             data_required="smallcase order history export (not downloadable for past orders)."),
        dict(limitation="Portfolio value on rebalance dates is inferred from the traded legs",
             impact="Rebalance reconstruction validates weight fidelity but does not "
                    "independently prove quantities for untouched holdings.",
             data_required="Daily closing prices, as above."),
        dict(limitation="Tax is an estimate, not a liability",
             impact="The Rs 1.25 lakh exemption and loss set-off are taxpayer-level and "
                    "annual; the true figure depends on the whole portfolio.",
             data_required="Complete capital-gains position for the financial year."),
    ]
    if missing_px:
        rows.append(dict(
            limitation=f"No closing price for: {', '.join(missing_px)}",
            impact="Current value excludes these holdings.",
            data_required="Closing price for those symbols on the valuation date."))
    if reported and "dividends" not in reported:
        rows.append(dict(limitation="No smallcase-reported figures supplied",
                         impact="Reconciliation column is blank.",
                         data_required="Screenshot or export of the Investments page."))
    unapplied = cal[cal["status"] != "applied"]
    if len(unapplied):
        rows.append(dict(
            limitation=f"{len(unapplied)} model rebalance(s) have no matching investor trade",
            impact="Those model changes were not implemented, or were implemented outside "
                   "the tradebook period.",
            data_required="Tradebook covering the full holding period."))
    return pd.DataFrame(rows)
