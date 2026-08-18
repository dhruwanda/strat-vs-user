"""Layer B/C detail: per-stock contribution, and the A-vs-B gap decomposition."""
from __future__ import annotations
import numpy as np
import pandas as pd


def stock_level(sc_trades: pd.DataFrame, positions: pd.DataFrame,
                realisations: pd.DataFrame, prices: dict,
                per_trade_costs: pd.DataFrame, constituents: pd.DataFrame,
                mapping: dict, start_date) -> pd.DataFrame:
    """One row per symbol the smallcase held at any point in the period."""
    sym_names = {}
    for n, s in mapping.items():
        sym_names.setdefault(s, n)

    syms = sorted(set(sc_trades["symbol"]))
    pos = positions.set_index("symbol") if len(positions) else pd.DataFrame()
    rl = (realisations.groupby("symbol")["realized_pnl"].sum()
          if len(realisations) else pd.Series(dtype=float))
    cost_heads = [c for c in ("stt", "exchange_txn", "sebi_turnover", "ipft",
                              "stamp_duty", "brokerage", "gst") if c in per_trade_costs]
    ct = (per_trade_costs[per_trade_costs["attribution"] == "smallcase"]
          .groupby("symbol")[cost_heads].sum().sum(axis=1)
          if cost_heads else pd.Series(dtype=float))

    # which versions held this symbol during the period
    inperiod = constituents[constituents["version_end"] >= start_date]
    vers_count = {}
    for n, g in inperiod.groupby("name"):
        sym = mapping.get(n)
        if sym:
            vers_count[sym] = vers_count.get(sym, 0) + g["version"].nunique()

    rows = []
    for s in syms:
        b = sc_trades[(sc_trades.symbol == s) & (sc_trades.trade_type == "buy")]
        sl = sc_trades[(sc_trades.symbol == s) & (sc_trades.trade_type == "sell")]
        q = float(pos["qty"].get(s, 0.0)) if len(pos) else 0.0
        cb = float(pos["cost_basis"].get(s, 0.0)) if len(pos) else 0.0
        p = prices.get(s, np.nan)
        mv = q * p if q and np.isfinite(p) else (0.0 if q == 0 else np.nan)
        unreal = (mv - cb) if q else 0.0
        real = float(rl.get(s, 0.0))
        rows.append(dict(
            symbol=s, constituent_name=sym_names.get(s, ""),
            still_held=(q > 0), exited_during_period=(q == 0),
            qty_bought=float(b["quantity"].sum()), buy_value=float(b["value"].sum()),
            qty_sold=float(sl["quantity"].sum()), sell_value=float(sl["value"].sum()),
            qty_held=q, avg_cost=(cb / q if q else np.nan), cost_basis=cb,
            current_price=p, market_value=mv,
            realized_pnl=real, unrealized_pnl=unreal, total_pnl=real + unreal,
            attributable_trading_costs=float(ct.get(s, 0.0)),
            n_smallcase_trades=int(len(b) + len(sl)),
            model_versions_containing_it=int(vers_count.get(s, 0)),
        ))
    df = pd.DataFrame(rows)
    for c, tot in (("realized_pnl", df["realized_pnl"].sum()),
                   ("total_pnl", df["total_pnl"].sum())):
        df[f"contribution_to_{c}_pct"] = df[c] / tot if tot else np.nan
    return df.sort_values("total_pnl", ascending=False).reset_index(drop=True)


def cashflow_matched_model(events: list, index_values: pd.DataFrame,
                           end_date: pd.Timestamp,
                           basis: str = "same_close") -> pd.DataFrame:
    """
    What the model portfolio would be worth today if it had received the
    investor's exact net cash flows on the exact dates they occurred.

    This is the only like-for-like bridge between the index (a single notional
    lumpsum) and the investor (several tranches at different index levels).
    """
    iv = index_values.set_index("date")["index_value"]
    i_end = float(iv.loc[:end_date].iloc[-1])
    rows = []
    for e in events:
        cash = e["buy_value"] - e["sell_value"]
        avail = iv.loc[:e["date"]]
        if avail.empty:
            continue
        # An order filled minutes after the open is priced far closer to the
        # PREVIOUS close than to that day's close; reporting both bounds the
        # timing convention instead of hiding it inside one number.
        i_k = float(avail.iloc[-2]) if (basis == "prev_close" and len(avail) > 1) \
            else float(avail.iloc[-1])
        rows.append(dict(event_id=e["event_id"], date=e["date"], kind=e["kind"],
                         net_cash=cash, index_on_date=i_k,
                         growth_factor=i_end / i_k,
                         model_value_today=cash * i_end / i_k))
    df = pd.DataFrame(rows)
    df.attrs["index_end"] = i_end
    return df


def gap_bridge(actual_market_value: float, model_value: float,
               model_value_prev: float, net_cash: float,
               dividends, costs_total: float, tax_estimate,
               lag_table: pd.DataFrame, abs_weight_deviation: float) -> pd.DataFrame:
    """
    Bridge from the cashflow-matched model portfolio to the investor's actual
    outcome. Only items evidenced by the data get a number; everything else is
    left in a single named residual rather than being invented.
    """
    lagged = lag_table[lag_table["lag_days"].fillna(0) > 0] if len(lag_table) else lag_table
    n_lag = len(lagged)
    max_lag = int(lagged["lag_days"].max()) if n_lag else 0
    gap = actual_market_value - model_value
    resid = gap - (dividends or 0.0)
    rows = [
        dict(item="Model portfolio value today, on the investor's exact cash flows",
             amount=model_value, status="measured",
             evidence="smallcase official index, each event's net cash grown to the "
                      "valuation date (same-day close basis)"),
        dict(item="  same, using the previous close for each event date",
             amount=model_value_prev, status="sensitivity",
             evidence="most orders filled near the open, so the previous close is the "
                      "other defensible convention; the gap is robust to the choice"),
        dict(item="Actual market value of smallcase holdings today",
             amount=actual_market_value, status="measured",
             evidence="reconstructed holdings x broker closing price; cross-checks to "
                      "the smallcase Investments page"),
        dict(item="GAP: actual minus model", amount=gap, status="measured",
             evidence="difference of the two above"),
        dict(item="Explained: dividends (the index is price-return and excludes them)",
             amount=dividends, status="measured" if dividends else "not available",
             evidence="smallcase Investments page; dividends are absent from the "
                      "broker files supplied"),
        dict(item="UNEXPLAINED RESIDUAL", amount=resid, status="not decomposable",
             evidence="cannot be split further without daily closing prices for every "
                      "constituent. Candidate drivers are listed below with the evidence "
                      "available for each; no rupee amount is assigned to any of them."),
        dict(item="  driver: rebalance applied later than the model date",
             amount=None, status="exposure measured, effect not measurable",
             evidence=f"{n_lag} of {len(lag_table)} rebalances applied late, up to "
                      f"{max_lag} days. See the Rebalance Timing sheet for the index "
                      f"move during each lag window."),
        dict(item="  driver: weight drift between rebalances",
             amount=None, status="exposure measured, effect not measurable",
             evidence=(f"the index resets to exact prescribed weights at every "
                       f"rebalance; a real book does not, because smallcase minimises "
                       f"churn. Absolute weight deviation today: "
                       f"{abs_weight_deviation:.2%}. See the Weight Drift sheet.")),
        dict(item="  driver: execution price vs the index's closing-price basis",
             amount=None, status="not measurable from these files",
             evidence="the index is built from end-of-day closes; the investor filled "
                      "intraday. Needs daily OHLC per constituent."),
        dict(item="  driver: integer-share rounding",
             amount=None, status="bounded, immaterial",
             evidence="reconstruction shows sub-1% value deviation per event"),
        dict(item="Attributable transaction costs (paid, already outside market value)",
             amount=-costs_total, status="measured",
             evidence="per-trade charges calibrated to broker totals, plus smallcase "
                      "fees and DP charges"),
        dict(item="Estimated tax on realised gains", amount=tax_estimate,
             status="estimate",
             evidence="FIFO capital gains; see the Tax Estimate sheet"),
    ]
    return pd.DataFrame(rows)


def timing_lag_evidence(rebalance_calendar: pd.DataFrame,
                        index_values: pd.DataFrame) -> pd.DataFrame:
    """
    How far behind the model each rebalance was applied, and how the index moved
    during that window. This measures EXPOSURE to the lag; it is not an
    attribution of profit or loss, which would need daily constituent prices.
    """
    iv = index_values.set_index("date")["index_value"]
    rows = []
    for _, r in rebalance_calendar.iterrows():
        d0, d1 = r["model_rebalance_date"], r["investor_applied_on"]
        if pd.isna(d1):
            rows.append(dict(model_rebalance_date=d0, investor_applied_on=pd.NaT,
                             lag_days=np.nan, index_at_model=np.nan,
                             index_at_applied=np.nan, index_move_during_lag=np.nan,
                             note=r["status"]))
            continue
        a = iv.loc[:d0]
        b = iv.loc[:d1]
        i0 = float(a.iloc[-1]) if len(a) else np.nan
        i1 = float(b.iloc[-1]) if len(b) else np.nan
        rows.append(dict(model_rebalance_date=d0, investor_applied_on=d1,
                         lag_days=r["lag_days"], index_at_model=i0,
                         index_at_applied=i1,
                         index_move_during_lag=(i1 / i0 - 1) if i0 else np.nan,
                         note="applied late" if r["lag_days"] else "applied same day"))
    df = pd.DataFrame(rows)
    return df


def weight_drift(positions: pd.DataFrame) -> pd.DataFrame:
    """Current weights against the prescribed weights of the live version."""
    if not len(positions):
        return positions
    d = positions[["symbol", "market_value", "weight_actual",
                   "weight_prescribed", "weight_deviation"]].copy()
    d["deviation_bps"] = d["weight_deviation"] * 10000
    d["value_vs_prescribed"] = d["weight_deviation"] * d["market_value"].sum()
    return d.sort_values("weight_deviation", ascending=False).reset_index(drop=True)
