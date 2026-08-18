"""
Share-level simulation of the smallcase MODEL portfolio on the investor's own
cash-flow dates.

Price conventions (configurable, defaults follow smallcase's documented index
methodology and are tested against the observed first-buy basket):

  * INVEST events: the model deploys the same net cash on the same date the
    investor did (the strategy has no opinion about when fresh money arrives),
    at that day's CLOSING price - the strategy's own EOD basis. The
    recommendation engine itself used a live quote, but that is the investor's
    execution reality, not the model's; the difference lands in the execution
    effect, which is the point.
  * MODEL REBALANCES: applied on T1 = the first trading day AFTER the flagged
    model rebalance date, at T1's OHLC average - this is how smallcase's index
    transitions quantities per its published return-calculation methodology.

The model book is self-financing at rebalances, drives every touched
constituent to weight x portfolio value, sends dropped names to zero, and holds
integer shares (same rounding the recommendation engine applies).

Validation: while prices are available, sum(qty x close) must track the supplied
official index up to a scale factor. The tracking error is reported; a drifting
model book means a convention is wrong and the decomposition should not be
trusted until it is fixed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def simulate(events: list, mapping: dict, store, cfg,
             valuation_date: pd.Timestamp) -> dict:
    invest_field = getattr(cfg, "model_invest_price_field", "close")
    reb_field = getattr(cfg, "model_rebalance_price_field", "ohlc_avg")

    hold: dict[str, float] = {}
    trades, notes = [], []
    done_model_dates = set()

    for e in events:
        sym_w = {mapping[n]: w for n, w in e["active_weights"].items() if n in mapping}

        if e["kind"] == "invest":
            date = e["date"]
            field = invest_field
            cash = e["buy_value"] - e["sell_value"]
            px = {s: store.bar(s, date, field, f"model invest event {e['event_id']}")
                  for s in sym_w}
            v_before = sum(q * px.get(s, np.nan) for s, q in hold.items())
            target_total = (0.0 if np.isnan(v_before) else v_before) + cash
            for s, w in sorted(sym_w.items()):
                p = px.get(s, np.nan)
                if np.isnan(p):
                    notes.append(dict(event_id=e["event_id"], symbol=s,
                                      issue="no model price; leg skipped"))
                    continue
                tgt = target_total * w
                need = max(0.0, tgt - hold.get(s, 0.0) * p)
                q = float(np.round(need / p))
                if q:
                    trades.append(dict(event_id=e["event_id"], model_date=date,
                                       symbol=s, side="buy", qty=q, price=p,
                                       price_field=field, kind="invest",
                                       weight=w))
                    hold[s] = hold.get(s, 0.0) + q

        elif e["kind"] == "rebalance":
            md = e.get("model_rebalance_date")
            if md is None or md in done_model_dates:
                # a second investor event against the same model change: the
                # model itself trades once per rebalance
                continue
            done_model_dates.add(md)
            t1 = store.next_trading_day(next(iter(sym_w), ""), md)
            # use any held/target symbol's calendar; fall back per symbol below
            date = t1 if t1 is not None else md
            px = {}
            for s in set(sym_w) | set(hold):
                d1 = store.next_trading_day(s, md)
                px[s] = store.bar(s, d1 if d1 is not None else md, reb_field,
                                  f"model rebalance {md.date()}")
            V = sum(q * px.get(s, np.nan) for s, q in hold.items())
            if np.isnan(V):
                bad = [s for s, q in hold.items() if np.isnan(px.get(s, np.nan))]
                notes.append(dict(event_id=e["event_id"], symbol=",".join(bad),
                                  issue="portfolio value unpriceable on model "
                                        "date; rebalance skipped in model book"))
                continue
            for s in sorted(set(sym_w) | set(hold)):
                p = px.get(s, np.nan)
                w = sym_w.get(s, 0.0)
                cur = hold.get(s, 0.0)
                if np.isnan(p):
                    notes.append(dict(event_id=e["event_id"], symbol=s,
                                      issue="no model price; leg skipped"))
                    continue
                tgt_q = float(np.round(V * w / p)) if w > 0 else 0.0
                dq = tgt_q - cur
                if abs(dq) < 1e-9:
                    continue
                trades.append(dict(event_id=e["event_id"], model_date=date,
                                   symbol=s, side="buy" if dq > 0 else "sell",
                                   qty=abs(dq), price=p, price_field=reb_field,
                                   kind="rebalance", weight=w))
                hold[s] = tgt_q
            hold = {s: q for s, q in hold.items() if q > 1e-9}

    tdf = pd.DataFrame(trades)
    term = {s: store.bar(s, valuation_date, "close", "model terminal value")
            for s in hold}
    pos = pd.DataFrame([dict(symbol=s, qty=q, close=term[s],
                             market_value=q * term[s]) for s, q in sorted(hold.items())])
    pnl = _book_pnl(tdf, hold, term)
    return dict(trades=tdf, positions=pos, terminal_prices=term,
                pnl_by_symbol=pnl, pnl_total=float(pnl["pnl"].sum(skipna=True)),
                notes=pd.DataFrame(notes))


def _book_pnl(trades: pd.DataFrame, hold: dict, term: dict) -> pd.DataFrame:
    """Price-return P&L per symbol: sells - buys + terminal value of remainder."""
    rows = []
    syms = sorted(set(trades["symbol"]) | set(hold)) if len(trades) else sorted(hold)
    for s in syms:
        g = trades[trades["symbol"] == s] if len(trades) else trades
        buys = float((g.loc[g["side"] == "buy", "qty"] *
                      g.loc[g["side"] == "buy", "price"]).sum())
        sells = float((g.loc[g["side"] == "sell", "qty"] *
                       g.loc[g["side"] == "sell", "price"]).sum())
        q = hold.get(s, 0.0)
        tv = q * term.get(s, np.nan) if q else 0.0
        rows.append(dict(symbol=s, buy_value=buys, sell_value=sells,
                         qty_end=q, terminal_value=tv, pnl=sells - buys + tv))
    return pd.DataFrame(rows)


def validate_against_index(model_positions_daily_fn, index_values: pd.DataFrame,
                           store, trades: pd.DataFrame, first_date, last_date):
    """
    Rebuild daily sum(qty x close) from the model trade list and compare with the
    official index. Days with any missing constituent price are excluded and
    counted. Returns (daily frame, summary dict).
    """
    if not len(trades):
        return pd.DataFrame(), dict(status="no model trades")
    t = trades.sort_values("model_date")
    days = index_values[(index_values["date"] >= first_date) &
                        (index_values["date"] <= last_date)][["date", "index_value"]]
    hold: dict[str, float] = {}
    ti = 0
    tl = t.to_dict("records")
    rows = []
    for _, d in days.iterrows():
        while ti < len(tl) and tl[ti]["model_date"] <= d["date"]:
            x = tl[ti]
            hold[x["symbol"]] = hold.get(x["symbol"], 0.0) + \
                (x["qty"] if x["side"] == "buy" else -x["qty"])
            ti += 1
        live = {s: q for s, q in hold.items() if q > 1e-9}
        vals = {s: store.bar(s, d["date"], "close", "model daily validation")
                for s in live}
        if any(np.isnan(v) for v in vals.values()) or not live:
            rows.append(dict(date=d["date"], index_value=d["index_value"],
                             model_value=np.nan, priced=False))
            continue
        rows.append(dict(date=d["date"], index_value=d["index_value"],
                         model_value=sum(q * vals[s] for s, q in live.items()),
                         priced=True))
    df = pd.DataFrame(rows)
    ok = df[df["priced"]]
    if len(ok) < 5:
        return df, dict(status="insufficient priced days", priced_days=len(ok),
                        total_days=len(df))
    # scale once, then measure relative tracking
    scale = (ok["model_value"] / ok["index_value"]).median()
    rel = ok["model_value"] / (ok["index_value"] * scale) - 1.0
    return df, dict(status="ok", priced_days=int(len(ok)), total_days=int(len(df)),
                    scale=float(scale),
                    tracking_mean_abs=float(rel.abs().mean()),
                    tracking_max_abs=float(rel.abs().max()))
