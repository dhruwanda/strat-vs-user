"""
Two parallel books over the same trades, because two different cost conventions
are in play and conflating them is the commonest error in this analysis.

  AVERAGE-COST BOOK (smallcase's convention)
      Runs over smallcase-attributed trades only. A sale books
      (sell price - running average cost) x qty. This is what the smallcase
      Investments page reports as Realized Returns, and what makes its
      Current Investment figure reproducible.

  FIFO BOOK (statutory convention for Indian capital gains)
      Runs over EVERY trade in the scrip, because units in a demat account are
      fungible: the tax authority does not know which lot the smallcase ordered.
      Where an investor also bought the same scrip independently, a smallcase
      sale can consume those personal lots first.

      Same-day offsetting legs are netted out first and reported as intraday.
      A buy and a sell of one scrip on one day never reach delivery, so they are
      squared off against each other rather than against older delivery lots -
      this is how the broker's own statement computes it, and it changes both the
      cost basis of the remaining lots and the tax character of the leg
      (speculative business income, not capital gains).
"""
from __future__ import annotations
from collections import deque
import numpy as np
import pandas as pd


def average_cost_book(trades: pd.DataFrame) -> tuple:
    """trades: smallcase-attributed only. Returns (positions, realisations)."""
    book, real = {}, []
    for _, r in trades.sort_values(["exec_ts", "symbol"]).iterrows():
        b = book.setdefault(r["symbol"], {"qty": 0.0, "cost": 0.0})
        if r["trade_type"] == "buy":
            b["qty"] += r["quantity"]
            b["cost"] += r["value"]
        else:
            avg = b["cost"] / b["qty"] if b["qty"] > 0 else np.nan
            pnl = (r["price"] - avg) * r["quantity"] if b["qty"] > 0 else np.nan
            real.append(dict(exec_ts=r["exec_ts"], date=r["trade_date"],
                             symbol=r["symbol"], qty=r["quantity"],
                             sell_price=r["price"], avg_cost=avg,
                             realized_pnl=pnl, sell_value=r["value"],
                             cost_released=(avg * r["quantity"] if b["qty"] > 0 else np.nan)))
            if b["qty"] > 0:
                b["cost"] -= avg * r["quantity"]
                b["qty"] -= r["quantity"]
            if b["qty"] <= 1e-9:
                b["qty"], b["cost"] = 0.0, 0.0
    pos = pd.DataFrame([dict(symbol=s, qty=v["qty"], cost_basis=v["cost"],
                             avg_price=(v["cost"] / v["qty"] if v["qty"] else np.nan))
                        for s, v in book.items() if v["qty"] > 1e-9])
    return (pos.sort_values("symbol").reset_index(drop=True) if len(pos) else
            pd.DataFrame(columns=["symbol", "qty", "cost_basis", "avg_price"]),
            pd.DataFrame(real))


def split_intraday(trades: pd.DataFrame) -> tuple:
    """
    Split each scrip-day into an intraday leg (the offsetting quantity) and the
    residual delivery leg. Returns (delivery_trades, intraday_legs).
    """
    intraday, delivery = [], []
    for (sym, day), g in trades.groupby(["symbol", "trade_date"], sort=False):
        b = g[g.trade_type == "buy"]
        s = g[g.trade_type == "sell"]
        bq, sq = b["quantity"].sum(), s["quantity"].sum()
        m = min(bq, sq)
        if m > 1e-9:
            bp = b["value"].sum() / bq
            sp = s["value"].sum() / sq
            intraday.append(dict(symbol=sym, date=day, qty=m, buy_price=bp,
                                 sell_price=sp, pnl=(sp - bp) * m,
                                 turnover=m * (bp + sp),
                                 sale_from_smallcase=bool(s["in_basket"].any())
                                 if "in_basket" in s else None))
        # residual, keeping trade-level prices weighted to the surviving side
        if bq - m > 1e-9:
            row = b.iloc[0].to_dict()
            row.update(quantity=bq - m, price=bp if m > 1e-9 else b["value"].sum() / bq,
                       trade_type="buy")
            row["value"] = row["quantity"] * row["price"]
            delivery.append(row)
        elif bq > 1e-9 and m <= 1e-9:
            delivery.extend(b.to_dict("records"))
        if sq - m > 1e-9:
            row = s.iloc[0].to_dict()
            row.update(quantity=sq - m, price=sp if m > 1e-9 else s["value"].sum() / sq,
                       trade_type="sell")
            row["value"] = row["quantity"] * row["price"]
            delivery.append(row)
        elif sq > 1e-9 and m <= 1e-9:
            delivery.extend(s.to_dict("records"))
    d = pd.DataFrame(delivery)
    if len(d):
        d = d.sort_values(["exec_ts", "symbol"]).reset_index(drop=True)
    return d, pd.DataFrame(intraday)


def fifo_book(all_trades: pd.DataFrame, smallcase_ts: set) -> tuple:
    """Returns (open_lots, delivery_realisations, intraday_legs)."""
    t = all_trades.copy()
    t["in_basket"] = t["exec_ts"].isin(smallcase_ts)
    delivery, intraday = split_intraday(t)

    lots, real = {}, []
    for _, r in delivery.iterrows():
        q = lots.setdefault(r["symbol"], deque())
        sc = bool(r.get("in_basket", False))
        if r["trade_type"] == "buy":
            q.append([r["trade_date"], r["quantity"], r["price"], sc])
            continue
        remaining = r["quantity"]
        while remaining > 1e-9 and q:
            lot = q[0]
            take = min(remaining, lot[1])
            real.append(dict(
                symbol=r["symbol"], buy_date=lot[0], sell_date=r["trade_date"],
                qty=take, buy_price=lot[2], sell_price=r["price"],
                cost=take * lot[2], proceeds=take * r["price"],
                gain=take * (r["price"] - lot[2]),
                holding_days=(r["trade_date"] - lot[0]).days,
                lot_from_smallcase=lot[3], sale_from_smallcase=sc))
            lot[1] -= take
            remaining -= take
            if lot[1] <= 1e-9:
                q.popleft()
        if remaining > 1e-9:
            real.append(dict(
                symbol=r["symbol"], buy_date=pd.NaT, sell_date=r["trade_date"],
                qty=remaining, buy_price=np.nan, sell_price=r["price"],
                cost=np.nan, proceeds=remaining * r["price"], gain=np.nan,
                holding_days=np.nan, lot_from_smallcase=None, sale_from_smallcase=sc))
    open_lots = pd.DataFrame(
        [dict(symbol=s, buy_date=l[0], qty=l[1], buy_price=l[2],
              cost=l[1] * l[2], from_smallcase=l[3])
         for s, dq in lots.items() for l in dq if l[1] > 1e-9])
    return open_lots, pd.DataFrame(real), intraday
