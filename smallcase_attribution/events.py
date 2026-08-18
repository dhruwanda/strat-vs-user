"""
Identify which broker trades belong to the smallcase, and group them into events.

Rule (deliberately transparent rather than clever):

  A trade belongs to the smallcase if it executed inside a BASKET - a set of
  trades sharing one execution timestamp covering at least `basket_min_symbols`
  distinct symbols. A smallcase order routes every leg simultaneously, so all
  legs of one order carry the same execution second. A standalone order in the
  same stock does not.

Everything else is classified 'outside smallcase' and listed for review with the
reason, so an investor who traded a constituent independently can see exactly
what was excluded and why.

Corroborating evidence recorded per event but never used to override the rule:
  * a smallcase transaction fee posted on the same date (charged on first buy,
    invest-more and SIP orders only - never on rebalance or exit)
  * proximity to a model rebalance date
  * whether every symbol traded is a current or immediately-prior constituent
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _active_version(constituents: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    m = (constituents["version_start"] <= day) & (constituents["version_end"] >= day)
    if m.any():
        return constituents[m]
    prior = constituents[constituents["version_start"] <= day]
    if prior.empty:
        return constituents.iloc[0:0]
    last = prior["version_start"].max()
    return constituents[constituents["version_start"] == last]


def _prev_version(constituents: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    starts = sorted(constituents["version_start"].unique())
    cur = _active_version(constituents, day)
    if cur.empty:
        return cur
    s = cur["version_start"].iloc[0]
    idx = starts.index(s)
    if idx == 0:
        return constituents.iloc[0:0]
    return constituents[constituents["version_start"] == starts[idx - 1]]


def _clusters(trades: pd.DataFrame, cfg):
    """Candidate legs (>= pair_leg_min_symbols simultaneous symbols), grouped
    into clusters within the merge window, then qualified:

      qualifies if  max leg size >= basket_min_symbols
                OR  the cluster has a buy leg AND a sell leg, each with
                    >= pair_leg_min_symbols simultaneous symbols.

    Returns (qualified clusters, rejected clusters) as lists of timestamp
    lists. Rejected clusters are reported, never silently dropped."""
    g = trades.groupby("exec_ts").agg(
        n=("symbol", "nunique"),
        sides=("trade_type", lambda x: frozenset(x))).reset_index()
    legs = g[g["n"] >= cfg.pair_leg_min_symbols].sort_values("exec_ts")
    clusters, cur = [], []
    for t in legs["exec_ts"]:
        if cur and (t - cur[-1]).total_seconds() <= cfg.event_merge_seconds:
            cur.append(t)
        else:
            if cur:
                clusters.append(cur)
            cur = [t]
    if cur:
        clusters.append(cur)
    ok, rejected = [], []
    info = legs.set_index("exec_ts")
    for c in clusters:
        nmax = int(info.loc[c, "n"].max())
        has_buy = any("buy" in info.loc[t, "sides"] for t in c)
        has_sell = any("sell" in info.loc[t, "sides"] for t in c)
        if nmax >= cfg.basket_min_symbols or (has_buy and has_sell):
            ok.append(c)
        else:
            rejected.append(c)
    return ok, rejected


def tag_smallcase_trades(trades: pd.DataFrame, cfg) -> pd.DataFrame:
    """Adds `in_basket` and `basket_ts` using the cluster qualification rule."""
    t = trades.copy()
    ok, _ = _clusters(t, cfg)
    stamps = {ts for c in ok for ts in c}
    t["in_basket"] = t["exec_ts"].isin(stamps)
    t["basket_ts"] = np.where(t["in_basket"], t["exec_ts"], pd.NaT)
    return t


def detect_events(trades: pd.DataFrame, constituents: pd.DataFrame,
                  fees: pd.DataFrame, index_values: pd.DataFrame, cfg) -> list:
    """
    Groups basket legs into events. A rebalance places its sell leg and buy leg a
    couple of seconds apart, so legs within `event_merge_seconds` form one event.
    Returns a list of dicts, chronological, each with the pre/post holding state.
    """
    t = tag_smallcase_trades(trades, cfg)
    basket = t[t["in_basket"]].copy()
    if basket.empty:
        return []
    groups, _ = _clusters(t, cfg)

    reb_dates = set(index_values.loc[index_values["rebalance_flag"], "date"])
    fee_dates = set(fees["posting_date"]) if len(fees) else set()

    held: dict = {}
    events = []
    prev_names = None
    for gi, g in enumerate(groups):
        sub = basket[basket["exec_ts"].isin(g)]
        t0 = sub["exec_ts"].min()
        day = t0.normalize()
        ver = _active_version(constituents, day)
        prev_ver = _prev_version(constituents, day)
        names = set(ver["name"])
        weights = dict(zip(ver["name"], ver["weight"]))

        buy = sub[sub.trade_type == "buy"].groupby("symbol")["quantity"].sum().to_dict()
        sell = sub[sub.trade_type == "sell"].groupby("symbol")["quantity"].sum().to_dict()
        px = (sub.groupby("symbol")
                 .apply(lambda x: x["value"].sum() / x["quantity"].sum(), include_groups=False)
                 .to_dict())

        before = dict(held)
        after = dict(held)
        for s, q in buy.items():
            after[s] = after.get(s, 0.0) + q
        for s, q in sell.items():
            after[s] = after.get(s, 0.0) - q
        entered = {s for s in buy if before.get(s, 0.0) <= 1e-9}
        exited = {s for s in sell if abs(after.get(s, 0.0)) <= 1e-9}

        kind = "invest" if not sell else ("exit" if not buy else "rebalance")
        # distance to the nearest model rebalance date at or before this trade
        past = [d for d in reb_dates if d <= day]
        near = max(past) if past else None

        events.append(dict(
            event_id=gi + 1, ts=t0, date=day, legs=list(g), kind=kind,
            buy_qty=buy, sell_qty=sell, price=px,
            buy_value=float(sub[sub.trade_type == "buy"]["value"].sum()),
            sell_value=float(sub[sub.trade_type == "sell"]["value"].sum()),
            held_before={k: v for k, v in before.items() if v > 1e-9},
            held_after={k: v for k, v in after.items() if v > 1e-9},
            entered=entered, exited=exited,
            active_names=names, active_weights=weights,
            version=ver["version"].iloc[0] if len(ver) else None,
            prev_names=set(prev_ver["name"]),
            added_names=(names - prev_names) if prev_names is not None else set(),
            removed_names=(prev_names - names) if prev_names is not None else set(),
            fee_on_date=day in fee_dates,
            model_rebalance_date=near,
            lag_days=(day - near).days if near is not None else None,
            n_symbols=int(sub["symbol"].nunique()),
        ))
        held = {k: v for k, v in after.items() if v > 1e-9}
        prev_names = names
    return events


def attribute_trades(trades: pd.DataFrame, events: list, mapping: dict,
                     constituents: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per-trade attribution table with an explicit reason on every row."""
    t = tag_smallcase_trades(trades, cfg)
    ts2ev = {}
    for e in events:
        for leg in e["legs"]:
            ts2ev[leg] = e
    sym_to_name = {}
    for n, s in mapping.items():
        sym_to_name.setdefault(s, n)
    ever = set(mapping.values())

    rows = []
    for _, r in t.iterrows():
        e = ts2ev.get(r["exec_ts"])
        if e is not None:
            in_ver = sym_to_name.get(r["symbol"]) in e["active_names"]
            in_prev = sym_to_name.get(r["symbol"]) in e["prev_names"]
            if in_ver:
                reason = "basket order; symbol is a current constituent"
            elif in_prev:
                reason = "basket order; symbol was a constituent before this rebalance"
            else:
                reason = ("basket order; symbol not in the current or prior version "
                          "- REVIEW")
            rows.append(dict(r, attribution="smallcase", event_id=e["event_id"],
                             event_kind=e["kind"], reason=reason,
                             flag=("review" if not (in_ver or in_prev) else "")))
        else:
            known = r["symbol"] in ever
            rows.append(dict(r, attribution="outside smallcase", event_id=None,
                             event_kind=None,
                             reason=("standalone order (single symbol at this execution "
                                     "time); symbol is a smallcase constituent elsewhere "
                                     "in the period"
                                     if known else
                                     "standalone order; symbol is never a constituent"),
                             flag=("review" if known else "")))
    out = pd.DataFrame(rows)
    return out.sort_values(["exec_ts", "symbol"]).reset_index(drop=True)


def detect_deferred_legs(attributed: pd.DataFrame, rec_lines: pd.DataFrame,
                         events: list, cfg) -> tuple:
    """
    A leg can fail inside the basket (circuit limit, illiquidity) and be
    executed standalone over the following days. Attach such trades to their
    event instead of misreading them as user modifications.

    A standalone trade attaches to an event only if ALL hold:
      * the event's reconstruction shows a model shortfall in that symbol and
        direction of at least max(2 shares, 2% of the model quantity)
      * the trade executes within repair_window_days after the event
      * its quantity does not exceed the remaining shortfall by more than
        repair_qty_tolerance
    Everything else stays outside. Returns (updated attribution, repair report).
    """
    if not len(rec_lines):
        return attributed, pd.DataFrame()
    ev_by_id = {e["event_id"]: e for e in events}
    shortfalls = []
    for _, r in rec_lines.iterrows():
        if r["event_kind"] == "invest":
            rec, obs = r.get("qty_recommended"), r.get("qty_observed")
            side = "buy"
        else:
            rec = r.get("qty_after_recommended")
            obs = r.get("qty_after")
            if pd.isna(rec) or pd.isna(obs):
                continue
            pre = r.get("qty_before", 0.0)
            side = "buy" if rec >= pre else "sell"
            rec, obs = abs(rec - pre), abs(obs - pre)
        if pd.isna(rec) or pd.isna(obs):
            continue
        short = rec - obs
        if short >= max(2.0, 0.02 * rec):
            shortfalls.append(dict(event_id=r["event_id"], symbol=r["symbol"],
                                   side=side, shortfall=float(short)))
    if not shortfalls:
        return attributed, pd.DataFrame()

    out = attributed.copy()
    repairs = []
    outside = out["attribution"] == "outside smallcase"
    for sf in shortfalls:
        e = ev_by_id[sf["event_id"]]
        lo, hi = e["date"], e["date"] + pd.Timedelta(days=cfg.repair_window_days)
        cand = out[outside & (out["symbol"] == sf["symbol"])
                   & (out["trade_type"] == sf["side"])
                   & (out["trade_date"] > lo) & (out["trade_date"] <= hi)
                   ].sort_values("exec_ts")
        remaining = sf["shortfall"] * (1 + cfg.repair_qty_tolerance)
        for i, row in cand.iterrows():
            if row["quantity"] > remaining:
                continue
            out.loc[i, ["attribution", "event_id", "event_kind", "flag"]] =                 ["smallcase", sf["event_id"], e["kind"],
                 "deferred leg (possible circuit limit)"]
            out.loc[i, "reason"] = (
                "standalone order matching a model shortfall of "
                f"{sf['shortfall']:.0f} in this event, within "
                f"{cfg.repair_window_days} days - attached as a deferred leg")
            remaining -= row["quantity"]
            repairs.append(dict(event_id=sf["event_id"], symbol=sf["symbol"],
                                side=sf["side"], trade_date=row["trade_date"],
                                qty=row["quantity"],
                                shortfall=sf["shortfall"]))
    return out, pd.DataFrame(repairs)
