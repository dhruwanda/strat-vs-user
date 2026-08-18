"""
Model-vs-actual implementation attribution (per event, per constituent, buys
and sells separately).

DEFINITIONS
-----------
MODEL QUANTITY: the reconstructed smallcase-prescribed quantity (see
reconstruct.py; validated against the investor's own unmodified orders).

MODEL REFERENCE PRICE: the price at which the smallcase MODEL/INDEX carries the
same transaction. Established from documentation plus independent replication
of the official index (strategy_replication.py):
  * rebalances: the index applies a rebalance on T1, the first trading day
    after the flagged date, at that day's OHLC average (T1-close is empirically
    indistinguishable; OHLC average is what the documentation states)
  * invest events: the index is an EOD close series, so the model carries fresh
    money at the event day's close
This is a MODEL-CONSTRUCTION price, not an execution price. The quantities
SHOWN to the user are a different thing: they are computed at order time from
live prices (empirics: 69/84 exact with execution VWAP + standard rounding vs
39/84 for the next best basis).

PER PAIRED LEG (event x symbol x side), with s = +1 buy, -1 sell:

  implementation price effect = s x model_qty x (ref price - actual fill)
      positive = the investor transacted at a better price than the model.

  quantity deviation           = actual_qty - model_qty
  quantity cash component      = (actual_qty - model_qty) x actual fill

  RECONCILIATION IDENTITY (exact, per leg):
      actual value - model value
        = actual_qty x fill - model_qty x ref
        = model_qty x (fill - ref)            [price component]
        + (actual_qty - model_qty) x fill     [quantity component]
  The price component equals -s x (implementation price effect).

Unpaired legs: model-only (skipped) carry their whole model value as a
negative quantity component; actual-only (drift, incl. attached deferred legs
beyond the model) carry their whole actual value. No terminal price enters any
of this; longer-horizon P&L context lives in the model book, separately.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _ref_price(store, symbol, event, cal_next, cfg):
    if event["kind"] == "invest":
        return store.bar(symbol, event["date"], "close",
                         f"model ref (invest close), event {event['event_id']}")
    t0 = event.get("model_rebalance_date")
    if t0 is None:
        return np.nan
    t1 = cal_next(t0)
    if t1 is None:
        return np.nan
    field = getattr(cfg, "model_rebalance_price_field", "ohlc_avg")
    return store.bar(symbol, t1, field,
                     f"model ref (rebalance T1 {field}), event {event['event_id']}")


def model_legs_from_reconstruction(rec_lines: pd.DataFrame) -> pd.DataFrame:
    """Model buy AND sell quantities for every event, from the reconstruction."""
    rows = []
    for _, r in rec_lines.iterrows():
        if r["event_kind"] == "invest":
            q = r.get("qty_recommended")
            if pd.isna(q) or q <= 0:
                continue
            rows.append(dict(event_id=r["event_id"], symbol=r["symbol"],
                             side="buy", model_qty=float(q)))
        else:
            rec, pre = r.get("qty_after_recommended"), r.get("qty_before", 0.0)
            if pd.isna(rec):
                continue
            dq = float(rec) - float(pre)
            if abs(dq) < 0.5:
                continue
            rows.append(dict(event_id=r["event_id"], symbol=r["symbol"],
                             side="buy" if dq > 0 else "sell",
                             model_qty=abs(dq)))
    return pd.DataFrame(rows)


def _actual_legs(sc_trades: pd.DataFrame) -> pd.DataFrame:
    g = (sc_trades.groupby(["event_id", "symbol", "trade_type"])
         .agg(qty=("quantity", "sum"), value=("value", "sum"),
              date=("trade_date", "min"),
              deferred=("flag", lambda x: any("deferred" in str(v) for v in x)))
         .reset_index())
    g["price"] = g["value"] / g["qty"]
    return g


def attribute_implementation(rec_lines: pd.DataFrame, sc_trades: pd.DataFrame,
                             events: list, store, cfg,
                             calendar: list | None = None) -> dict:
    # The trading calendar comes from the smallcase index timeline, which lists
    # every trading day. Deriving it from the price file would silently shift
    # T1 whenever the price file is sparse.
    cal = sorted(calendar) if calendar is not None else \
        sorted(store.prices["date"].unique())

    def cal_next(d):
        c = [x for x in cal if x > pd.Timestamp(d)]
        return c[0] if c else None

    ml = model_legs_from_reconstruction(rec_lines)
    al = _actual_legs(sc_trades)
    ev_by_id = {e["event_id"]: e for e in events}

    def _residual_excl(e, sym):
        """Net cash of the event's OTHER legs (sells - buys), at event prices."""
        r = 0.0
        for x, q in e["sell_qty"].items():
            if x != sym:
                r += q * e["price"][x]
        for x, q in e["buy_qty"].items():
            if x != sym:
                r -= q * e["price"][x]
        return r

    def _range_reproducible(sym, day, target_value, held_or_zero, qty_a):
        """Can any price inside the day's low-high range make the model formula
        yield the executed quantity? target_value is the leg's model target in
        rupees; monotone in price, so the two endpoints bound the range."""
        lo = store.bar(sym, day, "low", "snap range test")
        hi = store.bar(sym, day, "high", "snap range test")
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
            return False
        qs = sorted(float(np.round(target_value / p - held_or_zero))
                    for p in (lo, hi))
        return qs[0] <= qty_a <= qs[1]

    def _classify(a, e, qm, s_dir):
        """Returns (model_qty_final, deviation_class)."""
        qty_a = float(a["qty"])
        dq = qty_a - qm
        if abs(dq) < 0.5:
            return qm, "match"
        if not getattr(cfg, "quantity_snap", True):
            return qm, "user modification (snapping disabled)"
        if abs(dq) <= 1.0:
            return qty_a, "snapped: +/-1 rounding boundary"
        # day-range price-noise test
        rec = rec_lines[(rec_lines["event_id"] == a["event_id"])
                        & (rec_lines["symbol"] == a["symbol"])]
        if len(rec):
            r0 = rec.iloc[0]
            if e["kind"] == "invest":
                held = float(r0.get("held_before", 0.0) or 0.0)
                tv = (float(r0["qty_recommended"]) + held) * float(r0["price"])
                if _range_reproducible(a["symbol"], a["date"], tv, held, qty_a):
                    return qty_a, "snapped: within day's traded price range"
            else:
                tv = float(r0.get("qty_after_recommended", np.nan)) * float(r0["price"])
                post_a = float(r0.get("qty_before", 0.0) or 0.0) + s_dir * qty_a
                if np.isfinite(tv) and _range_reproducible(
                        a["symbol"], a["date"], tv, 0.0, post_a):
                    return qty_a, "snapped: within day's traded price range"
        # cash-balancer test (rebalances only)
        if e["kind"] != "invest":
            resid = _residual_excl(e, a["symbol"])
            leg_cash = s_dir * qty_a * float(a["price"]) * -1.0  # buy consumes cash
            tol = max(cfg.balancer_abs_tol, cfg.balancer_rel_tol * abs(resid))
            if abs(leg_cash + resid) <= tol and abs(resid) > cfg.balancer_abs_tol:
                return qty_a, "snapped: cash balancer (absorbs the event's residual)"
        return qm, "user modification"

    rows, used = [], set()
    for _, a in al.iterrows():
        e = ev_by_id[a["event_id"]]
        s_dir = 1.0 if a["trade_type"] == "buy" else -1.0
        ref = _ref_price(store, a["symbol"], e, cal_next, cfg)
        m = ml[(ml["event_id"] == a["event_id"]) & (ml["symbol"] == a["symbol"])
               & (ml["side"] == a["trade_type"])] if len(ml) else ml
        qm_raw = float(m["model_qty"].iloc[0]) if len(m) else 0.0
        if len(m):
            used.add((a["event_id"], a["symbol"], a["trade_type"]))
        qm, dev_class = _classify(a, e, qm_raw, s_dir)
        dq = float(a["qty"]) - qm
        price_eff = s_dir * qm * (ref - a["price"]) if np.isfinite(ref) else np.nan
        price_comp = qm * (a["price"] - ref) if np.isfinite(ref) else np.nan
        qty_comp = dq * a["price"]
        val_diff = (float(a["qty"]) * a["price"] - qm * ref
                    if np.isfinite(ref) else np.nan)
        rows.append(dict(
            event_id=a["event_id"], event_kind=e["kind"], symbol=a["symbol"],
            side=a["trade_type"], model_date=e.get("model_rebalance_date"),
            actual_date=a["date"], model_qty=qm, actual_qty=float(a["qty"]),
            model_qty_reconstructed=qm_raw,
            qty_deviation=dq, deviation_class=dev_class,
            model_ref_price=ref, actual_price=float(a["price"]),
            implementation_price_effect=price_eff,
            price_component=price_comp, quantity_component=qty_comp,
            value_difference=val_diff, deferred_leg=bool(a["deferred"]),
            pairing="paired" if qm_raw else "actual only",
            status=("ok" if np.isfinite(ref) else
                    "model ref price missing - excluded from totals")))
    for _, m in ml.iterrows():
        if (m["event_id"], m["symbol"], m["side"]) in used:
            continue
        e = ev_by_id.get(m["event_id"])
        ref = _ref_price(store, m["symbol"], e, cal_next, cfg) if e else np.nan
        leg_val = float(m["model_qty"]) * ref if np.isfinite(ref) else np.nan
        if getattr(cfg, "quantity_snap", True) and (
                float(m["model_qty"]) <= 1.0 or
                (np.isfinite(leg_val) and leg_val <= cfg.no_trade_value_threshold)):
            rows.append(dict(
                event_id=m["event_id"], event_kind=e["kind"] if e else "",
                symbol=m["symbol"], side=m["side"],
                model_date=e.get("model_rebalance_date") if e else pd.NaT,
                actual_date=pd.NaT, model_qty=0.0,
                model_qty_reconstructed=float(m["model_qty"]), actual_qty=0.0,
                qty_deviation=0.0,
                deviation_class="snapped: untraded model leg below threshold",
                model_ref_price=ref, actual_price=np.nan,
                implementation_price_effect=0.0, price_component=0.0,
                quantity_component=0.0, value_difference=0.0,
                deferred_leg=False, pairing="model only", status="ok"))
            continue
        rows.append(dict(
            event_id=m["event_id"], event_kind=e["kind"] if e else "",
            symbol=m["symbol"], side=m["side"],
            model_date=e.get("model_rebalance_date") if e else pd.NaT,
            actual_date=pd.NaT, model_qty=float(m["model_qty"]),
            model_qty_reconstructed=float(m["model_qty"]), actual_qty=0.0,
            qty_deviation=-float(m["model_qty"]),
            deviation_class="user modification (model leg not traded)",
            model_ref_price=ref,
            actual_price=np.nan, implementation_price_effect=0.0,
            price_component=0.0,
            quantity_component=(-float(m["model_qty"]) * ref
                                if np.isfinite(ref) else np.nan),
            value_difference=(-float(m["model_qty"]) * ref
                              if np.isfinite(ref) else np.nan),
            deferred_leg=False, pairing="model only",
            status=("ok" if np.isfinite(ref) else
                    "model ref price missing - excluded from totals")))
    lines = pd.DataFrame(rows)
    snap_counts = lines["deviation_class"].value_counts().rename_axis(
        "deviation_class").reset_index(name="legs")

    ok = lines[lines["status"] == "ok"]
    ident = (ok["value_difference"]
             - ok["price_component"] - ok["quantity_component"]).abs().max()
    summary = pd.DataFrame([
        dict(measure="Implementation price effect - BUYS",
             amount=float(ok.loc[ok.side == "buy",
                                 "implementation_price_effect"].sum())),
        dict(measure="Implementation price effect - SELLS",
             amount=float(ok.loc[ok.side == "sell",
                                 "implementation_price_effect"].sum())),
        dict(measure="Implementation price effect - TOTAL",
             amount=float(ok["implementation_price_effect"].sum())),
        dict(measure="Quantity cash component - TOTAL",
             amount=float(ok["quantity_component"].sum())),
        dict(measure="Total model-vs-actual value difference",
             amount=float(ok["value_difference"].sum())),
        dict(measure="Per-leg identity check, max abs error",
             amount=float(ident) if np.isfinite(ident) else np.nan),
    ])
    by_event = (ok.groupby(["event_id", "event_kind"])
                [["implementation_price_effect", "quantity_component"]]
                .sum().reset_index())
    by_stock = (ok.groupby("symbol")
                [["implementation_price_effect", "quantity_component"]]
                .sum().sort_values("implementation_price_effect")
                .reset_index())
    by_side = (ok.groupby(["event_kind", "side"])
               [["implementation_price_effect", "quantity_component"]]
               .sum().reset_index())
    return dict(lines=lines, summary=summary, by_event=by_event,
                by_stock=by_stock, by_side=by_side, snap_counts=snap_counts,
                excluded=lines[lines["status"] != "ok"],
                identity_max_error=float(ident) if np.isfinite(ident) else np.nan)


def rebalance_timeline(index_values, constituents, events, store, cfg,
                       start) -> pd.DataFrame:
    """T0, effective date, T1, price conventions and investor application."""
    cal = sorted(index_values["date"].unique())

    def cal_next(d):
        c = [x for x in cal if x > pd.Timestamp(d)]
        return c[0] if c else None

    flags = index_values.loc[index_values["rebalance_flag"] &
                             (index_values["date"] >= start), "date"]
    applied = {}
    for e in events:
        if e["kind"] != "invest" and e.get("model_rebalance_date") is not None:
            applied.setdefault(e["model_rebalance_date"], e)
    rows = []
    for t0 in flags:
        ver = constituents[constituents["version_start"] == t0]
        e = applied.get(t0)
        rows.append(dict(
            T0_model_rebalance_date=t0,
            version_effective_from=t0,
            T1_first_trading_day=cal_next(t0),
            index_quantity_price_basis="T1 OHLC average (documentation; T1 close "
                                       "indistinguishable in replication)",
            user_quantity_price_basis="live prices at order time (execution VWAP "
                                      "is the observable stand-in)",
            rounding_rule="standard rounding (69/84 exact on invest events; "
                          "floor vs round inconclusive on rebalances, both "
                          "within +/-1 share)",
            constituents=int(len(ver)),
            investor_applied_on=e["date"] if e else pd.NaT,
            lag_trading_days=np.nan if not e else
            len([d for d in cal if t0 < d <= e["date"]]) - 1,
            status="applied" if e else "no matching investor trade"))
    return pd.DataFrame(rows)


RULES = {"floor": np.floor, "round": np.round, "ceil": np.ceil}


def quantity_convention_evidence(events: list, mapping: dict, store, cfg) -> tuple:
    """Empirical test: which price basis and rounding rule reproduce the
    observed quantities? Returns (rebalance table, invest table)."""
    cal = sorted(store.prices["date"].unique())

    def prev_td(d):
        c = [x for x in cal if x < pd.Timestamp(d)]
        return c[-1] if c else None

    def next_td(d):
        c = [x for x in cal if x > pd.Timestamp(d)]
        return c[0] if c else None

    def basis_price(s, e, basis):
        T0, T2 = e.get("model_rebalance_date"), e["date"]
        if basis == "exec_vwap":
            p = e["price"].get(s)
            return p if p is not None else store.bar(s, T2, "close", "evidence")
        if basis == "t2_prev_close":
            d = prev_td(T2)
            return store.bar(s, d, "close", "evidence") if d else np.nan
        if basis == "t2_close":
            return store.bar(s, T2, "close", "evidence")
        if basis == "t0_close":
            return store.bar(s, T0, "close", "evidence") if T0 else np.nan
        if basis in ("t1_ohlc", "t1_close"):
            d = next_td(T0) if T0 else None
            return store.bar(s, d, "ohlc_avg" if basis == "t1_ohlc" else "close",
                             "evidence") if d else np.nan

    reb, inv = [], []
    for basis in ("exec_vwap", "t2_prev_close", "t2_close", "t0_close",
                  "t1_ohlc", "t1_close"):
        for rn, rule in RULES.items():
            tot = ok = 0
            for e in events:
                if e["kind"] != "rebalance":
                    continue
                W = {mapping[n]: w for n, w in e["active_weights"].items()
                     if n in mapping}
                px = {s: basis_price(s, e, basis)
                      for s in set(e["held_before"]) | set(W)}
                V = sum(q * px.get(s, np.nan) for s, q in e["held_before"].items())
                if np.isnan(V):
                    continue
                for s in set(e["buy_qty"]) | set(e["sell_qty"]):
                    w, p = W.get(s, 0.0), px.get(s, np.nan)
                    if np.isnan(p):
                        continue
                    tgt = float(rule(V * w / p)) if w > 0 else 0.0
                    tot += 1
                    ok += int(abs(tgt - e["held_after"].get(s, 0.0)) < 0.5)
            reb.append(dict(price_basis=basis, rounding=rn,
                            exact_matches=ok, quantities_tested=tot))
    for basis in ("exec_vwap", "t2_prev_close", "t2_close"):
        for rn, rule in RULES.items():
            tot = ok = 0
            for e in events:
                if e["kind"] != "invest":
                    continue
                W = {mapping[n]: w for n, w in e["active_weights"].items()
                     if n in mapping}
                px = {s: basis_price(s, e, basis)
                      for s in set(W) | set(e["held_before"])}
                V = sum(q * px.get(s, np.nan) for s, q in e["held_before"].items())
                if np.isnan(V):
                    continue
                for s, w in W.items():
                    p = px.get(s, np.nan)
                    if np.isnan(p):
                        continue
                    need = max(0.0, (V + e["buy_value"]) * w
                               - e["held_before"].get(s, 0.0) * p)
                    tot += 1
                    ok += int(abs(float(rule(need / p))
                                  - e["buy_qty"].get(s, 0.0)) < 0.5)
            inv.append(dict(price_basis=basis, rounding=rn,
                            exact_matches=ok, quantities_tested=tot))
    r = pd.DataFrame(reb).sort_values("exact_matches", ascending=False)
    i = pd.DataFrame(inv).sort_values("exact_matches", ascending=False)
    return r.reset_index(drop=True), i.reset_index(drop=True)
