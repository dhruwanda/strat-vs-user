"""
Reconstruct the quantities smallcase would have recommended at each event.

Verified against smallcase's published methodology and against the observed
first-buy basket:

  FIRST BUY / INVEST MORE
      target_value_i = (V_before + A) * w_i
      buy_value_i    = max(0, target_value_i - held_i * p_i)
      qty_i          = round(buy_value_i / p_i)
  where V_before is the pre-event market value of the smallcase holding and A is
  the amount deployed. On a first buy V_before is zero and this collapses to the
  familiar A * w / p. On an invest-more it does not: smallcase tops the portfolio
  up TOWARDS the prescribed weights rather than buying every constituent
  pro-rata, which is why the naive formula fails from the second event onwards.

  REBALANCE
      target_value_i = V * w_i for every constituent it touches; names dropped
      from the version go to zero. V is the portfolio value on the rebalance day.

  Irreducible limitation: smallcase sizes the order from a live quote at order
  construction; the tradebook only records the execution price. Quantities are
  therefore reproduced to within a share or so, and the honest measure of fit is
  the VALUE deviation, not exact share equality.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _fit(observed, recommended, price, cfg):
    tol = max(cfg.qty_tolerance_shares, cfg.qty_tolerance_pct * max(recommended, 1))
    return abs(observed - recommended) <= tol


def reconstruct_investment(event: dict, mapping: dict, cfg,
                           amount: float | None = None) -> pd.DataFrame:
    """Top-up-to-target reconstruction for a buy-only event."""
    sym_w = {mapping[n]: w for n, w in event["active_weights"].items() if n in mapping}
    px = event["price"]
    held = event["held_before"]
    A = event["buy_value"] if amount is None else amount

    priced = {s: p for s, p in px.items()}
    v_before = sum(q * priced[s] for s, q in held.items() if s in priced)
    unpriced = sorted(s for s, q in held.items() if s not in priced and q > 0)

    rows = []
    for s in sorted(set(sym_w) | set(event["buy_qty"])):
        w = sym_w.get(s, np.nan)
        p = priced.get(s, np.nan)
        h = held.get(s, 0.0)
        obs = event["buy_qty"].get(s, 0.0)
        if np.isnan(p) or np.isnan(w):
            rows.append(dict(symbol=s, weight=w, price=p, held_before=h,
                             qty_observed=obs, qty_recommended=np.nan,
                             qty_diff=np.nan, value_diff=np.nan,
                             status="cannot reconstruct: missing price or weight"))
            continue
        target = (v_before + A) * w
        need = max(0.0, target - h * p)
        rec = float(np.round(need / p))
        rows.append(dict(symbol=s, weight=w, price=p, held_before=h,
                         qty_observed=obs, qty_recommended=rec,
                         qty_diff=obs - rec, value_diff=(obs - rec) * p,
                         status="consistent" if _fit(obs, rec, p, cfg) else "differs"))
    df = pd.DataFrame(rows)
    df.attrs["v_before"] = v_before
    df.attrs["amount"] = A
    df.attrs["unpriced_holdings"] = unpriced
    return df


def reconstruct_rebalance(event: dict, mapping: dict, cfg) -> pd.DataFrame:
    """
    Rebalance reconstruction.

    The pre-event value of untouched holdings cannot be computed without EOD
    prices for every constituent on the event date, which the broker files do not
    contain. Two portfolio-value estimates are therefore produced:

      V_probe  - median over the TRADED constituents of (post-qty * price / weight).
                 If the rebalance really did drive each touched name to its
                 prescribed weight, these agree; their dispersion is a direct
                 measure of weight fidelity. Derived from the observed trades, so
                 it validates weight fidelity rather than proving quantities.
      V_direct - sum over all holdings of qty * price, using event-day prices where
                 available and the most recent earlier observed price otherwise.
                 The staleness of each fallback price is reported.
    """
    sym_w = {mapping[n]: w for n, w in event["active_weights"].items() if n in mapping}
    px = event["price"]
    after = event["held_after"]

    probes = []
    for s, q in after.items():
        w, p = sym_w.get(s), px.get(s)
        if w and p and q > 0 and s in set(event["buy_qty"]) | set(event["sell_qty"]):
            probes.append(q * p / w)
    v_probe = float(np.median(probes)) if probes else np.nan
    cv = float(np.std(probes) / np.mean(probes)) if len(probes) > 1 else np.nan

    rows = []
    for s in sorted(set(sym_w) | set(after) | set(event["sell_qty"])):
        w = sym_w.get(s, 0.0)
        p = px.get(s, np.nan)
        pre = event["held_before"].get(s, 0.0)
        post = after.get(s, 0.0)
        obs_delta = post - pre
        if np.isnan(p) or np.isnan(v_probe):
            rows.append(dict(symbol=s, weight=w, price=p, qty_before=pre,
                             qty_after=post, qty_delta_observed=obs_delta,
                             qty_after_recommended=np.nan, qty_diff=np.nan,
                             value_diff=np.nan, weight_actual=np.nan,
                             status="cannot reconstruct: no event-day price"))
            continue
        rec_after = float(np.round(v_probe * w / p)) if w > 0 else 0.0
        rows.append(dict(symbol=s, weight=w, price=p, qty_before=pre,
                         qty_after=post, qty_delta_observed=obs_delta,
                         qty_after_recommended=rec_after,
                         qty_diff=post - rec_after,
                         value_diff=(post - rec_after) * p,
                         weight_actual=(post * p / v_probe) if v_probe else np.nan,
                         status="consistent" if _fit(post, rec_after, p, cfg) else "differs"))
    df = pd.DataFrame(rows)
    df.attrs["v_probe"] = v_probe
    df.attrs["probe_cv"] = cv
    df.attrs["n_probes"] = len(probes)
    return df


def reconstruct_all(events: list, mapping: dict, cfg) -> tuple:
    """Returns (per-line frame, per-event summary frame)."""
    lines, summ = [], []
    for e in events:
        if e["kind"] == "invest":
            d = reconstruct_investment(e, mapping, cfg)
            basis = "top-up to prescribed weights"
            vinfo = d.attrs["v_before"]
            extra = dict(portfolio_value_basis="sum of held qty x event-day price",
                         portfolio_value=vinfo,
                         unpriced_holdings=", ".join(d.attrs["unpriced_holdings"]))
            # sensitivity: nearest round amount a human would have typed
            best = None
            for step in cfg.round_amount_grid:
                A = round(e["buy_value"] / step) * step
                if A <= 0:
                    continue
                alt = reconstruct_investment(e, mapping, cfg, amount=A)
                ok = int((alt["status"] == "consistent").sum())
                if best is None or ok > best[1]:
                    best = (A, ok, float(alt["value_diff"].abs().sum()))
            extra["round_amount_tested"] = best[0] if best else np.nan
            extra["round_amount_consistent"] = best[1] if best else np.nan
        elif e["kind"] == "rebalance":
            d = reconstruct_rebalance(e, mapping, cfg)
            basis = "rebalance to prescribed weights"
            extra = dict(portfolio_value_basis="median of traded-leg implied value",
                         portfolio_value=d.attrs["v_probe"],
                         probe_dispersion_cv=d.attrs["probe_cv"],
                         n_probes=d.attrs["n_probes"],
                         unpriced_holdings="", round_amount_tested=np.nan,
                         round_amount_consistent=np.nan)
        else:
            d = pd.DataFrame()
            basis = "not reconstructed"
            extra = dict(portfolio_value_basis="", portfolio_value=np.nan,
                         unpriced_holdings="", round_amount_tested=np.nan,
                         round_amount_consistent=np.nan)

        if len(d):
            d = d.copy()
            d.insert(0, "event_id", e["event_id"])
            d.insert(1, "event_ts", e["ts"])
            d.insert(2, "event_kind", e["kind"])
            lines.append(d)
            n = int((~d["status"].str.startswith("cannot")).sum())
            n_bad = int(len(d) - n)
            ok = int((d["status"] == "consistent").sum())
            vd = float(d["value_diff"].abs().sum(skipna=True))
        else:
            n = ok = n_bad = 0
            vd = np.nan

        gross = e["buy_value"] + e["sell_value"]
        summ.append(dict(
            event_id=e["event_id"], event_ts=e["ts"], date=e["date"], kind=e["kind"],
            version=e["version"], n_symbols=e["n_symbols"],
            buy_value=e["buy_value"], sell_value=e["sell_value"],
            net_cash=e["buy_value"] - e["sell_value"],
            smallcase_fee_on_date=e["fee_on_date"],
            model_rebalance_date=e["model_rebalance_date"], lag_days=e["lag_days"],
            ambiguous_rebalance_pairing=bool(e.get("rebalance_flags_skipped")),
            reconstruction_basis=basis, lines_reconstructible=n,
            lines_not_reconstructible=n_bad, lines_consistent=ok,
            pct_lines_consistent=(ok / n if n else np.nan),
            abs_value_deviation=vd,
            value_deviation_pct_of_event=(vd / gross if gross else np.nan),
            **extra))
    line_df = pd.concat(lines, ignore_index=True) if lines else pd.DataFrame()
    return line_df, pd.DataFrame(summ)
