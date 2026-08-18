"""
Layer A: how the smallcase MODEL portfolio performed.

The supplied index is smallcase's official series, so it is used as-is rather
than rebuilt from constituents. Per smallcase's published methodology:
  * the index starts at 100 on the smallcase's inception date and is a PRICE
    return series computed from end-of-day closes: index = sum(qty x close)
  * quantities are held constant between rebalances
  * at a rebalance the new quantities are set on the day after the rebalance date
    using that day's OHLC average, so the transition is priced at a traded average
    rather than a single close
  * transaction fees and other costs are NOT included, and no money was actually
    invested to produce it
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def normalise(index_values: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    idx = index_values[index_values["date"] >= start_date].copy()
    if idx.empty:
        raise ValueError("no index values on or after the first investment date")
    base = float(idx["index_value"].iloc[0])
    idx["normalised_index"] = idx["index_value"] / base * 100.0
    idx["cumulative_return"] = idx["normalised_index"] / 100.0 - 1.0
    idx.attrs["base_date"] = idx["date"].iloc[0]
    idx.attrs["base_index"] = base
    return idx.reset_index(drop=True)


def summarise(norm: pd.DataFrame) -> dict:
    first, last = norm.iloc[0], norm.iloc[-1]
    days = (last["date"] - first["date"]).days
    r = last["normalised_index"] / 100.0 - 1.0
    return {
        "start_date": first["date"], "end_date": last["date"],
        "base_index_value": norm.attrs["base_index"],
        "end_index_value": last["index_value"],
        "normalised_start": 100.0, "normalised_end": last["normalised_index"],
        "absolute_return": r, "calendar_days": days,
        "annualised_return": ((1 + r) ** (365.0 / days) - 1) if days >= 1 else np.nan,
        "trading_days": len(norm),
        "max_drawdown": float((norm["normalised_index"] /
                               norm["normalised_index"].cummax() - 1).min()),
        "rebalances_in_period": int(norm["rebalance_flag"].sum()),
    }


def rebalance_calendar(index_values: pd.DataFrame, constituents: pd.DataFrame,
                       start_date: pd.Timestamp, events: list) -> pd.DataFrame:
    """Model rebalance dates against the date the investor actually applied them."""
    flags = index_values.loc[index_values["rebalance_flag"] &
                             (index_values["date"] >= start_date), "date"]
    starts = sorted(constituents["version_start"].unique())
    applied = {}
    for e in events:
        if e["kind"] in ("rebalance", "exit") and e["model_rebalance_date"] is not None:
            applied.setdefault(e["model_rebalance_date"], e)
    rows = []
    for d in flags:
        ver = constituents[constituents["version_start"] == d]
        e = applied.get(d)
        rows.append(dict(
            model_rebalance_date=d,
            version=ver["version"].iloc[0] if len(ver) else None,
            constituents=len(ver),
            index_value_on_date=float(
                index_values.loc[index_values["date"] == d, "index_value"].iloc[0]),
            investor_applied_on=e["date"] if e else pd.NaT,
            lag_days=e["lag_days"] if e else np.nan,
            status="applied" if e else "no matching investor trade found"))
    # a version boundary with no index flag would be a data inconsistency
    extra = [d for d in starts if d >= start_date and d not in set(flags)]
    for d in extra:
        rows.append(dict(model_rebalance_date=d,
                         version=constituents.loc[constituents["version_start"] == d,
                                                  "version"].iloc[0],
                         constituents=int((constituents["version_start"] == d).sum()),
                         index_value_on_date=np.nan, investor_applied_on=pd.NaT,
                         lag_days=np.nan,
                         status="version boundary not flagged in the index sheet"))
    return pd.DataFrame(rows).sort_values("model_rebalance_date").reset_index(drop=True)
