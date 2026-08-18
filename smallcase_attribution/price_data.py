"""
EOD market data: request planning, fetching, caching.

Design constraints honoured:
  * Only the symbol/date spans the attribution actually needs are requested.
    Under yfinance one request covers one symbol's full span, so per symbol we
    request min(needed)..max(needed) once - the same call count as any sliver.
  * Everything is cached to a local CSV; repeated runs hit the cache only.
  * Prices must be UNADJUSTED (auto_adjust=False) because trade prices in the
    tradebook are unadjusted. Corporate actions in the span are detected via
    the actions feed and flagged; they are never silently adjusted around.
  * A missing symbol or date is reported as missing, never substituted.

Provider is pluggable; yfinance is the default because it is free. Its NSE
coverage is unofficial - failures are surfaced per symbol.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

CACHE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
ACTION_COLUMNS = ["symbol", "date", "action", "value"]


# ---------------------------------------------------------------------------
# request planning
# ---------------------------------------------------------------------------
def plan_requests(events: list, index_values: pd.DataFrame,
                  valuation_date: pd.Timestamp, buffer_days: int = 5) -> pd.DataFrame:
    """
    Minimal per-symbol date spans needed by the decomposition:
      * every symbol traded in any smallcase event: from just before its first
        event to the valuation date (needed for terminal closes and for the
        model book's daily validation against the index while it is held)
      * model rebalance reference dates (T+1 after each flagged date)
    Returns one row per symbol with start/end and the reason.
    """
    need: dict[str, list] = {}
    for e in events:
        syms = set(e["buy_qty"]) | set(e["sell_qty"])
        d0 = e["date"]
        if e.get("model_rebalance_date") is not None:
            d0 = min(d0, e["model_rebalance_date"])
        for s in syms:
            lo, hi = need.get(s, (d0, valuation_date))
            need[s] = [min(lo, d0), valuation_date]
    rows = [dict(symbol=s, start=lo - pd.Timedelta(days=buffer_days),
                 end=valuation_date + pd.Timedelta(days=1),
                 reason="traded in smallcase events; span covers model reference "
                        "dates, actual dates and terminal valuation")
            for s, (lo, hi) in sorted(need.items())]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
def _yahoo_symbol(symbol: str, overrides: dict) -> str:
    if symbol in overrides:
        return overrides[symbol]
    return f"{symbol}.NS"


def fetch_yfinance(plan: pd.DataFrame, overrides: dict | None = None,
                   pause: float = 0.4) -> tuple:
    """Runs only where Yahoo endpoints are reachable (i.e. on the user's machine)."""
    import time
    import yfinance as yf
    overrides = overrides or {}
    px_rows, act_rows, missing = [], [], []
    for _, r in plan.iterrows():
        ysym = _yahoo_symbol(r["symbol"], overrides)
        try:
            t = yf.Ticker(ysym)
            h = t.history(start=r["start"].strftime("%Y-%m-%d"),
                          end=r["end"].strftime("%Y-%m-%d"), auto_adjust=False)
        except Exception as ex:  # noqa: BLE001
            missing.append(dict(symbol=r["symbol"], yahoo_symbol=ysym,
                                error=str(ex)[:200]))
            continue
        if h is None or h.empty:
            missing.append(dict(symbol=r["symbol"], yahoo_symbol=ysym,
                                error="no rows returned"))
            continue
        h = h.reset_index()
        for _, x in h.iterrows():
            px_rows.append(dict(symbol=r["symbol"],
                                date=pd.Timestamp(x["Date"]).tz_localize(None).normalize(),
                                open=float(x["Open"]), high=float(x["High"]),
                                low=float(x["Low"]), close=float(x["Close"]),
                                volume=float(x.get("Volume", np.nan))))
            for col, name in (("Dividends", "dividend"), ("Stock Splits", "split")):
                v = float(x.get(col, 0) or 0)
                if v:
                    act_rows.append(dict(symbol=r["symbol"],
                                         date=pd.Timestamp(x["Date"]).tz_localize(None).normalize(),
                                         action=name, value=v))
        time.sleep(pause)
    return (pd.DataFrame(px_rows, columns=CACHE_COLUMNS),
            pd.DataFrame(act_rows, columns=ACTION_COLUMNS),
            pd.DataFrame(missing))


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
class PriceStore:
    """CSV-backed store of unadjusted EOD bars keyed by (symbol, date)."""

    def __init__(self, prices: pd.DataFrame, actions: pd.DataFrame | None = None):
        p = prices.copy()
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
        p["symbol"] = p["symbol"].astype(str).str.upper()
        self.prices = p.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
        self._by_sym = {s: g.set_index("date") for s, g in self.prices.groupby("symbol")}
        a = actions if actions is not None else pd.DataFrame(columns=ACTION_COLUMNS)
        if len(a):
            a = a.copy()
            a["date"] = pd.to_datetime(a["date"]).dt.normalize()
        self.actions = a
        self.misses: list[dict] = []

    @classmethod
    def from_csv(cls, path: str, actions_path: str | None = None):
        px = pd.read_csv(path)
        act = (pd.read_csv(actions_path) if actions_path and os.path.exists(actions_path)
               else None)
        return cls(px, act)

    def to_csv(self, path: str, actions_path: str | None = None):
        self.prices.to_csv(path, index=False)
        if actions_path is not None:
            self.actions.to_csv(actions_path, index=False)

    def has(self, symbol: str) -> bool:
        return symbol in self._by_sym

    def bar(self, symbol: str, date: pd.Timestamp, field: str = "close",
            purpose: str = "") -> float:
        """Exact-date lookup. A miss is recorded and returns NaN - never a guess."""
        g = self._by_sym.get(symbol)
        d = pd.Timestamp(date).normalize()
        if g is None or d not in g.index:
            self.misses.append(dict(symbol=symbol, date=d, field=field, purpose=purpose))
            return np.nan
        if field == "ohlc_avg":
            r = g.loc[d]
            return float((r["open"] + r["high"] + r["low"] + r["close"]) / 4.0)
        return float(g.loc[d, field])

    def next_trading_day(self, symbol: str, date: pd.Timestamp):
        g = self._by_sym.get(symbol)
        if g is None:
            return None
        after = g.index[g.index > pd.Timestamp(date).normalize()]
        return after[0] if len(after) else None

    def splits_in_span(self) -> pd.DataFrame:
        if not len(self.actions):
            return pd.DataFrame(columns=ACTION_COLUMNS)
        return self.actions[self.actions["action"] == "split"].reset_index(drop=True)

    def missing_report(self) -> pd.DataFrame:
        return (pd.DataFrame(self.misses).drop_duplicates()
                if self.misses else
                pd.DataFrame(columns=["symbol", "date", "field", "purpose"]))
