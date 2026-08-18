"""
Dividend statement handling.

Input: the Zerodha Console dividend statement (CSV or XLSX) with columns
Symbol, Ex-date, Qty, Dividend per share, Total dividend. Symbols may carry a
trailing '#' marker; it is stripped.

Attribution: a dividend belongs to the smallcase only to the extent the
smallcase held the shares. For each row, the smallcase-attributed quantity as
of the day before the ex-date is compared with the statement quantity, and the
amount is attributed pro-rata:

    attributed = total_dividend x min(1, smallcase_qty / statement_qty)

This is what keeps a dividend on independently bought units (or on holdings
that predate the smallcase) out of the smallcase's return - e.g. a dividend on
a personally traded constituent attributes to zero.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

_ALIASES = {
    "symbol": ["symbol"],
    "ex_date": ["ex-date", "ex date", "exdate"],
    "qty": ["qty", "quantity"],
    "dps": ["dividend per share", "dps"],
    "amount": ["total dividend", "amount", "net dividend"],
}


def load_dividends(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    low = {str(c).strip().lower(): c for c in raw.columns}
    col = {}
    for want, alts in _ALIASES.items():
        for a in alts:
            if a in low:
                col[want] = low[a]
                break
    for req in ("symbol", "ex_date", "amount"):
        if req not in col:
            raise ValueError(f"dividend statement is missing a '{req}' column")
    d = pd.DataFrame({k: raw[v] for k, v in col.items()})
    d["symbol"] = (d["symbol"].astype(str).str.strip().str.upper()
                   .str.replace(r"[#\*]+$", "", regex=True))
    d["ex_date"] = pd.to_datetime(d["ex_date"])
    for c in ("qty", "dps", "amount"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values("ex_date").reset_index(drop=True)


def attribute_dividends(dividends: pd.DataFrame, sc_trades: pd.DataFrame) -> pd.DataFrame:
    """Adds smallcase_qty_on_ex_date, attributed_amount, and the reason."""
    t = sc_trades.sort_values("exec_ts")
    out = dividends.copy()
    sc_q, reasons = [], []
    for _, r in out.iterrows():
        h = t[(t["symbol"] == r["symbol"]) & (t["trade_date"] < r["ex_date"])]
        q = float((h["signed_qty"]).sum()) if len(h) else 0.0
        sc_q.append(q)
        stmt_q = r.get("qty", np.nan)
        if q <= 0:
            reasons.append("symbol not held by the smallcase before the ex-date; "
                           "dividend is personal")
        elif np.isfinite(stmt_q) and q < stmt_q - 0.5:
            reasons.append("smallcase held only part of the entitled quantity; "
                           "attributed pro-rata")
        else:
            reasons.append("fully attributable to the smallcase holding")
    out["smallcase_qty_on_ex_date"] = sc_q
    frac = np.where(out.get("qty", pd.Series(np.nan, index=out.index)).fillna(0) > 0,
                    np.clip(out["smallcase_qty_on_ex_date"] /
                            out["qty"].replace(0, np.nan), 0, 1), 0.0)
    out["attributed_amount"] = out["amount"] * frac
    out["attribution_note"] = reasons
    return out
