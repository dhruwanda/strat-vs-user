"""
Independently reproduce the official smallcase index from constituents, weights
and EOD prices, to VERIFY (not assume) the index-construction convention.

Per smallcase's published return-calculation methodology, at a rebalance dated
T0 the index sets new quantities on T1 (the next trading day) in three steps:
intermediate index at T1's OHLC average -> new fractional quantities
q_i = intermediate x w_i / OHLC_i(T1) -> final index = sum(q_i x close_i(T1)).
Between rebalances quantities are constant and the index is sum(q x close).

This module replays that arithmetic under several candidate transition
conventions and reports which one reproduces the official series. The anchor
scales quantities so the replica equals the official index on the first
transition day; everything after that is out-of-sample.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CONVENTIONS = ("t1_ohlc", "t1_close", "t0_close")


def replicate(index_values: pd.DataFrame, constituents: pd.DataFrame,
              mapping: dict, store, start: pd.Timestamp, end: pd.Timestamp,
              convention: str = "t1_ohlc") -> pd.DataFrame:
    cal = index_values[(index_values["date"] >= start) &
                       (index_values["date"] <= end)].reset_index(drop=True)
    flags = list(cal.loc[cal["rebalance_flag"], "date"])
    if not flags:
        raise ValueError("no rebalance flags in the window")

    def version_on(d):
        m = (constituents["version_start"] <= d) & (constituents["version_end"] >= d)
        return constituents[m]

    def t1_of(t0):
        after = cal.loc[cal["date"] > t0, "date"]
        return after.iloc[0] if len(after) else None

    def trans_price(sym, t0):
        if convention == "t0_close":
            return store.bar(sym, t0, "close", f"replication {convention}")
        t1 = t1_of(t0)
        if t1 is None:
            return np.nan
        field = "ohlc_avg" if convention == "t1_ohlc" else "close"
        return store.bar(sym, t1, field, f"replication {convention}")

    # ---- anchor at the first flag: scale to the official index ----
    t0 = flags[0]
    ver = version_on(t0 if convention == "t0_close" else (t1_of(t0) or t0))
    w = {mapping[n]: wt for n, wt in zip(ver["name"], ver["weight"]) if n in mapping}
    p = {s: trans_price(s, t0) for s in w}
    eff = t0 if convention == "t0_close" else t1_of(t0)
    close0 = {s: store.bar(s, eff, "close", "replication anchor") for s in w}
    if any(np.isnan(list(p.values()))) or any(np.isnan(list(close0.values()))):
        bad = [s for s in w if np.isnan(p[s]) or np.isnan(close0[s])]
        raise ValueError(f"anchor prices missing for {bad}")
    i_off = float(cal.loc[cal["date"] == eff, "index_value"].iloc[0])
    # q_i proportional to w_i / p_i, scaled so sum(q x close(eff)) = official(eff)
    raw = {s: w[s] / p[s] for s in w}
    scale = i_off / sum(raw[s] * close0[s] for s in w)
    q = {s: raw[s] * scale for s in w}

    rows, seg = [], str(t0.date())
    for _, d in cal.iterrows():
        day = d["date"]
        if day < eff:
            continue
        if d["rebalance_flag"] and day != t0:
            # 3-step transition using the replica's own state (out-of-sample)
            ver = version_on(day if convention == "t0_close" else (t1_of(day) or day))
            w = {mapping[n]: wt for n, wt in zip(ver["name"], ver["weight"])
                 if n in mapping}
            pt = {s: trans_price(s, day) for s in set(w) | set(q)}
            inter = sum(qq * pt.get(s, np.nan) for s, qq in q.items())
            if np.isnan(inter) or any(np.isnan(pt.get(s, np.nan)) for s in w):
                rows.append(dict(date=day, segment=seg, official=d["index_value"],
                                 replica=np.nan,
                                 note="transition price missing; replica paused"))
                continue
            q = {s: inter * w[s] / pt[s] for s in w}
            seg = str(day.date())
        c = {s: store.bar(s, day, "close", "replication daily") for s in q}
        if any(np.isnan(v) for v in c.values()):
            rows.append(dict(date=day, segment=seg, official=d["index_value"],
                             replica=np.nan, note="daily close missing"))
            continue
        rows.append(dict(date=day, segment=seg, official=d["index_value"],
                         replica=sum(q[s] * c[s] for s in q), note=""))
    df = pd.DataFrame(rows)
    df["diff_abs"] = df["replica"] - df["official"]
    df["diff_pct"] = df["replica"] / df["official"] - 1.0
    df["convention"] = convention
    return df


def compare_conventions(index_values, constituents, mapping, store,
                        start, end) -> tuple:
    """Run every candidate convention; report which reproduces the index."""
    all_daily, summ = [], []
    for c in CONVENTIONS:
        try:
            d = replicate(index_values, constituents, mapping, store, start, end, c)
        except ValueError as ex:
            summ.append(dict(convention=c, status=str(ex)))
            continue
        ok = d.dropna(subset=["replica"])
        seg = (ok.groupby("segment")["diff_pct"]
                 .agg(["mean", lambda s: s.abs().max()]))
        worst = ok.loc[ok["diff_pct"].abs().idxmax()] if len(ok) else None
        summ.append(dict(convention=c, status="ok", days_priced=int(len(ok)),
                         days_total=int(len(d)),
                         mean_abs_diff_pct=float(ok["diff_pct"].abs().mean()),
                         max_abs_diff_pct=float(ok["diff_pct"].abs().max()),
                         worst_date=(worst["date"] if worst is not None else pd.NaT),
                         worst_segment=(worst["segment"] if worst is not None else "")))
        all_daily.append(d)
    return (pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame(),
            pd.DataFrame(summ))
