"""
Resolve smallcase constituent names to broker trading symbols.

The constituent file gives company names ('One 97 Communications Ltd'); the
tradebook gives symbols ('PAYTM'). No shared key exists, and name similarity
alone fails on renamed or brand-named listings.

Strategy: score every (name, symbol) pair on structural evidence plus name
similarity, then solve a global one-to-one assignment. Structural evidence is
far stronger than names, because a smallcase order fires exactly the constituent
set of the active version, and each rebalance's additions and removals show up
as first buys and sell-to-zero events.

Any assignment won by a small margin is reported for review rather than trusted.
"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_STOP = {"ltd", "limited", "ltd.", "co", "company", "corporation", "corp",
         "the", "of", "plc", "private", "pvt", "inc"}


def _tokens(name: str) -> List[str]:
    s = name.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return [w for w in s.split() if w and w not in _STOP]


def _align_score(sym: str, toks: tuple) -> float:
    """Best fraction of the symbol explained by in-order prefixes of the name's
    tokens. Handles AUBANK <- AU Small Finance BANK and CANBK <- CANara BanK."""
    n, m = len(sym), len(toks)
    if n == 0 or m == 0:
        return 0.0

    @lru_cache(maxsize=None)
    def f(i, j):
        if i == n:
            return 0.0
        if j == m:
            return -2.0 * (n - i)
        best = f(i, j + 1) - 0.15                     # skip this token
        tok = toks[j]
        for k in range(1, min(len(tok), n - i) + 1):
            if sym[i:i + k] != tok[:k]:
                break
            bonus = k + (0.6 if k == len(tok) else 0.0) + (0.5 if (i == 0 and j == 0) else 0.0)
            best = max(best, bonus + f(i + k, j + 1))
        return best

    return max(0.0, f(0, 0)) / n


def _acronym_score(sym: str, toks: tuple) -> float:
    """ONGC <- Oil Natural Gas Corporation."""
    ini = "".join(t[0] for t in toks)
    if sym == ini:
        return 1.0
    if len(sym) >= 3 and ini.startswith(sym):
        return 0.85
    if len(ini) >= 3 and sym.startswith(ini):
        return 0.80
    return 0.0


def name_similarity(name: str, symbol: str) -> float:
    toks = tuple(_tokens(name))
    if not toks:
        return 0.0
    sym = re.sub(r"-e$", "", symbol.lower())
    return max(_align_score(sym, toks), _acronym_score(sym, toks))


def resolve(events: List[dict], constituents: pd.DataFrame,
            overrides: Dict[str, str] | None = None,
            margin_warn: float = 0.25) -> tuple:
    """
    events: output of events.detect_events (needs keys buy_qty, sell_qty,
            held_before, held_after, active_names, added_names, removed_names)
    Returns (mapping, diagnostics_frame).
    """
    overrides = {k: v.upper() for k, v in (overrides or {}).items()}

    all_syms: Set[str] = set()
    for e in events:
        all_syms |= set(e["buy_qty"]) | set(e["sell_qty"])
    all_names: Set[str] = set()
    for e in events:
        all_names |= set(e["active_names"])

    names = sorted(n for n in all_names if n not in overrides)
    syms = sorted(s for s in all_syms if s not in set(overrides.values()))
    if not names or not syms:
        return dict(overrides), pd.DataFrame(
            columns=["constituent_name", "symbol", "score", "margin",
                     "name_similarity", "confidence"])

    # events where the whole basket was bought and its size equals the version
    # size: the symbol set and the name set must then coincide exactly.
    full = [e for e in events
            if not e["sell_qty"] and len(e["buy_qty"]) == len(e["active_names"])]
    reb = [e for e in events if e["sell_qty"]]

    S = np.zeros((len(names), len(syms)))
    for i, n in enumerate(names):
        for j, s in enumerate(syms):
            f_full = np.mean([(s in e["buy_qty"]) == (n in e["active_names"])
                              for e in full]) if full else 0.0
            f_hold = np.mean([(s in e["held_after"]) == (n in e["active_names"])
                              for e in events])
            f_add = np.mean([(s in e["entered"]) == (n in e["added_names"])
                             for e in reb]) if reb else 0.0
            f_exit = np.mean([(s in e["exited"]) == (n in e["removed_names"])
                              for e in reb]) if reb else 0.0
            S[i, j] = (2.0 * f_full + 2.0 * f_hold + 3.0 * f_add + 3.0 * f_exit
                       + 3.0 * name_similarity(n, s))

    ri, ci = linear_sum_assignment(-S)
    mapping = dict(overrides)
    diag = []
    base = S[ri, ci].sum()
    for i, j in zip(ri, ci):
        mapping[names[i]] = syms[j]
    # margin = loss in total score if this name were forced elsewhere
    assign = {i: j for i, j in zip(ri, ci)}
    for i, j in assign.items():
        best_alt = -np.inf
        for i2, j2 in assign.items():
            if i2 == i:
                continue
            alt = base - S[i, j] - S[i2, j2] + S[i, j2] + S[i2, j]
            best_alt = max(best_alt, alt)
        margin = base - best_alt if np.isfinite(best_alt) else np.inf
        diag.append(dict(constituent_name=names[i], symbol=syms[j],
                         score=round(float(S[i, j]), 4),
                         margin=round(float(margin), 4),
                         name_similarity=round(name_similarity(names[i], syms[j]), 4),
                         confidence="review" if margin < margin_warn else "high"))
    for n, s in overrides.items():
        diag.append(dict(constituent_name=n, symbol=s, score=np.nan, margin=np.inf,
                         name_similarity=round(name_similarity(n, s), 4),
                         confidence="user override"))
    d = pd.DataFrame(diag).sort_values(["confidence", "margin"]).reset_index(drop=True)
    return mapping, d
