"""Missing/bad-data treatment policies for DQ-001 (all strictly causal).

Every policy turns *observations as delivered* into the close series a strategy sees.
Backward filling or interpolation with future values is never used.

* ``strict_skip`` – accept only fresh observations (as-of == session) that pass a jump filter;
  otherwise the strategy gets NaN and expresses **no opinion** (engine holds the position).
  Indicator windows average whatever valid observations exist (min coverage 90%).
* ``bounded_ffill`` – same acceptance tests, then forward-fill the last accepted *historical*
  value for at most ``max_ffill`` sessions; beyond that, behaves like strict_skip.
* ``naive_ffill`` – RESEARCH-ONLY COUNTERFACTUAL: accepts every delivered value (stale or
  corrupted) as fresh and forward-fills gaps without limit. Shows the cost of ignoring data
  quality; never a tradable policy.

Jump filter: an observation moving more than ``jump_threshold`` from the last accepted value is
held back; if the *next* observation confirms the new level (within ``confirm_tolerance``) it is
accepted as a genuine move, so real regime shifts are not rejected forever.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantlab.core.labels import RESEARCH_ONLY_LABEL


@dataclass(frozen=True)
class Policy:
    name: str
    use_asof: bool
    jump_filter: bool
    max_ffill: int | None  # None = unbounded; 0 = no fill
    research_only: bool = False
    jump_threshold: float = 0.10
    confirm_tolerance: float = 0.02

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.research_only:
            d["label"] = RESEARCH_ONLY_LABEL
        return d


POLICIES = {
    "strict_skip": Policy("strict_skip", use_asof=True, jump_filter=True, max_ffill=0),
    "bounded_ffill": Policy("bounded_ffill", use_asof=True, jump_filter=True, max_ffill=3),
    "naive_ffill": Policy("naive_ffill", use_asof=False, jump_filter=False, max_ffill=None, research_only=True),
}


@dataclass
class ObservedSeries:
    value: np.ndarray  # delivered close per session (NaN if no bar arrived)
    asof: np.ndarray  # grid index the value truly belongs to (== own index if fresh)


def observed_from_events(grid: pd.DatetimeIndex, perturbed: pd.DataFrame, events: pd.DataFrame,
                         symbol: str) -> ObservedSeries:
    g = perturbed[perturbed["symbol"] == symbol].set_index("session_date")["close"].reindex(grid)
    value = g.to_numpy(dtype=float)
    asof = np.arange(len(grid), dtype=float)
    asof[~np.isfinite(value)] = np.nan
    if len(events):
        ev = events[(events["symbol"] == symbol) & (events["kind"] == "delayed")]
        pos = grid.get_indexer(pd.DatetimeIndex(ev["session_date"]))
        for p, lag in zip(pos, ev["lag"]):
            if p >= 0:
                asof[p] = p - int(lag)
    return ObservedSeries(value, asof)


def apply_policy(obs: ObservedSeries, policy: Policy) -> tuple[np.ndarray, dict]:
    """Return (series the strategy sees, diagnostics). Single forward pass => causal."""
    n = len(obs.value)
    out = np.full(n, np.nan)
    last_acc = np.nan
    last_acc_i = -(10**9)
    held_back = np.nan
    diag = {"stale_rejected": 0, "jump_rejected": 0, "jump_confirmed": 0, "filled": 0, "no_opinion": 0}
    for t in range(n):
        v = obs.value[t]
        accept = np.isfinite(v)
        if accept and policy.use_asof and obs.asof[t] != t:
            accept = False
            diag["stale_rejected"] += 1
        if accept and policy.jump_filter and np.isfinite(last_acc) and abs(v / last_acc - 1) > policy.jump_threshold:
            if np.isfinite(held_back) and abs(v / held_back - 1) <= policy.confirm_tolerance:
                diag["jump_confirmed"] += 1  # two consecutive observations agree -> genuine move
            else:
                held_back = v
                accept = False
                diag["jump_rejected"] += 1
        if accept:
            out[t] = v
            last_acc, last_acc_i = v, t
            held_back = np.nan
            continue
        age = t - last_acc_i
        if np.isfinite(last_acc) and (policy.max_ffill is None or age <= policy.max_ffill):
            out[t] = last_acc
            diag["filled"] += 1
        else:
            diag["no_opinion"] += 1
    return out, diag
