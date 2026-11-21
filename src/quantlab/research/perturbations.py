"""Seeded, reproducible data perturbations for DQ-001.

Perturbations never modify the source snapshot. They return a *derived* bar set plus an event
log describing exactly what was changed (persisted in the derived snapshot's manifest).

Kinds
* ``missing``   – the bar never arrives (row removed).
* ``delayed``   – the vendor delivers a stale bar for session t: its values are the true bar of
  session t-k (k in 1..max_lag), but it carries an as-of stamp of t-k. A timestamp-aware
  policy can detect it; a naive one treats it as fresh.
* ``corrupted`` – close/adj_close multiplied by (1 ± u), u ~ U(min, max); OHLC bounds are kept
  consistent so the corruption is not trivially detectable by schema checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    missing_rate: float = 0.0
    delay_rate: float = 0.0
    delay_max_lag: int = 3
    corrupt_rate: float = 0.0
    corrupt_min: float = 0.05
    corrupt_max: float = 0.25

    @property
    def is_clean(self) -> bool:
        return self.missing_rate == 0 and self.delay_rate == 0 and self.corrupt_rate == 0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_SCENARIOS = [
    Scenario("clean"),
    Scenario("missing_1pct", missing_rate=0.01),
    Scenario("missing_5pct", missing_rate=0.05),
    Scenario("missing_10pct", missing_rate=0.10),
    Scenario("delayed_5pct", delay_rate=0.05, delay_max_lag=3),
    Scenario("corrupted_2pct", corrupt_rate=0.02),
]


def _stable_int(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def perturb(
    bars: pd.DataFrame,
    scenario: Scenario,
    seed: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (perturbed_bars, events). Only sessions inside the window are perturbed."""
    events = []
    out_frames = []
    for sym, g in bars.groupby("symbol", sort=True):
        g = g.sort_values("session_date").reset_index(drop=True).copy()
        if scenario.is_clean:
            out_frames.append(g)
            continue
        rng = np.random.default_rng([seed, _stable_int(scenario.name, sym)])
        in_win = np.where((g["session_date"] >= window_start) & (g["session_date"] <= window_end))[0]
        # never perturb the very first rows of the series (a delayed bar needs a predecessor)
        in_win = in_win[in_win >= scenario.delay_max_lag]
        n = len(in_win)
        order = rng.permutation(in_win)
        n_miss = int(round(scenario.missing_rate * n))
        n_delay = int(round(scenario.delay_rate * n))
        n_corr = int(round(scenario.corrupt_rate * n))
        miss_idx = np.sort(order[:n_miss])
        delay_idx = np.sort(order[n_miss : n_miss + n_delay])
        corr_idx = np.sort(order[n_miss + n_delay : n_miss + n_delay + n_corr])

        value_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        orig = g[value_cols].copy()
        for i in delay_idx:
            lag = int(rng.integers(1, scenario.delay_max_lag + 1))
            g.loc[i, value_cols] = orig.loc[i - lag, value_cols].to_numpy()
            events.append({"symbol": sym, "session_date": g.loc[i, "session_date"], "kind": "delayed", "lag": lag,
                           "factor": np.nan})
        for i in corr_idx:
            u = rng.uniform(scenario.corrupt_min, scenario.corrupt_max)
            f = 1 + u if rng.random() < 0.5 else 1 - u
            g.loc[i, ["close", "adj_close"]] = g.loc[i, ["close", "adj_close"]].to_numpy() * f
            g.loc[i, "high"] = max(g.loc[i, "high"], g.loc[i, "close"], g.loc[i, "open"])
            g.loc[i, "low"] = min(g.loc[i, "low"], g.loc[i, "close"], g.loc[i, "open"])
            events.append({"symbol": sym, "session_date": g.loc[i, "session_date"], "kind": "corrupted", "lag": 0,
                           "factor": float(f)})
        for i in miss_idx:
            events.append({"symbol": sym, "session_date": g.loc[i, "session_date"], "kind": "missing", "lag": 0,
                           "factor": np.nan})
        g = g.drop(index=miss_idx)
        out_frames.append(g)
    ev = pd.DataFrame(events, columns=["symbol", "session_date", "kind", "lag", "factor"])
    if len(ev):
        ev = ev.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    return pd.concat(out_frames, ignore_index=True), ev
