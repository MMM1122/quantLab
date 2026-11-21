"""Small, explicit statistics helpers (seeded, deterministic)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import stats


def mean_ci(values, level: float = 0.95) -> dict[str, float | int | None]:
    """Student-t confidence interval for the mean across independent seeds."""
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    m = float(x.mean())
    if n == 1:
        return {"n": 1, "mean": m, "std": None, "ci_low": None, "ci_high": None}
    sd = float(x.std(ddof=1))
    h = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n))
    return {"n": n, "mean": m, "std": sd, "ci_low": m - h, "ci_high": m + h}


def sharpe(returns: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def block_bootstrap_ci(
    returns: np.ndarray,
    stat: Callable[[np.ndarray], float] = sharpe,
    n_boot: int = 1000,
    block: int = 21,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Moving-block bootstrap percentile CI (preserves short-range autocorrelation)."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < block * 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        out[b] = stat(sample)
    lo, hi = np.nanpercentile(out, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
