"""Statistical tests for comparing strategies (daily returns; all deterministic given a seed).

* ``sharpe_difference_test`` — Jobson & Korkie (1981) z-test with Memmel's (2003) correction for
  H0: SR_a = SR_b on *paired* daily returns. Assumes iid, approximately normal returns.
* ``paired_block_bootstrap`` — circular block bootstrap of the paired return series (keeps
  cross-correlation and short-range autocorrelation) for the Sharpe difference: percentile CI and
  a two-sided sign p-value. Fewer distributional assumptions than the z-test.
* ``probabilistic_sharpe`` / ``deflated_sharpe`` — Bailey & López de Prado (2012, 2014). PSR is the
  probability that the true Sharpe exceeds a benchmark given sample length, skew and kurtosis.
  DSR raises the benchmark to the Sharpe expected from the *best of N trials* under the null,
  correcting for multiple testing (e.g. a parameter grid or many compared strategies).

Sharpe values inside the formulas are per-period (daily); outputs also report annualised values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.core.errors import InputValidationError

PERIODS = 252
EULER_GAMMA = 0.5772156649015329


def _sr(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def _paired(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 30:
        raise InputValidationError("need at least 30 overlapping return observations")
    return df["a"].to_numpy(float), df["b"].to_numpy(float)


def sharpe_difference_test(ra: pd.Series, rb: pd.Series) -> dict[str, Any]:
    a, b = _paired(ra, rb)
    t = len(a)
    s1, s2 = _sr(a), _sr(b)
    rho = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    theta = (2 - 2 * rho + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / t
    if theta <= 0:
        z, p = 0.0, 1.0
    else:
        z = (s1 - s2) / math.sqrt(theta)
        p = float(2 * stats.norm.sf(abs(z)))
    return {"test": "Jobson-Korkie (Memmel-corrected)", "n_obs": t, "sharpe_a": s1 * math.sqrt(PERIODS),
            "sharpe_b": s2 * math.sqrt(PERIODS), "sharpe_diff": (s1 - s2) * math.sqrt(PERIODS),
            "correlation": rho, "z": float(z), "p_value": float(p)}


def paired_block_bootstrap(ra: pd.Series, rb: pd.Series, n_boot: int = 2000, block: int = 21,
                           seed: int = 0, level: float = 0.95) -> dict[str, Any]:
    a, b = _paired(ra, rb)
    n = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n  # circular blocks
        diffs[k] = (_sr(a[idx]) - _sr(b[idx])) * math.sqrt(PERIODS)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {"test": "paired circular block bootstrap", "n_obs": n, "n_boot": int(len(diffs)), "block": block,
            "seed": seed, "sharpe_diff": (_sr(a) - _sr(b)) * math.sqrt(PERIODS), "ci_low": float(lo),
            "ci_high": float(hi), "p_value": p}


def probabilistic_sharpe(r: pd.Series, sr_benchmark_daily: float = 0.0) -> float:
    x = pd.Series(r).dropna().to_numpy(float)
    t = len(x)
    if t < 30:
        raise InputValidationError("need at least 30 return observations")
    sr = _sr(x)
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=False, bias=False))  # non-excess
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if not np.isfinite(sr) or denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark_daily) * math.sqrt(t - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_sr_daily: float) -> float:
    """Expected maximum daily Sharpe among N independent zero-skill trials (False Strategy Theorem)."""
    if n_trials <= 1 or var_sr_daily <= 0:
        return 0.0
    return math.sqrt(var_sr_daily) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                                      + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpes_annual: list[float] | None = None) -> dict[str, Any]:
    """DSR for a strategy selected among ``n_trials``. The variance of Sharpe across trials comes
    from ``trial_sharpes_annual`` when given; otherwise the iid sampling variance (1/T) is used,
    which is the minimum plausible dispersion (conservative only if trials are independent)."""
    if n_trials < 1:
        raise InputValidationError("n_trials must be >= 1")
    x = pd.Series(r).dropna().to_numpy(float)
    t = len(x)
    if trial_sharpes_annual and len(trial_sharpes_annual) >= 2:
        daily = np.asarray(trial_sharpes_annual, float) / math.sqrt(PERIODS)
        var_sr = float(np.var(daily[np.isfinite(daily)], ddof=1))
        source = "dispersion of the supplied trial Sharpe ratios"
    else:
        var_sr = 1.0 / max(t - 1, 1)
        source = "iid sampling variance 1/(T-1) (no trial Sharpe ratios supplied)"
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return {"n_trials": n_trials, "n_obs": t, "sharpe": _sr(x) * math.sqrt(PERIODS),
            "psr_vs_zero": probabilistic_sharpe(x, 0.0),
            "expected_max_sharpe_null": sr0 * math.sqrt(PERIODS),
            "deflated_sharpe": probabilistic_sharpe(x, sr0), "trial_variance_source": source}


def compare_strategies(navs: dict[str, pd.Series], n_trials: int | None = None, seed: int = 0,
                       n_boot: int = 2000) -> dict[str, Any]:
    """Pairwise tests of every strategy against the first one + DSR for each.

    ``n_trials`` defaults to the number of strategies compared; pass the true number of
    configurations you tried (including discarded ones) for an honest correction.
    """
    if len(navs) < 2:
        raise InputValidationError("compare needs at least two equity curves")
    rets = {k: v.astype(float).pct_change().dropna() for k, v in navs.items()}
    names = list(rets)
    common = rets[names[0]].index
    for k in names[1:]:
        common = common.intersection(rets[k].index)
    if len(common) < 30:
        raise InputValidationError("runs overlap on fewer than 30 sessions; use runs on the same window")
    rets = {k: v.loc[common] for k, v in rets.items()}
    n = n_trials or len(names)
    sharpes = [_sr(v.to_numpy()) * math.sqrt(PERIODS) for v in rets.values()]
    ref = names[0]
    pairwise = []
    for k in names[1:]:
        jk = sharpe_difference_test(rets[k], rets[ref])
        bs = paired_block_bootstrap(rets[k], rets[ref], n_boot=n_boot, seed=seed)
        pairwise.append({"strategy": k, "versus": ref, "sharpe_diff": jk["sharpe_diff"], "jk_z": jk["z"],
                         "jk_p_value": jk["p_value"], "bootstrap_ci_low": bs["ci_low"],
                         "bootstrap_ci_high": bs["ci_high"], "bootstrap_p_value": bs["p_value"],
                         "correlation": jk["correlation"]})
    dsr = {k: deflated_sharpe(rets[k], n, sharpes) for k in names}
    return {"window": [common[0].strftime("%Y-%m-%d"), common[-1].strftime("%Y-%m-%d")], "n_obs": len(common),
            "n_trials": n, "reference": ref, "pairwise": pairwise, "deflated_sharpe": dsr, "seed": seed,
            "notes": ["Jobson-Korkie assumes iid normal returns; the block bootstrap relaxes this.",
                      "Deflated Sharpe corrects for selecting the best of n_trials; pass the real number of "
                      "configurations tried, not just the ones shown."]}
