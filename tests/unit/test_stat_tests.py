from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantlab.core.errors import InputValidationError
from quantlab.research.stat_tests import (
    compare_strategies,
    deflated_sharpe,
    expected_max_sharpe,
    paired_block_bootstrap,
    probabilistic_sharpe,
    sharpe_difference_test,
)

IDX = pd.bdate_range("2015-01-01", periods=2000)


def _series(mu, sd, seed, n=2000):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=IDX[:n])


def test_jk_identical_series_is_not_significant():
    r = _series(0.0004, 0.01, 1)
    t = sharpe_difference_test(r, r)
    assert t["z"] == 0 and t["p_value"] == 1.0 and t["correlation"] == pytest.approx(1.0)


def test_jk_matches_memmel_formula_and_detects_large_difference():
    a, b = _series(0.002, 0.01, 2), _series(-0.001, 0.01, 3)
    t = sharpe_difference_test(a, b)
    s1, s2 = a.mean() / a.std(), b.mean() / b.std()
    rho = np.corrcoef(a, b)[0, 1]
    theta = (2 - 2 * rho + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / len(a)
    assert t["z"] == pytest.approx((s1 - s2) / math.sqrt(theta))
    assert t["p_value"] < 1e-6 and t["sharpe_diff"] > 0
    rev = sharpe_difference_test(b, a)
    assert rev["z"] == pytest.approx(-t["z"]) and rev["p_value"] == pytest.approx(t["p_value"])


def test_bootstrap_is_seeded_and_brackets_the_difference():
    a, b = _series(0.001, 0.01, 4), _series(0.0002, 0.01, 5)
    x = paired_block_bootstrap(a, b, n_boot=500, seed=9)
    y = paired_block_bootstrap(a, b, n_boot=500, seed=9)
    assert x == y
    assert x["ci_low"] < x["sharpe_diff"] < x["ci_high"]
    same = paired_block_bootstrap(a, a, n_boot=200, seed=1)
    assert same["ci_low"] == same["ci_high"] == 0.0


def test_psr_and_dsr_behaviour():
    good = _series(0.001, 0.01, 6)
    assert probabilistic_sharpe(good) > 0.99
    noise = _series(0.0, 0.01, 7)
    assert 0.0 < probabilistic_sharpe(noise) < 1.0
    one = deflated_sharpe(good, 1)
    assert one["deflated_sharpe"] == pytest.approx(one["psr_vs_zero"])  # N=1 -> PSR
    many = deflated_sharpe(good, 100)
    assert many["deflated_sharpe"] < one["deflated_sharpe"]
    assert many["expected_max_sharpe_null"] > 0
    # hand check of the False Strategy Theorem with unit variance
    n = 10
    exp = (1 - 0.5772156649015329) * stats.norm.ppf(1 - 1 / n) + 0.5772156649015329 * stats.norm.ppf(1 - 1 / (n * math.e))
    assert expected_max_sharpe(n, 1.0) == pytest.approx(exp)
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_compare_strategies_uses_common_window():
    navs = {
        "a": (1 + _series(0.0008, 0.01, 8)).cumprod(),
        "b": (1 + _series(0.0002, 0.01, 9)).cumprod().iloc[100:],
        "c": (1 + _series(0.0001, 0.012, 10)).cumprod(),
    }
    out = compare_strategies(navs, n_boot=200, seed=3)
    assert out["n_obs"] == 2000 - 100 - 1 and out["n_trials"] == 3 and out["reference"] == "a"
    assert [p["strategy"] for p in out["pairwise"]] == ["b", "c"]
    assert set(out["deflated_sharpe"]) == {"a", "b", "c"}
    with pytest.raises(InputValidationError):
        compare_strategies({"a": navs["a"]})
    with pytest.raises(InputValidationError):
        compare_strategies({"a": navs["a"].iloc[:10], "b": navs["b"]})
