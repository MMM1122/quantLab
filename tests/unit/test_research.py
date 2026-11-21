from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.core.errors import InputValidationError
from quantlab.research.perturbations import Scenario, perturb
from quantlab.research.policies import POLICIES, ObservedSeries, apply_policy, observed_from_events
from quantlab.research.splits import chronological_split, walk_forward
from quantlab.research.statistics import block_bootstrap_ci, mean_ci


@pytest.fixture(scope="module")
def spy(synth_bars):
    return synth_bars[synth_bars.symbol == "SPY"].reset_index(drop=True)


def _win(b):
    return b.session_date.iloc[300], b.session_date.iloc[-1]


def test_perturbation_reproducible_and_seed_dependent(spy):
    sc = Scenario("mix", missing_rate=0.05, delay_rate=0.03, corrupt_rate=0.02)
    original = spy.copy()
    a, ea = perturb(spy, sc, 11, *_win(spy))
    b, eb = perturb(spy, sc, 11, *_win(spy))
    c, ec = perturb(spy, sc, 12, *_win(spy))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ea, eb)
    assert not ea.equals(ec)
    pd.testing.assert_frame_equal(spy, original)  # source never mutated
    n_win = ((spy.session_date >= _win(spy)[0]) & (spy.session_date <= _win(spy)[1])).sum()
    counts = ea.kind.value_counts()
    assert counts["missing"] == round(0.05 * n_win)
    assert counts["delayed"] == round(0.03 * n_win)
    assert counts["corrupted"] == round(0.02 * n_win)
    assert len(a) == len(spy) - counts["missing"]
    assert (ea.session_date >= _win(spy)[0]).all()  # only inside the window
    assert not ea.duplicated(["symbol", "session_date"]).any()  # disjoint selections


def test_delayed_bar_carries_old_values(spy):
    pert, ev = perturb(spy, Scenario("d", delay_rate=0.05, delay_max_lag=3), 5, *_win(spy))
    for _, e in ev.iterrows():
        i = spy.index[spy.session_date == e.session_date][0]
        row = pert[pert.session_date == e.session_date].iloc[0]
        assert row.close == spy.loc[i - e.lag, "close"]


def _obs(values, asof=None):
    v = np.asarray(values, dtype=float)
    a = np.arange(len(v), dtype=float) if asof is None else np.asarray(asof, dtype=float)
    a[~np.isfinite(v)] = np.nan
    return ObservedSeries(v, a)


def test_strict_policy_rejects_stale_and_never_fills():
    o = _obs([100, 101, np.nan, 99, 102], asof=[0, 1, 2, 2, 4])  # t=3 is a stale copy of t=2's slot
    out, d = apply_policy(o, POLICIES["strict_skip"])
    assert np.isnan(out[2]) and np.isnan(out[3]) and out[4] == 102
    assert d["stale_rejected"] == 1


def test_bounded_fill_limit_and_naive_unbounded():
    vals = [100, np.nan, np.nan, np.nan, np.nan, np.nan, 103]
    b, _ = apply_policy(_obs(vals), POLICIES["bounded_ffill"])
    n, _ = apply_policy(_obs(vals), POLICIES["naive_ffill"])
    assert list(b[1:4]) == [100, 100, 100] and np.isnan(b[4]) and np.isnan(b[5])
    assert (n[1:6] == 100).all()


def test_naive_policy_trusts_stale_and_corrupted_values():
    o = _obs([100, 101, 140, 102], asof=[0, 1, 2, 3])
    naive, _ = apply_policy(o, POLICIES["naive_ffill"])
    strict, d = apply_policy(o, POLICIES["strict_skip"])
    assert naive[2] == 140
    assert np.isnan(strict[2]) and strict[3] == 102 and d["jump_rejected"] == 1
    assert POLICIES["naive_ffill"].to_dict()["label"].startswith("RESEARCH-ONLY")


def test_jump_filter_accepts_confirmed_level_shift():
    o = _obs([100, 100, 130, 131, 132])
    out, d = apply_policy(o, POLICIES["strict_skip"])
    assert np.isnan(out[2]) and out[3] == 131 and out[4] == 132 and d["jump_confirmed"] == 1


def test_policies_are_causal_and_never_backfill(spy):
    pert, ev = perturb(spy, Scenario("m", missing_rate=0.1, delay_rate=0.05, corrupt_rate=0.02), 3, *_win(spy))
    grid = pd.DatetimeIndex(spy.session_date)
    obs = observed_from_events(grid, pert, ev, "SPY")
    for p in POLICIES.values():
        full, _ = apply_policy(obs, p)
        for k in (310, 500, 1000):
            part, _ = apply_policy(ObservedSeries(obs.value[:k], obs.asof[:k]), p)
            np.testing.assert_array_equal(full[:k], part)
    first_missing = _obs([np.nan, np.nan, 100, 101])
    for p in POLICIES.values():
        out, _ = apply_policy(first_missing, p)
        assert np.isnan(out[0]) and np.isnan(out[1])  # nothing filled from the future


def test_mean_ci_and_bootstrap():
    ci = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert ci["mean"] == 2.5 and ci["n"] == 4
    assert ci["ci_low"] == pytest.approx(2.5 - 3.182446 * np.std([1, 2, 3, 4], ddof=1) / 2, rel=1e-5)
    assert mean_ci([5.0])["ci_low"] is None
    r = np.random.default_rng(0).normal(0.0005, 0.01, 1000)
    assert block_bootstrap_ci(r, seed=1) == block_bootstrap_ci(r, seed=1)
    lo, hi = block_bootstrap_ci(r, seed=1)
    assert lo < hi


def test_chronological_split_has_no_overlap_or_leakage():
    s = pd.bdate_range("2015-01-01", periods=1000)
    segs = chronological_split(s, 0.6, 0.2, embargo=5)
    assert [x.name for x in segs] == ["train", "validation", "test"]
    for a, b in zip(segs, segs[1:]):
        assert a.end < b.start
        gap = s.get_loc(b.start) - s.get_loc(a.end) - 1
        assert gap == 5
    folds = walk_forward(s, train_size=500, test_size=100, embargo=2)
    for tr, te in folds:
        assert tr.end < te.start
    assert all(f1[1].end < f2[1].start for f1, f2 in zip(folds, folds[1:]))
    with pytest.raises(InputValidationError):
        chronological_split(s, 0.9, 0.2)
