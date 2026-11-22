from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.market import build_panel
from quantlab.core.errors import InputValidationError
from quantlab.strategies.registry import STRATEGIES, create_strategy, list_strategies, normalize_strategy_id
from quantlab.strategies.rsi_mean_reversion import rsi

FAST_PARAMS = {
    "buy-hold": {},
    "sma-crossover": {"fast": 10, "slow": 40},
    "rsi-mean-reversion": {"window": 14, "entry": 35, "exit": 55, "max_hold": 10},
    "momentum": {"lookback": 60, "skip": 5, "top_k": 2},
    "vol-target": {"base": "sma-crossover", "base_params": {"fast": 10, "slow": 40}, "window": 20},
}


@pytest.fixture(scope="module")
def prices(synth_bars):
    return build_panel(synth_bars, "unadjusted").signal_price


@pytest.mark.parametrize("sid", sorted(STRATEGIES))
@pytest.mark.parametrize("params_kind", ["default", "fast"])
def test_strategy_is_causal(prices, sid, params_kind):
    """Output at row t must be identical whether or not rows > t exist (no look-ahead)."""
    strat = create_strategy(sid, FAST_PARAMS[sid] if params_kind == "fast" else None)
    full = strat.target_weights(prices)
    for k in (60, 255, 300, 517, 900, len(prices) - 3):
        part = strat.target_weights(prices.iloc[:k])
        pd.testing.assert_frame_equal(full.iloc[:k], part, check_freq=False)


@pytest.mark.parametrize("sid", sorted(STRATEGIES))
def test_future_changes_do_not_alter_past_signals(prices, sid):
    strat = create_strategy(sid, FAST_PARAMS[sid])
    base = strat.target_weights(prices)
    k = 400
    shocked = prices.copy()
    shocked.iloc[k + 1 :] *= 0.5  # crash after k
    out = strat.target_weights(shocked)
    pd.testing.assert_frame_equal(base.iloc[: k + 1], out.iloc[: k + 1], check_freq=False)


@pytest.mark.parametrize("sid", sorted(STRATEGIES))
def test_weights_long_only_no_leverage(prices, sid):
    w = create_strategy(sid, FAST_PARAMS[sid]).target_weights(prices)
    vals = w.to_numpy()
    fin = vals[np.isfinite(vals)]
    assert (fin >= 0).all() and (fin <= 1).all()
    assert (w.fillna(0).sum(axis=1) <= 1 + 1e-12).all()


def test_nan_input_yields_no_opinion(prices):
    p = prices.copy()
    p.iloc[500, 0] = np.nan
    for sid in ("sma-crossover", "rsi-mean-reversion", "buy-hold"):
        w = create_strategy(sid, FAST_PARAMS[sid]).target_weights(p)
        assert np.isnan(w.iloc[500, 0]), sid


def test_sma_crossover_signal_logic():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.DataFrame({"X": np.r_[np.linspace(100, 80, 30), np.linspace(80, 120, 30)]}, index=idx)
    w = create_strategy("sma-crossover", {"fast": 3, "slow": 10, "min_coverage": 1.0}).target_weights(up)
    assert w["X"].iloc[:9].isna().all()  # warm-up
    assert w["X"].iloc[25] == 0.0 and w["X"].iloc[-1] == 1.0


def test_rsi_values_and_strategy_entry_exit():
    idx = pd.bdate_range("2020-01-01", periods=40)
    px = pd.DataFrame({"X": np.r_[np.linspace(100, 70, 20), np.linspace(70, 100, 20)]}, index=idx)
    r = rsi(px, 5, 1.0)
    assert r["X"].iloc[10] == 0.0  # only losses
    assert r["X"].iloc[-1] == 100.0  # only gains
    w = create_strategy("rsi-mean-reversion", {"window": 5, "entry": 30, "exit": 55, "max_hold": 50}).target_weights(px)
    assert w["X"].iloc[10] == 1.0  # oversold -> long
    assert w["X"].iloc[-1] == 0.0  # recovered -> flat


def test_rsi_time_stop():
    idx = pd.bdate_range("2020-01-01", periods=40)
    px = pd.DataFrame({"X": np.linspace(100, 60, 40)}, index=idx)  # persistently oversold
    w = create_strategy("rsi-mean-reversion", {"window": 5, "entry": 30, "exit": 55, "max_hold": 3}).target_weights(px)
    # keeps re-entering because each session is a fresh oversold reading
    assert (w["X"].iloc[6:] == 1.0).all()


def test_momentum_rebalances_monthly_and_picks_leaders():
    idx = pd.bdate_range("2020-01-01", periods=120)
    px = pd.DataFrame(
        {"A": np.linspace(100, 200, 120), "B": np.linspace(100, 150, 120), "C": np.linspace(100, 50, 120)}, index=idx
    )
    w = create_strategy("momentum", {"lookback": 40, "skip": 5, "top_k": 2}).target_weights(px)
    decided = w.dropna(how="all")
    assert all(d.day <= 3 for d in decided.index)  # first session of month
    last = decided.iloc[-1]
    assert last["A"] == 0.5 and last["B"] == 0.5 and last["C"] == 0.0


def test_registry_aliases_and_validation():
    assert normalize_strategy_id("ma-crossover") == "sma-crossover"
    assert {s["id"] for s in list_strategies()} == set(STRATEGIES)
    with pytest.raises(InputValidationError):
        create_strategy("does-not-exist")
    with pytest.raises(InputValidationError):
        create_strategy("sma-crossover", {"fast": 200, "slow": 50})
    with pytest.raises(InputValidationError):
        create_strategy("sma-crossover", {"bogus": 1})


def test_vol_target_scales_down_never_levers_and_reduces_volatility(synth_bars):
    from quantlab.backtest.engine import BacktestConfig, run_backtest

    panel = build_panel(synth_bars, "unadjusted", symbols=["QQQ"])
    vt = create_strategy("vol-target", {"target_vol": 0.08, "window": 63, "step": 0.1})
    w = vt.target_weights(panel.signal_price)["QQQ"].dropna()
    assert (w <= 1.0).all() and (w >= 0).all()
    assert np.allclose(w * 10, np.round(w * 10))  # quantised to 0.1 steps
    assert w.mean() < 1.0  # QQQ fixture vol (~22%) > 8% target -> scaled down
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0, start="2019-01-02")
    raw = run_backtest(panel, create_strategy("buy-hold"), cfg).equity.nav.pct_change().std()
    tgt = run_backtest(panel, vt, cfg).equity.nav.pct_change().std()
    assert tgt < raw * 0.75
    with pytest.raises(InputValidationError):
        create_strategy("vol-target", {"base": "vol-target"})
    with pytest.raises(InputValidationError):
        create_strategy("vol-target", {"target_vol": 2.0})
