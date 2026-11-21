from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import BacktestConfig, run_backtest
from quantlab.backtest.execution import CostModel
from quantlab.backtest.market import build_panel
from quantlab.backtest.metrics import performance_metrics
from quantlab.core.errors import InputValidationError
from quantlab.strategies.base import Strategy, StrategyParams
from quantlab.strategies.buy_hold import BuyAndHold
from quantlab.strategies.registry import create_strategy
from tests.conftest import make_bars


@dataclass
class _P(StrategyParams):
    pass


class Scripted(Strategy):
    """Target weight by row position (independent of prices => trivially causal)."""

    id = "scripted"
    name = "scripted"
    version = "test"
    description = "test"

    def __init__(self, schedule: dict[int, float], symbol: str = "SPY"):
        self.schedule = schedule
        self.symbol = symbol
        self.params = _P()

    @property
    def warmup_sessions(self) -> int:
        return 0

    def target_weights(self, prices):
        out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for i, w in self.schedule.items():
            if i < len(out):
                out.iloc[i, out.columns.get_loc(self.symbol)] = w
        return out


def test_round_trip_nav_is_exact():
    closes = [100.0] * 6
    opens = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    panel = build_panel(make_bars("SPY", closes, opens=opens), "unadjusted")
    cfg = BacktestConfig(initial_cash=10_000, commission_bps=10, slippage_bps=20)
    res = run_backtest(panel, Scripted({0: 1.0, 2: 0.0}), cfg)
    t = res.trades
    assert list(t.side) == ["BUY", "SELL"]
    buy_px = 101 * 1.002
    q = int(np.floor(10_000 / (buy_px * 1.001)))
    assert t.qty.iloc[0] == q == 98
    cash = 10_000 - q * buy_px - q * buy_px * 0.001
    sell_px = 103 * 0.998
    cash = cash + q * sell_px - q * sell_px * 0.001
    assert res.equity["nav"].iloc[-1] == pytest.approx(cash, abs=1e-9)
    assert res.equity["cash"].iloc[-1] == pytest.approx(cash, abs=1e-9)
    # buy was cash-limited (sized at close 100, filled at 101.2 + fees)
    assert res.orders.status.iloc[0] == "PARTIALLY_FILLED" and res.orders.reason.iloc[0] == "CASH_LIMITED"
    m = performance_metrics(res.equity, res.trades, res.orders)
    assert m["total_commission"] == pytest.approx(t.commission.sum())


def test_no_trade_on_signal_bar_executes_next_open(synth_bars):
    panel = build_panel(synth_bars, "unadjusted", symbols=["SPY"])
    res = run_backtest(panel, create_strategy("sma-crossover", {"fast": 10, "slow": 40}), BacktestConfig())
    assert len(res.trades) > 4
    sess = panel.sessions
    for _, tr in res.trades.iterrows():
        di = sess.get_loc(tr.decision_session)
        assert tr.session == sess[di + 1]  # strictly the next session
        assert tr.session > tr.decision_session
        assert tr.ref_open == pytest.approx(panel.open.loc[tr.session, "SPY"])


def test_next_open_changes_fill_but_next_close_cannot_change_signal(synth_bars):
    spy = synth_bars[synth_bars.symbol == "SPY"].reset_index(drop=True)
    strat = create_strategy("sma-crossover", {"fast": 10, "slow": 40})
    base = run_backtest(build_panel(spy, "unadjusted"), strat, BacktestConfig())
    tr = base.trades.iloc[2]
    exec_day = tr.session

    bumped = spy.copy()
    m = bumped.session_date == exec_day
    bumped.loc[m, "open"] *= 1.01
    bumped.loc[m, "high"] = np.maximum(bumped.loc[m, "high"], bumped.loc[m, "open"])
    r2 = run_backtest(build_panel(bumped, "unadjusted"), strat, BacktestConfig())
    t2 = r2.trades[r2.trades.session == exec_day].iloc[0]
    assert t2.fill_price == pytest.approx(tr.fill_price * 1.01)

    # change the close of the execution day: the decision (made the evening before) is unchanged
    shocked = spy.copy()
    shocked.loc[shocked.session_date == exec_day, ["close", "adj_close", "low"]] *= 0.8
    r3 = run_backtest(build_panel(shocked, "unadjusted"), strat, BacktestConfig())
    dec = tr.decision_session
    pd.testing.assert_frame_equal(base.targets.loc[:dec], r3.targets.loc[:dec])
    assert (r3.trades.decision_session == dec).any()


def test_insufficient_cash_rejects_buy_and_never_overdraws():
    panel = build_panel(make_bars("SPY", [100.0] * 4, opens=[100.0, 200.0, 200.0, 200.0]), "unadjusted")
    res = run_backtest(panel, Scripted({0: 1.0}), BacktestConfig(initial_cash=150, commission_bps=0, slippage_bps=0))
    assert res.trades.empty
    assert res.orders.status.iloc[0] == "REJECTED" and res.orders.reason.iloc[0] == "INSUFFICIENT_CASH"
    assert (res.equity.cash >= 0).all()


def test_sell_never_creates_short():
    closes = [100.0] * 6
    panel = build_panel(make_bars("SPY", closes), "unadjusted")
    res = run_backtest(panel, Scripted({0: 0.5, 2: 0.0, 3: 0.0}), BacktestConfig())
    assert (res.positions["SPY"] >= 0).all()
    assert res.positions["SPY"].iloc[-1] == 0


def test_split_does_not_create_return_spike_and_odd_split_pays_cash_in_lieu():
    closes = [100.0] * 5 + [50.0] * 5
    panel = build_panel(make_bars("QQQ", closes, splits={5: 2.0}), "unadjusted")
    res = run_backtest(panel, BuyAndHold(), BacktestConfig(commission_bps=0, slippage_bps=0))
    nav = res.equity.nav
    assert nav.pct_change().dropna().abs().max() < 1e-12
    assert res.positions["QQQ"].iloc[-1] == 2 * res.positions["QQQ"].iloc[3]

    closes = [99.0] * 5 + [66.0] * 5
    panel = build_panel(make_bars("QQQ", closes, splits={5: 1.5}), "unadjusted")
    res = run_backtest(panel, BuyAndHold(), BacktestConfig(initial_cash=3300, commission_bps=0, slippage_bps=0))
    assert res.positions["QQQ"].iloc[3] == 33
    assert res.positions["QQQ"].iloc[-1] == 49  # 49.5 -> 49 + cash in lieu
    ev = res.events[res.events.type == "SPLIT"].iloc[0]
    assert ev.cash == pytest.approx(0.5 * 66.0)
    assert res.equity.nav.pct_change().dropna().abs().max() < 1e-12


def test_split_adjusted_basis_does_not_split_holdings_again():
    bars = make_bars("QQQ", [50.0] * 10, splits={5: 2.0})
    panel = build_panel(bars, "split_adjusted")
    res = run_backtest(panel, BuyAndHold(), BacktestConfig(commission_bps=0, slippage_bps=0))
    assert res.positions["QQQ"].iloc[-1] == res.positions["QQQ"].iloc[2]
    assert res.events.empty


def test_dividends_credited_once_raw_mode_and_never_in_adjusted_mode():
    closes = [100.0] * 5 + [99.0] * 5
    bars = make_bars("SPY", closes, dividends={5: 1.0})
    raw = run_backtest(build_panel(bars, "unadjusted", "raw_with_dividends"), BuyAndHold(),
                       BacktestConfig(commission_bps=0, slippage_bps=0))
    q = raw.positions["SPY"].iloc[3]
    assert raw.equity.dividends.sum() == pytest.approx(q * 1.0)
    assert raw.equity.nav.pct_change().dropna().abs().max() < 1e-12  # ex-div drop offset by cash
    adj = run_backtest(build_panel(bars, "unadjusted", "adjusted_total_return"), BuyAndHold(),
                       BacktestConfig(commission_bps=0, slippage_bps=0))
    assert adj.equity.dividends.sum() == 0.0 and adj.events.empty
    # both modes give the same economic return (up to integer-share rounding)
    r_raw = raw.equity.nav.iloc[-1] / raw.equity.nav.iloc[0]
    r_adj = adj.equity.nav.iloc[-1] / adj.equity.nav.iloc[0]
    assert r_raw == pytest.approx(r_adj, abs=0.01)


def test_raw_and_adjusted_modes_agree_on_long_buy_and_hold(synth_bars):
    spy = synth_bars[synth_bars.symbol == "SPY"]
    cfg = BacktestConfig(initial_cash=1_000_000, commission_bps=0, slippage_bps=0)
    raw = run_backtest(build_panel(spy, "unadjusted", "raw_with_dividends"), BuyAndHold(), cfg)
    adj = run_backtest(build_panel(spy, "unadjusted", "adjusted_total_return"), BuyAndHold(), cfg)
    tr_raw = raw.equity.nav.iloc[-1] / raw.equity.nav.iloc[0]
    tr_adj = adj.equity.nav.iloc[-1] / adj.equity.nav.iloc[0]
    # raw mode holds dividends as cash (not reinvested) so it trails slightly; no double count
    assert tr_raw < tr_adj * 1.001
    assert tr_raw == pytest.approx(tr_adj, rel=0.03)


def test_missing_execution_bar_expires_and_retries():
    bars = make_bars("SPY", [100.0] * 8)
    bars = bars.drop(index=1).reset_index(drop=True)  # session 1 missing
    panel = build_panel(bars, "unadjusted")
    res = run_backtest(panel, Scripted({0: 0.5}), BacktestConfig())
    assert res.orders.status.iloc[0] == "EXPIRED" and res.orders.reason.iloc[0] == "MISSING_BAR"
    assert len(res.trades) == 1 and res.trades.session.iloc[0] == panel.sessions[3]
    assert any("expired" in w for w in res.warnings)


def test_invariants_hold_on_multi_asset_runs(synth_bars):
    panel = build_panel(synth_bars, "unadjusted")
    for sid, p in [("momentum", {"lookback": 60, "skip": 5}), ("rsi-mean-reversion", {}), ("sma-crossover", {})]:
        res = run_backtest(panel, create_strategy(sid, p), BacktestConfig())
        eq = res.equity
        assert (eq.cash >= -1e-6).all()
        assert (res.positions.to_numpy() >= 0).all()
        mv = (res.positions * panel.close.reindex(eq.index).ffill()).sum(axis=1)
        assert np.allclose(eq.cash + mv, eq.nav, atol=1e-6)


def test_deterministic_replay(synth_bars):
    panel = build_panel(synth_bars, "unadjusted")
    a = run_backtest(panel, create_strategy("momentum", {"lookback": 60, "skip": 5}), BacktestConfig())
    b = run_backtest(panel, create_strategy("momentum", {"lookback": 60, "skip": 5}), BacktestConfig())
    pd.testing.assert_frame_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.trades, b.trades)


def test_costs_reduce_performance(synth_bars):
    panel = build_panel(synth_bars, "unadjusted", symbols=["SPY"])
    s = create_strategy("rsi-mean-reversion")
    net = run_backtest(panel, s, BacktestConfig(commission_bps=5, slippage_bps=10))
    gross = run_backtest(panel, s, BacktestConfig(commission_bps=0, slippage_bps=0))
    assert net.equity.nav.iloc[-1] < gross.equity.nav.iloc[-1]


def test_window_validation(synth_bars):
    panel = build_panel(synth_bars, "unadjusted", symbols=["SPY"])
    with pytest.raises(InputValidationError):
        run_backtest(panel, BuyAndHold(), BacktestConfig(start="2030-01-01"))
    with pytest.raises(InputValidationError):
        BacktestConfig(max_weight=1.5)


def test_cost_model_affordability():
    cm = CostModel(commission_bps=10, slippage_bps=0, min_commission=1.0)
    q = cm.max_affordable_qty(1000, 99.0)
    assert q * 99 + cm.commission(q * 99) <= 1000
    assert (q + 1) * 99 + cm.commission((q + 1) * 99) > 1000


def test_metrics_on_known_series():
    idx = pd.bdate_range("2020-01-01", periods=253)
    nav = pd.Series(100 * 1.0004 ** np.arange(253), index=idx)
    nav.iloc[100:] *= 0.9  # one 10% drawdown
    eq = pd.DataFrame({"nav": nav, "gross_exposure": 1.0, "dividends": 0.0})
    m = performance_metrics(eq)
    assert m["years"] == pytest.approx(1.0)
    assert m["total_return"] == pytest.approx(nav.iloc[-1] / 100 - 1)
    assert m["cagr"] == pytest.approx(m["total_return"])  # exactly one year
    assert m["max_drawdown"] == pytest.approx(-0.1, abs=1e-3)
    assert m["time_in_market"] == 1.0
