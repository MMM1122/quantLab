from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quantlab.context import build_context
from quantlab.core.errors import (
    AccountingInvariantError,
    InputValidationError,
    LiveTradingDisabledError,
    NotFoundError,
)
from quantlab.core.labels import REPLAY_LABEL
from quantlab.data.validation import run_quality_checks
from quantlab.paper.broker import PaperBroker, ReplayFinished
from tests.conftest import make_bars


def _snapshot(ctx, bars, basis="unadjusted"):
    s, _ = ctx.snapshots.create(bars, "synthetic", basis, True)
    ctx.snapshots.record_quality(s.id, run_quality_checks(bars, basis))
    return s


@pytest.fixture
def env(ctx):
    closes = np.full(12, 100.0)
    opens = np.array([100.0, 101.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    spy = make_bars("SPY", closes, opens=opens, volume=1e6)
    tlt = make_bars("TLT", np.full(12, 50.0), volume=1e6)
    snap = _snapshot(ctx, pd.concat([spy, tlt], ignore_index=True))
    broker = PaperBroker(ctx)
    acc = broker.create_account("test", snap.id, "10000")
    return ctx, broker, acc, snap


def test_create_account_and_labels(env):
    ctx, broker, acc, snap = env
    assert acc["cash"] == "10000.00" and acc["nav"] == "10000.00" and acc["mode"] == "PAPER"
    assert acc["label"] == REPLAY_LABEL and acc["risk_limits"]["max_position_weight"] == 0.25
    with pytest.raises(LiveTradingDisabledError):
        broker.create_account("live", snap.id, 1000, mode="LIVE")
    with pytest.raises(LiveTradingDisabledError):
        broker.submit_order(acc["id"], "SPY", "BUY", 1, mode="LIVE")
    with pytest.raises(InputValidationError):
        broker.create_account("x", snap.id, -5)


def test_buy_fills_next_open_with_exact_decimal_accounting(env):
    _, broker, acc, _ = env
    o = broker.submit_order(acc["id"], "SPY", "BUY", 20, client_order_id="c1")
    assert o["status"] == "SUBMITTED"
    st = broker.account_state(acc["id"])
    assert st["cash"] == "10000.00"  # nothing fills until the next session open
    s = broker.advance_session(acc["id"])[0]
    assert s["fills"][0]["price"] == "101.0202"  # open 101 + 2 bps slippage
    st = broker.account_state(acc["id"])
    # notional 2020.404 -> 2020.40 ; commission 1bp = 0.2020 -> rounded up 0.21
    assert st["cash"] == "7979.39"
    assert st["holdings"][0]["quantity"] == "20"
    assert Decimal(st["nav"]) == Decimal("7979.39") + 20 * Decimal("100")
    assert broker.list_orders(acc["id"])[0]["status"] == "FILLED"
    assert broker.reconcile(acc["id"])["ok"]


def test_no_shorting(env):
    ctx, broker, acc, _ = env
    o = broker.submit_order(acc["id"], "SPY", "SELL", 1)
    assert o["status"] == "REJECTED" and o["reason"].startswith("NO_SHORTING")
    broker.submit_order(acc["id"], "SPY", "BUY", 10)
    broker.advance_session(acc["id"])
    broker.submit_order(acc["id"], "SPY", "SELL", 6)
    o = broker.submit_order(acc["id"], "SPY", "SELL", 5)  # 10 held, 6 pending -> only 4 available
    assert o["status"] == "REJECTED" and "NO_SHORTING" in o["reason"]
    assert any(e["rule"] == "NO_SHORTING" for e in broker.risk_events(acc["id"]))


def test_max_25pct_per_asset(env):
    _, broker, acc, _ = env
    assert broker.submit_order(acc["id"], "SPY", "BUY", 25)["status"] == "SUBMITTED"  # 2500 = 25%
    o = broker.submit_order(acc["id"], "SPY", "BUY", 1)  # pending + new = 26 -> 26%
    assert o["status"] == "REJECTED" and o["reason"].startswith("MAX_POSITION_WEIGHT")


def test_no_negative_cash(ctx):
    bars = make_bars("SPY", np.full(6, 100.0))
    snap = _snapshot(ctx, bars)
    broker = PaperBroker(ctx, {"max_position_weight": 1.0})
    acc = broker.create_account("cash", snap.id, 1000)
    o = broker.submit_order(acc["id"], "SPY", "BUY", 10)  # 1000 + slippage + fee > 1000
    assert o["status"] == "REJECTED" and o["reason"].startswith("INSUFFICIENT_CASH")
    assert broker.submit_order(acc["id"], "SPY", "BUY", 9)["status"] == "SUBMITTED"
    o = broker.submit_order(acc["id"], "SPY", "BUY", 1)  # pending buys count against cash
    assert o["status"] == "REJECTED"
    broker.advance_session(acc["id"])
    assert Decimal(broker.account_state(acc["id"])["cash"]) >= 0


def test_stale_data_rejected(ctx):
    spy = make_bars("SPY", np.full(60, 100.0))
    tlt = make_bars("TLT", np.full(60, 50.0)).drop(index=[2])  # TLT bar missing on session 2 (WARN-level gap)
    snap = _snapshot(ctx, pd.concat([spy, tlt], ignore_index=True))
    broker = PaperBroker(ctx)
    acc = broker.create_account("stale", snap.id, 10_000)
    broker.advance_session(acc["id"], 2)
    o = broker.submit_order(acc["id"], "TLT", "BUY", 1)
    assert o["status"] == "REJECTED" and o["reason"].startswith("STALE_DATA")
    assert broker.submit_order(acc["id"], "SPY", "BUY", 1)["status"] == "SUBMITTED"
    o = broker.submit_order(acc["id"], "XLE", "BUY", 1)
    assert o["status"] == "REJECTED" and o["reason"].startswith("UNKNOWN_SYMBOL")


def test_duplicate_order_prevention_and_idempotency(env):
    ctx, broker, acc, _ = env
    a = broker.submit_order(acc["id"], "SPY", "BUY", 5, client_order_id="abc")
    b = broker.submit_order(acc["id"], "SPY", "BUY", 5, client_order_id="abc")
    assert a["id"] == b["id"] and b.get("idempotent_replay")
    assert len(broker.list_orders(acc["id"])) == 1
    c = broker.submit_order(acc["id"], "SPY", "BUY", 5)  # same content, new id, same session
    assert c["status"] == "REJECTED" and c["reason"].startswith("DUPLICATE_ORDER")


def test_duplicate_fill_callback_does_not_double_count(env):
    _, broker, acc, _ = env
    o = broker.submit_order(acc["id"], "SPY", "BUY", 10)
    assert broker.apply_execution(o["id"], "ext-1", 4, "100.00", "0.05", "2024-01-03") is True
    before = broker.account_state(acc["id"])
    assert broker.apply_execution(o["id"], "ext-1", 4, "100.00", "0.05", "2024-01-03") is False
    after = broker.account_state(acc["id"])
    assert before["cash"] == after["cash"] == "9599.95" and after["holdings"][0]["quantity"] == "4"
    order = [x for x in broker.list_orders(acc["id"]) if x["id"] == o["id"]][0]
    assert order["status"] == "PARTIALLY_FILLED" and order["remaining_quantity"] == "6"
    with pytest.raises(AccountingInvariantError):
        broker.apply_execution(o["id"], "ext-2", 7, "100.00", "0", "2024-01-03")  # 4 + 7 > 10
    assert broker.account_state(acc["id"])["cash"] == "9599.95"  # rolled back


def test_partial_fills_across_sessions(ctx):
    bars = make_bars("SPY", np.full(8, 100.0), volume=100)  # 10% participation -> 10 shares per session
    snap = _snapshot(ctx, bars)
    broker = PaperBroker(ctx)
    acc = broker.create_account("pf", snap.id, 100_000)
    o = broker.submit_order(acc["id"], "SPY", "BUY", 25)
    statuses = []
    for _ in range(3):
        broker.advance_session(acc["id"])
        cur = [x for x in broker.list_orders(acc["id"]) if x["id"] == o["id"]][0]
        statuses.append((cur["status"], cur["filled_quantity"], cur["remaining_quantity"]))
    assert statuses == [("PARTIALLY_FILLED", "10", "15"), ("PARTIALLY_FILLED", "20", "5"), ("FILLED", "25", "0")]
    assert len(broker.executions(acc["id"])) == 3
    trades = [x for x in broker.ledger(acc["id"]) if x["event_type"] == "TRADE_BUY"]
    assert len(trades) == 3
    assert broker.reconcile(acc["id"])["ok"]


def test_kill_switch(env):
    _, broker, acc, _ = env
    o = broker.submit_order(acc["id"], "SPY", "BUY", 5)
    st = broker.set_kill_switch(acc["id"], True, "test")
    assert st["kill_switch"] and st["orders_canceled"] == 1
    assert [x for x in broker.list_orders(acc["id"]) if x["id"] == o["id"]][0]["status"] == "CANCELED"
    r = broker.submit_order(acc["id"], "TLT", "BUY", 1)
    assert r["status"] == "REJECTED" and r["reason"].startswith("KILL_SWITCH")
    broker.advance_session(acc["id"])
    assert broker.account_state(acc["id"])["cash"] == "10000.00"  # nothing filled
    broker.set_kill_switch(acc["id"], False, "resume")
    assert broker.submit_order(acc["id"], "TLT", "BUY", 1)["status"] == "SUBMITTED"


def test_dividend_and_split_handling(ctx):
    closes = [100.0] * 4 + [99.0] * 2 + [49.5] * 4
    bars = make_bars("QQQ", closes, dividends={4: 1.0}, splits={6: 2.0}, volume=1e6)
    snap = _snapshot(ctx, bars)
    broker = PaperBroker(ctx, {"max_position_weight": 1.0, "commission_bps": 0, "slippage_bps": 0})
    acc = broker.create_account("ca", snap.id, 10_000)
    broker.submit_order(acc["id"], "QQQ", "BUY", 33)
    out = broker.advance_session(acc["id"], 8)
    divs = [d for s in out for d in s["dividends"]]
    splits = [d for s in out for d in s["splits"]]
    assert divs == [{"symbol": "QQQ", "amount": "33.00"}]
    assert splits and splits[0]["qty_after"] == "66"
    st = broker.account_state(acc["id"])
    assert st["holdings"][0]["quantity"] == "66"
    assert Decimal(st["nav"]) == Decimal("10000") - 33 * 100 + 33 + 66 * Decimal("49.5")
    nav = broker.nav_history(acc["id"])["nav"]
    assert nav.max() - nav.min() < 1.0  # no artificial jumps from the dividend or split
    assert broker.reconcile(acc["id"])["ok"]


def test_replay_end_and_input_validation(env):
    _, broker, acc, _ = env
    broker.advance_session(acc["id"], 11)
    with pytest.raises(ReplayFinished):
        broker.advance_session(acc["id"])
    with pytest.raises(InputValidationError):
        broker.submit_order(acc["id"], "SPY", "BUY", "1.5")
    with pytest.raises(InputValidationError):
        broker.submit_order(acc["id"], "SPY", "HOLD", 1)
    with pytest.raises(InputValidationError):
        broker.submit_order(acc["id"], "SPY", "BUY", 1, order_type="LIMIT")
    with pytest.raises(InputValidationError):
        broker.submit_order("not-a-uuid", "SPY", "BUY", 1)
    with pytest.raises(NotFoundError):
        broker.submit_order("00000000-0000-0000-0000-000000000000", "SPY", "BUY", 1)


def test_state_persists_across_restart(env, tmp_path):
    ctx, broker, acc, _ = env
    broker.submit_order(acc["id"], "SPY", "BUY", 10, client_order_id="persist-1")
    broker.advance_session(acc["id"])
    broker.submit_order(acc["id"], "TLT", "BUY", 5)
    before = broker.account_state(acc["id"])
    ctx.db.close()
    ctx2 = build_context(ctx.settings.data_dir)  # simulates an app restart
    b2 = PaperBroker(ctx2)
    after = b2.account_state(acc["id"])
    assert after == before
    assert b2.submit_order(acc["id"], "SPY", "BUY", 10, client_order_id="persist-1").get("idempotent_replay")
    b2.advance_session(acc["id"])
    assert {h["symbol"] for h in b2.account_state(acc["id"])["holdings"]} == {"SPY", "TLT"}
    assert b2.reconcile(acc["id"])["ok"]
