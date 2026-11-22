from __future__ import annotations

import sqlite3
from decimal import Decimal

import numpy as np
import pytest

from quantlab.core.errors import InputValidationError
from quantlab.data.validation import run_quality_checks
from quantlab.paper.broker import PaperBroker, limit_fill_price
from quantlab.storage import db as dbmod
from quantlab.storage.db import Database
from tests.conftest import make_bars


def _setup(ctx, opens, lows, highs, volume=1e6, cash=10_000, **risk):
    n = len(opens)
    bars = make_bars("SPY", np.full(n, 100.0), opens=opens, volume=volume)
    bars["low"] = np.minimum(lows, np.minimum(bars["open"], bars["close"]))
    bars["high"] = np.maximum(highs, np.maximum(bars["open"], bars["close"]))
    snap, _ = ctx.snapshots.create(bars, "synthetic", "unadjusted", True)
    ctx.snapshots.record_quality(snap.id, run_quality_checks(bars, "unadjusted"))
    broker = PaperBroker(ctx, {"max_position_weight": 1.0, **risk})
    return broker, broker.create_account("lim", snap.id, cash)


def _order(broker, aid, oid):
    return [o for o in broker.list_orders(aid) if o["id"] == oid][0]


def test_limit_fill_rule():
    D = Decimal
    assert limit_fill_price("BUY", D("99"), D("101.0202"), D("98"), D("102")) == D("99")  # trades through
    assert limit_fill_price("BUY", D("102"), D("101.0202"), D("98"), D("102")) == D("101.0202")  # open better
    assert limit_fill_price("BUY", D("98"), D("101.0202"), D("98"), D("102")) is None  # touch only
    assert limit_fill_price("SELL", D("103"), D("100.98"), D("98"), D("104")) == D("103")
    assert limit_fill_price("SELL", D("100"), D("100.98"), D("98"), D("104")) == D("100.98")
    assert limit_fill_price("SELL", D("104"), D("100.98"), D("98"), D("104")) is None


def test_buy_limit_fills_at_limit_when_low_trades_through(ctx):
    b, acc = _setup(ctx, opens=[100, 101, 100, 100], lows=[99, 98, 99, 99], highs=[101, 102, 101, 101])
    o = b.submit_order(acc["id"], "SPY", "BUY", 10, order_type="LIMIT", limit_price="99")
    assert o["status"] == "SUBMITTED" and o["limit_price"] == "99" and o["time_in_force"] == "GTC"
    fill = b.advance_session(acc["id"])[0]["fills"][0]
    assert fill["price"] == "99"
    assert _order(b, acc["id"], o["id"])["status"] == "FILLED"
    # 10 x 99 = 990.00 ; commission 1bp = 0.099 -> 0.10
    assert b.account_state(acc["id"])["cash"] == "9009.90"


def test_buy_limit_above_open_fills_at_open(ctx):
    b, acc = _setup(ctx, opens=[100, 101, 100], lows=[99, 100, 99], highs=[101, 102, 101])
    b.submit_order(acc["id"], "SPY", "BUY", 5, order_type="LIMIT", limit_price="102")
    assert b.advance_session(acc["id"])[0]["fills"][0]["price"] == "101.0202"


def test_gtc_limit_keeps_working_and_day_limit_expires(ctx):
    b, acc = _setup(ctx, opens=[100, 100, 100, 100], lows=[99, 98, 96, 99], highs=[101, 101, 101, 101])
    gtc = b.submit_order(acc["id"], "SPY", "BUY", 5, order_type="LIMIT", limit_price="97", time_in_force="GTC")
    day = b.submit_order(acc["id"], "SPY", "BUY", 6, order_type="LIMIT", limit_price="97", time_in_force="DAY")
    s1 = b.advance_session(acc["id"])[0]
    assert not s1["fills"]
    assert _order(b, acc["id"], gtc["id"])["status"] == "SUBMITTED"
    d = _order(b, acc["id"], day["id"])
    assert d["status"] == "CANCELED" and d["reason"].startswith("DAY_ORDER_EXPIRED")
    s2 = b.advance_session(acc["id"])[0]
    assert [f["price"] for f in s2["fills"]] == ["97"]
    assert _order(b, acc["id"], gtc["id"])["status"] == "FILLED"


def test_sell_limit(ctx):
    b, acc = _setup(ctx, opens=[100, 100, 100, 100], lows=[99, 99, 99, 99], highs=[101, 101, 104, 101])
    b.submit_order(acc["id"], "SPY", "BUY", 10)
    b.advance_session(acc["id"])
    o = b.submit_order(acc["id"], "SPY", "SELL", 10, order_type="LIMIT", limit_price="103")
    fills = b.advance_session(acc["id"])[0]["fills"]
    assert fills[0]["price"] == "103" and _order(b, acc["id"], o["id"])["status"] == "FILLED"
    assert b.reconcile(acc["id"])["ok"]


def test_day_market_order_remainder_canceled_after_volume_cap(ctx):
    b, acc = _setup(ctx, opens=[100] * 4, lows=[99] * 4, highs=[101] * 4, volume=100, cash=100_000)
    o = b.submit_order(acc["id"], "SPY", "BUY", 25, time_in_force="DAY")
    b.advance_session(acc["id"])
    cur = _order(b, acc["id"], o["id"])
    assert cur["status"] == "CANCELED" and cur["filled_quantity"] == "10" and "DAY_ORDER_EXPIRED" in cur["reason"]


def test_limit_risk_checks_and_validation(ctx):
    b, acc = _setup(ctx, opens=[100] * 3, lows=[99] * 3, highs=[101] * 3, cash=1000)
    far = b.submit_order(acc["id"], "SPY", "BUY", 1, order_type="LIMIT", limit_price="130")
    assert far["status"] == "REJECTED" and far["reason"].startswith("LIMIT_PRICE_OUT_OF_BAND")
    # cash check uses the limit price (worst case you agreed to pay)
    assert b.submit_order(acc["id"], "SPY", "BUY", 10, order_type="LIMIT", limit_price="105")["status"] == "REJECTED"
    assert b.submit_order(acc["id"], "SPY", "BUY", 9, order_type="LIMIT", limit_price="105")["status"] == "SUBMITTED"
    for kw in ({"order_type": "LIMIT"}, {"limit_price": "99"}, {"time_in_force": "IOC"},
               {"order_type": "LIMIT", "limit_price": "99.12345"}, {"order_type": "STOP"}):
        with pytest.raises(InputValidationError):
            b.submit_order(acc["id"], "SPY", "BUY", 1, **kw)


def test_v1_database_migrates_without_data_loss(tmp_path):
    path = tmp_path / "v1.db"
    v1_orders = dbmod.new_orders_ddl()
    v1_orders = (v1_orders.replace("order_type IN ('MARKET','LIMIT')", "order_type IN ('MARKET')")
                 .replace("  limit_price TEXT,\n", "")
                 .replace("  time_in_force TEXT NOT NULL DEFAULT 'GTC' CHECK (time_in_force IN ('DAY','GTC')),\n", "")
                 .replace(",\n  CHECK ((order_type = 'LIMIT') = (limit_price IS NOT NULL))", ""))
    assert "limit_price" not in v1_orders
    con = sqlite3.connect(path)
    con.executescript(dbmod.SCHEMA.replace(dbmod.new_orders_ddl(), v1_orders))
    con.execute("PRAGMA user_version = 1")
    con.execute("INSERT INTO data_snapshots(id, provider, dataset_hash, storage_uri, price_basis, retrieved_at, symbols, "
                "start_date, end_date, n_rows, created_at) VALUES ('s','p','h','u','unadjusted','t','[]','a','b',0,'t')")
    con.execute("INSERT INTO accounts(id, name, mode, base_currency, status, initial_cash, snapshot_id, current_session, "
                "created_at) VALUES ('a','n','PAPER','USD','ACTIVE','1','s','2024-01-02','t')")
    con.execute("INSERT INTO orders(id, account_id, symbol, client_order_id, side, quantity, order_type, status, "
                "decision_session, created_at, updated_at) VALUES ('o','a','SPY','c','BUY','5','MARKET','FILLED',"
                "'2024-01-02','t','t')")
    con.execute("INSERT INTO executions(id, order_id, execution_ref, quantity, price, fee, currency, session_date, "
                "executed_at) VALUES ('e','o','r','5','100','0','USD','2024-01-03','t')")
    con.commit()
    con.close()
    db = Database(path)
    assert db.query_one("PRAGMA user_version")["user_version"] == 2
    row = db.query_one("SELECT * FROM orders WHERE id='o'")
    assert row["order_type"] == "MARKET" and row["limit_price"] is None and row["time_in_force"] == "GTC"
    assert db.query_one("SELECT order_id FROM executions WHERE id='e'")["order_id"] == "o"
    assert db.query("PRAGMA foreign_key_check") == []
    db.execute("INSERT INTO orders(id, account_id, symbol, client_order_id, side, quantity, order_type, limit_price, "
               "status, decision_session, created_at, updated_at) VALUES ('o2','a','SPY','c2','BUY','1','LIMIT','99',"
               "'SUBMITTED','2024-01-02','t','t')")
    with pytest.raises(sqlite3.IntegrityError):  # LIMIT without a price violates the CHECK
        db.execute("INSERT INTO orders(id, account_id, symbol, client_order_id, side, quantity, order_type, status, "
                   "decision_session, created_at, updated_at) VALUES ('o3','a','SPY','c3','BUY','1','LIMIT',"
                   "'SUBMITTED','2024-01-02','t','t')")
