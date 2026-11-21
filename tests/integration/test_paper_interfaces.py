from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quantlab.core.labels import REPLAY_LABEL, SYNTHETIC_LABEL
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def snap(ctx):
    return ingest(["SPY", "QQQ", "TLT", "GLD", "IWM"], "2022-01-01", "2023-12-29", SyntheticProvider(seed=7),
                  ctx.snapshots, ctx.db).snapshot


def test_paper_api(ctx, snap):
    from quantlab.api.app import create_app

    with TestClient(create_app(ctx=ctx)) as c:
        assert c.post("/paper/accounts", json={"name": "a", "snapshot_id": snap.id, "mode": "LIVE"}).status_code == 403
        acc = c.post("/paper/accounts", json={"name": "a", "snapshot_id": snap.id, "initial_cash": 10000}).json()
        assert acc["meta"]["label"] == REPLAY_LABEL
        aid = acc["data"]["id"]
        o = c.post("/paper/orders", json={"account_id": aid, "symbol": "SPY", "side": "BUY", "quantity": 5,
                                          "client_order_id": "k1"}).json()["data"]
        assert o["status"] == "SUBMITTED"
        again = c.post("/paper/orders", json={"account_id": aid, "symbol": "SPY", "side": "BUY", "quantity": 5,
                                              "client_order_id": "k1"}).json()["data"]
        assert again["id"] == o["id"]
        assert c.post("/paper/orders", json={"account_id": aid, "symbol": "SPY", "side": "BUY", "quantity": 5,
                                             "mode": "LIVE"}).status_code == 403
        adv = c.post("/paper/advance", json={"account_id": aid, "sessions": 2}).json()["data"]
        assert adv["sessions"][0]["fills"] and adv["account"]["holdings"][0]["quantity"] == "5"
        bad = c.post("/paper/orders", json={"account_id": aid, "symbol": "SPY", "side": "SELL", "quantity": 50}).json()
        assert bad["data"]["status"] == "REJECTED"
        ev = c.get("/risk/events", params={"account_id": aid}).json()["data"]
        assert any(e["rule"] == "NO_SHORTING" for e in ev)
        ks = c.post("/paper/kill-switch", json={"account_id": aid, "active": True, "reason": "test"}).json()["data"]
        assert ks["kill_switch"] is True
        assert len(c.get("/paper/orders", params={"account_id": aid}).json()["data"]) == 2
        assert c.get("/paper/account", params={"account_id": "bad"}).status_code == 422


def test_paper_cli_and_dashboard(ctx, snap, capsys, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from quantlab.cli.main import main as cli

    dd = ["--data-dir", str(ctx.settings.data_dir)]
    assert cli([*dd, "paper", "create", "--name", "cli", "--snapshot", snap.id, "--cash", "5000"]) == 0
    acc = json.loads(capsys.readouterr().out)
    assert cli([*dd, "paper", "order", "--account", acc["id"], "--symbol", "GLD", "--side", "BUY", "--quantity", "3"]) == 0
    capsys.readouterr()
    assert cli([*dd, "paper", "order", "--account", acc["id"], "--symbol", "GLD", "--side", "SELL",
                "--quantity", "99"]) == 3  # rejected -> non-zero exit
    capsys.readouterr()
    assert cli([*dd, "paper", "advance", "--account", acc["id"], "--sessions", "3"]) == 0
    out = capsys.readouterr()
    assert REPLAY_LABEL in out.err and SYNTHETIC_LABEL in out.err
    assert cli([*dd, "paper", "reconcile", "--account", acc["id"]]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(ctx.settings.data_dir))
    view = Path(__file__).resolve().parents[2] / "apps/dashboard/views/paper_trading.py"
    at = AppTest.from_file(str(view), default_timeout=120)
    at.run()
    assert not at.exception, [x.value for x in at.exception]
    assert any(REPLAY_LABEL in e.value for e in at.error)
    adv = [b for b in at.button if b.label == "Advance replay clock"][0]
    adv.click().run()
    assert not at.exception
    assert any("Advanced to" in s.value for s in at.success)


def test_limit_orders_via_api_and_cli(ctx, snap, capsys):
    from quantlab.api.app import create_app
    from quantlab.cli.main import main as cli

    with TestClient(create_app(ctx=ctx)) as c:
        aid = c.post("/paper/accounts", json={"name": "l", "snapshot_id": snap.id}).json()["data"]["id"]
        state = c.get("/paper/account", params={"account_id": aid}).json()["data"]
        assert state["holdings"] == []
        o = c.post("/paper/orders", json={"account_id": aid, "symbol": "SPY", "side": "BUY", "quantity": 1,
                                          "order_type": "LIMIT", "limit_price": "1.00"}).json()["data"]
        assert o["status"] == "REJECTED" and o["reason"].startswith("LIMIT_PRICE_OUT_OF_BAND")
        bad = c.post("/paper/orders", json={"account_id": aid, "symbol": "SPY", "side": "BUY", "quantity": 1,
                                            "order_type": "LIMIT"})
        assert bad.status_code == 422
    dd = ["--data-dir", str(ctx.settings.data_dir)]
    assert cli([*dd, "paper", "order", "--account", aid, "--symbol", "SPY", "--side", "BUY", "--quantity", "1",
                "--type", "MARKET", "--tif", "DAY"]) == 0
    assert json.loads(capsys.readouterr().out)["time_in_force"] == "DAY"
