from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quantlab.core.errors import InputValidationError, NotFoundError
from quantlab.core.labels import REPLAY_LABEL
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider
from quantlab.paper.autopilot import StrategyAutopilot
from quantlab.paper.broker import PaperBroker
from quantlab.paper.compare import paper_vs_backtest

pytestmark = pytest.mark.integration
ALL = ["SPY", "QQQ", "IWM", "TLT", "GLD"]


@pytest.fixture
def snap(ctx):
    return ingest(ALL, "2018-01-01", "2021-12-31", SyntheticProvider(seed=7), ctx.snapshots, ctx.db).snapshot


def _setup(ctx, snap, cap=1.0, strategy="sma-crossover", symbols=("SPY",), params=None):
    b = PaperBroker(ctx, {"max_position_weight": cap})
    acc = b.create_account("auto", snap.id, 10_000)
    ap = StrategyAutopilot(ctx, b)
    ap.attach(acc["id"], strategy, list(symbols), params if params is not None else {"fast": 20, "slow": 100})
    return b, ap, acc["id"]


def test_autopilot_matches_backtest_when_mechanics_coincide(ctx, snap):
    b, ap, aid = _setup(ctx, snap)
    out = ap.run(aid, 600)
    assert out["orders_submitted"] >= 4 and out["orders_rejected"] == 0
    r = paper_vs_backtest(ctx, aid)
    r.pop("_series")
    # same fills, same costs: only cent rounding separates the two NAV paths
    assert abs(r["final_relative_difference"]) < 1e-4
    assert r["explanations"]["paper_orders"] == r["explanations"]["backtest_fills"]
    assert r["daily_return_correlation"] > 0.9999
    assert b.reconcile(aid)["ok"]


def test_autopilot_respects_25pct_cap_and_risk_rules(ctx, snap):
    b, ap, aid = _setup(ctx, snap, cap=0.25, strategy="momentum", symbols=ALL, params={"lookback": 126, "top_k": 4})
    ap.run(aid, 500)
    for snap_row in ctx.db.query("SELECT nav, holdings FROM portfolio_snapshots WHERE account_id=?", (aid,)):
        nav = Decimal(snap_row["nav"])
        for h in json.loads(snap_row["holdings"]).values():
            # targets capped at 24.875% and drift trimmed every session -> at most one session of drift above 25%
            assert Decimal(h["quantity"]) * Decimal(h["close"]) <= nav * Decimal("0.29")
    assert any(o["side"] == "SELL" for o in b.list_orders(aid))
    rej = [o for o in b.list_orders(aid) if o["status"] == "REJECTED"]
    assert all(not o["reason"].startswith("MAX_POSITION_WEIGHT") for o in rej)
    r = paper_vs_backtest(ctx, aid)
    assert r["explanations"]["paper_rejected"] == len(rej)  # gaps are explained by recorded events
    assert r["explanations"]["autopilot_drift_trims"] >= 1
    assert b.reconcile(aid)["ok"]


def test_step_is_idempotent_and_kill_switch_blocks(ctx, snap):
    b, ap, aid = _setup(ctx, snap)
    b.advance_session(aid, 120)  # past the SMA warm-up
    first = ap.step(aid)
    again = ap.step(aid)
    assert again["skipped"] == "already stepped this session" and again["orders"] == []
    n_orders = len(b.list_orders(aid))
    b.set_kill_switch(aid, True, "test")
    ap.run(aid, 200)
    after = b.list_orders(aid)
    new = after[n_orders:]
    assert all(o["status"] == "REJECTED" and o["reason"].startswith("KILL_SWITCH") for o in new)
    assert first["session"] != ap.get(aid)["state"]["last_step_session"]


def test_autopilot_validation_and_disable(ctx, snap):
    b = PaperBroker(ctx)
    acc = b.create_account("v", snap.id, 10_000)
    ap = StrategyAutopilot(ctx, b)
    with pytest.raises(NotFoundError):
        ap.step(acc["id"])
    with pytest.raises(InputValidationError):
        ap.attach(acc["id"], "nope", ["SPY"])
    with pytest.raises(InputValidationError):
        ap.attach(acc["id"], "sma-crossover", ["XLE"])
    ap.attach(acc["id"], "buy-hold", ["SPY"], {})
    ap.set_enabled(acc["id"], False)
    assert ap.step(acc["id"])["skipped"] == "autopilot disabled"


def test_autopilot_api_cli_dashboard(ctx, snap, capsys, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from quantlab.api.app import create_app
    from quantlab.cli.main import main as cli

    with TestClient(create_app(ctx=ctx)) as c:
        aid = c.post("/paper/accounts", json={"name": "api", "snapshot_id": snap.id}).json()["data"]["id"]
        r = c.post("/paper/autopilot", json={"account_id": aid, "strategy": "buy-hold", "symbols": ["SPY", "TLT"]})
        assert r.status_code == 200
        run = c.post("/paper/autopilot/run", json={"account_id": aid, "sessions": 5}).json()
        assert run["meta"]["label"] == REPLAY_LABEL and run["data"]["orders_submitted"] == 2
        cmp_ = c.get("/paper/autopilot/compare", params={"account_id": aid}).json()["data"]
        assert "_series" not in cmp_ and cmp_["final_nav"]["paper"] > 0
    dd = ["--data-dir", str(ctx.settings.data_dir)]
    assert cli([*dd, "paper", "autopilot", "run", "--account", aid, "--sessions", "3"]) == 0
    capsys.readouterr()
    assert cli([*dd, "paper", "autopilot", "compare", "--account", aid]) == 0
    assert json.loads(capsys.readouterr().out)["n_sessions"] >= 8
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(ctx.settings.data_dir))
    view = Path(__file__).resolve().parents[2] / "apps/dashboard/views/paper_trading.py"
    at = AppTest.from_file(str(view), default_timeout=120)
    at.run()
    assert not at.exception, [x.value for x in at.exception]
    [btn for btn in at.button if btn.label == "Compare with backtest"][0].click().run()
    assert not at.exception, [x.value for x in at.exception]
