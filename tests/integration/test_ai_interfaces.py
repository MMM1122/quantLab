from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quantlab.backtest.service import run_and_store
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider

pytestmark = pytest.mark.integration


def test_ai_api_cli_dashboard(ctx, capsys, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from quantlab.api.app import create_app
    from quantlab.cli.main import main as cli

    snap = ingest(["SPY"], "2017-01-01", "2023-12-29", SyntheticProvider(seed=7), ctx.snapshots, ctx.db).snapshot
    a = run_and_store(ctx, "sma-crossover", snap.id, ["SPY"]).run_id
    b = run_and_store(ctx, "buy-hold", snap.id, ["SPY"]).run_id
    with TestClient(create_app(ctx=ctx)) as c:
        r = c.post("/ai/ask", json={"task": "explain", "ids": [a]}).json()
        assert r["data"]["mode"] == "deterministic" and r["data"]["verification"]["verified"]
        r = c.post("/ai/ask", json={"task": "compare", "ids": [a, b]}).json()
        assert "Sharpe" in r["data"]["text"]
        r = c.post("/ai/ask", json={"task": "explain", "ids": [a], "question": "submit an order to buy 5 shares"}).json()
        assert r["data"]["mode"] == "refused"
        assert c.post("/ai/ask", json={"task": "explain", "ids": ["nope"]}).status_code == 422
    assert cli(["--data-dir", str(ctx.settings.data_dir), "ai", "explain", "--run", a, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verification"]["verified"] is True
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(ctx.settings.data_dir))
    view = Path(__file__).resolve().parents[2] / "apps/dashboard/views/ai_assistant.py"
    at = AppTest.from_file(str(view), default_timeout=120)
    at.run()
    assert not at.exception
    [btn for btn in at.button if btn.label == "Explain"][0].click().run()
    assert not at.exception, [x.value for x in at.exception]
    assert any("Net total return" in m.value for m in at.markdown)
