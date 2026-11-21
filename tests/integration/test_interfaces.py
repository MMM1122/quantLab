"""API, CLI, exports and Streamlit smoke tests (offline, synthetic fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quantlab.cli.main import main as cli
from quantlab.core.labels import SYNTHETIC_LABEL
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider

REPO = Path(__file__).resolve().parents[2]
VIEWS = REPO / "apps" / "dashboard" / "views"


@pytest.fixture
def seeded(ctx):
    snap = ingest(["SPY", "QQQ", "TLT"], "2018-01-01", "2023-12-29", SyntheticProvider(seed=7),
                  ctx.snapshots, ctx.db).snapshot
    return ctx, snap


# ----------------------------------------------------------------- API
@pytest.fixture
def client(seeded):
    from quantlab.api.app import create_app

    ctx, snap = seeded
    app = create_app(ctx=ctx)
    with TestClient(app) as c:
        c.app_state = app.state
        c.snap = snap
        yield c


def test_api_health_and_metadata(client):
    r = client.get("/health").json()
    assert r["data"]["status"] == "ok" and r["meta"]["live_trading_enabled"] is False
    assert {s["id"] for s in client.get("/strategies").json()["data"]} >= {"buy-hold", "sma-crossover"}
    snaps = client.get("/data/snapshots").json()["data"]
    assert snaps[0]["data_label"] == SYNTHETIC_LABEL
    q = client.get(f"/data/quality/{client.snap.id}").json()
    assert q["data"]["status"] in ("PASSED", "PASSED_WITH_WARNINGS") and q["meta"]["data_label"] == SYNTHETIC_LABEL
    assert any(a["symbol"] == "SPY" for a in client.get("/assets").json()["data"]["assets"])


def test_api_backtest_job_lifecycle(client):
    body = {"strategy": "ma_crossover", "snapshot_id": client.snap.id, "symbols": ["SPY"], "start": "2019-01-02",
            "capital": 10000, "commission_bps": 1.0, "slippage_bps": 2.0, "mode": "BACKTEST", "seed": 42,
            "params": {"fast": 20, "slow": 100}}
    r = client.post("/backtests", json=body)
    assert r.status_code == 202
    job = client.app_state.jobs.wait(r.json()["data"]["job_id"])
    assert job["status"] == "COMPLETED", job
    run_id = job["result_ref"]
    j = client.get(f"/jobs/{job['id']}").json()["data"]
    assert j["status"] == "COMPLETED" and j["result_ref"] == run_id
    res = client.get(f"/backtests/{run_id}").json()
    assert res["data"]["status"] == "COMPLETED" and res["meta"]["run_id"] == run_id
    assert res["meta"]["assumptions"] and res["meta"]["data_label"] == SYNTHETIC_LABEL
    trades = client.get(f"/backtests/{run_id}/trades").json()
    assert len(trades["data"]) == res["data"]["result"]["metrics"]["net"]["n_fills"]
    assert "computation_seconds" in trades["meta"]


def test_api_failed_job_is_recorded(client):
    body = {"strategy": "buy-hold", "snapshot_id": client.snap.id, "symbols": ["SPY"], "start": "2030-01-01"}
    job = client.app_state.jobs.wait(client.post("/backtests", json=body).json()["data"]["job_id"])
    assert job["status"] == "FAILED" and "evaluation window" in job["error"]


@pytest.mark.parametrize(
    "patch,code",
    [
        ({"snapshot_id": "not-a-uuid"}, 422),
        ({"snapshot_id": "00000000-0000-0000-0000-000000000000"}, 404),
        ({"strategy": "unknown"}, 422),
        ({"mode": "LIVE"}, 403),
        ({"end": "2999-01-01"}, 422),
        ({"capital": -5}, 422),
    ],
)
def test_api_rejects_invalid_requests(client, patch, code):
    body = {"strategy": "buy-hold", "snapshot_id": client.snap.id, "symbols": ["SPY"], **patch}
    assert client.post("/backtests", json=body).status_code == code


def test_api_has_no_live_order_path(client):
    r = client.post("/orders/live", json={})
    assert r.status_code == 403 and r.json()["error"] == "LiveTradingDisabledError"


def test_api_ingest_job(client):
    r = client.post("/data/ingest", json={"symbols": ["GLD"], "start": "2020-01-01", "end": "2020-06-30",
                                          "provider": "synthetic"})
    job = client.app_state.jobs.wait(r.json()["data"]["job_id"])
    assert job["status"] == "COMPLETED"
    assert client.get(f"/data/quality/{job['result_ref']}").status_code == 200
    bad = client.post("/data/ingest", json={"symbols": ["GLD"], "start": "2020-06-30", "end": "2020-01-01"})
    assert bad.status_code == 422


# ----------------------------------------------------------------- CLI
def test_cli_offline_flow(tmp_path, capsys):
    dd = str(tmp_path / "cli")
    assert cli(["--data-dir", dd, "data", "ingest", "--symbols", "SPY", "QQQ", "IWM", "GLD", "TLT",
                "--start", "2015-01-01", "--end", "2024-12-31", "--fixture", "synthetic"]) == 0
    out = capsys.readouterr()
    info = json.loads(out.out)
    assert info["data_label"] == SYNTHETIC_LABEL and SYNTHETIC_LABEL in out.err
    sid = info["snapshot_id"]
    assert cli(["--data-dir", dd, "data", "check", "--snapshot", sid]) == 0
    assert json.loads(capsys.readouterr().out)["hash_verified"] is True
    assert cli(["--data-dir", dd, "backtest", "run", "--strategy", "ma-crossover", "--symbol", "SPY"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["data_label"] == SYNTHETIC_LABEL and run["net"]["final_nav"] > 0
    assert cli(["--data-dir", dd, "backtest", "run", "--strategy", "rsi", "--symbol", "SPY", "--param", "entry=25"]) == 0
    run2 = json.loads(capsys.readouterr().out)
    assert cli(["--data-dir", dd, "backtest", "compare", "--runs", run["run_id"], run2["run_id"]]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    for f in ("md", "html", "csv", "json"):
        assert cli(["--data-dir", dd, "report", "export", "--run", run["run_id"], "--format", f]) == 0
        files = json.loads(capsys.readouterr().out)["files"]
        for p in files:
            text = Path(p).read_text()
            assert "SYNTHETIC" in Path(p).name
            if f != "json":
                assert SYNTHETIC_LABEL in text
            else:
                assert json.loads(text)["data_label"] == SYNTHETIC_LABEL


def test_cli_errors_are_clean(tmp_path, capsys):
    assert cli(["--data-dir", str(tmp_path), "data", "check", "--snapshot", "nope"]) == 2
    assert "InputValidationError" in capsys.readouterr().err


# ----------------------------------------------------------------- Streamlit
def _apptest(path: Path, data_dir: str, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("QUANTLAB_DATA_DIR", data_dir)
    at = AppTest.from_file(str(path), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [w.value for w in at.warning] + [i.value for i in at.info]
    parts += [e.value for e in at.error] + [s.value for s in at.success] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


def test_dashboard_pages_render_empty_state(tmp_path, monkeypatch):
    dd = str(tmp_path / "empty")
    for page in VIEWS.glob("*.py"):
        _apptest(page, dd, monkeypatch)


def test_dashboard_backtest_lab_runs_and_labels(seeded, monkeypatch):
    ctx, snap = seeded
    dd = str(ctx.settings.data_dir)
    at = _apptest(VIEWS / "backtest_lab.py", dd, monkeypatch)
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("completed" in s.value for s in at.success)
    at = _apptest(VIEWS / "backtest_lab.py", dd, monkeypatch)
    t = _texts(at)
    assert SYNTHETIC_LABEL in t and "BACKTEST / SIMULATION" in t
    for page in ("overview.py", "market_explorer.py", "strategy_studio.py"):
        at = _apptest(VIEWS / page, dd, monkeypatch)
    assert "SYNTHETIC" in _texts(_apptest(VIEWS / "market_explorer.py", dd, monkeypatch))

