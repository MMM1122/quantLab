from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantlab.ai.assistant import ResearchAssistant
from quantlab.backtest.engine import BacktestConfig
from quantlab.backtest.market import build_panel
from quantlab.core.errors import InputValidationError
from quantlab.core.labels import SYNTHETIC_LABEL
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider
from quantlab.reports.export import export_experiment
from quantlab.research.dq_001 import get_experiment
from quantlab.research.walk_forward import WalkForwardConfig, run_walk_forward, select_params

pytestmark = pytest.mark.integration
GRID = {"fast": [10, 20, 50], "slow": [100, 200]}


@pytest.fixture
def snap(ctx):
    return ingest(["SPY", "QQQ", "TLT"], "2015-01-01", "2023-12-29", SyntheticProvider(seed=7), ctx.snapshots,
                  ctx.db).snapshot


def _cfg(snap, **kw):
    base = dict(snapshot_id=snap.id, strategy="sma-crossover", symbols=["SPY"], param_grid=GRID,
                train_sessions=504, test_sessions=252)
    return WalkForwardConfig(**{**base, **kw})


def test_walk_forward_structure_and_persistence(ctx, snap):
    res = run_walk_forward(ctx, _cfg(snap))
    folds = res["folds"]
    assert len(folds) >= 5 and res["n_configs"] == 6
    for f in folds:
        assert f["train_end"] < f["test_start"] <= f["test_end"]
        assert f["chosen_params"] in [{"fast": a, "slow": b} for a in GRID["fast"] for b in GRID["slow"]]
        assert 0.0 <= f["oos_percentile_rank"] <= 1.0
    for a, b in zip(folds, folds[1:]):
        assert a["test_end"] < b["test_start"]  # test windows never overlap
        assert b["start_nav"] == pytest.approx(a["end_nav"])  # stitched NAV is continuous
    exp = get_experiment(ctx, res["experiment_id"])
    assert exp["status"] == "COMPLETED" and exp["code"] == "WALK-FORWARD"
    oos = pd.read_parquet(f"{exp['artifact_uri']}/oos_equity.parquet")
    assert res["oos_metrics"]["final_nav"] == pytest.approx(float(oos.nav.iloc[-1]))
    assert res["data_label"] == SYNTHETIC_LABEL and res["findings"]
    for fmt in ("md", "html", "csv", "json"):
        for p in export_experiment(ctx, res["experiment_id"], fmt):
            t = p.read_text()
            assert SYNTHETIC_LABEL in t or json.loads(t)["data_label"] == SYNTHETIC_LABEL
    s = ResearchAssistant(ctx.settings.db_path, provider=None).summarize_experiment(res["experiment_id"])
    assert s.verification["verified"], s.verification
    assert "Walk-forward validation" in s.text


def test_selection_cannot_see_test_period(ctx, snap):
    """Perturbing data after the train end must not change which parameters are selected."""
    bars = ctx.snapshots.load(snap.id)
    bars = bars[bars.symbol == "SPY"].reset_index(drop=True)
    configs = [{"fast": a, "slow": b} for a in GRID["fast"] for b in GRID["slow"]]
    panel = build_panel(bars, snap.price_basis)
    train_end = panel.sessions[800]
    base = BacktestConfig()
    best, table = select_params(panel, "sma-crossover", configs, panel.sessions[0].strftime("%Y-%m-%d"),
                                train_end.strftime("%Y-%m-%d"), base, "sharpe")
    shocked = bars.copy()
    after = shocked.session_date > train_end
    rng = np.random.default_rng(0)
    f = np.exp(np.cumsum(rng.normal(-0.002, 0.03, after.sum())))
    for c in ("open", "high", "low", "close", "adj_close"):
        shocked.loc[after, c] = shocked.loc[after, c].to_numpy() * f
    p2 = build_panel(shocked, snap.price_basis)
    best2, table2 = select_params(p2, "sma-crossover", configs, panel.sessions[0].strftime("%Y-%m-%d"),
                                  train_end.strftime("%Y-%m-%d"), base, "sharpe")
    assert best == best2
    assert [r["train_objective"] for r in table] == [r["train_objective"] for r in table2]


def test_walk_forward_reproducible(ctx, snap):
    a = run_walk_forward(ctx, _cfg(snap, expanding=False))
    b = run_walk_forward(ctx, _cfg(snap, expanding=False))
    assert a["folds"] == b["folds"] and a["oos_metrics"] == b["oos_metrics"]


def test_walk_forward_validation(ctx, snap):
    with pytest.raises(InputValidationError):
        run_walk_forward(ctx, _cfg(snap, param_grid={"fast": list(range(1, 30)), "slow": list(range(30, 40))}))
    with pytest.raises(InputValidationError):
        run_walk_forward(ctx, _cfg(snap, param_grid={"fast": [300], "slow": [100]}))  # all invalid
    with pytest.raises(InputValidationError):
        run_walk_forward(ctx, _cfg(snap, param_grid={"fast": 20}))
    with pytest.raises(InputValidationError):
        run_walk_forward(ctx, _cfg(snap, objective="profit"))
    with pytest.raises(InputValidationError):
        run_walk_forward(ctx, _cfg(snap, train_sessions=2000, test_sessions=500))  # not enough data


def test_walk_forward_api_and_cli(ctx, snap, capsys):
    from fastapi.testclient import TestClient

    from quantlab.api.app import create_app
    from quantlab.cli.main import main as cli

    app = create_app(ctx=ctx)
    with TestClient(app) as c:
        r = c.post("/experiments/walk-forward", json={"snapshot_id": snap.id, "strategy": "rsi",
                                                      "param_grid": {"entry": [25, 30]}, "train_sessions": 504})
        assert r.status_code == 202
        job = app.state.jobs.wait(r.json()["data"]["job_id"], timeout=300)
        assert job["status"] == "COMPLETED", job.get("error")
        assert c.post("/experiments/walk-forward", json={"snapshot_id": snap.id, "strategy": "nope"}).status_code == 422
    assert cli(["--data-dir", str(ctx.settings.data_dir), "experiment", "run", "--id", "walk-forward",
                "--snapshot", snap.id, "--strategy", "momentum", "--symbols", "SPY", "QQQ", "TLT",
                "--grid", '{"lookback": [126, 252]}', "--train", "504"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["folds"] >= 4 and out["data_label"] == SYNTHETIC_LABEL


def test_research_lab_walk_forward_tab_renders(ctx, snap, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    run_walk_forward(ctx, _cfg(snap))
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(ctx.settings.data_dir))
    view = Path(__file__).resolve().parents[2] / "apps/dashboard/views/research_lab.py"
    at = AppTest.from_file(str(view), default_timeout=120)
    at.run()
    assert not at.exception, [x.value for x in at.exception]
    assert any(m.label == "OOS Sharpe" for m in at.metric)
    submit = [b for b in at.button if b.label == "Run walk-forward"][0]
    submit.click().run()
    assert not at.exception, [x.value for x in at.exception]
    assert any("Walk-forward" in s.value and "completed" in s.value for s in at.success)
