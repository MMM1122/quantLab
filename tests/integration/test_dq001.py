from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantlab.core.labels import RESEARCH_ONLY_LABEL, SYNTHETIC_LABEL
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider
from quantlab.reports.export import export_experiment
from quantlab.research.dq_001 import DQ001Config, experiment_runs, get_experiment, run_dq001
from quantlab.research.perturbations import Scenario

pytestmark = pytest.mark.integration

SMALL = dict(
    symbol="SPY",
    strategies=["sma-crossover", "rsi-mean-reversion"],
    seeds=[11, 23],
    scenarios=[Scenario("clean"), Scenario("missing_5pct", missing_rate=0.05),
               Scenario("delayed_5pct", delay_rate=0.05), Scenario("corrupted_2pct", corrupt_rate=0.02)],
    strategy_params={"sma-crossover": {"fast": 20, "slow": 100}},
)


@pytest.fixture
def snap(ctx):
    return ingest(["SPY", "TLT"], "2017-01-01", "2023-12-29", SyntheticProvider(seed=7), ctx.snapshots, ctx.db).snapshot


def test_dq001_end_to_end(ctx, snap):
    before = ctx.snapshots.get(snap.id).dataset_hash
    res = run_dq001(ctx, DQ001Config(snapshot_id=snap.id, **SMALL))
    eid = res["experiment_id"]
    # 1 clean (x1 seed) + 3 scenarios x 2 seeds, each x 3 policies x 2 strategies
    assert res["n_runs"] == (1 + 3 * 2) * 3 * 2
    assert res["source_snapshot_verified_unchanged"] and ctx.snapshots.get(snap.id).dataset_hash == before
    assert res["data_label"] == SYNTHETIC_LABEL
    assert res["research_only_policies"] == {"naive_ffill": RESEARCH_ONLY_LABEL}
    # clean data + strict policy reproduces the baseline exactly for SMA (no genuine 10% jumps in fixture)
    agg = pd.DataFrame(res["aggregate"])
    clean = agg[(agg.scenario == "clean") & (agg.strategy == "sma-crossover")]
    assert (clean["delta_total_return_mean"].abs() < 1e-12).all()
    assert (clean["signal_agreement_mean"] == 1.0).all()
    # derived snapshots: persisted, linked to parent, carry the event log; parent untouched
    derived = [s for s in ctx.snapshots.list() if s.parent_id == snap.id]
    assert len(derived) == 3 * 2
    for d in derived:
        assert d.manifest["transformation"] == "dq001.perturb" and d.manifest["events"]
        assert d.is_synthetic and d.provider.endswith("+dq001")
    # detection: every injected missing bar is found by the calendar-aware validator
    for det in res["detection"]:
        assert det["detected_missing"] == det["injected_missing"]
    # persisted rows and artifacts
    runs = experiment_runs(ctx, eid)
    assert len(runs) == res["n_runs"]
    e = get_experiment(ctx, eid)
    assert e["status"] == "COMPLETED" and e["design"]["missing_data_rules"]
    for f in ("runs.csv", "aggregate.csv", "policy_comparison.csv", "result.json"):
        assert (Path(e["artifact_uri"]) / f).exists()
    assert res["findings"] and any("CI" in f for f in res["findings"])
    for fmt in ("md", "html", "csv", "json"):
        for p in export_experiment(ctx, eid, fmt):
            text = p.read_text()
            assert SYNTHETIC_LABEL in text or json.loads(text).get("data_label") == SYNTHETIC_LABEL


def test_dq001_is_reproducible(ctx, snap):
    cfg = dict(SMALL, seeds=[5], scenarios=[Scenario("clean"), Scenario("missing_10pct", missing_rate=0.1)])
    a = run_dq001(ctx, DQ001Config(snapshot_id=snap.id, **cfg))
    b = run_dq001(ctx, DQ001Config(snapshot_id=snap.id, **cfg))
    ra = experiment_runs(ctx, a["experiment_id"]).drop(columns=["experiment_id"])
    rb = experiment_runs(ctx, b["experiment_id"]).drop(columns=["experiment_id"])
    pd.testing.assert_frame_equal(ra, rb)
    assert a["findings"] == b["findings"]


def test_dq001_validation_errors(ctx, snap):
    from quantlab.core.errors import InputValidationError

    with pytest.raises(InputValidationError):
        run_dq001(ctx, DQ001Config(snapshot_id=snap.id, symbol="QQQ"))
    with pytest.raises(InputValidationError):
        run_dq001(ctx, DQ001Config(snapshot_id=snap.id, policies=["backfill_future"]))
    with pytest.raises(InputValidationError):
        run_dq001(ctx, DQ001Config(snapshot_id=snap.id, seeds=[1, 1]))


def test_dq001_api_cli_and_dashboard(ctx, snap, monkeypatch, capsys):
    from fastapi.testclient import TestClient
    from streamlit.testing.v1 import AppTest

    from quantlab.api.app import create_app
    from quantlab.cli.main import main as cli

    app = create_app(ctx=ctx)
    with TestClient(app) as c:
        r = c.post("/experiments/dq-001", json={"snapshot_id": snap.id, "seeds": [3], "scenarios": ["missing_5pct"],
                                                "strategies": ["sma-crossover"]})
        assert r.status_code == 202
        job = app.state.jobs.wait(r.json()["data"]["job_id"], timeout=300)
        assert job["status"] == "COMPLETED", job.get("error")
        e = c.get(f"/experiments/{job['result_ref']}").json()
        assert e["data"]["status"] == "COMPLETED" and e["meta"]["data_label"] == SYNTHETIC_LABEL
    assert cli(["--data-dir", str(ctx.settings.data_dir), "experiment", "run", "--id", "dq-001", "--snapshot", snap.id,
                "--seeds", "1", "--scenarios", "corrupted_2pct", "--strategies", "sma-crossover"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_runs"] == 2 * 3
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(ctx.settings.data_dir))
    view = Path(__file__).resolve().parents[2] / "apps/dashboard/views/research_lab.py"
    at = AppTest.from_file(str(view), default_timeout=120)
    at.run()
    assert not at.exception, [x.value for x in at.exception]
    assert any(SYNTHETIC_LABEL in w.value for w in at.warning)
