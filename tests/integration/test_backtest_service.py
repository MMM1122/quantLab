from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import BacktestConfig
from quantlab.backtest.service import get_run, list_runs, load_artifact, run_and_store
from quantlab.core.errors import InputValidationError, NotFoundError, QualityGateError
from quantlab.core.labels import BACKTEST_LABEL, SYNTHETIC_LABEL
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider
from quantlab.data.validation import run_quality_checks
from tests.conftest import make_bars


@pytest.fixture
def snap(ctx):
    r = ingest(["SPY", "QQQ", "IWM", "TLT", "GLD"], "2016-01-01", "2023-12-29", SyntheticProvider(seed=7),
               ctx.snapshots, ctx.db)
    return r.snapshot


@pytest.mark.integration
def test_three_strategies_persist_with_provenance(ctx, snap):
    outs = {}
    for sid in ("buy-hold", "sma-crossover", "rsi-mean-reversion"):
        out = run_and_store(ctx, sid, snap.id, ["SPY"], config=BacktestConfig(start="2017-01-03"))
        outs[sid] = out
        run = get_run(ctx, out.run_id)
        assert run["status"] == "COMPLETED"
        res = run["result"]
        assert res["label"] == BACKTEST_LABEL and res["data_label"] == SYNTHETIC_LABEL and res["is_synthetic"]
        assert res["snapshot"]["dataset_hash"] == snap.dataset_hash
        assert res["config_hash"] and res["assumptions"] and res["seed"] == 42
        m = res["metrics"]
        assert set(m) == {"net", "gross", "benchmark", "relative"}
        # metrics are derived from stored artifacts, not invented
        eq = load_artifact(ctx, out.run_id, "equity")
        assert m["net"]["final_nav"] == pytest.approx(float(eq.nav.iloc[-1]))
        assert m["net"]["final_nav"] <= m["gross"]["final_nav"] + 1e-9
        for f in ("equity", "trades", "orders", "positions", "benchmark_equity", "equity_gross"):
            assert (Path(run["artifact_uri"]) / f"{f}.parquet").exists()
    # buy & hold on SPY vs SPY benchmark: identical
    bh = outs["buy-hold"].summary["metrics"]
    assert bh["net"]["total_return"] == pytest.approx(bh["benchmark"]["total_return"])
    assert len(list_runs(ctx)) == 3


@pytest.mark.integration
def test_rerun_is_reproducible(ctx, snap):
    a = run_and_store(ctx, "sma-crossover", snap.id, ["SPY"], {"fast": 20, "slow": 100})
    b = run_and_store(ctx, "sma-crossover", snap.id, ["SPY"], {"fast": 20, "slow": 100})
    assert a.run_id != b.run_id
    assert a.summary["config_hash"] == b.summary["config_hash"]
    assert json.dumps(a.summary["metrics"], sort_keys=True) == json.dumps(b.summary["metrics"], sort_keys=True)
    pd.testing.assert_frame_equal(load_artifact(ctx, a.run_id, "trades"), load_artifact(ctx, b.run_id, "trades"))


@pytest.mark.integration
def test_momentum_multi_asset(ctx, snap):
    out = run_and_store(ctx, "momentum", snap.id, ["SPY", "QQQ", "IWM", "TLT", "GLD"])
    assert out.summary["metrics"]["net"]["n_fills"] > 10


def test_rejects_bad_inputs_and_records_failures(ctx, snap):
    with pytest.raises(InputValidationError):
        run_and_store(ctx, "nope", snap.id, ["SPY"])
    with pytest.raises(InputValidationError):
        run_and_store(ctx, "buy-hold", "not-a-uuid", ["SPY"])
    with pytest.raises(NotFoundError):
        run_and_store(ctx, "buy-hold", "00000000-0000-0000-0000-000000000000", ["SPY"])
    with pytest.raises(InputValidationError):
        run_and_store(ctx, "buy-hold", snap.id, ["XLE"])
    with pytest.raises(InputValidationError):
        run_and_store(ctx, "buy-hold", snap.id, ["SPY"], config=BacktestConfig(start="2035-01-01"))
    failed = [r for r in list_runs(ctx) if r["status"] == "FAILED"]
    assert len(failed) == 1 and "evaluation window" in failed[0]["failure_reason"]


def test_quality_gate_blocks_backtest(ctx):
    bars = make_bars("SPY", np.linspace(100, 120, 60))
    bars.loc[10, "close"] = -5
    s, _ = ctx.snapshots.create(bars, "synthetic", "unadjusted", True)
    ctx.snapshots.record_quality(s.id, run_quality_checks(bars, "unadjusted"))
    with pytest.raises(QualityGateError):
        run_and_store(ctx, "buy-hold", s.id, ["SPY"])
    assert list_runs(ctx) == []  # no run started without a valid snapshot
