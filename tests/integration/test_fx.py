from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.backtest.service import metrics_in_currency, run_and_store
from quantlab.core.errors import DataUnavailableError, InputValidationError, NotFoundError
from quantlab.data import calendar
from quantlab.data.fx import fx_series, ingest_fx, is_fx_snapshot, to_cad, yahoo_fx
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider, YFinanceProvider
from quantlab.data.refresh import refresh_snapshot
from quantlab.paper.broker import PaperBroker

pytestmark = pytest.mark.integration


@pytest.fixture
def setup(ctx):
    snap = ingest(["SPY", "TLT"], "2018-01-01", "2021-12-31", SyntheticProvider(seed=7), ctx.snapshots, ctx.db).snapshot
    fx, rep, _ = ingest_fx(ctx, "2017-06-01", "2022-06-30", synthetic=True)
    return ctx, snap, fx, rep


def test_to_cad_as_of_rules():
    idx = pd.DatetimeIndex(calendar.sessions("2024-01-02", "2024-01-31"))
    nav = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    fx = pd.Series(1.35, index=idx).drop(idx[[5, 6]])  # two sessions without a rate
    cad, info = to_cad(nav, fx)
    assert np.allclose(cad, nav * 1.35) and info["stale_rate_sessions"] == 2
    with pytest.raises(InputValidationError):  # FX starts after the NAV -> would need a future rate
        to_cad(nav, fx.iloc[3:])
    with pytest.raises(InputValidationError):  # gap longer than max_stale
        to_cad(nav, fx.drop(idx[8:16]), max_stale=5)


def test_yahoo_fx_aligns_to_xnys_sessions():
    days = pd.bdate_range("2024-07-01", "2024-07-10")  # includes 2024-07-04 (US holiday, FX trades)
    raw = pd.DataFrame({"Open": 1.36, "High": 1.37, "Low": 1.35, "Close": 1.365},
                       index=days.tz_localize("Europe/London"))
    bars = yahoo_fx("2024-07-01", "2024-07-10", downloader=lambda *a: raw)
    assert pd.Timestamp("2024-07-04") not in set(bars.session_date)
    assert set(bars.session_date) <= set(calendar.sessions("2024-07-01", "2024-07-10"))
    assert (bars.source == "yfinance").all() and (bars.symbol == "USDCAD").all()
    with pytest.raises(DataUnavailableError):
        yahoo_fx("2024-07-01", "2024-07-10", downloader=lambda *a: pd.DataFrame())


def test_fx_snapshot_and_cad_metrics(setup):
    ctx, snap, fx, rep = setup
    assert is_fx_snapshot(fx) and fx.is_synthetic and rep.status == "PASSED"
    _, _, created = ingest_fx(ctx, "2017-06-01", "2022-06-30", synthetic=True)
    assert not created  # deduplicated
    run = run_and_store(ctx, "buy-hold", snap.id, ["SPY"]).run_id
    usd = metrics_in_currency(ctx, run, "USD")["metrics"]["net"]
    cad = metrics_in_currency(ctx, run, "CAD")
    curves = cad.pop("_curves")
    rate = fx_series(ctx, fx.id)
    start, end = pd.Timestamp(usd["start"]), pd.Timestamp(usd["end"])
    expected = (1 + usd["total_return"]) * rate[end] / rate[start] - 1
    assert cad["metrics"]["net"]["total_return"] == pytest.approx(expected)
    assert "total_commission" not in cad["metrics"]["net"]  # USD cash amounts are not converted
    assert cad["fx"]["snapshot_id"] == fx.id and set(curves) >= {"net", "benchmark"}
    with pytest.raises(InputValidationError):
        metrics_in_currency(ctx, run, "EUR")


def test_fx_guards(setup):
    ctx, snap, fx, _ = setup
    with pytest.raises(InputValidationError):
        run_and_store(ctx, "buy-hold", fx.id, ["USDCAD"])
    with pytest.raises(InputValidationError):
        refresh_snapshot(ctx, fx.id, YFinanceProvider())
    # a real-data run must not be converted with synthetic FX
    frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Adj Close": 100.0,
                          "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0},
                         index=pd.DatetimeIndex(calendar.sessions("2020-01-02", "2020-06-30")).tz_localize("America/New_York"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real = ingest(["SPY"], "2020-01-02", "2020-06-30", YFinanceProvider(downloader=lambda *a: frame),
                      ctx.snapshots, ctx.db).snapshot
    rid = run_and_store(ctx, "buy-hold", real.id, ["SPY"]).run_id
    with pytest.raises(NotFoundError, match="real USDCAD"):
        metrics_in_currency(ctx, rid, "CAD")


def test_paper_nav_in_cad(setup):
    ctx, snap, fx, _ = setup
    b = PaperBroker(ctx)
    acc = b.create_account("cad", snap.id, 10_000)
    b.submit_order(acc["id"], "SPY", "BUY", 5)
    b.advance_session(acc["id"], 5)
    hist, info = b.nav_history_cad(acc["id"])
    rate = fx_series(ctx, fx.id)
    last = hist.iloc[-1]
    assert last["nav_cad"] == pytest.approx(last["nav"] * rate[pd.Timestamp(last["session_date"])])
    assert info["fx_snapshot_id"] == fx.id


def test_fx_cli_api_dashboard(setup, capsys, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from quantlab.api.app import create_app
    from quantlab.cli.main import main as cli

    ctx, snap, fx, _ = setup
    dd = ["--data-dir", str(ctx.settings.data_dir)]
    assert cli([*dd, "data", "ingest-fx", "--start", "2015-01-01", "--end", "2016-12-30", "--fixture", "synthetic"]) == 0
    assert json.loads(capsys.readouterr().out)["pair"].startswith("USDCAD")
    rid = run_and_store(ctx, "sma-crossover", snap.id, ["SPY"]).run_id
    assert cli([*dd, "backtest", "show", "--run", rid, "--currency", "CAD"]) == 0
    assert json.loads(capsys.readouterr().out)["currency"] == "CAD"
    with TestClient(create_app(ctx=ctx)) as c:
        r = c.get(f"/backtests/{rid}/metrics", params={"currency": "CAD"}).json()["data"]
        assert r["currency"] == "CAD" and "_curves" not in r
        j = c.post("/data/fx/ingest", json={"start": "2016-01-01", "end": "2016-06-30"}).json()["data"]["job_id"]
        assert c.app.state.jobs.wait(j)["status"] == "COMPLETED"
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(ctx.settings.data_dir))
    view = Path(__file__).resolve().parents[2] / "apps/dashboard/views/backtest_lab.py"
    at = AppTest.from_file(str(view), default_timeout=120)
    at.run()
    at.radio(key="bl_cur").set_value("CAD").run()
    assert not at.exception, [x.value for x in at.exception]
    assert any("Unhedged CAD view" in c.value for c in at.caption)
