from __future__ import annotations

import os
import stat

import numpy as np
import pandas as pd
import pytest

from quantlab.core.errors import (
    DataIntegrityError,
    DataUnavailableError,
    InputValidationError,
    LiveTradingDisabledError,
    QualityGateError,
)
from quantlab.core.labels import SYNTHETIC_LABEL, data_label
from quantlab.data import calendar
from quantlab.data.adjustments import back_adjusted_close, total_return_index
from quantlab.data.ingestion import ingest, require_tradable_snapshot, validate_date_range
from quantlab.data.providers import SyntheticProvider, YFinanceProvider
from quantlab.data.validation import run_quality_checks
from quantlab.storage import duckdb_store
from tests.conftest import make_bars


def issue_types(report):
    return {(i.issue_type, str(i.severity)) for i in report.issues}


# ----------------------------------------------------------------- calendar
def test_calendar_excludes_weekends_and_holidays():
    s = calendar.sessions("2024-07-01", "2024-07-08")
    days = [d.strftime("%Y-%m-%d") for d in s]
    assert "2024-07-04" not in days  # Independence Day
    assert "2024-07-06" not in days and "2024-07-07" not in days  # weekend
    assert days == ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"]
    assert not calendar.is_session("2024-11-28")  # Thanksgiving


def test_missing_day_assessed_against_exchange_calendar_not_weekends():
    bars = make_bars("SPY", np.linspace(100, 110, 30), start="2024-06-24")
    rep = run_quality_checks(bars, "unadjusted")
    assert not any(i.issue_type == "MISSING_SESSIONS" for i in rep.issues), rep.to_dict()
    # Remove one real session -> flagged
    rep2 = run_quality_checks(bars.drop(index=5), "unadjusted")
    miss = [i for i in rep2.issues if i.issue_type == "MISSING_SESSIONS"]
    assert len(miss) == 1 and miss[0].row_count == 1
    assert miss[0].details["dates"] == [bars["session_date"].iloc[5].strftime("%Y-%m-%d")]


# ----------------------------------------------------------------- validation
def test_quality_checks_flag_each_problem():
    bars = make_bars("SPY", np.linspace(100, 120, 40))
    bad = bars.copy()
    bad = pd.concat([bad, bad.iloc[[3]]], ignore_index=True)  # duplicate
    bad.loc[5, "close"] = -1.0  # nonpositive
    bad.loc[6, "high"] = bad.loc[6, "low"] - 1  # OHLC violation
    bad.loc[7, "volume"] = 0  # zero volume
    bad.loc[8, "adj_close"] = bad.loc[7, "adj_close"] * 1.6  # extreme return
    weekend = bad.iloc[[0]].copy()
    weekend["session_date"] = pd.Timestamp("2024-01-06")  # Saturday
    bad = pd.concat([bad, weekend], ignore_index=True)
    rep = run_quality_checks(bad, "unadjusted")
    types = issue_types(rep)
    assert ("DUPLICATE_BAR", "ERROR") in types
    assert ("NONPOSITIVE_PRICE", "ERROR") in types
    assert ("OHLC_BOUNDS", "ERROR") in types
    assert ("ZERO_VOLUME", "WARN") in types
    assert ("EXTREME_RETURN", "WARN") in types
    assert ("NON_SESSION_ROW", "ERROR") in types
    assert rep.blocking and rep.status == "FAILED"


def test_extreme_return_flagged_not_deleted():
    closes = np.full(30, 100.0)
    closes[10:] = 140.0
    bars = make_bars("SPY", closes)
    rep = run_quality_checks(bars, "unadjusted")
    assert any(i.issue_type == "EXTREME_RETURN" for i in rep.issues)
    assert rep.n_rows == 30  # nothing removed


def test_stale_data_warning_when_tail_missing():
    bars = make_bars("SPY", np.linspace(100, 101, 20), start="2024-01-02")
    rep = run_quality_checks(bars, "unadjusted", "2024-01-02", "2024-03-29")
    assert any(i.issue_type == "STALE_DATA" for i in rep.issues)


def test_split_consistency_checks():
    closes = [100.0] * 10 + [50.0] * 10
    unadj = make_bars("QQQ", closes, splits={10: 2.0})
    assert not any(i.issue_type == "SPLIT_INCONSISTENT" for i in run_quality_checks(unadj, "unadjusted").issues)
    # Same raw jump but snapshot claims split-adjusted prices -> inconsistent
    rep = run_quality_checks(unadj, "split_adjusted")
    assert any(i.issue_type == "SPLIT_INCONSISTENT" and i.severity == "ERROR" for i in rep.issues)
    # unadjusted basis but no jump
    flat = make_bars("QQQ", [100.0] * 20, splits={10: 2.0})
    assert any(i.issue_type == "SPLIT_INCONSISTENT" for i in run_quality_checks(flat, "unadjusted").issues)


def test_synthetic_fixture_passes_quality(synth_bars):
    rep = run_quality_checks(synth_bars, "unadjusted", expected_source="synthetic")
    assert not rep.blocking, [i.to_dict() for i in rep.issues if i.severity in ("ERROR", "BLOCK")]


# ----------------------------------------------------------------- adjustments
def test_total_return_index_handles_split_without_spike_and_counts_dividend_once():
    close = np.array([100.0, 100.0, 50.0, 50.0, 49.0])
    div = np.array([0, 0, 0, 0, 1.0])
    spl = np.array([1, 1, 2.0, 1, 1])
    tr = total_return_index(close, div, spl, "unadjusted")
    r = tr[1:] / tr[:-1] - 1
    assert abs(r[1]) < 1e-12  # split day: no artificial return
    assert abs(r[3]) < 1e-12  # ex-div day: price drop exactly offset by dividend, not double counted
    # split-adjusted basis must NOT apply the split again
    tr2 = total_return_index(np.array([50.0, 50.0, 50.0]), np.zeros(3), np.array([1, 2.0, 1]), "split_adjusted")
    assert np.allclose(tr2, 50.0)


def test_total_return_matches_back_adjusted_close_ratio(synth_bars):
    g = synth_bars[synth_bars.symbol == "QQQ"]
    tr = total_return_index(g.close.values, g.dividend.values, g.split_ratio.values, "unadjusted")
    adj = back_adjusted_close(g.close.values, g.dividend.values, g.split_ratio.values, "unadjusted")
    # total return over the window must agree between the causal index and Yahoo-style adj close
    assert tr[-1] / tr[0] == pytest.approx(adj[-1] / adj[0], rel=2e-3)


# ----------------------------------------------------------------- providers
def _fake_yf_frame(n=10, start="2024-03-01"):
    idx = pd.DatetimeIndex(calendar.sessions(start, "2024-06-01")[:n]).tz_localize("America/New_York")
    c = np.linspace(500, 510, n)
    return pd.DataFrame(
        {
            "Open": c - 1, "High": c + 2, "Low": c - 3, "Close": c, "Adj Close": c * 0.99,
            "Volume": np.full(n, 5e7), "Dividends": np.r_[np.zeros(n - 1), 1.5], "Stock Splits": np.zeros(n),
        },
        index=idx,
    )


def test_yfinance_provider_maps_columns_with_mocked_downloader():
    prov = YFinanceProvider(downloader=lambda s, a, b: _fake_yf_frame())
    with pytest.warns(UserWarning, match="personal research"):
        res = prov.fetch(["SPY"], "2024-03-01", "2024-03-20")
    b = res.bars
    assert res.price_basis == "split_adjusted" and res.is_synthetic is False
    assert list(b.columns)[:3] == ["symbol", "session_date", "open"]
    assert (b.split_ratio == 1.0).all() and b.dividend.iloc[-1] == 1.5
    assert b.session_date.dt.tz is None and b.session_date.iloc[0] == pd.Timestamp("2024-03-01")
    assert (b.source == "yfinance").all()


def test_provider_failure_raises_and_creates_no_snapshot(ctx):
    def boom(*_):
        raise ConnectionError("CONNECT tunnel failed, response 403")

    with pytest.raises(DataUnavailableError), pytest.warns(UserWarning):
        ingest(["SPY"], "2024-01-01", "2024-02-01", YFinanceProvider(downloader=boom), ctx.snapshots, ctx.db)
    assert ctx.snapshots.list() == []
    assert ctx.db.query("SELECT event FROM audit_events WHERE event='ingest.failed'")


def test_synthetic_provider_deterministic_and_labelled():
    a = SyntheticProvider(seed=3).fetch(["SPY"], "2020-01-01", "2020-12-31").bars
    b = SyntheticProvider(seed=3).fetch(["SPY", "GLD"], "2020-01-01", "2020-12-31").bars
    c = SyntheticProvider(seed=4).fetch(["SPY"], "2020-01-01", "2020-12-31").bars
    pd.testing.assert_frame_equal(a, b[b.symbol == "SPY"].reset_index(drop=True))
    assert not np.allclose(a.close, c.close)
    assert data_label(True) == SYNTHETIC_LABEL == "SYNTHETIC — NOT REAL MARKET PERFORMANCE"


# ----------------------------------------------------------------- snapshots / ingestion
def test_ingest_creates_immutable_hashed_snapshot_and_dedupes(ctx):
    prov = SyntheticProvider(seed=11)
    r1 = ingest(["SPY", "TLT"], "2021-01-01", "2021-12-31", prov, ctx.snapshots, ctx.db)
    assert r1.created and r1.snapshot.is_synthetic and len(r1.snapshot.dataset_hash.split(":")[0]) == 64
    r2 = ingest(["TLT", "SPY"], "2021-01-01", "2021-12-31", prov, ctx.snapshots, ctx.db)
    assert not r2.created and r2.snapshot.id == r1.snapshot.id
    assert len(ctx.snapshots.list()) == 1
    bars = ctx.snapshots.load(r1.snapshot.id)
    assert not bars.duplicated(["symbol", "session_date"]).any()
    mode = os.stat(r1.snapshot.storage_uri).st_mode
    assert not (mode & stat.S_IWUSR)  # read-only on disk
    assert ctx.snapshots.quality_status(r1.snapshot.id) in ("PASSED", "PASSED_WITH_WARNINGS")
    assert require_tradable_snapshot(ctx.snapshots, r1.snapshot.id).id == r1.snapshot.id


def test_tampered_snapshot_detected(ctx):
    r = ingest(["SPY"], "2021-01-01", "2021-06-30", SyntheticProvider(seed=1), ctx.snapshots, ctx.db)
    path = r.snapshot.storage_uri
    df = pd.read_parquet(path)
    df.loc[3, "close"] = df.loc[3, "close"] * 1.01
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    df.to_parquet(path, index=False)
    with pytest.raises(DataIntegrityError, match="hash mismatch"):
        ctx.snapshots.load(r.snapshot.id)


def test_failed_quality_blocks_snapshot_use(ctx):
    bars = make_bars("SPY", np.linspace(100, 110, 30))
    bars.loc[4, "low"] = bars.loc[4, "high"] + 5
    snap, _ = ctx.snapshots.create(bars, "synthetic", "unadjusted", True)
    ctx.snapshots.record_quality(snap.id, run_quality_checks(bars, "unadjusted"))
    with pytest.raises(QualityGateError):
        require_tradable_snapshot(ctx.snapshots, snap.id)


def test_cannot_mix_real_and_synthetic_rows(ctx):
    a = make_bars("SPY", [100, 101, 102], source="synthetic")
    b = make_bars("TLT", [90, 91, 92], source="yfinance")
    with pytest.raises(DataIntegrityError):
        ctx.snapshots.create(pd.concat([a, b]), "mixed", "unadjusted", True)


def test_duckdb_queries_snapshot(ctx):
    r = ingest(["SPY", "GLD"], "2022-01-01", "2022-03-31", SyntheticProvider(seed=2), ctx.snapshots, ctx.db)
    q = duckdb_store.query_bars(r.snapshot.storage_uri, ["GLD"], "2022-02-01", "2022-02-28")
    assert set(q.symbol) == {"GLD"} and len(q) == len(calendar.sessions("2022-02-01", "2022-02-28"))
    summ = duckdb_store.symbol_summary(r.snapshot.storage_uri)
    assert list(summ.symbol) == ["GLD", "SPY"]


# ----------------------------------------------------------------- input validation / safety
@pytest.mark.parametrize(
    "start,end",
    [("2024-05-01", "2024-01-01"), ("not-a-date", "2024-01-01"), ("2024-01-01", "2999-01-01"), ("1980-01-01", "2000-01-01")],
)
def test_invalid_dates_fail_clearly(start, end):
    with pytest.raises(InputValidationError):
        validate_date_range(start, end)


@pytest.mark.parametrize("sym", ["", "spy$", "THIS_IS_TOO_LONG", "1ABC"])
def test_invalid_symbols_rejected(ctx, sym):
    with pytest.raises(InputValidationError):
        ingest([sym], "2021-01-01", "2021-02-01", SyntheticProvider(), ctx.snapshots, ctx.db)


def test_invalid_uuid_rejected(ctx):
    with pytest.raises(InputValidationError):
        ctx.snapshots.get("not-a-uuid")


def test_live_trading_flag_refuses_to_start(monkeypatch, tmp_path):
    from quantlab.core.config import load_settings

    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(LiveTradingDisabledError):
        load_settings(tmp_path)
