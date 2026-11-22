from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.core.errors import InputValidationError
from quantlab.data import calendar
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider, YFinanceProvider
from quantlab.data.refresh import diff_snapshots, lineage, refresh_snapshot

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------- API auth
@pytest.fixture
def client_factory(ctx):
    from quantlab.api.app import create_app

    def make(token):
        return TestClient(create_app(ctx=ctx, api_token=token))

    return make


def test_api_token_required_when_configured(client_factory):
    with client_factory("s3cret") as c:
        h = c.get("/health").json()
        assert h["data"]["auth_required"] is True and "db" not in h["data"]  # no paths leaked publicly
        assert c.get("/strategies").status_code == 401
        assert c.get("/strategies", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/strategies", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.post("/orders/live", json={}).status_code == 401  # auth first, then 403 inside
        assert c.get("/openapi.json").status_code == 200


def test_api_open_without_token(client_factory):
    with client_factory("") as c:
        assert c.get("/strategies").status_code == 200
        assert c.get("/health").json()["data"]["auth_required"] is False


# ----------------------------------------------------------------- refresh / revisions
def _frame(n, start="2024-01-02", bump_day=None, extra_div_day=None):
    sess = calendar.sessions(start, "2024-12-31")[:n]
    idx = pd.DatetimeIndex(sess).tz_localize("America/New_York")
    c = 400 + 0.5 * np.arange(n, dtype=float)  # same price for the same day regardless of n
    if bump_day is not None:
        c[bump_day] *= 1.01  # provider revised a historical close
    div = np.zeros(n)
    adj = c.copy()
    if extra_div_day is not None:
        div[extra_div_day] = 2.0
        f = 1 - 2.0 / c[extra_div_day - 1]
        adj[:extra_div_day] = c[:extra_div_day] * f  # back-adjustment rescales earlier rows
    return pd.DataFrame({"Open": c - 1, "High": c + 2, "Low": c - 3, "Close": c, "Adj Close": adj,
                         "Volume": np.full(n, 5e7), "Dividends": div, "Stock Splits": np.zeros(n)}, index=idx)


class FakeYahoo(YFinanceProvider):
    def __init__(self, frame_fn):
        super().__init__(downloader=lambda sym, a, b: frame_fn(sym))


def _ingest_real(ctx, frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ingest(["SPY"], "2024-01-02", calendar.sessions("2024-01-02", "2024-12-31")[len(frame) - 1]
                      .strftime("%Y-%m-%d"), FakeYahoo(lambda s: frame), ctx.snapshots, ctx.db).snapshot


def test_refresh_detects_appends_revisions_and_adjustment_changes(ctx):
    base = _ingest_real(ctx, _frame(60))
    newer = _frame(65, bump_day=10, extra_div_day=62)
    end = calendar.sessions("2024-01-02", "2024-12-31")[64].strftime("%Y-%m-%d")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = refresh_snapshot(ctx, "latest", FakeYahoo(lambda s: newer), end)
    assert res.base_snapshot_id == base.id and res.new_snapshot_id != base.id
    t = res.diff["totals"]
    assert t["appended"] == 5 and t["removed"] == 0
    assert t["raw_revised"] == 1 and t["raw_revised_by_column"]["close"] == 1
    assert t["adj_only_revised"] == 59  # all 60 overlapping (pre-dividend) rows except the revised one
    assert res.diff["status"] == "REVISIONS_DETECTED"
    sample = res.diff["symbols"]["SPY"]["raw_revised_samples"][0]
    assert "close" in sample and sample["close"][1] == pytest.approx(sample["close"][0] * 1.01)
    # old snapshot untouched and still loadable (hash verified); lineage + risk event recorded
    assert len(ctx.snapshots.load(base.id)) == 60
    assert lineage(ctx, base.id)[0]["new_snapshot_id"] == res.new_snapshot_id
    assert ctx.db.query("SELECT rule FROM risk_events WHERE rule='PROVIDER_REVISION'")


def test_refresh_with_identical_data_reports_unchanged(ctx):
    base = _ingest_real(ctx, _frame(40))
    end = base.end
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = refresh_snapshot(ctx, base.id, FakeYahoo(lambda s: _frame(40)), end)
    assert res.new_snapshot_id == base.id and not res.ingest.created
    assert res.diff["status"] == "UNCHANGED"


def test_refresh_rejects_synthetic_and_derived(ctx):
    s = ingest(["SPY"], "2020-01-01", "2020-12-31", SyntheticProvider(), ctx.snapshots, ctx.db).snapshot
    with pytest.raises(InputValidationError):
        refresh_snapshot(ctx, s.id, YFinanceProvider())


def test_removed_rows_flagged():
    a = pd.DataFrame({"symbol": "SPY", "session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                      **{c: [1.0, 2.0, 3.0] for c in ("open", "high", "low", "close", "adj_close", "volume")},
                      "dividend": 0.0, "split_ratio": 1.0})
    d = diff_snapshots(a, a.drop(index=1))
    assert d["totals"]["removed"] == 1 and d["status"] == "REVISIONS_DETECTED"
    assert d["symbols"]["SPY"]["removed_dates"] == ["2024-01-03"]
