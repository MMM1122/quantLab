"""Known Yahoo/yfinance quirks reproduced with crafted frames (the real feed is unreachable in CI)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quantlab.data import calendar
from quantlab.data.ingestion import ingest
from quantlab.data.providers import YFinanceProvider, clean_provider_rows
from quantlab.data.validation import run_quality_checks

NY = "America/New_York"


def _frame(dates, close=None, div=None, split=None, nan_rows=()):
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(NY)
    n = len(idx)
    c = np.asarray(close if close is not None else np.linspace(500, 510, n), float)
    f = pd.DataFrame({"Open": c - 1, "High": c + 2, "Low": c - 3, "Close": c, "Adj Close": c, "Volume": 5e7,
                      "Dividends": div if div is not None else 0.0, "Stock Splits": split if split is not None else 0.0},
                     index=idx)
    for i in nan_rows:
        f.iloc[i, :6] = np.nan
    return f


def _fetch(frame, now="2024-03-15 20:00"):
    prov = YFinanceProvider(downloader=lambda *a: frame, now=lambda: pd.Timestamp(now, tz=NY))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prov.fetch(["SPY"], "2024-03-01", "2024-03-20")


def test_intraday_partial_bar_is_dropped():
    sess = calendar.sessions("2024-03-01", "2024-03-15")
    res = _fetch(_frame(sess), now="2024-03-15 11:30")  # market still open on the 15th
    assert res.bars.session_date.max() == pd.Timestamp("2024-03-14")
    assert any("not yet closed" in w for w in res.warnings)
    assert res.parameters["last_completed_session"] == "2024-03-14"
    after_close = _fetch(_frame(sess), now="2024-03-15 16:30")
    assert after_close.bars.session_date.max() == pd.Timestamp("2024-03-15")


def test_dividend_only_row_is_merged_into_its_session():
    sess = list(calendar.sessions("2024-03-01", "2024-03-14"))
    dates = sess + [sess[5]]  # second row for the same session carrying only the dividend
    div = [0.0] * len(sess) + [1.75]
    f = _frame(dates, close=list(np.linspace(500, 510, len(sess))) + [np.nan], div=div, nan_rows=[len(sess)])
    res = _fetch(f)
    b = res.bars
    assert not b.duplicated(["symbol", "session_date"]).any()
    assert b.loc[b.session_date == sess[5], "dividend"].iloc[0] == 1.75
    assert any("merged action-only row" in w for w in res.warnings)
    assert res.parameters["provider_cleanup"]  # recorded in snapshot provenance


def test_weekend_dividend_moves_to_next_session_and_empty_rows_dropped():
    sess = list(calendar.sessions("2024-03-01", "2024-03-14"))
    dates = sess + ["2024-03-09", "2024-03-10"]  # Saturday dividend row, Sunday empty row
    close = list(np.linspace(500, 510, len(sess))) + [np.nan, np.nan]
    div = [0.0] * len(sess) + [1.5, 0.0]
    res = _fetch(_frame(dates, close=close, div=div, nan_rows=[len(sess), len(sess) + 1]))
    b = res.bars
    assert b.loc[b.session_date == pd.Timestamp("2024-03-11"), "dividend"].iloc[0] == 1.5  # next session (Mon)
    assert set(b.session_date) <= set(calendar.sessions("2024-03-01", "2024-03-14"))
    assert any("moved from non-session day 2024-03-09" in w for w in res.warnings)
    assert any("empty provider row" in w for w in res.warnings)
    assert not run_quality_checks(b, "split_adjusted").blocking


def test_exact_duplicates_collapse_but_conflicts_are_flagged():
    sess = list(calendar.sessions("2024-03-01", "2024-03-14"))
    exact = _fetch(_frame(sess + [sess[3]], close=list(np.linspace(500, 510, len(sess))) + [np.linspace(500, 510, len(sess))[3]]))
    assert not exact.bars.duplicated(["symbol", "session_date"]).any()
    assert any("exact duplicate" in w for w in exact.warnings)
    conflict = _fetch(_frame(sess + [sess[3]], close=list(np.linspace(500, 510, len(sess))) + [999.0]))
    rep = run_quality_checks(conflict.bars, "split_adjusted")
    assert any(i.issue_type == "DUPLICATE_BAR" and i.severity == "ERROR" for i in rep.issues)  # never silently picked


def test_all_rows_in_progress_means_unavailable(ctx):
    from quantlab.core.errors import DataUnavailableError

    f = _frame(["2024-03-15"])
    prov = YFinanceProvider(downloader=lambda *a: f, now=lambda: pd.Timestamp("2024-03-15 10:00", tz=NY))
    with warnings.catch_warnings(), pytest.raises(DataUnavailableError):
        warnings.simplefilter("ignore")
        ingest(["SPY"], "2024-03-14", "2024-03-15", prov, ctx.snapshots, ctx.db)


def test_cleanup_is_a_noop_on_clean_data():
    sess = calendar.sessions("2024-03-01", "2024-03-14")
    df = YFinanceProvider._normalize("SPY", _frame(sess))
    out, notes = clean_provider_rows("SPY", df, pd.Timestamp("2024-03-14"))
    pd.testing.assert_frame_equal(out, df.reset_index(drop=True))
    assert notes == []
