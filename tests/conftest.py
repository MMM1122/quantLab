from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("LIVE_TRADING_ENABLED", "false")
os.environ["QUANTLAB_LLM_PROVIDER"] = "none"

from quantlab.context import build_context  # noqa: E402
from quantlab.data import calendar  # noqa: E402
from quantlab.data.providers import SyntheticProvider  # noqa: E402


@pytest.fixture
def ctx(tmp_path):
    return build_context(tmp_path / "data")


@pytest.fixture(scope="session")
def synth_bars() -> pd.DataFrame:
    return SyntheticProvider(seed=7).fetch(["SPY", "QQQ", "TLT"], "2018-01-01", "2023-12-29").bars


def make_bars(
    symbol: str,
    closes: list[float] | np.ndarray,
    start: str = "2024-01-02",
    opens: list[float] | np.ndarray | None = None,
    dividends: dict[int, float] | None = None,
    splits: dict[int, float] | None = None,
    source: str = "synthetic",
    volume: float = 1e6,
) -> pd.DataFrame:
    """Hand-made bars on real XNYS sessions for exact accounting tests."""
    closes = np.asarray(closes, dtype=float)
    sess = calendar.sessions(start, pd.Timestamp(start) + pd.Timedelta(days=len(closes) * 2 + 10))[: len(closes)]
    opens_arr = np.asarray(opens if opens is not None else closes, dtype=float)
    div = np.zeros(len(closes))
    spl = np.ones(len(closes))
    for i, v in (dividends or {}).items():
        div[i] = v
    for i, v in (splits or {}).items():
        spl[i] = v
    from quantlab.data.adjustments import back_adjusted_close

    adj = back_adjusted_close(closes, div, spl, "unadjusted")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "session_date": sess,
            "open": opens_arr,
            "high": np.maximum(opens_arr, closes) * 1.001,
            "low": np.minimum(opens_arr, closes) * 0.999,
            "close": closes,
            "adj_close": adj,
            "volume": volume,
            "dividend": div,
            "split_ratio": spl,
            "currency": "USD",
            "source": source,
        }
    )
