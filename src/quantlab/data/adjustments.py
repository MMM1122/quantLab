"""Corporate-action adjustments.

Two distinct uses, never mixed:

1. ``back_adjusted_close`` – Yahoo/CRSP-style back adjustment (uses *future* events to
   rescale *past* prices). Only ever used for display or for the explicitly labelled
   ``adjusted_total_return`` research mode, never for causal signals.
2. ``total_return_index`` – a forward, causal total-return index. Value at t depends only
   on data up to t. Strategies compute signals on this series so splits do not create
   fake crossovers and dividends are not ignored.

Dividend double counting is avoided by construction: the backtest engine either trades raw
(or split-adjusted) prices and credits cash dividends, or trades adjusted prices and credits
nothing. See ``quantlab.backtest.engine``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.domain.models import PriceBasis


def back_adjusted_close(
    close: np.ndarray, dividend: np.ndarray, split_ratio: np.ndarray, price_basis: str
) -> np.ndarray:
    """Back-adjusted close for one symbol (rows in session order, no gaps assumed).

    factor_t = prod over events e with ex-date > t of (1 - div_e / close_{e-1}) * (1 / split_e)
    The split term is applied only for ``unadjusted`` price basis.
    """
    n = len(close)
    factor = np.ones(n)
    running = 1.0
    apply_splits = price_basis == PriceBasis.UNADJUSTED
    for t in range(n - 1, 0, -1):
        # event at t affects all rows < t
        m = 1.0
        if dividend[t] and dividend[t] > 0 and close[t - 1] > 0:
            m *= 1.0 - dividend[t] / close[t - 1]
        if apply_splits and split_ratio[t] and split_ratio[t] != 1.0:
            m *= 1.0 / split_ratio[t]
        running *= m
        factor[t - 1] = running
    return close * factor


def total_return_index(
    close: np.ndarray,
    dividend: np.ndarray,
    split_ratio: np.ndarray,
    price_basis: str,
) -> np.ndarray:
    """Causal total-return index on a session grid; NaN where ``close`` is NaN.

    Between consecutive observed closes p (at s) and c (at t):
        TR_t = TR_s * (c * S + D) / p
    where S is the product of split ratios in (s, t] (only for unadjusted basis) and D is the
    sum of dividends in (s, t] expressed per *current* share. Starts at the first observed close.
    """
    n = len(close)
    out = np.full(n, np.nan)
    apply_splits = price_basis == PriceBasis.UNADJUSTED
    tr = np.nan
    prev = np.nan
    acc_split = 1.0
    acc_div = 0.0
    for t in range(n):
        sr = split_ratio[t] if apply_splits and np.isfinite(split_ratio[t]) and split_ratio[t] > 0 else 1.0
        dv = dividend[t] if np.isfinite(dividend[t]) else 0.0
        # dividends are quoted per share at ex-date; convert previously accumulated
        # dividends (per old share) into per-current-share terms when a split happens
        acc_div = acc_div / sr + dv
        acc_split *= sr
        c = close[t]
        if not np.isfinite(c) or c <= 0:
            continue
        if not np.isfinite(prev):
            tr = c
        else:
            tr = tr * (c * acc_split + acc_div * acc_split) / prev
        out[t] = tr
        prev = c
        acc_split = 1.0
        acc_div = 0.0
    return out


def adjusted_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale open/high/low/close by adj_close/close (adjusted_total_return research mode)."""
    out = bars.copy()
    f = out["adj_close"] / out["close"]
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] * f
    out["dividend"] = 0.0
    out["split_ratio"] = 1.0
    return out
