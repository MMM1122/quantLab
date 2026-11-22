"""Portfolio accounting shared by the backtest engine (float) and the paper broker (Decimal).

Pure functions over numbers so both numeric types work. They encode the invariants that tests
guard everywhere: positions >= 0, cash >= -tolerance, every held position has a price, and
NAV = cash + Σ quantity × price.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from quantlab.core.errors import AccountingInvariantError

CASH_TOLERANCE = 1e-6


def position_values(quantities: np.ndarray, prices: np.ndarray, when: str = "") -> np.ndarray:
    """Market value per position (float arrays). Raises if a held position cannot be valued."""
    q = np.asarray(quantities)
    p = np.asarray(prices, dtype=float)
    if np.any(q < 0):
        raise AccountingInvariantError(f"negative position {when}".strip())
    held = q > 0
    if np.any(held & ~np.isfinite(p)):
        raise AccountingInvariantError(f"cannot value a held position without any price {when}".strip())
    return np.where(held, q * np.nan_to_num(p, nan=0.0), 0.0)


def net_asset_value(cash: Any, quantities: Mapping[str, Any], prices: Mapping[str, Any]) -> Any:
    """NAV = cash + Σ qty × price, for float or Decimal inputs (symbols without a price must be flat)."""
    total = cash
    for sym, q in quantities.items():
        if q == 0:
            continue
        if q < 0:
            raise AccountingInvariantError(f"negative position in {sym}")
        px = prices.get(sym)
        if px is None:
            raise AccountingInvariantError(f"no price to value {sym}")
        total = total + q * px
    return total


def weights(values: Mapping[str, Any], nav: Any) -> dict[str, float | None]:
    """Position weights as floats; None when NAV is not positive."""
    return {s: (float(v / nav) if nav > 0 else None) for s, v in values.items()}


def assert_cash_non_negative(cash: Any, when: str = "", tolerance: float = CASH_TOLERANCE) -> None:
    if cash < -tolerance:
        raise AccountingInvariantError(f"negative cash {cash} {when}".strip())
