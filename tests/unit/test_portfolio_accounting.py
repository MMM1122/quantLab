from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from quantlab.core.errors import AccountingInvariantError
from quantlab.portfolio.accounting import assert_cash_non_negative, net_asset_value, position_values, weights


def test_position_values_and_invariants():
    v = position_values(np.array([2, 0, 3]), np.array([10.0, np.nan, 5.0]))
    assert list(v) == [20.0, 0.0, 15.0]  # flat position may lack a price
    with pytest.raises(AccountingInvariantError):
        position_values(np.array([-1]), np.array([10.0]))
    with pytest.raises(AccountingInvariantError, match="without any price"):
        position_values(np.array([1]), np.array([np.nan]))


def test_nav_works_for_decimal_and_float():
    nav = net_asset_value(Decimal("100.50"), {"SPY": Decimal(2), "TLT": Decimal(0)}, {"SPY": Decimal("10.25")})
    assert nav == Decimal("121.00")
    assert net_asset_value(1.0, {"A": 2}, {"A": 3.0}) == 7.0
    with pytest.raises(AccountingInvariantError):
        net_asset_value(0.0, {"A": 1}, {})
    assert weights({"A": Decimal(25), "B": Decimal(75)}, Decimal(100)) == {"A": 0.25, "B": 0.75}
    assert weights({"A": 1.0}, 0.0) == {"A": None}


def test_cash_invariant():
    assert_cash_non_negative(-1e-9)  # within float tolerance
    assert_cash_non_negative(Decimal("0.00"), tolerance=0)
    with pytest.raises(AccountingInvariantError):
        assert_cash_non_negative(Decimal("-0.01"), tolerance=0)
