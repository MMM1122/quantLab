"""Pre-trade risk rules shared by the backtest engine and the paper broker.

Each rule returns a ``RiskDecision``. Rules are pure functions over numbers (float or
Decimal) so they can be unit-tested and reused. They run BEFORE an order is created.
"""

from __future__ import annotations

from quantlab.domain.models import OrderSide, RiskDecision

OK = RiskDecision(True)


def long_only(side: OrderSide, quantity, held, pending_sells=0) -> RiskDecision:
    if quantity <= 0:
        return RiskDecision(False, "POSITIVE_QUANTITY", "order quantity must be > 0")
    if side == OrderSide.SELL and quantity > held - pending_sells:
        return RiskDecision(
            False, "NO_SHORTING", f"sell {quantity} exceeds available position {held - pending_sells}"
        )
    return OK


def sufficient_cash(side: OrderSide, est_cost, cash, pending_buy_cost=0) -> RiskDecision:
    if side == OrderSide.BUY and est_cost > cash - pending_buy_cost:
        return RiskDecision(
            False, "INSUFFICIENT_CASH", f"estimated cost {est_cost} exceeds available cash {cash - pending_buy_cost}"
        )
    return OK


def max_position_weight(post_trade_value, nav, max_weight) -> RiskDecision:
    if nav <= 0:
        return RiskDecision(False, "NAV_NONPOSITIVE", "NAV must be positive")
    if post_trade_value > nav * max_weight:
        return RiskDecision(
            False,
            "MAX_POSITION_WEIGHT",
            f"post-trade position {post_trade_value} exceeds {max_weight:.0%} of NAV {nav}",
        )
    return OK


def fresh_data(bar_session_index: int | None, current_session_index: int, max_staleness: int) -> RiskDecision:
    if bar_session_index is None:
        return RiskDecision(False, "NO_MARKET_DATA", "no bar available for this symbol")
    stale = current_session_index - bar_session_index
    if stale > max_staleness:
        return RiskDecision(False, "STALE_DATA", f"latest bar is {stale} session(s) old (max {max_staleness})")
    return OK


def kill_switch(active: bool) -> RiskDecision:
    if active:
        return RiskDecision(False, "KILL_SWITCH", "kill switch is active; new orders are blocked")
    return OK


def first_failure(*decisions: RiskDecision) -> RiskDecision:
    for d in decisions:
        if not d.approved:
            return d
    return OK
