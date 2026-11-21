"""Simulation fill model (an explicit ASSUMPTION, not a broker).

Fill price = next-session open adjusted adversely by ``slippage_bps`` (buys pay more, sells
receive less). Commission = max(min_commission, commission_bps x notional). Integer shares.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    min_commission: float = 0.0

    def __post_init__(self):
        if self.commission_bps < 0 or self.slippage_bps < 0 or self.min_commission < 0:
            raise ValueError("costs must be non-negative")

    def fill_price(self, ref_price: float, side: str) -> float:
        adj = self.slippage_bps / 1e4
        return ref_price * (1 + adj) if side == "BUY" else ref_price * (1 - adj)

    def commission(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        return max(self.min_commission, abs(notional) * self.commission_bps / 1e4)

    def max_affordable_qty(self, cash: float, price: float) -> int:
        if cash <= 0 or price <= 0:
            return 0
        q = int(math.floor(cash / (price * (1 + self.commission_bps / 1e4))))
        while q > 0 and q * price + self.commission(q * price) > cash + 1e-9:
            q -= 1
        return q

    def describe(self) -> dict:
        d = asdict(self)
        d["fill_model"] = "next-session open +/- slippage_bps (adverse), integer shares"
        return d


ZERO_COST = CostModel(0.0, 0.0, 0.0)
