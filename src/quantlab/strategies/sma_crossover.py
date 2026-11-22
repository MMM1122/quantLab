from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.core.errors import InputValidationError
from quantlab.strategies.base import Strategy, StrategyParams, rolling_mean


@dataclass
class SMAParams(StrategyParams):
    fast: int = 50
    slow: int = 200
    min_coverage: float = 0.9


class SMACrossover(Strategy):
    params: SMAParams
    id = "sma-crossover"
    name = "SMA Crossover"
    version = "1.0.0"
    description = (
        "Long (equal share of capital per symbol) while SMA(fast) > SMA(slow) of the total-return "
        "index; flat otherwise. Evaluated after each close, executed next open."
    )

    def __init__(self, params: SMAParams | None = None, **kwargs):
        self.params = params or SMAParams(**kwargs)
        p = self.params
        if not (1 <= p.fast < p.slow):
            raise InputValidationError("SMA crossover requires 1 <= fast < slow")
        if not (0 < p.min_coverage <= 1):
            raise InputValidationError("min_coverage must be in (0, 1]")

    @property
    def warmup_sessions(self) -> int:
        return self.params.slow

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        fast = rolling_mean(prices, p.fast, p.min_coverage)
        slow = rolling_mean(prices, p.slow, p.min_coverage)
        n = prices.shape[1]
        w = pd.DataFrame(np.where(fast > slow, 1.0 / n, 0.0), index=prices.index, columns=prices.columns)
        valid = fast.notna() & slow.notna() & prices.notna()
        return w.where(valid)
