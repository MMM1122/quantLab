from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantlab.strategies.base import Strategy, StrategyParams


@dataclass
class BuyHoldParams(StrategyParams):
    weights: dict[str, float] = field(default_factory=dict)  # empty => equal weight


class BuyAndHold(Strategy):
    id = "buy-hold"
    name = "Buy & Hold"
    version = "1.0.0"
    description = "Buy once (equal weight unless specified) at the first open after data is available; never rebalance."

    def __init__(self, params: BuyHoldParams | None = None, **kwargs):
        self.params = params or BuyHoldParams(**kwargs)

    @property
    def warmup_sessions(self) -> int:
        return 1

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = list(prices.columns)
        w = self.params.weights or {c: 1.0 / len(cols) for c in cols}
        out = pd.DataFrame(index=prices.index, columns=cols, dtype=float)
        for c in cols:
            out[c] = float(w.get(c, 0.0))
        return out.where(prices.notna())
