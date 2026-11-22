from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.errors import InputValidationError
from quantlab.strategies.base import Strategy, StrategyParams


@dataclass
class VolTargetParams(StrategyParams):
    base: str = "buy-hold"
    base_params: dict[str, Any] = field(default_factory=dict)
    target_vol: float = 0.10  # annualised
    window: int = 63  # sessions of trailing realised volatility
    step: float = 0.10  # weights rounded DOWN to multiples of `step` to limit churn
    min_coverage: float = 0.9


class VolatilityTarget(Strategy):
    params: VolTargetParams
    id = "vol-target"
    name = "Volatility Target"
    version = "1.0.0"
    description = (
        "Scales a base strategy's weights by min(1, target_vol / trailing realised volatility) — never "
        "levered. Weights are rounded down to `step` so small volatility changes do not cause daily trades."
    )

    def __init__(self, params: VolTargetParams | None = None, **kwargs):
        self.params = params or VolTargetParams(**kwargs)
        p = self.params
        if p.base == self.id:
            raise InputValidationError("vol-target cannot wrap itself")
        if not (0 < p.target_vol <= 1 and p.window >= 10 and 0 < p.step <= 1 and 0 < p.min_coverage <= 1):
            raise InputValidationError("vol-target requires 0<target_vol<=1, window>=10, 0<step<=1")
        from quantlab.strategies.registry import create_strategy  # local import: registry imports this module

        self._base = create_strategy(p.base, p.base_params)

    @property
    def warmup_sessions(self) -> int:
        return max(self.params.window + 1, self._base.warmup_sessions)

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        base_w = self._base.target_weights(prices)
        rets = prices.pct_change(fill_method=None)
        vol = rets.rolling(p.window, min_periods=max(2, math.ceil(p.min_coverage * p.window))).std() * math.sqrt(252)
        scale = (p.target_vol / vol.replace(0.0, np.nan)).clip(upper=1.0)
        w = base_w * scale
        w = np.floor(w / p.step + 1e-9) * p.step
        return w.where(base_w.notna() & vol.notna())
