from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.core.errors import InputValidationError
from quantlab.strategies.base import Strategy, StrategyParams


@dataclass
class MomentumParams(StrategyParams):
    lookback: int = 252  # ~12 months
    skip: int = 21  # skip the most recent ~1 month (12-1 momentum)
    top_k: int = 2
    absolute_filter: bool = True  # only hold assets with positive momentum, else cash


class CrossSectionalMomentum(Strategy):
    id = "momentum"
    name = "Cross-sectional Momentum (12-1)"
    version = "1.0.0"
    description = (
        "On the first session of each month, rank symbols by total return from t-lookback to "
        "t-skip and hold the top_k equally weighted (cash for those with non-positive momentum)."
    )

    def __init__(self, params: MomentumParams | None = None, **kwargs):
        self.params = params or MomentumParams(**kwargs)
        p = self.params
        if not (0 <= p.skip < p.lookback and p.top_k >= 1):
            raise InputValidationError("momentum requires 0 <= skip < lookback and top_k >= 1")

    @property
    def warmup_sessions(self) -> int:
        return self.params.lookback + 1

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        # lagged lookups may fall on a missing observation; use the most recent prior value
        # (at most 5 sessions back) — still strictly historical data.
        hist = prices.ffill(limit=5)
        score = hist.shift(p.skip) / hist.shift(p.lookback) - 1.0
        idx = prices.index
        first_of_month = np.r_[True, idx.month[1:] != idx.month[:-1]]
        out = pd.DataFrame(np.nan, index=idx, columns=prices.columns)
        k = min(p.top_k, prices.shape[1])
        for i in np.where(first_of_month)[0]:
            row = score.iloc[i]
            cur = prices.iloc[i]
            if row.isna().any() or cur.isna().any():
                continue  # cannot rank with incomplete information -> no opinion
            ranked = row.sort_values(ascending=False, kind="mergesort")
            chosen = list(ranked.index[:k])
            w = pd.Series(0.0, index=prices.columns)
            for c in chosen:
                if not p.absolute_filter or ranked[c] > 0:
                    w[c] = 1.0 / k
            out.iloc[i] = w.to_numpy()
        return out
