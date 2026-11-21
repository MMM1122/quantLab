from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.core.errors import InputValidationError
from quantlab.strategies.base import Strategy, StrategyParams, rolling_mean


@dataclass
class RSIParams(StrategyParams):
    window: int = 14
    entry: float = 30.0  # enter when RSI < entry (oversold)
    exit: float = 55.0  # exit once RSI > exit
    max_hold: int = 20  # time stop, sessions after the latest oversold reading
    min_coverage: float = 0.9


def rsi(prices: pd.DataFrame, window: int, min_coverage: float) -> pd.DataFrame:
    """Cutler's RSI (simple moving averages of gains/losses).

    Simple averages give the indicator a finite memory, which keeps it causal and makes
    results identical whether computed on full or truncated history.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = rolling_mean(gain, window, min_coverage)
    al = rolling_mean(loss, window, min_coverage)
    rs = ag / al.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(~((al == 0) & (ag > 0)), 100.0)
    out = out.where(~((al == 0) & (ag == 0)), 50.0)
    return out.where(ag.notna() & al.notna())


class RSIMeanReversion(Strategy):
    id = "rsi-mean-reversion"
    name = "RSI Mean Reversion"
    version = "1.0.0"
    description = (
        "Long after RSI(window) closes below `entry`; hold until RSI rises above `exit` or "
        "`max_hold` sessions pass since the latest oversold reading. Flat otherwise."
    )

    def __init__(self, params: RSIParams | None = None, **kwargs):
        self.params = params or RSIParams(**kwargs)
        p = self.params
        if not (2 <= p.window and 0 < p.entry < p.exit < 100 and p.max_hold >= 1):
            raise InputValidationError("RSI params require window>=2, 0<entry<exit<100, max_hold>=1")

    @property
    def warmup_sessions(self) -> int:
        return self.params.window + 1

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        r = rsi(prices, p.window, p.min_coverage)
        n = prices.shape[1]
        out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        pos = np.arange(len(prices))
        for c in prices.columns:
            rv = r[c].to_numpy()
            entry = rv < p.entry
            # index of the latest oversold reading at or before t
            last_entry = pd.Series(np.where(entry, pos, np.nan)).ffill().to_numpy()
            age = pos - last_entry
            # exit if RSI exceeded `exit` at any point after the latest entry
            above = (rv > p.exit).astype(float)
            seg = np.cumsum(entry)
            exited = pd.Series(above).groupby(seg).cummax().to_numpy() > 0
            hold = np.isfinite(last_entry) & (age < p.max_hold) & ~exited
            w = np.where(hold, 1.0 / n, 0.0)
            valid = np.isfinite(rv) & prices[c].notna().to_numpy()
            out[c] = np.where(valid, w, np.nan)
        return out
