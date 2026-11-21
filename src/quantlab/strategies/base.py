"""Strategy interface.

A strategy maps a panel of *signal prices* (causal total-return index; sessions x symbols,
NaN where no fresh observation is available) to *target weights* (same shape).

Contract (enforced by tests/unit/test_strategies.py):
* Row t of the output depends only on input rows <= t (no look-ahead).
* Weights are in [0, 1] and each row sums to <= 1 (long-only, no leverage).
* NaN means "no opinion / cannot decide" — the engine keeps the current position. This is
  how missing or stale data is handled by the strict policies in DQ-001.
* The engine executes a decision made at the close of t no earlier than the open of t+1.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import pandas as pd


@dataclass
class StrategyParams:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Strategy(ABC):
    id: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str]
    params: StrategyParams

    @property
    @abstractmethod
    def warmup_sessions(self) -> int:
        """Sessions of history needed before the first valid signal."""

    @abstractmethod
    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame: ...

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "parameters": self.params.to_dict(),
            "warmup_sessions": self.warmup_sessions,
        }


def rolling_mean(s: pd.Series | pd.DataFrame, window: int, min_coverage: float):
    """Rolling mean over the last ``window`` sessions using available observations.

    Requires at least ceil(min_coverage * window) observations. With complete data and
    min_coverage=1 this is the textbook SMA; with gaps it averages what was observed.
    """
    return s.rolling(window, min_periods=max(1, math.ceil(min_coverage * window))).mean()
