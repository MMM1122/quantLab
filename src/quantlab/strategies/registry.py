from __future__ import annotations

from typing import Any

from quantlab.core.errors import InputValidationError
from quantlab.strategies.base import Strategy
from quantlab.strategies.buy_hold import BuyAndHold, BuyHoldParams
from quantlab.strategies.momentum import CrossSectionalMomentum, MomentumParams
from quantlab.strategies.rsi_mean_reversion import RSIMeanReversion, RSIParams
from quantlab.strategies.sma_crossover import SMACrossover, SMAParams

STRATEGIES: dict[str, tuple[type[Strategy], type]] = {
    BuyAndHold.id: (BuyAndHold, BuyHoldParams),
    SMACrossover.id: (SMACrossover, SMAParams),
    RSIMeanReversion.id: (RSIMeanReversion, RSIParams),
    CrossSectionalMomentum.id: (CrossSectionalMomentum, MomentumParams),
}
ALIASES = {
    "buy_hold": "buy-hold",
    "buy-and-hold": "buy-hold",
    "ma-crossover": "sma-crossover",
    "ma_crossover": "sma-crossover",
    "sma_crossover": "sma-crossover",
    "rsi": "rsi-mean-reversion",
    "rsi_mean_reversion": "rsi-mean-reversion",
    "mean-reversion": "rsi-mean-reversion",
}


def normalize_strategy_id(name: str) -> str:
    sid = ALIASES.get(name, name)
    if sid not in STRATEGIES:
        raise InputValidationError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}")
    return sid


def create_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    sid = normalize_strategy_id(name)
    cls, pcls = STRATEGIES[sid]
    try:
        return cls(pcls(**(params or {})))
    except TypeError as exc:
        raise InputValidationError(f"invalid parameters for {sid}: {exc}") from exc


def list_strategies() -> list[dict[str, Any]]:
    return [cls(pcls()).describe() for cls, pcls in STRATEGIES.values()]
