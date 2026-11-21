"""Typed domain contracts.

Field semantics
---------------
* ``session_date``: the exchange trading session (America/New_York calendar date, XNYS),
  stored tz-naive. Never a raw calendar day.
* Event timestamps (``*_at``) are timezone-aware UTC.
* Money/quantities in the paper-trading ledger are ``Decimal``; analytic series
  (backtest equity curves, metrics) are float64 with acknowledged rounding.
* ``price_basis`` describes the OHLC columns of a snapshot:
  - ``split_adjusted``: prices already adjusted for splits but NOT dividends (Yahoo "Close").
    Split events are informational; holdings must NOT be split-adjusted again.
  - ``unadjusted``: as-traded prices; split events must be applied to holdings.
  In both cases ``adj_close`` is split+dividend back-adjusted (total-return style).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

BAR_COLUMNS = [
    "symbol",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividend",
    "split_ratio",
    "currency",
    "source",
]
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


class PriceBasis(StrEnum):
    SPLIT_ADJUSTED = "split_adjusted"
    UNADJUSTED = "unadjusted"


class PriceMode(StrEnum):
    """How the backtest engine treats corporate actions."""

    RAW_WITH_DIVIDENDS = "raw_with_dividends"  # trade raw/split-adj prices, credit cash dividends
    ADJUSTED_TOTAL_RETURN = "adjusted_total_return"  # trade adj prices, NO dividend credit (research mode)


class Severity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    BLOCK = "BLOCK"


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Mode(StrEnum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    RESEARCH = "RESEARCH"
    LIVE = "LIVE"  # exists only so it can be explicitly rejected


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    CREATED = "CREATED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"


OPEN_ORDER_STATUSES = (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)


@dataclass(frozen=True)
class Asset:
    symbol: str
    exchange: str = "ARCX"
    currency: str = "USD"
    asset_type: str = "ETF"


@dataclass
class DataSnapshot:
    id: str
    provider: str
    dataset_hash: str
    storage_uri: str
    parameters: dict[str, Any]
    price_basis: str
    is_synthetic: bool
    retrieved_at: str
    symbols: list[str]
    start: str
    end: str
    n_rows: int
    parent_id: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityIssue:
    issue_type: str
    severity: Severity
    message: str
    symbol: str | None = None
    session_date: str | None = None
    row_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = str(self.severity)
        return d


@dataclass
class QualityReport:
    issues: list[DataQualityIssue]
    n_rows: int
    symbols: list[str]
    checks_run: list[str]

    @property
    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for i in self.issues:
            out[str(i.severity)] += 1
        return out

    @property
    def blocking(self) -> bool:
        return any(i.severity in (Severity.ERROR, Severity.BLOCK) for i in self.issues)

    @property
    def status(self) -> str:
        if self.blocking:
            return "FAILED"
        return "PASSED_WITH_WARNINGS" if any(i.severity == Severity.WARN for i in self.issues) else "PASSED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blocking": self.blocking,
            "counts": self.counts,
            "n_rows": self.n_rows,
            "symbols": self.symbols,
            "checks_run": self.checks_run,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: OrderSide
    quantity: Decimal
    client_order_id: str
    order_type: str = "MARKET"


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    rule: str | None = None
    reason: str | None = None


@dataclass
class Order:
    id: str
    account_id: str
    symbol: str
    client_order_id: str
    side: OrderSide
    quantity: Decimal
    filled_quantity: Decimal
    order_type: str
    status: OrderStatus
    reason: str | None
    decision_session: date
    created_at: datetime


@dataclass
class Execution:
    id: str
    order_id: str
    execution_ref: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    session_date: date
    executed_at: datetime
