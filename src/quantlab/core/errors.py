"""Typed exception hierarchy. Never swallow these silently."""


class QuantLabError(Exception):
    """Base class for all QuantLab errors."""


class ConfigError(QuantLabError):
    pass


class LiveTradingDisabledError(QuantLabError):
    """Raised for any attempt to enable or use live trading. V1 has no live broker."""


class InputValidationError(QuantLabError):
    """Invalid user input (symbol, dates, ids, parameters)."""


class DataUnavailableError(QuantLabError):
    """Market data could not be fetched. Must never lead to trading actions."""


class DataIntegrityError(QuantLabError):
    """Stored data does not match its recorded hash, or violates a data contract."""


class QualityGateError(QuantLabError):
    """A snapshot failed blocking data-quality checks."""


class NotFoundError(QuantLabError):
    pass


class RiskRejection(QuantLabError):
    """An order intent was rejected by a pre-trade risk rule."""

    def __init__(self, rule: str, message: str):
        super().__init__(f"{rule}: {message}")
        self.rule = rule
        self.message = message


class AccountingInvariantError(QuantLabError):
    """Cash/position/NAV invariant violated. Indicates a bug; never ignore."""
