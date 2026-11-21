"""Identifier helpers."""

from __future__ import annotations

import re
import uuid

from quantlab.core.errors import InputValidationError

_NAMESPACE = uuid.UUID("6f1c3b8e-6a55-4f39-9c2b-0b9c2f6a7d10")
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def new_id() -> str:
    return str(uuid.uuid4())


def deterministic_id(*parts: object) -> str:
    """UUIDv5 derived from the given parts (stable across runs)."""
    return str(uuid.uuid5(_NAMESPACE, "|".join(str(p) for p in parts)))


def validate_uuid(value: str, field: str = "id") -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise InputValidationError(f"{field} must be a valid UUID, got {value!r}") from exc


def validate_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise InputValidationError(f"symbol must be a string, got {symbol!r}")
    s = symbol.strip().upper()
    if not _SYMBOL_RE.match(s):
        raise InputValidationError(f"invalid symbol {symbol!r}")
    return s
