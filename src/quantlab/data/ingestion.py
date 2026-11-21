"""Ingest -> canonicalise -> snapshot -> validate -> persist quality report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quantlab.core.errors import DataUnavailableError, InputValidationError, QualityGateError
from quantlab.core.ids import validate_symbol
from quantlab.data.providers import MarketDataProvider
from quantlab.data.snapshots import SnapshotStore
from quantlab.data.validation import run_quality_checks
from quantlab.domain.models import DataSnapshot, QualityReport
from quantlab.storage.db import Database


@dataclass
class IngestResult:
    snapshot: DataSnapshot
    quality: QualityReport
    created: bool
    warnings: list[str]


def validate_date_range(start: str, end: str | None, allow_future: bool = False) -> tuple[str, str]:
    try:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp(date.today())
    except (ValueError, TypeError) as exc:
        raise InputValidationError(f"invalid date: {exc}") from exc
    if pd.isna(s) or pd.isna(e):
        raise InputValidationError("dates must be valid YYYY-MM-DD")
    if s >= e:
        raise InputValidationError(f"start {s.date()} must be before end {e.date()}")
    if not allow_future and e.normalize() > pd.Timestamp(date.today()):
        raise InputValidationError(f"end date {e.date()} is in the future")
    if s < pd.Timestamp("1995-01-01"):
        raise InputValidationError("start must be on or after 1995-01-01")
    return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")


def ingest(
    symbols: list[str],
    start: str,
    end: str | None,
    provider: MarketDataProvider,
    store: SnapshotStore,
    db: Database,
    quality_params: dict | None = None,
) -> IngestResult:
    syms = sorted({validate_symbol(s) for s in symbols})
    if not syms:
        raise InputValidationError("at least one symbol is required")
    s, e = validate_date_range(start, end, allow_future=provider.name == "synthetic")
    try:
        res = provider.fetch(syms, s, e)
    except DataUnavailableError:
        db.audit("system", "ingest.failed", details={"provider": provider.name, "symbols": syms})
        raise
    returned = set(res.bars["symbol"].unique())
    if returned != set(syms):
        raise DataUnavailableError(f"provider returned symbols {sorted(returned)} != requested {syms}")
    snap, created = store.create(
        res.bars,
        provider=res.provider,
        price_basis=res.price_basis,
        is_synthetic=res.is_synthetic,
        parameters=res.parameters,
        retrieved_at=res.retrieved_at,
    )
    report = run_quality_checks(
        res.bars, res.price_basis, s, e, expected_source=res.bars["source"].iloc[0], params=quality_params
    )
    store.record_quality(snap.id, report)
    db.audit("system", "ingest.completed", "data_snapshot", snap.id,
             {"created": created, "quality": report.status, "symbols": syms})
    return IngestResult(snap, report, created, res.warnings)


def require_tradable_snapshot(store: SnapshotStore, snapshot_id: str) -> DataSnapshot:
    """Gate: no backtest or trade without a validated snapshot."""
    snap = store.get(snapshot_id)
    status = store.quality_status(snap.id)
    if status is None:
        raise QualityGateError(f"snapshot {snap.id} has no quality report; run `quantlab data check` first")
    if status == "FAILED":
        raise QualityGateError(f"snapshot {snap.id} failed blocking data-quality checks")
    return snap
