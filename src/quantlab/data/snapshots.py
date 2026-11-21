"""Immutable, content-addressed Parquet snapshots with SQLite metadata.

* The SHA-256 content hash is computed from the canonical DataFrame (not Parquet bytes), so
  it is stable across encoder versions.
* Ingesting identical content twice returns the existing snapshot (no duplicate bars).
* Files are written once and made read-only; every load re-verifies the hash.
* Derived snapshots (e.g. DQ-001 perturbations) record ``parent_id`` and a manifest
  (transformation, parameters, seed). Source snapshots are never modified.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.core.errors import DataIntegrityError, NotFoundError
from quantlab.core.hashing import dataframe_sha256
from quantlab.core.ids import deterministic_id, validate_uuid
from quantlab.data.providers import canonicalize_bars
from quantlab.domain.models import DataSnapshot, QualityReport
from quantlab.storage.db import Database, dumps, loads, utcnow


class SnapshotStore:
    def __init__(self, root: str | Path, db: Database):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = db

    # ------------------------------------------------------------------ write
    def create(
        self,
        bars: pd.DataFrame,
        provider: str,
        price_basis: str,
        is_synthetic: bool,
        parameters: dict[str, Any] | None = None,
        retrieved_at: str | None = None,
        parent_id: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> tuple[DataSnapshot, bool]:
        """Persist bars. Returns (snapshot, created). ``created`` is False on dedupe."""
        canon = canonicalize_bars(bars)
        if canon["source"].nunique() > 1:
            raise DataIntegrityError("refusing to mix data sources in one snapshot")
        if is_synthetic != (canon["source"].iloc[0] == "synthetic"):
            raise DataIntegrityError("synthetic flag does not match row provenance")
        h = dataframe_sha256(canon)
        # identity includes basis/synthetic flag so the same numbers can't be relabelled
        content_key = f"{h}:{price_basis}:{int(is_synthetic)}"
        existing = self.db.query_one("SELECT id FROM data_snapshots WHERE dataset_hash = ?", (content_key,))
        if existing:
            return self.get(existing["id"]), False

        sid = deterministic_id("snapshot", content_key)
        folder = self.root / sid
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "bars.parquet"
        canon.to_parquet(path, index=False)
        man = {
            "dataset_hash": h,
            "provider": provider,
            "price_basis": price_basis,
            "is_synthetic": is_synthetic,
            "parameters": parameters or {},
            "parent_id": parent_id,
            "transformation": manifest or {},
            "columns": list(canon.columns),
            "n_rows": len(canon),
            "written_at": datetime.now(UTC).isoformat(),
        }
        (folder / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True, default=str))
        for f in (path, folder / "manifest.json"):
            os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        snap = DataSnapshot(
            id=sid,
            provider=provider,
            dataset_hash=content_key,
            storage_uri=str(path),
            parameters=parameters or {},
            price_basis=price_basis,
            is_synthetic=is_synthetic,
            retrieved_at=retrieved_at or datetime.now(UTC).isoformat(),
            symbols=sorted(canon["symbol"].unique().tolist()),
            start=canon["session_date"].min().strftime("%Y-%m-%d"),
            end=canon["session_date"].max().strftime("%Y-%m-%d"),
            n_rows=len(canon),
            parent_id=parent_id,
            manifest=manifest or {},
        )
        with self.db.transaction() as con:
            con.execute(
                "INSERT INTO data_snapshots(id, provider, dataset_hash, storage_uri, parameters, price_basis, "
                "is_synthetic, retrieved_at, symbols, start_date, end_date, n_rows, parent_id, manifest, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snap.id, snap.provider, snap.dataset_hash, snap.storage_uri, dumps(snap.parameters),
                    snap.price_basis, int(snap.is_synthetic), snap.retrieved_at, dumps(snap.symbols),
                    snap.start, snap.end, snap.n_rows, parent_id, dumps(snap.manifest), utcnow(),
                ),
            )
            self.db.audit("system", "snapshot.created", "data_snapshot", snap.id,
                          {"provider": provider, "hash": h, "parent_id": parent_id}, con=con)
        return self.get(sid), True

    def record_quality(self, snapshot_id: str, report: QualityReport) -> None:
        with self.db.transaction() as con:
            con.execute("DELETE FROM quality_issues WHERE snapshot_id = ?", (snapshot_id,))
            for i, issue in enumerate(report.issues):
                con.execute(
                    "INSERT INTO quality_issues(id, snapshot_id, symbol, session_date, issue_type, severity, "
                    "message, row_count, details, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        deterministic_id("qi", snapshot_id, i, issue.issue_type, issue.symbol),
                        snapshot_id, issue.symbol, issue.session_date, issue.issue_type, str(issue.severity),
                        issue.message, issue.row_count, dumps(issue.details), utcnow(),
                    ),
                )
            con.execute("UPDATE data_snapshots SET quality_status = ? WHERE id = ?", (report.status, snapshot_id))

    # ------------------------------------------------------------------ read
    def _row_to_snapshot(self, r: dict[str, Any]) -> DataSnapshot:
        return DataSnapshot(
            id=r["id"], provider=r["provider"], dataset_hash=r["dataset_hash"], storage_uri=r["storage_uri"],
            parameters=loads(r["parameters"]), price_basis=r["price_basis"], is_synthetic=bool(r["is_synthetic"]),
            retrieved_at=r["retrieved_at"], symbols=loads(r["symbols"]), start=r["start_date"],
            end=r["end_date"], n_rows=r["n_rows"], parent_id=r["parent_id"], manifest=loads(r["manifest"]),
            created_at=r["created_at"],
        )

    def get(self, snapshot_id: str) -> DataSnapshot:
        sid = validate_uuid(snapshot_id, "snapshot_id")
        r = self.db.query_one("SELECT * FROM data_snapshots WHERE id = ?", (sid,))
        if r is None:
            raise NotFoundError(f"snapshot {sid} not found")
        return self._row_to_snapshot(r)

    def quality_status(self, snapshot_id: str) -> str | None:
        r = self.db.query_one("SELECT quality_status FROM data_snapshots WHERE id = ?", (snapshot_id,))
        return r["quality_status"] if r else None

    def quality_issues(self, snapshot_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT symbol, session_date, issue_type, severity, message, row_count, details "
            "FROM quality_issues WHERE snapshot_id = ? ORDER BY severity, issue_type, symbol",
            (snapshot_id,),
        )
        for r in rows:
            r["details"] = loads(r["details"])
        return rows

    def list(self, include_derived: bool = True) -> list[DataSnapshot]:
        sql = "SELECT * FROM data_snapshots"
        if not include_derived:
            sql += " WHERE parent_id IS NULL"
        sql += " ORDER BY created_at DESC"
        return [self._row_to_snapshot(r) for r in self.db.query(sql)]

    def load(self, snapshot_id: str, verify: bool = True) -> pd.DataFrame:
        snap = self.get(snapshot_id)
        path = Path(snap.storage_uri)
        if not path.exists():
            raise DataIntegrityError(f"snapshot file missing: {path}")
        df = pd.read_parquet(path)
        df = canonicalize_bars(df)
        if verify:
            h = dataframe_sha256(df)
            if not snap.dataset_hash.startswith(h + ":"):
                raise DataIntegrityError(
                    f"snapshot {snap.id} content hash mismatch (stored {snap.dataset_hash[:12]}…, "
                    f"computed {h[:12]}…). The file was modified; refusing to use it."
                )
        return df
