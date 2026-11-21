"""Refresh a real-data snapshot and detect provider revisions.

Snapshots are immutable, so a refresh never edits the old one: it re-downloads the full history
(base start → new end) into a NEW snapshot and records a lineage row with a diff:

* ``appended``   — sessions after the old end (expected).
* ``removed``    — rows present before but missing now (suspicious).
* ``raw_revised``— rows whose OHLCV, dividend or split changed (provider revised history — WARN).
* ``adj_only``   — rows where only ``adj_close`` changed. Expected after a new dividend/split,
  because back-adjustment rescales all earlier prices; reported separately so it is not
  mistaken for a data error.

A refresh with identical content dedupes to the existing snapshot and reports no changes.
Run it daily from cron/launchd if you like — no scheduler service is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.context import AppContext
from quantlab.core.errors import InputValidationError, NotFoundError
from quantlab.core.ids import deterministic_id
from quantlab.data import calendar
from quantlab.data.ingestion import IngestResult, ingest
from quantlab.data.providers import MarketDataProvider
from quantlab.storage.db import dumps, loads, utcnow

RAW_COLS = ["open", "high", "low", "close", "volume", "dividend", "split_ratio"]
REL_TOL = 1e-6



@dataclass
class RefreshResult:
    base_snapshot_id: str
    ingest: IngestResult
    diff: dict[str, Any]

    @property
    def new_snapshot_id(self) -> str:
        return self.ingest.snapshot.id


def _changed(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.astype(float), b.astype(float)
    both_nan = a.isna() & b.isna()
    denom = np.maximum(np.abs(a), 1e-12)
    return ~both_nan & ((a.isna() != b.isna()) | ((a - b).abs() / denom > REL_TOL))


def diff_snapshots(old: pd.DataFrame, new: pd.DataFrame, sample: int = 10) -> dict[str, Any]:
    key = ["symbol", "session_date"]
    o = old.set_index(key)
    n = new.set_index(key)
    common = o.index.intersection(n.index)
    removed = o.index.difference(n.index)
    added = n.index.difference(o.index)
    out: dict[str, Any] = {"symbols": {}, "totals": {}}
    raw_mask = pd.Series(False, index=common)
    col_counts = {}
    for c in RAW_COLS:
        ch = _changed(o.loc[common, c], n.loc[common, c])
        col_counts[c] = int(ch.sum())
        raw_mask |= ch
    adj_ch = _changed(o.loc[common, "adj_close"], n.loc[common, "adj_close"])
    adj_only = adj_ch & ~raw_mask
    old_end = old.groupby("symbol")["session_date"].max()
    for sym in sorted(set(old["symbol"]) | set(new["symbol"])):
        def _of(idx):
            return [d for s, d in idx if s == sym]

        appended = [d for d in _of(added) if sym in old_end and d > old_end[sym]]
        backfilled = [d for d in _of(added) if not (sym in old_end and d > old_end[sym])]
        rev = [d for (s, d) in raw_mask[raw_mask].index if s == sym]
        details = []
        for d in rev[:sample]:
            row = {"session_date": d.strftime("%Y-%m-%d")}
            for c in RAW_COLS:
                a, b = o.loc[(sym, d), c], n.loc[(sym, d), c]
                if _changed(pd.Series([a]), pd.Series([b])).iloc[0]:
                    row[c] = [float(a), float(b)]
            details.append(row)
        out["symbols"][sym] = {
            "appended": len(appended),
            "appended_range": [min(appended).strftime("%Y-%m-%d"), max(appended).strftime("%Y-%m-%d")] if appended else None,
            "backfilled": len(backfilled),
            "removed": len(_of(removed)),
            "removed_dates": [d.strftime("%Y-%m-%d") for d in _of(removed)[:sample]],
            "raw_revised": len(rev),
            "raw_revised_samples": details,
            "adj_only_revised": int(sum(1 for (s, _d) in adj_only[adj_only].index if s == sym)),
        }
    out["totals"] = {
        "appended": sum(v["appended"] for v in out["symbols"].values()),
        "backfilled": sum(v["backfilled"] for v in out["symbols"].values()),
        "removed": int(len(removed)),
        "raw_revised": int(raw_mask.sum()),
        "adj_only_revised": int(adj_only.sum()),
        "raw_revised_by_column": col_counts,
    }
    t = out["totals"]
    out["status"] = "REVISIONS_DETECTED" if (t["raw_revised"] or t["removed"]) else (
        "UPDATED" if t["appended"] or t["backfilled"] or t["adj_only_revised"] else "UNCHANGED")
    return out


def latest_real_snapshot(ctx: AppContext):
    for s in ctx.snapshots.list(include_derived=False):
        if not s.is_synthetic:
            return s
    raise NotFoundError("no real-data snapshot to refresh; run `quantlab data ingest` first")


def refresh_snapshot(ctx: AppContext, base_snapshot_id: str, provider: MarketDataProvider,
                     end: str | None = None) -> RefreshResult:
    base = latest_real_snapshot(ctx) if base_snapshot_id == "latest" else ctx.snapshots.get(base_snapshot_id)
    if base.is_synthetic or base.parent_id:
        raise InputValidationError("only real, non-derived snapshots can be refreshed (synthetic fixtures are "
                                   "regenerated, derived snapshots are immutable experiment inputs)")
    if provider.name != base.provider:
        raise InputValidationError(f"snapshot came from {base.provider!r}; refresh with the same provider")
    end = end or calendar.last_completed_session().strftime("%Y-%m-%d")
    old = ctx.snapshots.load(base.id)
    res = ingest(base.symbols, base.start, end, provider, ctx.snapshots, ctx.db)
    new = ctx.snapshots.load(res.snapshot.id)
    diff = diff_snapshots(old, new)
    diff["base_end"], diff["new_end"] = base.end, res.snapshot.end
    with ctx.db.transaction() as con:
        con.execute(
            "INSERT OR REPLACE INTO snapshot_lineage(id, base_snapshot_id, new_snapshot_id, diff, created_at) "
            "VALUES (?,?,?,?,?)",
            (deterministic_id("lineage", base.id, res.snapshot.id), base.id, res.snapshot.id, dumps(diff), utcnow()),
        )
        if diff["status"] == "REVISIONS_DETECTED":
            con.execute(
                "INSERT INTO risk_events(id, account_id, rule, severity, action, details, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (deterministic_id("rev", base.id, res.snapshot.id), None, "PROVIDER_REVISION", "WARN",
                 "SNAPSHOT_REFRESHED", dumps({"base": base.id, "new": res.snapshot.id, **diff["totals"]}), utcnow()),
            )
        ctx.db.audit("system", "snapshot.refreshed", "data_snapshot", res.snapshot.id,
                     {"base": base.id, "status": diff["status"]}, con=con)
    return RefreshResult(base.id, res, diff)


def lineage(ctx: AppContext, snapshot_id: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM snapshot_lineage"
    params: tuple = ()
    if snapshot_id:
        sql += " WHERE base_snapshot_id=? OR new_snapshot_id=?"
        params = (snapshot_id, snapshot_id)
    rows = ctx.db.query(sql + " ORDER BY created_at DESC", params)
    for r in rows:
        r["diff"] = loads(r["diff"])
    return rows
