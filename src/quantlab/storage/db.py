"""SQLite metadata store (single-user, local).

The PostgreSQL reference schema (``schema.sql`` in the spec package) is adapted to SQLite:
UUIDs are TEXT, JSONB is TEXT (JSON), NUMERIC money is TEXT holding a ``Decimal`` string so
no precision is lost. Market bars are NOT stored here; they live in immutable Parquet snapshots.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantlab.core.ids import new_id

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  exchange TEXT NOT NULL,
  currency TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  UNIQUE(symbol, exchange)
);
CREATE TABLE IF NOT EXISTS data_snapshots (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  dataset_hash TEXT NOT NULL UNIQUE,
  storage_uri TEXT NOT NULL,
  parameters TEXT NOT NULL DEFAULT '{}',
  price_basis TEXT NOT NULL,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  retrieved_at TEXT NOT NULL,
  symbols TEXT NOT NULL,
  start_date TEXT NOT NULL,
  end_date TEXT NOT NULL,
  n_rows INTEGER NOT NULL,
  parent_id TEXT REFERENCES data_snapshots(id),
  manifest TEXT NOT NULL DEFAULT '{}',
  quality_status TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot_lineage (
  id TEXT PRIMARY KEY,
  base_snapshot_id TEXT NOT NULL REFERENCES data_snapshots(id),
  new_snapshot_id TEXT NOT NULL REFERENCES data_snapshots(id),
  diff TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quality_issues (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES data_snapshots(id),
  symbol TEXT,
  session_date TEXT,
  issue_type TEXT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('INFO','WARN','ERROR','BLOCK')),
  message TEXT NOT NULL,
  row_count INTEGER NOT NULL DEFAULT 0,
  details TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quality_snapshot ON quality_issues(snapshot_id, severity);
CREATE TABLE IF NOT EXISTS strategies (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  parameters TEXT NOT NULL DEFAULT '{}',
  code_reference TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(name, version, parameters)
);
CREATE TABLE IF NOT EXISTS backtest_runs (
  id TEXT PRIMARY KEY,
  strategy_id TEXT REFERENCES strategies(id),
  strategy_name TEXT NOT NULL,
  snapshot_id TEXT NOT NULL REFERENCES data_snapshots(id),
  config TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  metrics TEXT,
  artifact_uri TEXT,
  git_commit TEXT,
  random_seed INTEGER,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','COMPLETED','FAILED')),
  failure_reason TEXT,
  warnings TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_backtest_time ON backtest_runs(created_at DESC);
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  code TEXT NOT NULL,
  hypothesis TEXT NOT NULL,
  design TEXT NOT NULL,
  snapshot_id TEXT REFERENCES data_snapshots(id),
  status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','COMPLETED','FAILED')),
  results TEXT,
  artifact_uri TEXT,
  failure_reason TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE TABLE IF NOT EXISTS experiment_runs (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  scenario TEXT NOT NULL,
  policy TEXT NOT NULL,
  strategy TEXT NOT NULL,
  seed INTEGER NOT NULL,
  derived_snapshot_id TEXT REFERENCES data_snapshots(id),
  outputs TEXT,
  UNIQUE(experiment_id, scenario, policy, strategy, seed)
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','COMPLETED','FAILED')),
  params TEXT NOT NULL DEFAULT '{}',
  result_ref TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  mode TEXT NOT NULL CHECK (mode IN ('PAPER')),
  base_currency TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('ACTIVE','PAUSED','DISABLED')),
  initial_cash TEXT NOT NULL,
  snapshot_id TEXT NOT NULL REFERENCES data_snapshots(id),
  current_session TEXT NOT NULL,
  kill_switch INTEGER NOT NULL DEFAULT 0,
  config TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  symbol TEXT NOT NULL,
  client_order_id TEXT NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  quantity TEXT NOT NULL,
  filled_quantity TEXT NOT NULL DEFAULT '0',
  order_type TEXT NOT NULL CHECK (order_type IN ('MARKET','LIMIT')),
  limit_price TEXT,
  time_in_force TEXT NOT NULL DEFAULT 'GTC' CHECK (time_in_force IN ('DAY','GTC')),
  status TEXT NOT NULL CHECK (status IN ('CREATED','REJECTED','SUBMITTED','PARTIALLY_FILLED','FILLED','CANCELED')),
  reason TEXT,
  decision_session TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(account_id, client_order_id),
  CHECK ((order_type = 'LIMIT') = (limit_price IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_orders_account ON orders(account_id, created_at DESC);
CREATE TABLE IF NOT EXISTS executions (
  id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL REFERENCES orders(id),
  execution_ref TEXT NOT NULL UNIQUE,
  quantity TEXT NOT NULL,
  price TEXT NOT NULL,
  fee TEXT NOT NULL,
  currency TEXT NOT NULL,
  session_date TEXT NOT NULL,
  executed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
  account_id TEXT NOT NULL REFERENCES accounts(id),
  symbol TEXT NOT NULL,
  quantity TEXT NOT NULL,
  cost_basis TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (account_id, symbol)
);
CREATE TABLE IF NOT EXISTS cash_ledger (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  event_ref TEXT NOT NULL UNIQUE,
  currency TEXT NOT NULL,
  amount TEXT NOT NULL,
  event_type TEXT NOT NULL,
  reference_id TEXT,
  session_date TEXT,
  recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_account ON cash_ledger(account_id, recorded_at);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  session_date TEXT NOT NULL,
  nav TEXT NOT NULL,
  cash_value TEXT NOT NULL,
  gross_exposure TEXT NOT NULL,
  holdings TEXT NOT NULL,
  UNIQUE(account_id, session_date)
);
CREATE TABLE IF NOT EXISTS risk_events (
  id TEXT PRIMARY KEY,
  account_id TEXT,
  rule TEXT NOT NULL,
  severity TEXT NOT NULL,
  action TEXT NOT NULL,
  details TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
  id TEXT PRIMARY KEY,
  actor TEXT NOT NULL,
  event TEXT NOT NULL,
  entity_type TEXT,
  entity_id TEXT,
  details TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_events(entity_type, entity_id, created_at);
"""


def new_orders_ddl() -> str:
    start = SCHEMA.index("CREATE TABLE IF NOT EXISTS orders (")
    end = SCHEMA.index(");", start) + 2
    return SCHEMA[start:end]


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def loads(s: str | None) -> Any:
    return None if s is None else json.loads(s)


class Database:
    """Thin wrapper around a SQLite file. One connection per thread."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self.connect() as con:
            version = con.execute("PRAGMA user_version").fetchone()[0]
            has_tables = con.execute("SELECT 1 FROM sqlite_master WHERE name='orders'").fetchone() is not None
            if has_tables and version < SCHEMA_VERSION:
                self._migrate(con, version)
            con.executescript(SCHEMA)
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _migrate(con: sqlite3.Connection, version: int) -> None:
        """Forward-only schema migrations (SQLite cannot ALTER a CHECK constraint, so rebuild)."""
        if version < 2:
            # v2: LIMIT orders + time_in_force on `orders`
            con.execute("PRAGMA foreign_keys = OFF")
            con.execute("BEGIN IMMEDIATE")
            try:
                # SQLite's recommended 12-step procedure: new table, copy, drop old, rename new.
                con.execute(new_orders_ddl().replace("CREATE TABLE IF NOT EXISTS orders (", "CREATE TABLE orders_v2 ("))
                con.execute(
                    "INSERT INTO orders_v2(id, account_id, symbol, client_order_id, side, quantity, filled_quantity, "
                    "order_type, limit_price, time_in_force, status, reason, decision_session, created_at, updated_at) "
                    "SELECT id, account_id, symbol, client_order_id, side, quantity, filled_quantity, order_type, NULL, "
                    "'GTC', status, reason, decision_session, created_at, updated_at FROM orders"
                )
                con.execute("DROP TABLE orders")
                con.execute("ALTER TABLE orders_v2 RENAME TO orders")
                bad = con.execute("PRAGMA foreign_key_check").fetchall()
                if bad:
                    raise sqlite3.IntegrityError(f"foreign key violations after migration: {bad[:5]}")
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            finally:
                con.execute("PRAGMA foreign_keys = ON")

    def _conn(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA journal_mode = WAL")
            con.execute("PRAGMA busy_timeout = 30000")
            self._local.con = con
        return con

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        yield self._conn()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction (BEGIN IMMEDIATE ... COMMIT / ROLLBACK)."""
        with self._write_lock:
            con = self._conn()
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            else:
                con.execute("COMMIT")

    def query(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple | list = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple | list = ()) -> None:
        with self.transaction() as con:
            con.execute(sql, params)

    def audit(
        self,
        actor: str,
        event: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        details: dict | None = None,
        con: sqlite3.Connection | None = None,
    ) -> None:
        sql = (
            "INSERT INTO audit_events(id, actor, event, entity_type, entity_id, details, created_at) "
            "VALUES (?,?,?,?,?,?,?)"
        )
        params = (new_id(), actor, event, entity_type, entity_id, dumps(details or {}), utcnow())
        if con is not None:
            con.execute(sql, params)
        else:
            self.execute(sql, params)

    def close(self) -> None:
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None
