# Architecture

QuantLab is a **modular monolith**: one Python package (`src/quantlab`) with strict internal
boundaries, three thin interfaces (CLI, FastAPI, Streamlit) and local storage only.

```
            ┌──────────── interfaces (no domain logic) ────────────┐
            │  cli/  (argparse)   api/ (FastAPI + jobs)   apps/dashboard (Streamlit) │
            └───────────────┬──────────────────────────────┬────────┘
                            │                              │
   data/ ── ingest → validate → snapshot      backtest/ ── panel → strategy → engine → metrics
     providers (yfinance | synthetic)            strategies/ (pure, causal target weights)
     calendar (XNYS), validation, adjustments    risk/ (pre-trade rules, shared)
                            │                              │
   research/ ── DQ-001 perturbations, policies, statistics, splits
   paper/    ── Decimal ledger, orders, fills, replay clock, kill switch, reconcile
   ai/       ── read-only tools (sqlite mode=ro) → deterministic draft → optional LLM → number check
   reports/  ── charts + CSV/JSON/MD/HTML exports
                            │
   storage/ ── SQLite metadata (WAL)   ·   immutable Parquet snapshots   ·   DuckDB queries
```

## Workflow
`Ingest → Validate → Research → Backtest → Compare → Paper Trade → Monitor`

1. **Ingest** (`data.ingestion.ingest`): provider fetch → canonical bars → content hash →
   read-only Parquet snapshot (deduplicated by hash) → quality report persisted.
2. **Validate** (`data.validation`): calendar-aware checks with INFO/WARN/ERROR/BLOCK severities.
   `require_tradable_snapshot` blocks backtests and paper accounts on FAILED snapshots.
3. **Research / Backtest** (`backtest.service.run_and_store`): builds a session-grid panel,
   computes strategy targets on a causal total-return index, runs the event loop, computes
   metrics (net, gross, benchmark), persists artifacts + provenance.
4. **DQ-001** (`research.dq_001`): perturbs a copy of the data, applies treatment policies, runs
   decisions on observed data and fills on pristine data, aggregates across seeds.
5. **Paper trade** (`paper.broker.PaperBroker`): virtual account stepping through historical
   sessions of a snapshot; risk rules run before any order is created.
6. **Monitor**: dashboard pages, risk events, audit log, reconciliation, AI explanations.

## Storage
| what | where | mutability |
|---|---|---|
| OHLCV + corporate actions | `data/snapshots/<id>/bars.parquet` + `manifest.json` | write-once, chmod 0444, hash-verified on load |
| metadata (snapshots, quality, runs, experiments, jobs, paper ledger, audit) | `data/quantlab.db` (SQLite, WAL) | transactional |
| backtest artifacts | `data/runs/<run_id>/*.parquet`, `result.json` | write-once |
| experiment artifacts | `data/experiments/<id>/*.csv`, `result.json` | write-once |
| exports | `data/exports/` | regenerated on demand |

`data/` is gitignored. The SQLite schema adapts the PostgreSQL reference `schema.sql`
(UUID→TEXT, JSONB→TEXT JSON, NUMERIC→Decimal strings).

## Safety boundaries
* `LIVE_TRADING_ENABLED=true` makes startup fail. There is no broker adapter at all.
* `mode=LIVE` requests are rejected by the API (403) and by the paper broker.
* The AI package opens SQLite read-only and is AST-checked to never import order code.
* Synthetic data is labelled in snapshots, API meta, CLI banners, charts, exports and UI.

## Concurrency
Single user. API long-running work (ingest, backtests, DQ-001) runs in a 2-thread pool with
persisted job states; jobs left RUNNING by a crash are marked FAILED at next startup. SQLite
writes are serialised by `BEGIN IMMEDIATE` + an in-process lock.
