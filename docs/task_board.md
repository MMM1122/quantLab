# Task board

## Done (V1 / P0) — each phase committed after lint + tests
- [x] Phase 1 — foundation, config, typed contracts, providers, XNYS validation, snapshots, SQLite, DuckDB
- [x] Phase 2 — Buy & Hold, SMA crossover, RSI mean reversion (+ 12-1 momentum), event-driven engine, metrics, benchmark
- [x] Phase 3 — Streamlit (7 pages), FastAPI with jobs, CLI, exports
- [x] Phase 4 — DQ-001 perturbations, policies, multi-seed statistics, derived snapshots, reports
- [x] Phase 5 — paper accounts, Decimal ledger, risk rules, idempotency, partial fills, kill switch, replay
- [x] Phase 6 — read-only AI assistant, deterministic mode, verified LLM adapter
- [x] Acceptance test script (`scripts/acceptance_test.py`) incl. process restart persistence
- [x] GitHub Actions CI (ruff, pytest, offline acceptance on Python 3.11 / 3.13)
- [x] Walk-forward out-of-sample validation (train-only selection, leakage test, OOS rank diagnostic)
- [x] Paper LIMIT orders, DAY/GTC time in force, fat-finger price band, SQLite schema migration v1→v2
- [x] Deployment (Docker, Compose, Streamlit Cloud), API bearer token, snapshot refresh + revision detection
- [x] Volatility-target overlay; statistical test module (JK/Memmel, block bootstrap, PSR/DSR)
- [x] Paper strategy autopilot (risk-checked, drift trims, idempotent) + paper-vs-backtest consistency report
- [x] CAD reporting currency (USDCAD snapshots, CAD metrics/curves, paper NAV in CAD)
- [x] Hardening: mypy-clean package (CI), Yahoo quirk handling (partial bars, action-only rows, duplicates)

## Next (prioritised)
| prio | item |
|---|---|
| P0 | Run acceptance test with **real** yfinance data on a networked machine; review DQ reports for real SPY |
| P0 | Review data-quality WARNs on real snapshots (Yahoo occasionally revises/omits bars) |
| P1 | Alembic/PostgreSQL migration path; FastAPI auth token for local API |
| P1 | Scheduled daily ingestion + paper-account monitoring (still historical/paper only) |
| P1 | Next.js dashboard against the existing API |
| P2 | Broker adapter **interface tests only**, paper endpoints, compliance review — live trading stays disabled |
