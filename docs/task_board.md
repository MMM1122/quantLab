# Task board

## Done (V1 / P0) — each phase committed after lint + tests
- [x] Phase 1 — foundation, config, typed contracts, providers, XNYS validation, snapshots, SQLite, DuckDB
- [x] Phase 2 — Buy & Hold, SMA crossover, RSI mean reversion (+ 12-1 momentum), event-driven engine, metrics, benchmark
- [x] Phase 3 — Streamlit (7 pages), FastAPI with jobs, CLI, exports
- [x] Phase 4 — DQ-001 perturbations, policies, multi-seed statistics, derived snapshots, reports
- [x] Phase 5 — paper accounts, Decimal ledger, risk rules, idempotency, partial fills, kill switch, replay
- [x] Phase 6 — read-only AI assistant, deterministic mode, verified LLM adapter
- [x] Acceptance test script (`scripts/acceptance_test.py`) incl. process restart persistence

## Next (prioritised)
| prio | item |
|---|---|
| P0 | Run acceptance test with **real** yfinance data on a networked machine; review DQ reports for real SPY |
| P0 | Review data-quality WARNs on real snapshots (Yahoo occasionally revises/omits bars) |
| P1 | Walk-forward evaluation UI using `research.splits` + parameter sweeps with multiple-testing caveats |
| P1 | Alembic/PostgreSQL migration path; FastAPI auth token for local API |
| P1 | LIMIT orders and day/GTC time-in-force in the paper broker |
| P1 | Scheduled daily ingestion + paper-account monitoring (still historical/paper only) |
| P1 | Next.js dashboard against the existing API |
| P2 | Volatility-target / regime baselines; CAD reporting currency |
| P2 | Broker adapter **interface tests only**, paper endpoints, compliance review — live trading stays disabled |
