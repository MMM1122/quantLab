# AGENTS.md — rules for coding agents working on QuantLab

QuantLab is a local-first research, backtesting and **paper**-trading platform for US-listed ETFs
(daily bars, USD, long-only, no leverage). It must stay honest: no fabricated metrics, no fake
test results, no live trading.

## Hard safety boundaries
- `LIVE_TRADING_ENABLED` must remain `false`; `quantlab.core.config` refuses to start otherwise.
- There is no real broker adapter. Do not add one without a separate, human-approved release.
- The AI assistant (`quantlab.ai`) is read-only. It must never import or call `quantlab.paper`
  order functions. `tests/unit/test_ai.py` enforces this.
- Never commit `.env`, API keys, `data/` (snapshots, DBs, run artifacts) or generated reports.

## Module boundaries (`src/quantlab/`)
| package | responsibility | may depend on |
|---|---|---|
| `core` | config, errors, ids, hashing, labels | — |
| `domain` | typed contracts/enums | core |
| `storage` | SQLite metadata DB, DuckDB queries over Parquet | core |
| `data` | providers, calendar, validation, adjustments, snapshots, ingestion | core, domain, storage |
| `strategies` | signal generation (target weights) — pure, causal | domain |
| `risk` | pre-trade rules shared by backtest and paper | domain |
| `portfolio` | analytic accounting helpers | domain |
| `backtest` | event-driven engine, fills, metrics, persistence | data, strategies, risk, portfolio, storage |
| `research` | DQ-001 perturbations, policies, statistics | backtest, data |
| `paper` | Decimal ledger, orders, fills, replay, kill switch | data, risk, storage |
| `ai` | read-only tools, deterministic summaries, optional LLM adapter | storage (read-only) |
| `reports` | CSV/JSON/Markdown/HTML exports | backtest, research |
| `api`, `cli`, `apps/dashboard` | interfaces only; no domain logic | everything above |

## Invariants that tests guard
- Signals computed after the close of session *t* execute no earlier than the open of *t+1*.
- Strategy output at row *t* depends only on rows ≤ *t* (`tests/unit/test_strategies.py`).
- Cash never negative; positions never negative; NAV = cash + Σ qty·price.
- Either cash dividends are credited on raw prices, or adjusted prices are used — never both.
- Snapshots are immutable and hash-verified on load; derived snapshots keep `parent_id`.
- Synthetic data is labelled `SYNTHETIC — NOT REAL MARKET PERFORMANCE` everywhere it appears.

## Workflow
```
make install   # venv + editable install
make lint      # ruff check .
make test      # pytest -q
make smoke     # scripts/smoke_test.sh (offline, synthetic)
```
Run lint + tests before every commit. Report failures honestly.
