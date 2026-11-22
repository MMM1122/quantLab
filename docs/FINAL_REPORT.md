# QuantLab v1.0.0 — final report

Status as of 2026-11-21. Every claim below was exercised by an automated test or by the acceptance script.
Anything not tested is listed under *Not verified*.

> Research and simulation only. No live trading, no broker connection, not investment advice.
> The development sandbox could not reach Yahoo Finance. Real data was verified instead by the
> GitHub Actions workflow `real-data.yml` (see §4b). Sections that say so use the
> **SYNTHETIC — NOT REAL MARKET PERFORMANCE** fixture.

## 1. Implemented features
| area | implemented and tested |
|---|---|
| **Market data** | yfinance provider for daily OHLCV plus dividends and splits, mockable for tests; seeded synthetic fixture; XNYS trading calendar; content-hashed, read-only Parquet snapshots (dedupe, re-verified on load); SQLite metadata; DuckDB queries; explicit handling of Yahoo quirks (partial intraday bars, action-only rows, weekend dividends, duplicates); `data refresh` with provider-revision diff and lineage |
| **Data quality** | duplicates, missing sessions vs. the exchange calendar, non-session rows, OHLC bounds, non-positive/non-finite values, volume, stale tail, extreme returns (flagged, not deleted), flatlines, split/adjustment consistency, provenance; a quality gate blocks backtests and paper accounts on failed snapshots |
| **Strategies** | Buy & Hold, SMA crossover, RSI mean reversion, 12-1 momentum, volatility-target overlay; each one is tested for causality (no look-ahead) |
| **Backtest engine** | event-driven daily loop; signals after the close executed at the next open; commission and slippage; integer shares; no overdraft and no shorting (invariants asserted every session); dividends and splits never double-counted (two exclusive price modes); trade ledger, equity curve, CAGR, Sharpe with confidence interval, Sortino, drawdown and duration, turnover, time in market; net vs. gross vs. benchmark; full provenance (snapshot hash, config hash, git commit, seed) |
| **Research** | **DQ-001**: missing, delayed and corrupted observations × strict / bounded / naive (research-only) policies × seeds, fixed window, derived snapshots, confidence intervals, signal fidelity, detection rates. **Walk-forward**: parameters chosen on training windows only, plus a leakage test. **Statistics**: Jobson–Korkie/Memmel test, block bootstrap, probabilistic and deflated Sharpe |
| **Paper trading** | PAPER-only accounts replaying historical sessions; Decimal cash ledger; MARKET and LIMIT orders with DAY/GTC; pre-trade risk checks (kill switch, stale data, duplicates, no shorting, no negative cash, ≤ 25% per asset, limit-price band); idempotent orders and fill callbacks; volume-capped partial fills; dividends and splits; reconciliation; persistence across restarts; strategy autopilot with drift trims; paper-vs-backtest consistency report |
| **AI assistant** | read-only (SQLite `mode=ro`, AST test forbids order imports); deterministic analytics when no API key is set; optional Anthropic/OpenAI adapters whose every number is checked against stored data; refuses order requests |
| **CAD reporting** | USDCAD snapshots; CAD metrics and equity curves for an unhedged Canadian investor; paper NAV in CAD (trading stays in USD) |
| **Interfaces** | Streamlit dashboard (Overview, Market Explorer, Strategy Studio, Backtest Lab, Research Lab, Paper Trading, AI Assistant); FastAPI with persisted jobs and an optional bearer token; CLI; CSV/JSON/Markdown/HTML exports carrying labels |
| **Deployment & CI** | Dockerfile, docker-compose, Streamlit Cloud support; GitHub Actions runs ruff, mypy, pytest and the offline acceptance test on Python 3.11 and 3.13, and builds and health-checks the Docker image |

## 2. File structure
```
apps/dashboard/              streamlit_app.py + views/ (7 pages)
configs/                     base.yaml, risk.yaml
docs/                        architecture, backtest_assumptions, data_contracts, dq_001, api, limitations,
                             task_board, FINAL_REPORT
scripts/                     acceptance_test.py, smoke_test.sh
src/quantlab/
  core/        config (LIVE_TRADING guard), errors, ids, hashing, labels
  domain/      typed contracts and enums
  storage/     SQLite schema + migrations, DuckDB queries
  data/        providers, calendar, validation, adjustments, snapshots, ingestion, refresh, fx
  strategies/  base, buy_hold, sma_crossover, rsi_mean_reversion, momentum, vol_target, registry
  backtest/    market panel, engine, execution (cost model), metrics, service
  portfolio/   accounting (valuation, NAV, weights, invariants)
  risk/        pre-trade rules
  research/    dq_001, perturbations, policies, walk_forward, splits, statistics, stat_tests
  paper/       broker, autopilot, compare
  ai/          tools (read-only), provider adapters, assistant
  reports/     charts, export
  api/, cli/, ui/, jobs.py, context.py
tests/unit (126 tests), tests/integration (53 tests)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, pyproject.toml, requirements.txt, AGENTS.md, README.md
```

## 3. Setup and run
See README → *Setup*, *Run* and *Deployment* for the exact commands. In short:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
quantlab data ingest --symbols SPY QQQ IWM GLD TLT --start 2015-01-01      # real data (needs internet)
streamlit run apps/dashboard/streamlit_app.py                              # dashboard
uvicorn quantlab.api.app:app --port 8000                                   # API
python scripts/acceptance_test.py                                          # end-to-end acceptance
GIT_COMMIT=$(git rev-parse HEAD) docker compose up -d --build              # persistent deployment
```

## 4. Tests executed and results
| check | result |
|---|---|
| `ruff check .` | clean |
| `mypy` (70 files) | no issues |
| `pytest` | **179 passed** (126 unit + 53 integration), Python 3.13; CI also green on 3.11 |
| `scripts/smoke_test.sh` | PASSED (lint, tests, CLI flow, API + Streamlit health) |
| `scripts/acceptance_test.py` | **8/8 steps PASS** (real download attempted, then fell back to the synthetic fixture with the fallback recorded) |
| CI `docker` job | image builds; CLI runs inside it; dashboard and token-protected API healthy |

Acceptance evidence (SYNTHETIC fixture, SPY, 2016-01-04 → 2025-12-31, 1 bp commission + 2 bp slippage):

| strategy | total return | CAGR | Sharpe (95% CI) | max DD | turnover/yr | fills |
|---|---|---|---|---|---|---|
| Buy & Hold | 70.36% | 5.49% | 0.41 (−0.21 … 1.03) | −32.20% | 0.08 | 1 |
| SMA 50/200 | 62.72% | 5.00% | 0.44 (−0.18 … 1.06) | −22.99% | 1.38 | 14 |
| RSI mean reversion | 102.80% | 7.35% | 0.85 (0.23 … 1.47) | −18.63% | 9.31 | 96 |

DQ-001 ran 96 backtests. The paper account ended with cash 8,505.24 USD and NAV 9,941.99 USD, verified independently against the ledger. Shorting, oversized and above-25% orders were rejected as expected. A restart (new Python process and two API server processes) reproduced the identical state. These numbers describe the synthetic fixture only and say nothing about real markets.

## 4b. Real-data acceptance (GitHub Actions, `real-data.yml`, `--require-real`, run 44983196737)
- Real yfinance snapshot: SPY, QQQ, IWM, TLT and GLD, 14,950 rows, 2015-01-02 → 2026-11-20. Quality **PASSED with 0 warnings**; hash re-verified.
- **8/8 acceptance steps PASS on real data**. The paper account ended with cash 7,609.61 USD and NAV 9,997.45 USD (verified against the ledger). Restarting the Python process and the API reproduced the same state.
- SPY, 2016-01-04 → 2026-11-20, 1 bp commission + 2 bp slippage. Dividends are held as cash and not reinvested:

| strategy | total return | CAGR | Sharpe (95% CI) | max DD | turnover/yr | fills |
|---|---|---|---|---|---|---|
| Buy & Hold | 270.83% | 12.82% | 0.82 (0.22 … 1.41) | −31.57% | 0.04 | 1 |
| SMA 50/200 | 136.03% | 8.23% | 0.61 (0.02 … 1.21) | −33.12% | 0.85 | 11 |
| RSI mean reversion | 116.20% | 7.36% | 0.63 (0.03 … 1.22) | −28.30% | 5.96 | 68 |

- Real USDCAD (`CAD=X`): 2,986 sessions, quality PASSED with one warning (a few sessions without a rate; for 4 sessions in the evaluation window the conversion used the previous session's rate). In the CAD view, SPY buy & hold returned +283.64% in CAD versus +270.83% in USD.
- `data refresh` on the real snapshot reported `UPDATED` with 0 raw revisions and 5 rows where only `adj_close` changed.
- **Bug found by real data and fixed:** Yahoo timestamps `CAD=X` daily bars at London midnight. Converting those timestamps to New York time moved every rate to the previous calendar day, so about 20% of sessions had no rate. FX bars now keep the provider's local date. A test reproduces the bug and fails on the old code.
- These are historical backtests, not predictions or advice.

## 5. Not verified / limitations
- **Real LLM calls** were not exercised (no API key or network access). The adapters are tested with mocked HTTP responses.
- The **Streamlit Community Cloud** deployment was not performed; it requires the owner's account.
- Modelling simplifications: next-open fills with fixed slippage; no market impact; dividends held as cash; Yahoo prices are retroactively split-adjusted; no taxes or FX conversion costs; few seeds in DQ-001. See `docs/limitations.md`.

## 6. Remaining tasks
| priority | task |
|---|---|
| done | PRs #1–#6 merged via #7 (`main` 99d7966). The `v1.0.0` tag and GitHub Release still need to be created by the owner (the session could not push tags or create releases) |
| done | Real-data acceptance passed on GitHub runners (§4b). To re-run it: Actions → *Real-data acceptance* → *Run workflow* |
| P1 (deliberately not done) | PostgreSQL/Alembic and a Next.js UI. These were P1 in the original specification, but the project's tech stack names SQLite and Streamlit and forbids unnecessary infrastructure, so they were not built for a single-user local tool |
| P2 | Broker adapter interface tests only. Live trading stays disabled pending a separate, human-approved release |
