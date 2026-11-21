# QuantLab

Local-first quantitative research, backtesting and **paper**-trading platform for US-listed ETFs
(SPY, QQQ, IWM, TLT, GLD · daily bars · USD · long-only · no leverage).

> **Research and simulation only.** No live trading, no broker connection, not investment advice.
> Backtest and paper results are simulations, not real performance. Data from the offline fixture
> is labelled **SYNTHETIC — NOT REAL MARKET PERFORMANCE** everywhere it appears.

## Features
| area | what works (tested) |
|---|---|
| Market data | yfinance provider (split-adjusted OHLC, dividends, splits), seeded synthetic fixture, XNYS calendar, content-hashed read-only Parquet snapshots, dedupe, SQLite metadata, DuckDB queries |
| Data quality | duplicates, missing sessions vs exchange calendar, non-session rows, OHLC bounds, non-positive/non-finite values, volume, stale tail, extreme returns (flag only), flatlines, split/adjustment consistency, provenance; quality gate blocks bad snapshots |
| Strategies | Buy & Hold, SMA crossover (50/200), RSI mean reversion (Cutler RSI), 12-1 cross-sectional momentum — all causal (tested) |
| Backtests | event-driven daily loop, next-open execution, slippage + commission, integer shares, cash/position invariants, dividends & splits without double counting, net/gross/benchmark metrics, Sharpe CI, reproducible runs with full provenance |
| Walk-forward | parameter grid chosen on training windows only, evaluated on later unseen windows; stitched out-of-sample equity, OOS rank of chosen params, leakage test |
| DQ-001 | missing/delayed/corrupted perturbations, strict / bounded / naive (research-only) policies, multi-seed CIs, derived snapshots, detection rates, reports |
| Paper trading | PAPER-only accounts on historical replay, Decimal ledger, MARKET and LIMIT orders with DAY/GTC, pre-trade risk (no shorting, no margin/negative cash, ≤25% per asset, stale data, duplicates, limit price band, kill switch), idempotent orders & fills, partial fills, reconciliation, persistence across restarts |
| AI assistant | read-only tools (SQLite `mode=ro`), deterministic analytics without a key, optional Anthropic/OpenAI phrasing with every number verified against stored data, refuses order requests |
| Interfaces | Streamlit (Overview, Market Explorer, Strategy Studio, Backtest Lab, Research Lab, Paper Trading, AI Assistant), FastAPI with jobs, CLI, CSV/JSON/Markdown/HTML exports |

## Setup (macOS/Linux, Python ≥ 3.11)
```bash
git clone https://github.com/MMM1122/quantLab.git && cd quantLab
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
cp .env.example .env            # optional: LLM provider keys; LIVE_TRADING_ENABLED stays false
```
or simply `make install`.

## Run
```bash
# 1) data — real (needs internet; Yahoo terms apply) or offline synthetic fixture
quantlab data ingest --symbols SPY QQQ IWM GLD TLT --start 2015-01-01
quantlab data ingest --symbols SPY QQQ IWM GLD TLT --start 2015-01-01 --end 2025-12-31 --fixture synthetic
quantlab data list
quantlab data check --snapshot <snapshot_id>

# 2) backtests
quantlab backtest run --strategy ma-crossover --symbol SPY --snapshot <snapshot_id>
quantlab backtest run --strategy rsi --symbol SPY --param entry=25 --param exit=60
quantlab backtest run --strategy momentum --symbols SPY QQQ IWM TLT GLD
quantlab backtest compare --runs <run_a> <run_b>
quantlab report export --run <run_id> --format html     # md | html | csv | json

# 3) research
quantlab experiment run --id dq-001 --snapshot <snapshot_id>
quantlab experiment run --id walk-forward --snapshot <snapshot_id> --strategy sma-crossover \
    --grid '{"fast": [20, 50], "slow": [100, 200]}' --train 756 --test 252
quantlab experiment export --experiment <experiment_id> --format md

# 4) paper trading (historical replay)
quantlab paper create --name demo --snapshot <snapshot_id> --cash 10000 --start 2024-01-02
quantlab paper order --account <account_id> --symbol SPY --side BUY --quantity 5 --client-order-id my-1
quantlab paper order --account <account_id> --symbol SPY --side BUY --quantity 5 --type LIMIT --limit-price 470.50 --tif DAY
quantlab paper advance --account <account_id> --sessions 5
quantlab paper status --account <account_id>
quantlab paper kill-switch --account <account_id> on --reason "manual stop"
quantlab paper reconcile --account <account_id>

# 5) AI assistant (deterministic without API key)
quantlab ai explain --run <run_id>
quantlab ai compare --runs <run_a> <run_b>
quantlab ai summarize --experiment <experiment_id>

# interfaces
streamlit run apps/dashboard/streamlit_app.py          # or: make dashboard
uvicorn quantlab.api.app:app --port 8000               # or: make api  → http://127.0.0.1:8000/docs
```

## Tests and checks
```bash
ruff check .                         # lint
pytest -q                            # 141 unit + integration tests (offline)
bash scripts/smoke_test.sh           # lint + tests + CLI flow + API/Streamlit health
python scripts/acceptance_test.py    # full end-to-end acceptance incl. restart persistence
python scripts/acceptance_test.py --offline   # skip the yfinance attempt
```

## Layout
```
apps/dashboard/        Streamlit entry + one file per page
configs/               base.yaml (defaults), risk.yaml (paper risk limits)
docs/                  architecture, assumptions, data contracts, DQ-001, API, limitations, task board
scripts/               smoke_test.sh, acceptance_test.py
src/quantlab/
  core/ domain/ storage/ data/ strategies/ backtest/ risk/ research/ paper/ ai/ reports/ ui/ api/ cli/
tests/unit, tests/integration
```
Generated data lives in `data/` (gitignored): `snapshots/`, `runs/`, `experiments/`, `exports/`, `quantlab.db`.

Read `docs/backtest_assumptions.md` and `docs/limitations.md` before interpreting any result.
