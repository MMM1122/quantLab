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
| Strategies | Buy & Hold, SMA crossover (50/200), RSI mean reversion (Cutler RSI), 12-1 cross-sectional momentum, volatility-target overlay (never levered) — all causal (tested) |
| Statistics | Jobson–Korkie/Memmel Sharpe-difference test, paired block-bootstrap CI, probabilistic and deflated Sharpe (multiple-testing correction) |
| Backtests | event-driven daily loop, next-open execution, slippage + commission, integer shares, cash/position invariants, dividends & splits without double counting, net/gross/benchmark metrics, Sharpe CI, reproducible runs with full provenance |
| Walk-forward | parameter grid chosen on training windows only, evaluated on later unseen windows; stitched out-of-sample equity, OOS rank of chosen params, leakage test |
| DQ-001 | missing/delayed/corrupted perturbations, strict / bounded / naive (research-only) policies, multi-seed CIs, derived snapshots, detection rates, reports |
| Paper trading | PAPER-only accounts on historical replay, Decimal ledger, MARKET and LIMIT orders with DAY/GTC, strategy autopilot with paper-vs-backtest consistency report, pre-trade risk (no shorting, no margin/negative cash, ≤25% per asset, stale data, duplicates, limit price band, kill switch), idempotent orders & fills, partial fills, reconciliation, persistence across restarts |
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
quantlab data refresh --snapshot latest        # new snapshot + provider-revision report (real data only)

# 2) backtests
quantlab backtest run --strategy ma-crossover --symbol SPY --snapshot <snapshot_id>
quantlab backtest run --strategy rsi --symbol SPY --param entry=25 --param exit=60
quantlab backtest run --strategy momentum --symbols SPY QQQ IWM TLT GLD
quantlab backtest compare --runs <run_a> <run_b> --stats --trials 20   # tests + deflated Sharpe
quantlab backtest run --strategy vol-target --symbol SPY --param target_vol=0.10 --param base=sma-crossover
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
quantlab paper autopilot attach --account <account_id> --strategy momentum --symbols SPY QQQ IWM TLT GLD
quantlab paper autopilot run --account <account_id> --sessions 250
quantlab paper autopilot compare --account <account_id>   # paper vs backtest, with explained gaps

# 5) AI assistant (deterministic without API key)
quantlab ai explain --run <run_id>
quantlab ai compare --runs <run_a> <run_b>
quantlab ai summarize --experiment <experiment_id>

# interfaces
streamlit run apps/dashboard/streamlit_app.py          # or: make dashboard
uvicorn quantlab.api.app:app --port 8000               # or: make api  → http://127.0.0.1:8000/docs
```

## Deployment

QuantLab is a research and **paper**-trading tool. Every deployment option below keeps
`LIVE_TRADING_ENABLED=false`, and there is no broker integration to switch on.

| option | best for | persistence | effort |
|---|---|---|---|
| **A. Streamlit Community Cloud** | a shareable dashboard demo | ephemeral (lost on reboot/redeploy) | ~5 min, free |
| **B. Docker Compose** (your machine or a VPS) | daily use, dashboard + API | persistent volume | ~10 min |
| **C. Local** (see *Run* above) | development | `./data` | — |

Serverless platforms such as Vercel are not a good fit: Streamlit needs a long-running process with
websockets, and QuantLab needs a writable disk for SQLite and Parquet.

### A. Streamlit Community Cloud
1. Push to GitHub (already done for `MMM1122/quantLab`).
2. Open <https://share.streamlit.io> → **Create app** → *Deploy a public app from GitHub*.
3. Repository `MMM1122/quantLab`, branch `main`, main file path **`apps/dashboard/streamlit_app.py`**.
4. **Advanced settings** → Python **3.12**. Optional **Secrets** (TOML; root-level keys become environment variables):
   ```toml
   LIVE_TRADING_ENABLED = "false"
   # optional AI phrasing; without a key the assistant uses deterministic analytics
   QUANTLAB_LLM_PROVIDER = "anthropic"
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Deploy. On first visit open **Market Explorer → Ingest data** (`yfinance` for real data, or the
   labelled `synthetic` fixture).
6. Notes:
   - Storage is **ephemeral**: snapshots, runs and paper accounts disappear when the app restarts.
     Export the reports you want to keep.
   - Streamlit has no login. Anyone with the URL can ingest data and run jobs, so set the app's
     sharing to **private / invited viewers** in the app settings.
   - Dependencies come from `requirements.txt`; the entry file adds `src/` to the import path itself.

### B. Docker Compose (persistent; dashboard + API)
```bash
git clone https://github.com/MMM1122/quantLab.git && cd quantLab
cp .env.example .env                         # optional: QUANTLAB_API_TOKEN, LLM keys
echo "QUANTLAB_API_TOKEN=$(openssl rand -hex 24)" >> .env   # recommended when the API is reachable by others
GIT_COMMIT=$(git rev-parse HEAD) docker compose up -d --build
# dashboard → http://127.0.0.1:8501     API → http://127.0.0.1:8000/docs
docker compose exec dashboard quantlab data ingest --symbols SPY QQQ IWM GLD TLT --start 2015-01-01
docker compose logs -f                       # follow logs
docker compose down                          # stop (data stays in the `quantlab-data` volume)
```
- Data lives in the named volume `quantlab-data` (`/data` in the containers). Back it up with
  `docker run --rm -v quantlab-data:/data -v "$PWD":/backup busybox tar czf /backup/quantlab-data.tgz /data`.
- Both ports are bound to `127.0.0.1`. To serve on a VPS, put a reverse proxy with authentication
  (e.g. Caddy/nginx basic auth or an SSO proxy) in front of `8501`, and keep `QUANTLAB_API_TOKEN` set.
  API clients send `Authorization: Bearer <token>`.
- Code version is baked into the image (`GIT_COMMIT`) so run provenance still records it.
- Optional daily refresh (host cron), which reports provider revisions and exits 5 when it finds any:
  `30 18 * * 1-5 cd ~/quantLab && docker compose exec -T dashboard quantlab data refresh --snapshot latest`

Verification: CI builds this image, runs the CLI inside it, and health-checks the dashboard and the
token-protected API on every push (`.github/workflows/ci.yml`, job `docker`).

## Tests and checks
```bash
ruff check .                         # lint
pytest -q                            # 164 unit + integration tests (offline)
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
