# Limitations (honest list)

## Not verified in the build environment
* Real market data could not be downloaded in the development sandbox, but it **is verified on
  GitHub runners** by `.github/workflows/real-data.yml` (manual or PR-triggered, `--require-real`).
  See `docs/FINAL_REPORT.md` §4b for the real-data evidence.
* **LLM providers were not called for real** (no API key, no network). Adapters are tested with
  mocked HTTP transports; the assistant runs in deterministic mode without a key.
* The dashboard was tested with Streamlit's `AppTest` harness and a live server health check, not
  with a visual browser review.

## Modelling
* Fill model = next open ± fixed slippage; no market impact, no queue position, no halts.
* Paper LIMIT orders on daily bars: filled at the open if it is better than the limit, otherwise at
  the limit only if the bar trades *through* it (touching is not a fill); intraday path unknown.
* Walk-forward: each test fold starts in cash (re-entry cost at fold boundaries); sessions after
  the last full test fold are unused; grids are capped at 200 parameter sets.
* Backtests have no volume cap; paper trading caps fills at 10% of session volume.
* Paper cash checks do not count proceeds of pending sells, so an autopilot rotation can buy one session
  later than the backtest (reported in the paper-vs-backtest explanations).
* Dividends credited on the ex-date (pay date ignored) and held as cash (not reinvested).
* Yahoo prices are split-adjusted retroactively; true as-traded prices are not available from it.
* ETF-only universe reduces but does not remove survivorship/selection bias.
* No taxes, borrow or account fees. Trading and costs are in USD; CAD is a reporting view only
  (NAV_CAD = NAV_USD × USDCAD at the session's rate, previous rate for ≤ 5 sessions; no FX conversion
  costs, no hedging). Cost/turnover/dividend amounts are not converted.
* Sharpe CI uses an iid approximation; DQ-001 uses few seeds → wide intervals.
* Synthetic fixture = GBM with fat-tail shocks; it has no real market structure, so strategy
  results on it are meaningless as evidence of profitability.

## Product scope not implemented (V1)
* FastAPI auth (local single user only), PostgreSQL/Alembic migrations (SQLite schema is
  versioned via `PRAGMA user_version` with forward migrations), Next.js UI, stop orders,
  broker adapters (IBKR etc.), live trading, scheduler for automatic daily ingestion.

## Tested versions
Python 3.13.18 and 3.11.18 (full suite on both). pandas 3.1.1, numpy 2.5.5, scipy 1.18.1,
pyarrow 26.0.1, duckdb 2.0.1, streamlit 1.67.0, fastapi 0.165.0, plotly 7.1.0,
exchange_calendars 4.13.3, yfinance 1.8.0, pydantic 2.14.4.
