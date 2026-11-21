# Limitations (honest list)

## Not verified in the build environment
* **Real market data was not downloaded during development.** The build sandbox blocks Yahoo
  Finance (HTTP 403 via proxy). The yfinance provider is implemented and tested against mocked
  responses; the acceptance test attempts a real download first and records the fallback to the
  labelled synthetic fixture when it fails. Run `quantlab data ingest --symbols SPY QQQ IWM GLD TLT
  --start 2015-01-01` on your machine to use real data.
* **LLM providers were not called for real** (no API key, no network). Adapters are tested with
  mocked HTTP transports; the assistant runs in deterministic mode without a key.
* The dashboard was tested with Streamlit's `AppTest` harness and a live server health check, not
  with a visual browser review.

## Modelling
* Fill model = next open ± fixed slippage; no market impact, no queue position, no halts.
* Backtests have no volume cap; paper trading caps fills at 10% of session volume.
* Dividends credited on the ex-date (pay date ignored) and held as cash (not reinvested).
* Yahoo prices are split-adjusted retroactively; true as-traded prices are not available from it.
* ETF-only universe reduces but does not remove survivorship/selection bias.
* No taxes, borrow, FX (USD only; CAD conversion is future work), or account fees.
* Sharpe CI uses an iid approximation; DQ-001 uses few seeds → wide intervals.
* Synthetic fixture = GBM with fat-tail shocks; it has no real market structure, so strategy
  results on it are meaningless as evidence of profitability.

## Product scope not implemented (V1)
* FastAPI auth (local single user only), PostgreSQL/Alembic migrations (SQLite schema is
  versioned via `PRAGMA user_version`), Next.js UI, walk-forward optimisation UI, LIMIT orders,
  broker adapters (IBKR etc.), live trading, scheduler for automatic daily ingestion.

## Tested versions
Python 3.13.18 and 3.11.18 (full suite on both). pandas 3.1.1, numpy 2.5.5, scipy 1.18.1,
pyarrow 26.0.1, duckdb 2.0.1, streamlit 1.67.0, fastapi 0.165.0, plotly 7.1.0,
exchange_calendars 4.13.3, yfinance 1.8.0, pydantic 2.14.4.
