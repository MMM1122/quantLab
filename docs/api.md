# API (local FastAPI)

Start: `uvicorn quantlab.api.app:app --host 127.0.0.1 --port 8000` → interactive docs at `/docs`.

Auth: if `QUANTLAB_API_TOKEN` is set, send `Authorization: Bearer <token>` (401 otherwise); `/health`, `/docs`,
`/openapi.json` stay public. Without a token the API is meant for `127.0.0.1` only.

Every response: `{"data": ..., "meta": {"service", "version", "live_trading_enabled": false, "mode", ...}}`
plus, where relevant, `label`, `data_label`, `assumptions`, `run_id`, `computation_seconds`, `source`.
Errors: `{"error": <type>, "detail": <message>}` with 422 (invalid input), 404 (not found),
409 (quality gate), 403 (live trading), 503 (data provider unavailable).

| method | path | notes |
|---|---|---|
| GET | `/health` | DB reachable |
| GET | `/assets` | universe + which snapshots contain each symbol |
| POST | `/data/ingest` | `{symbols, start, end?, provider: yfinance|synthetic, seed}` → 202 + job id |
| POST | `/data/refresh` | `{snapshot_id: <id>|latest, end?}` → 202; new snapshot + revision diff |
| GET | `/data/lineage?snapshot_id=` | refresh history with diff totals |
| POST | `/data/fx/ingest` | `{start, end?, provider: yfinance|synthetic}` → 202; USDCAD snapshot |
| GET | `/data/snapshots` | includes `quality_status`, `data_label` |
| GET | `/data/quality/{snapshot_id}` | stored issues |
| GET | `/strategies` | ids, params, warm-up, versions |
| POST | `/backtests` | `{strategy, snapshot_id, symbols, start?, end?, capital, commission_bps, slippage_bps, price_mode, params, mode, seed}` → 202 + job id; `mode=LIVE` → 403 |
| GET | `/backtests`, `/backtests/{id}`, `/backtests/{id}/trades` | stored results only |
| GET | `/backtests/{id}/metrics?currency=USD|CAD` | return metrics in USD or unhedged CAD |
| GET | `/backtests/compare?run_ids=a,b[,c]&n_trials=` | Sharpe-difference tests vs first run + deflated Sharpe |
| POST | `/experiments/dq-001` | → 202 + job id + experiment id |
| POST | `/experiments/walk-forward` | `{snapshot_id, strategy, symbols, param_grid, train_sessions, test_sessions, embargo_sessions, expanding, objective}` → 202 |
| GET | `/experiments`, `/experiments/{id}` | DQ-001 and walk-forward |
| GET | `/jobs/{job_id}` | PENDING / RUNNING / COMPLETED / FAILED (+ error) |
| POST | `/paper/accounts` | create PAPER account on a snapshot (historical replay) |
| GET | `/paper/account?account_id=` | one account or all |
| POST | `/paper/orders` | `{account_id, symbol, side, quantity, client_order_id?, order_type: MARKET|LIMIT, limit_price?, time_in_force: GTC|DAY}`; risk-checked before creation |
| GET | `/paper/orders?account_id=` | |
| POST | `/paper/advance` | `{account_id, sessions}` step the replay clock |
| POST | `/paper/kill-switch` | `{account_id, active, reason}` |
| POST | `/paper/autopilot` | `{account_id, strategy, symbols, params}` attach a deterministic strategy |
| POST | `/paper/autopilot/run` | `{account_id, sessions}` decide at close → advance → repeat |
| GET | `/paper/autopilot/compare?account_id=` | paper vs backtest NAV with explained differences |
| GET | `/risk/events` | |
| POST | `/ai/ask` | `{task: explain|compare|summarize_experiment, ids, question?}` read-only |
| POST | `/orders/live` | always 403 — exists only to prove there is no live path |
