# API (local FastAPI)

Start: `uvicorn quantlab.api.app:app --host 127.0.0.1 --port 8000` → interactive docs at `/docs`.

Every response: `{"data": ..., "meta": {"service", "version", "live_trading_enabled": false, "mode", ...}}`
plus, where relevant, `label`, `data_label`, `assumptions`, `run_id`, `computation_seconds`, `source`.
Errors: `{"error": <type>, "detail": <message>}` with 422 (invalid input), 404 (not found),
409 (quality gate), 403 (live trading), 503 (data provider unavailable).

| method | path | notes |
|---|---|---|
| GET | `/health` | DB reachable |
| GET | `/assets` | universe + which snapshots contain each symbol |
| POST | `/data/ingest` | `{symbols, start, end?, provider: yfinance|synthetic, seed}` → 202 + job id |
| GET | `/data/snapshots` | includes `quality_status`, `data_label` |
| GET | `/data/quality/{snapshot_id}` | stored issues |
| GET | `/strategies` | ids, params, warm-up, versions |
| POST | `/backtests` | `{strategy, snapshot_id, symbols, start?, end?, capital, commission_bps, slippage_bps, price_mode, params, mode, seed}` → 202 + job id; `mode=LIVE` → 403 |
| GET | `/backtests`, `/backtests/{id}`, `/backtests/{id}/trades` | stored results only |
| POST | `/experiments/dq-001` | → 202 + job id + experiment id |
| GET | `/experiments`, `/experiments/{id}` | |
| GET | `/jobs/{job_id}` | PENDING / RUNNING / COMPLETED / FAILED (+ error) |
| POST | `/paper/accounts` | create PAPER account on a snapshot (historical replay) |
| GET | `/paper/account?account_id=` | one account or all |
| POST | `/paper/orders` | market orders only; risk-checked before creation |
| GET | `/paper/orders?account_id=` | |
| POST | `/paper/advance` | `{account_id, sessions}` step the replay clock |
| POST | `/paper/kill-switch` | `{account_id, active, reason}` |
| GET | `/risk/events` | |
| POST | `/ai/ask` | `{task: explain|compare|summarize_experiment, ids, question?}` read-only |
| POST | `/orders/live` | always 403 — exists only to prove there is no live path |
