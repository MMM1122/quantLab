# Data contracts

## Bars (long format, one row per symbol × session)
| column | type | meaning |
|---|---|---|
| `symbol` | str | ticker, uppercase |
| `session_date` | datetime64[ns], tz-naive | XNYS trading session (America/New_York date) — never a calendar day |
| `open`,`high`,`low`,`close` | float64 | per snapshot `price_basis` (see below) |
| `adj_close` | float64 | split + dividend **back-adjusted** close (display / adjusted research mode only) |
| `volume` | float64 | shares |
| `dividend` | float64 | cash dividend per share with ex-date = this session (0 otherwise) |
| `split_ratio` | float64 | new shares per old share with ex-date = this session (1 otherwise) |
| `currency` | str | `USD` |
| `source` | str | provider (`yfinance` / `synthetic`); one source per snapshot |

`price_basis`: `split_adjusted` (Yahoo OHLC — splits already applied, dividends not) or
`unadjusted` (as traded; synthetic fixture). Invariant checked: `low ≤ open, close ≤ high`.

## Snapshot
`id` (UUIDv5 of content key), `dataset_hash` = `<sha256>:<price_basis>:<is_synthetic>`,
`provider`, `parameters` (symbols, dates, seed…), `retrieved_at` (UTC), `symbols`, `start`, `end`,
`n_rows`, `parent_id` (derived snapshots), `manifest` (transformation + event log).
The SHA-256 is computed over canonical column bytes (not Parquet bytes) so it is stable.

## Quality issue
`issue_type`, `severity` ∈ {INFO, WARN, ERROR, BLOCK}, `message`, `symbol`, `session_date`,
`row_count`, `details` (sample dates, values). Checks: schema, provenance, duplicates, non-finite,
non-positive prices, OHLC bounds, volume, corporate-action sanity, non-session rows, missing
sessions (WARN ≤ 2%, ERROR above), stale tail, extreme returns (flag only), flatlines, split
consistency vs price basis, adj_close factor consistency with recorded dividends/splits.

## Paper ledger
Money strings are `Decimal` (cash 2 dp, prices 4 dp). `cash_ledger.event_ref` and
`executions.execution_ref` are UNIQUE (idempotency). `orders (account_id, client_order_id)` is
UNIQUE. Statuses: CREATED, REJECTED, SUBMITTED, PARTIALLY_FILLED, FILLED, CANCELED.

## Timestamps
Event timestamps (`*_at`) are ISO-8601 UTC. Sessions are dates in the exchange calendar.
