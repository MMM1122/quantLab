# Backtest assumptions (read before trusting any number)

All backtest outputs are labelled **BACKTEST / SIMULATION — NOT REAL PERFORMANCE**.

## Timing
* Strategies are evaluated **after the close** of session *t* using data up to and including *t*.
* Orders execute at the **open of session t+1** — never at the close that produced the signal.
  Tested: `test_no_trade_on_signal_bar_executes_next_open`,
  `test_next_open_changes_fill_but_next_close_cannot_change_signal`.
* Signals at the final session of the window are not executed (reported as a warning).
* Strategy output at row *t* is identical whether or not future rows exist (causality tests for
  every strategy with default and short parameters).

## Fills and costs (an explicit model, not a broker)
* Fill price = open(t+1) × (1 + slippage_bps/10⁴) for buys, × (1 − slippage_bps/10⁴) for sells.
* Commission = max(min_commission, commission_bps/10⁴ × notional). Defaults: 1 bp + 2 bp slippage.
* Integer shares. Orders are sized at close(t) and buys are reduced at fill time to what cash can
  pay for (status `PARTIALLY_FILLED`, reason `CASH_LIMITED`) — cash can never go negative.
* Sells are capped at held quantity — positions can never go negative.
* If the execution bar is missing the order expires (`MISSING_BAR`) and is retried next decision.
* No market impact, no volume limits in backtests (paper trading has a participation cap).
* Every run is also executed with zero costs ("gross") so net vs gross is always visible.

## Corporate actions — no double counting
Two mutually exclusive price modes:

| mode | prices traded | dividends | splits |
|---|---|---|---|
| `raw_with_dividends` (default) | as provided (Yahoo: split-adjusted, not dividend-adjusted) | cash credited on ex-date for shares held at prior close; **held as cash, not reinvested** | applied to holdings only if prices are `unadjusted` (cash in lieu for fractions) |
| `adjusted_total_return` (research mode) | OHLC × adj_close/close | **none credited** (already in prices) | none (already in prices) |

Strategies always observe a **causal total-return index** (built forward from close, dividends
and splits), so a split does not create a fake crossover and no future corporate action leaks
into past signals. Back-adjusted `adj_close` (which uses future events) is never used for signals.

## Metrics
* Daily NAV returns, 252 sessions/year. CAGR uses (n_sessions − 1)/252 years.
* Sharpe = mean(daily excess)/std(daily excess, ddof=1)·√252, risk-free converted geometrically
  (default 0%). Standard error per Lo (2002, iid): √((1 + SR²/2)/n), reported as a 95% CI.
* Max drawdown on NAV; duration = longest run of sessions below the prior peak.
* Turnover = Σ|traded notional| / average NAV / years. Time in market = share of sessions with
  positive exposure.
* Benchmark = buy-and-hold of `SPY` (configurable) over the same window with the same costs.
* A higher backtest Sharpe does not prove an exploitable edge; overlapping CIs mean no
  statistically reliable ranking.

## Reproducibility
Each run persists: strategy id/version/params, snapshot id + SHA-256, full config + config hash,
git commit (with `-dirty` suffix if uncommitted changes), seed, warnings, timing. Re-running the
same inputs produces identical equity curves and trades (tested).
