"""Deterministic event-driven daily backtest engine.

Session loop (for each session t in the evaluation window):

1. **Open of t — corporate actions.** Ex-date events apply to holdings carried from t-1:
   splits rescale share counts (only when prices are unadjusted; fractional remainders are
   paid as cash-in-lieu at the previous close / ratio), cash dividends are credited (only in
   ``raw_with_dividends`` mode — never together with adjusted prices).
2. **Open of t — executions.** Orders decided at the close of t-1 fill at open(t) with adverse
   slippage + commission. Sells first, then buys. Buys are cash-limited so cash never goes
   negative; sells are limited to held quantity so positions never go negative. If the bar
   for t is missing the order expires (reason ``MISSING_BAR``) and is retried next decision.
3. **Close of t — valuation.** NAV = cash + Σ qty · close(t) (last known close if missing,
   counted as a stale mark). Accounting invariants are asserted every session.
4. **Close of t — decision.** Target weights row t (computed only from data <= t) is compared
   with the current desired weights; changed symbols are sized at close(t) and queued for t+1.
   Decisions made at the final session are not executed.

Analytic quantities are float64 (documented rounding); the paper broker uses Decimal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.execution import CostModel
from quantlab.backtest.market import MarketPanel
from quantlab.core.errors import AccountingInvariantError, InputValidationError
from quantlab.domain.models import PriceMode
from quantlab.strategies.base import Strategy

CASH_TOL = 1e-6


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    min_commission: float = 0.0
    price_mode: str = PriceMode.RAW_WITH_DIVIDENDS.value
    risk_free_rate: float = 0.0
    max_weight: float = 1.0
    start: str | None = None
    end: str | None = None
    seed: int = 42  # recorded for reproducibility; the engine itself is deterministic

    def __post_init__(self):
        if self.initial_cash <= 0:
            raise InputValidationError("initial_cash must be > 0")
        if not (0 < self.max_weight <= 1):
            raise InputValidationError("max_weight must be in (0, 1] (no leverage)")
        PriceMode(self.price_mode)

    @property
    def costs(self) -> CostModel:
        return CostModel(self.commission_bps, self.slippage_bps, self.min_commission)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    equity: pd.DataFrame  # index session: cash, positions_value, nav, gross_exposure, stale_marks
    trades: pd.DataFrame
    orders: pd.DataFrame
    events: pd.DataFrame  # dividends, splits, cash-in-lieu
    positions: pd.DataFrame  # end-of-session quantities (sessions x symbols)
    targets: pd.DataFrame  # strategy output on the evaluation window
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def _window(sessions: pd.DatetimeIndex, start: str | None, end: str | None) -> tuple[int, int]:
    i0 = 0 if start is None else int(sessions.searchsorted(pd.Timestamp(start)))
    i1 = len(sessions) - 1 if end is None else int(sessions.searchsorted(pd.Timestamp(end), side="right")) - 1
    if i0 >= len(sessions) or i1 < i0 + 1:
        raise InputValidationError("evaluation window must contain at least two sessions of data")
    return i0, i1


def run_backtest(
    panel: MarketPanel,
    strategy: Strategy,
    config: BacktestConfig,
    signal_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run one backtest. ``signal_prices`` overrides what the strategy *observes* (used by
    DQ-001 to feed perturbed data while fills/valuation use ``panel`` as the truth)."""
    sessions = panel.sessions
    syms = panel.symbols
    n_sym = len(syms)
    obs = panel.signal_price if signal_prices is None else signal_prices.reindex(index=sessions, columns=syms)
    targets = strategy.target_weights(obs).reindex(index=sessions, columns=syms).astype(float)
    if ((targets < -1e-12) | (targets > 1 + 1e-12)).any().any():
        raise AccountingInvariantError(f"{strategy.id} produced weights outside [0, 1]")
    targets = targets.clip(upper=config.max_weight)
    row_sum = targets.fillna(0).sum(axis=1)
    over = row_sum > 1 + 1e-9
    if over.any():
        targets.loc[over] = targets.loc[over].div(row_sum[over], axis=0)

    i0, i1 = _window(sessions, config.start, config.end)
    costs = config.costs
    OPEN = panel.open.to_numpy(float)
    C = panel.close.to_numpy(float)
    DIV = panel.dividend.to_numpy(float)
    SPL = panel.split.to_numpy(float)
    T = targets.to_numpy(float)

    cash = float(config.initial_cash)
    qty = np.zeros(n_sym, dtype=np.int64)
    last_close = np.full(n_sym, np.nan)
    # seed last_close with history before the window so pre-window data can value positions
    for j in range(n_sym):
        hist = C[: i0 + 1, j]
        ok = np.where(np.isfinite(hist))[0]
        if len(ok):
            last_close[j] = hist[ok[-1]]
    desired = np.full(n_sym, np.nan)
    retry = np.zeros(n_sym, dtype=bool)
    pending: list[dict[str, Any]] = []
    order_seq = 0

    eq_rows, trades, orders, events, pos_rows = [], [], [], [], []
    warnings: list[str] = []

    for i in range(i0, i1 + 1):
        d = sessions[i]
        div_cash = 0.0
        # ---- 1. corporate actions at the open (ex-date = d), for holdings carried from d-1
        if i > i0:
            for j in range(n_sym):
                if qty[j] <= 0:
                    continue
                if panel.apply_splits_to_holdings and np.isfinite(SPL[i, j]) and SPL[i, j] != 1.0:
                    r = SPL[i, j]
                    new_q = qty[j] * r
                    whole = int(math.floor(new_q + 1e-9))
                    frac = new_q - whole
                    in_lieu = frac * (last_close[j] / r) if frac > 1e-9 and np.isfinite(last_close[j]) else 0.0
                    events.append({"session": d, "symbol": syms[j], "type": "SPLIT", "ratio": r,
                                   "qty_before": int(qty[j]), "qty_after": whole, "cash": in_lieu})
                    qty[j] = whole
                    cash += in_lieu
                    if np.isfinite(last_close[j]):
                        last_close[j] = last_close[j] / r
                if panel.credit_dividends and DIV[i, j] > 0:
                    amt = float(qty[j]) * DIV[i, j]
                    cash += amt
                    div_cash += amt
                    events.append({"session": d, "symbol": syms[j], "type": "DIVIDEND", "per_share": DIV[i, j],
                                   "qty": int(qty[j]), "cash": amt})

        # ---- 2. executions at the open of d (orders decided at close of d-1)
        if pending:
            pending.sort(key=lambda o: 0 if o["side"] == "SELL" else 1)
            for o in pending:
                j = o["j"]
                px = OPEN[i, j]
                base = {"order_id": o["order_id"], "decision_session": o["decision_session"], "session": d,
                        "symbol": syms[j], "side": o["side"], "requested_qty": o["qty"]}
                if not (np.isfinite(px) and px > 0):
                    orders.append({**base, "filled_qty": 0, "status": "EXPIRED", "reason": "MISSING_BAR"})
                    retry[j] = True
                    continue
                fill_px = costs.fill_price(px, o["side"])
                if o["side"] == "SELL":
                    q = int(min(o["qty"], qty[j]))
                    if q <= 0:
                        orders.append({**base, "filled_qty": 0, "status": "REJECTED", "reason": "NO_SHORTING"})
                        continue
                    notional = q * fill_px
                    fee = costs.commission(notional)
                    cash += notional - fee
                    qty[j] -= q
                else:
                    q = int(min(o["qty"], costs.max_affordable_qty(cash, fill_px)))
                    if q <= 0:
                        orders.append({**base, "filled_qty": 0, "status": "REJECTED", "reason": "INSUFFICIENT_CASH"})
                        continue
                    notional = q * fill_px
                    fee = costs.commission(notional)
                    cash -= notional + fee
                    qty[j] += q
                status = "FILLED" if q == o["qty"] else "PARTIALLY_FILLED"
                reason = None if status == "FILLED" else ("CASH_LIMITED" if o["side"] == "BUY" else "POSITION_LIMITED")
                orders.append({**base, "filled_qty": q, "status": status, "reason": reason})
                trades.append({
                    "session": d, "decision_session": o["decision_session"], "symbol": syms[j], "side": o["side"],
                    "qty": q, "ref_open": px, "fill_price": fill_px, "notional": notional, "commission": fee,
                    "slippage_cost": abs(fill_px - px) * q, "cash_after": cash, "order_id": o["order_id"],
                })
                if cash < -CASH_TOL:
                    raise AccountingInvariantError(f"negative cash {cash} after fill on {d:%Y-%m-%d}")
            pending = []

        # ---- 3. valuation at the close of d
        stale = 0
        for j in range(n_sym):
            if np.isfinite(C[i, j]) and C[i, j] > 0:
                last_close[j] = C[i, j]
            elif qty[j] > 0:
                stale += 1
        held_val = np.where(qty > 0, qty * np.nan_to_num(last_close, nan=0.0), 0.0)
        if np.any(qty < 0):
            raise AccountingInvariantError(f"negative position on {d:%Y-%m-%d}")
        if np.any((qty > 0) & ~np.isfinite(last_close)):
            raise AccountingInvariantError(f"cannot value position without any price on {d:%Y-%m-%d}")
        mv = float(held_val.sum())
        nav = cash + mv
        if cash < -CASH_TOL:
            raise AccountingInvariantError(f"negative cash {cash} on {d:%Y-%m-%d}")
        eq_rows.append({"session": d, "cash": cash, "positions_value": mv, "nav": nav,
                        "gross_exposure": mv / nav if nav > 0 else 0.0, "stale_marks": stale,
                        "dividends": div_cash})
        pos_rows.append(qty.copy())

        # ---- 4. decision at the close of d -> orders for d+1
        if i == i1:
            if np.any(np.isfinite(T[i]) & (np.abs(np.nan_to_num(T[i]) - np.nan_to_num(desired)) > 1e-12)):
                warnings.append(f"signal change at final session {d:%Y-%m-%d} not executed (no next session)")
            break
        changed = []
        for j in range(n_sym):
            t = T[i, j]
            if np.isfinite(t) and (not np.isfinite(desired[j]) or abs(t - desired[j]) > 1e-12):
                desired[j] = t
                changed.append(j)
            elif retry[j] and np.isfinite(desired[j]):
                changed.append(j)
        for j in changed:
            px = C[i, j]
            if not (np.isfinite(px) and px > 0):
                retry[j] = True  # cannot size without a fresh close; try again next session
                continue
            retry[j] = False
            target_q = int(math.floor(desired[j] * nav / px))
            delta = target_q - int(qty[j])
            if delta == 0:
                continue
            order_seq += 1
            pending.append({"order_id": order_seq, "j": j, "side": "BUY" if delta > 0 else "SELL",
                            "qty": abs(delta), "decision_session": d})

    equity = pd.DataFrame(eq_rows).set_index("session")
    positions = pd.DataFrame(pos_rows, index=equity.index, columns=syms)
    tr_cols = ["session", "decision_session", "symbol", "side", "qty", "ref_open", "fill_price", "notional",
               "commission", "slippage_cost", "cash_after", "order_id"]
    trades_df = pd.DataFrame(trades, columns=tr_cols)
    ord_cols = ["order_id", "decision_session", "session", "symbol", "side", "requested_qty", "filled_qty",
                "status", "reason"]
    orders_df = pd.DataFrame(orders, columns=ord_cols)
    events_df = pd.DataFrame(events, columns=["session", "symbol", "type", "ratio", "per_share", "qty",
                                              "qty_before", "qty_after", "cash"])
    if int(equity["stale_marks"].sum()):
        warnings.append(f"{int(equity['stale_marks'].sum())} position marks used a stale close (missing bars)")
    expired = (orders_df["status"] == "EXPIRED").sum() if len(orders_df) else 0
    if expired:
        warnings.append(f"{expired} orders expired because the execution bar was missing")
    return BacktestResult(
        equity=equity,
        trades=trades_df,
        orders=orders_df,
        events=events_df,
        positions=positions,
        targets=targets.iloc[i0 : i1 + 1],
        warnings=warnings,
        config=config.to_dict(),
    )
