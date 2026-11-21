"""Local paper broker on historical session replay.

This is a *simulator*. It never connects to a real broker and has no live mode. The account
clock (``current_session``) walks through the sessions of an immutable snapshot; prices are
historical, labelled ``HISTORICAL REPLAY — NOT LIVE MARKET CONDITIONS``.

Order lifecycle
  intent --(pre-trade risk checks)--> REJECTED                       (persisted for audit)
                                  \\-> SUBMITTED --advance--> FILLED
                                                  \\-> PARTIALLY_FILLED (volume cap) --advance--> ...
                                                  \\-> CANCELED (user, kill switch, cash at fill)
Market orders decided during session t fill at the open of t+1 (adverse slippage + commission),
capped at ``participation_rate`` x session volume (remainder keeps working).

Money is Decimal: prices 4 dp, cash 2 dp (ROUND_HALF_EVEN), commission rounded up to the cent.
The cash ledger is the source of truth for cash; ``reconcile`` recomputes cash and positions
from the ledger and executions and compares them with stored state.
"""

from __future__ import annotations

import sqlite3
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_UP, Decimal, InvalidOperation
from functools import lru_cache
from typing import Any

import pandas as pd

from quantlab.context import AppContext
from quantlab.core.errors import (
    AccountingInvariantError,
    InputValidationError,
    LiveTradingDisabledError,
    NotFoundError,
    QuantLabError,
)
from quantlab.core.ids import deterministic_id, new_id, validate_symbol, validate_uuid
from quantlab.core.labels import REPLAY_LABEL, data_label
from quantlab.data import calendar
from quantlab.data.ingestion import require_tradable_snapshot
from quantlab.domain.models import OPEN_ORDER_STATUSES, OrderSide, OrderStatus, PriceBasis
from quantlab.risk import rules
from quantlab.storage.db import dumps, loads, utcnow

CENT = Decimal("0.01")
PRICE_Q = Decimal("0.0001")
ZERO = Decimal("0")


def D(x: Any) -> Decimal:
    try:
        return x if isinstance(x, Decimal) else Decimal(str(x))
    except (InvalidOperation, ValueError) as exc:
        raise InputValidationError(f"not a number: {x!r}") from exc


def money(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_EVEN)


class ReplayFinished(QuantLabError):
    pass


class _Market:
    """Read-only view of one snapshot on its session grid."""

    def __init__(self, bars: pd.DataFrame, price_basis: str, is_synthetic: bool):
        self.price_basis = price_basis
        self.is_synthetic = is_synthetic
        self.symbols = sorted(bars["symbol"].unique())
        self.sessions = calendar.sessions(bars["session_date"].min(), bars["session_date"].max())
        self._idx = {d: i for i, d in enumerate(self.sessions)}
        self.bars = {(r.symbol, r.session_date): r for r in bars.itertuples(index=False)}
        self.last_idx: dict[str, list[int]] = {}
        for s in self.symbols:
            have = sorted(self._idx[d] for d in bars.loc[bars["symbol"] == s, "session_date"])
            self.last_idx[s] = have

    def index(self, session: pd.Timestamp) -> int:
        try:
            return self._idx[pd.Timestamp(session)]
        except KeyError as exc:
            raise InputValidationError(f"{session} is not a session in this snapshot") from exc

    def bar(self, symbol: str, i: int):
        return self.bars.get((symbol, self.sessions[i]))

    def latest_bar_index(self, symbol: str, i: int) -> int | None:
        import bisect

        have = self.last_idx.get(symbol, [])
        k = bisect.bisect_right(have, i)
        return have[k - 1] if k else None

    def last_close(self, symbol: str, i: int) -> Decimal | None:
        j = self.latest_bar_index(symbol, i)
        if j is None:
            return None
        return D(self.bar(symbol, j).close).quantize(PRICE_Q)


class PaperBroker:
    def __init__(self, ctx: AppContext, risk_overrides: dict[str, Any] | None = None):
        self.ctx = ctx
        self.db = ctx.db
        self.risk_defaults = {**ctx.settings.risk, **(risk_overrides or {})}
        if self.risk_defaults.get("allow_short") or self.risk_defaults.get("allow_margin"):
            raise InputValidationError("shorting/margin cannot be enabled")

    # ------------------------------------------------------------------ helpers
    @lru_cache(maxsize=8)  # noqa: B019 - broker objects are short-lived per context
    def _market(self, snapshot_id: str) -> _Market:
        snap = self.ctx.snapshots.get(snapshot_id)
        return _Market(self.ctx.snapshots.load(snapshot_id), snap.price_basis, snap.is_synthetic)

    def _account_row(self, account_id: str, con: sqlite3.Connection | None = None) -> dict[str, Any]:
        aid = validate_uuid(account_id, "account_id")
        sql = "SELECT * FROM accounts WHERE id=?"
        row = (con.execute(sql, (aid,)).fetchone() if con else None) or self.db.query_one(sql, (aid,))
        if row is None:
            raise NotFoundError(f"paper account {aid} not found")
        row = dict(row)
        row["config"] = loads(row["config"])
        return row

    @staticmethod
    def _cash(con, aid: str) -> Decimal:
        rows = con.execute("SELECT amount FROM cash_ledger WHERE account_id=?", (aid,)).fetchall()
        return sum((D(r[0]) for r in rows), ZERO)

    @staticmethod
    def _positions(con, aid: str) -> dict[str, Decimal]:
        rows = con.execute("SELECT symbol, quantity FROM positions WHERE account_id=?", (aid,)).fetchall()
        return {r[0]: D(r[1]) for r in rows if D(r[1]) != 0}

    def _risk_event(self, con, aid: str | None, rule: str, action: str, details: dict, severity: str = "WARN") -> None:
        con.execute("INSERT INTO risk_events(id, account_id, rule, severity, action, details, created_at) "
                    "VALUES (?,?,?,?,?,?,?)", (new_id(), aid, rule, severity, action, dumps(details), utcnow()))

    def _commission(self, cfg: dict, notional: Decimal) -> Decimal:
        if notional <= 0:
            return ZERO
        c = notional * D(cfg["commission_bps"]) / Decimal(10_000)
        return max(D(cfg["min_commission"]), c).quantize(CENT, rounding=ROUND_UP)

    def _fill_price(self, cfg: dict, ref: Decimal, side: str) -> Decimal:
        adj = D(cfg["slippage_bps"]) / Decimal(10_000)
        px = ref * (1 + adj) if side == OrderSide.BUY else ref * (1 - adj)
        return px.quantize(PRICE_Q, rounding=ROUND_HALF_EVEN)

    # ------------------------------------------------------------------ accounts
    def create_account(self, name: str, snapshot_id: str, initial_cash: Any = 10_000,
                       start_session: str | None = None, mode: str = "PAPER") -> dict[str, Any]:
        if str(mode).upper() != "PAPER":
            raise LiveTradingDisabledError("only PAPER accounts exist in QuantLab V1")
        cash = D(initial_cash)
        if cash <= 0:
            raise InputValidationError("initial_cash must be > 0")
        if not name or len(name) > 80:
            raise InputValidationError("account name must be 1-80 characters")
        snap = require_tradable_snapshot(self.ctx.snapshots, snapshot_id)
        mkt = self._market(snap.id)
        i = mkt.index(pd.Timestamp(start_session)) if start_session else 0
        aid = new_id()
        cfg = {k: self.risk_defaults[k] for k in ("max_position_weight", "max_data_staleness_sessions",
                                                  "participation_rate", "commission_bps", "min_commission",
                                                  "slippage_bps")}
        with self.db.transaction() as con:
            con.execute(
                "INSERT INTO accounts(id, name, mode, base_currency, status, initial_cash, snapshot_id, current_session, "
                "kill_switch, config, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (aid, name, "PAPER", "USD", "ACTIVE", str(money(cash)), snap.id,
                 mkt.sessions[i].strftime("%Y-%m-%d"), 0, dumps(cfg), utcnow()),
            )
            con.execute("INSERT INTO cash_ledger(id, account_id, event_ref, currency, amount, event_type, session_date, "
                        "recorded_at) VALUES (?,?,?,?,?,?,?,?)",
                        (new_id(), aid, f"{aid}:deposit", "USD", str(money(cash)), "DEPOSIT",
                         mkt.sessions[i].strftime("%Y-%m-%d"), utcnow()))
            self._snapshot_portfolio(con, aid, mkt, i)
            self.db.audit("user", "paper.account.created", "account", aid,
                          {"snapshot_id": snap.id, "initial_cash": str(cash)}, con=con)
        return self.account_state(aid)

    def list_accounts(self) -> list[dict[str, Any]]:
        return [self.account_state(r["id"]) for r in self.db.query("SELECT id FROM accounts ORDER BY created_at")]

    def account_state(self, account_id: str) -> dict[str, Any]:
        acc = self._account_row(account_id)
        mkt = self._market(acc["snapshot_id"])
        i = mkt.index(pd.Timestamp(acc["current_session"]))
        with self.db.connect() as con:
            cash = self._cash(con, acc["id"])
            pos = self._positions(con, acc["id"])
        holdings = []
        mv = ZERO
        for s, q in sorted(pos.items()):
            px = mkt.last_close(s, i)
            stale = (i - (mkt.latest_bar_index(s, i) or i)) if px is not None else None
            val = (q * px) if px is not None else ZERO
            mv += val
            holdings.append({"symbol": s, "quantity": str(q), "last_close": str(px), "market_value": str(money(val)),
                             "price_age_sessions": stale})
        nav = cash + mv
        for h in holdings:
            h["weight"] = float(D(h["market_value"]) / nav) if nav > 0 else None
        return {
            "id": acc["id"], "name": acc["name"], "mode": acc["mode"], "status": acc["status"],
            "label": REPLAY_LABEL, "data_label": data_label(mkt.is_synthetic), "is_synthetic": mkt.is_synthetic,
            "snapshot_id": acc["snapshot_id"], "replay_session": acc["current_session"],
            "replay_sessions_remaining": len(mkt.sessions) - 1 - i,
            "kill_switch": bool(acc["kill_switch"]), "base_currency": "USD",
            "initial_cash": acc["initial_cash"], "cash": str(money(cash)), "positions_value": str(money(mv)),
            "nav": str(money(nav)), "holdings": holdings, "risk_limits": acc["config"],
        }

    # ------------------------------------------------------------------ orders
    def submit_order(self, account_id: str, symbol: str, side: str, quantity: Any,
                     client_order_id: str | None = None, order_type: str = "MARKET",
                     actor: str = "user", mode: str = "PAPER") -> dict[str, Any]:
        if str(mode).upper() == "LIVE":
            raise LiveTradingDisabledError("live orders do not exist in QuantLab V1")
        if order_type != "MARKET":
            raise InputValidationError(f"order_type {order_type!r} not supported in V1 (MARKET only)")
        try:
            side_e = OrderSide(str(side).upper())
        except ValueError as exc:
            raise InputValidationError("side must be BUY or SELL") from exc
        qty = D(quantity)
        if qty != qty.to_integral_value() or qty <= 0:
            raise InputValidationError("quantity must be a positive whole number of shares")
        sym = validate_symbol(symbol)
        coid = client_order_id or new_id()
        if len(coid) > 64:
            raise InputValidationError("client_order_id too long")

        with self.db.transaction() as con:
            acc = self._account_row(account_id, con)
            aid = acc["id"]
            existing = con.execute("SELECT id FROM orders WHERE account_id=? AND client_order_id=?",
                                   (aid, coid)).fetchone()
            if existing:  # idempotent resubmission: return the original, create nothing
                return {**self._order(con, existing[0]), "idempotent_replay": True}
            cfg = acc["config"]
            mkt = self._market(acc["snapshot_id"])
            i = mkt.index(pd.Timestamp(acc["current_session"]))
            cash = self._cash(con, aid)
            pos = self._positions(con, aid)
            open_rows = con.execute(
                "SELECT symbol, side, quantity, filled_quantity FROM orders WHERE account_id=? AND status IN (?,?)",
                (aid, *[s.value for s in OPEN_ORDER_STATUSES])).fetchall()
            pend_sell = sum((D(r[2]) - D(r[3]) for r in open_rows if r[0] == sym and r[1] == "SELL"), ZERO)
            pend_buy_qty = sum((D(r[2]) - D(r[3]) for r in open_rows if r[0] == sym and r[1] == "BUY"), ZERO)
            pend_buy_cost = ZERO
            for r in open_rows:
                if r[1] == "BUY":
                    p = mkt.last_close(r[0], i) or ZERO
                    rem = D(r[2]) - D(r[3])
                    n = rem * self._fill_price(cfg, p, "BUY")
                    pend_buy_cost += n + self._commission(cfg, n)
            nav = cash + sum(((q * (mkt.last_close(s, i) or ZERO)) for s, q in pos.items()), ZERO)
            px = mkt.last_close(sym, i) if sym in mkt.symbols else None
            dup = con.execute(
                "SELECT id FROM orders WHERE account_id=? AND symbol=? AND side=? AND quantity=? AND decision_session=? "
                "AND status IN (?,?)", (aid, sym, side_e.value, str(qty), acc["current_session"],
                                        *[s.value for s in OPEN_ORDER_STATUSES])).fetchone()

            decision = rules.first_failure(
                rules.kill_switch(bool(acc["kill_switch"])),
                rules.RiskDecision(acc["status"] == "ACTIVE", "ACCOUNT_STATUS", f"account is {acc['status']}"),
                rules.RiskDecision(sym in mkt.symbols, "UNKNOWN_SYMBOL", f"{sym} not in replay snapshot"),
                rules.fresh_data(mkt.latest_bar_index(sym, i) if sym in mkt.symbols else None, i,
                                 int(cfg["max_data_staleness_sessions"])),
                rules.RiskDecision(dup is None, "DUPLICATE_ORDER",
                                   "identical open order already exists this session (use a distinct client_order_id "
                                   "only if intentional)"),
                rules.long_only(side_e, qty, pos.get(sym, ZERO), pend_sell),
            )
            if decision.approved and side_e == OrderSide.BUY:
                est_px = self._fill_price(cfg, px, "BUY")
                est = qty * est_px
                decision = rules.first_failure(
                    rules.sufficient_cash(side_e, money(est + self._commission(cfg, est)), cash, money(pend_buy_cost)),
                    rules.max_position_weight((pos.get(sym, ZERO) + pend_buy_qty + qty) * px, nav,
                                              D(cfg["max_position_weight"])),
                )
            oid = new_id()
            now = utcnow()
            status = OrderStatus.SUBMITTED if decision.approved else OrderStatus.REJECTED
            con.execute(
                "INSERT INTO orders(id, account_id, symbol, client_order_id, side, quantity, filled_quantity, order_type, "
                "status, reason, decision_session, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (oid, aid, sym, coid, side_e.value, str(qty), "0", "MARKET", status.value,
                 None if decision.approved else f"{decision.rule}: {decision.reason}", acc["current_session"], now, now),
            )
            if not decision.approved:
                self._risk_event(con, aid, decision.rule, "ORDER_REJECTED",
                                 {"order_id": oid, "symbol": sym, "side": side_e.value, "quantity": str(qty),
                                  "reason": decision.reason})
            self.db.audit(actor, "paper.order." + status.value.lower(), "order", oid,
                          {"symbol": sym, "side": side_e.value, "quantity": str(qty), "client_order_id": coid}, con=con)
            return self._order(con, oid)

    def cancel_order(self, account_id: str, order_id: str, reason: str = "USER_CANCELED") -> dict[str, Any]:
        oid = validate_uuid(order_id, "order_id")
        with self.db.transaction() as con:
            acc = self._account_row(account_id, con)
            o = self._order(con, oid)
            if o["account_id"] != acc["id"]:
                raise NotFoundError("order not found for this account")
            if o["status"] not in [s.value for s in OPEN_ORDER_STATUSES]:
                raise InputValidationError(f"order is {o['status']}; cannot cancel")
            con.execute("UPDATE orders SET status=?, reason=?, updated_at=? WHERE id=?",
                        (OrderStatus.CANCELED.value, reason, utcnow(), oid))
            self.db.audit("user", "paper.order.canceled", "order", oid, {"reason": reason}, con=con)
            return self._order(con, oid)

    def _order(self, con, oid: str) -> dict[str, Any]:
        r = con.execute("SELECT * FROM orders WHERE id=?", (oid,)).fetchone()
        if r is None:
            raise NotFoundError(f"order {oid} not found")
        d = dict(r)
        d["remaining_quantity"] = str(D(d["quantity"]) - D(d["filled_quantity"]))
        return d

    def list_orders(self, account_id: str) -> list[dict[str, Any]]:
        aid = self._account_row(account_id)["id"]
        rows = self.db.query("SELECT * FROM orders WHERE account_id=? ORDER BY created_at, id", (aid,))
        for r in rows:
            r["remaining_quantity"] = str(D(r["quantity"]) - D(r["filled_quantity"]))
        return rows

    # ------------------------------------------------------------------ executions
    def apply_execution(self, order_id: str, execution_ref: str, quantity: Any, price: Any, fee: Any,
                        session: str, con: sqlite3.Connection | None = None) -> bool:
        """Post one fill. Idempotent on ``execution_ref``: a duplicate callback changes nothing."""
        if con is None:
            with self.db.transaction() as c:
                return self.apply_execution(order_id, execution_ref, quantity, price, fee, session, c)
        q, px, f = D(quantity), D(price), D(fee)
        if q <= 0 or px <= 0 or f < 0:
            raise InputValidationError("execution quantity/price must be > 0 and fee >= 0")
        if con.execute("SELECT 1 FROM executions WHERE execution_ref=?", (execution_ref,)).fetchone():
            return False
        o = self._order(con, order_id)
        aid = o["account_id"]
        filled = D(o["filled_quantity"]) + q
        if filled > D(o["quantity"]):
            raise AccountingInvariantError(f"fill would exceed order quantity ({filled} > {o['quantity']})")
        notional = money(q * px)
        sign = Decimal(-1) if o["side"] == "BUY" else Decimal(1)
        now = utcnow()
        con.execute("INSERT INTO executions(id, order_id, execution_ref, quantity, price, fee, currency, session_date, "
                    "executed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (new_id(), order_id, execution_ref, str(q), str(px), str(f), "USD", session, now))
        con.execute("INSERT INTO cash_ledger(id, account_id, event_ref, currency, amount, event_type, reference_id, "
                    "session_date, recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (new_id(), aid, f"{execution_ref}:trade", "USD", str(sign * notional), f"TRADE_{o['side']}",
                     order_id, session, now))
        if f > 0:
            con.execute("INSERT INTO cash_ledger(id, account_id, event_ref, currency, amount, event_type, reference_id, "
                        "session_date, recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        (new_id(), aid, f"{execution_ref}:fee", "USD", str(-f), "COMMISSION", order_id, session, now))
        prow = con.execute("SELECT quantity, cost_basis FROM positions WHERE account_id=? AND symbol=?",
                           (aid, o["symbol"])).fetchone()
        pq, cb = (D(prow[0]), D(prow[1])) if prow else (ZERO, ZERO)
        if o["side"] == "BUY":
            nq, ncb = pq + q, cb + notional + f
        else:
            nq = pq - q
            ncb = cb * (nq / pq) if pq > 0 else ZERO
        if nq < 0:
            raise AccountingInvariantError(f"position for {o['symbol']} would become negative")
        con.execute("INSERT INTO positions(account_id, symbol, quantity, cost_basis, updated_at) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(account_id, symbol) DO UPDATE SET quantity=excluded.quantity, "
                    "cost_basis=excluded.cost_basis, updated_at=excluded.updated_at",
                    (aid, o["symbol"], str(nq), str(money(ncb)), now))
        status = OrderStatus.FILLED if filled == D(o["quantity"]) else OrderStatus.PARTIALLY_FILLED
        con.execute("UPDATE orders SET filled_quantity=?, status=?, updated_at=? WHERE id=?",
                    (str(filled), status.value, now, order_id))
        if self._cash(con, aid) < 0:
            raise AccountingInvariantError("fill would make cash negative")  # rolls back the transaction
        return True

    # ------------------------------------------------------------------ replay clock
    def advance_session(self, account_id: str, n: int = 1) -> list[dict[str, Any]]:
        if n < 1 or n > 2520:
            raise InputValidationError("n must be between 1 and 2520")
        return [self._advance_one(account_id) for _ in range(n)]

    def _advance_one(self, account_id: str) -> dict[str, Any]:
        with self.db.transaction() as con:
            acc = self._account_row(account_id, con)
            aid = acc["id"]
            if acc["status"] != "ACTIVE":
                raise InputValidationError(f"account is {acc['status']}")
            mkt = self._market(acc["snapshot_id"])
            i = mkt.index(pd.Timestamp(acc["current_session"]))
            if i + 1 >= len(mkt.sessions):
                raise ReplayFinished("replay reached the end of the snapshot; no further sessions")
            j = i + 1
            d = mkt.sessions[j].strftime("%Y-%m-%d")
            cfg = acc["config"]
            summary: dict[str, Any] = {"session": d, "fills": [], "dividends": [], "splits": [], "notes": []}

            # 1) corporate actions at the open for positions carried overnight
            for sym, q in self._positions(con, aid).items():
                b = mkt.bar(sym, j)
                if b is None:
                    continue
                if mkt.price_basis == PriceBasis.UNADJUSTED and b.split_ratio not in (0, 1.0):
                    r = D(b.split_ratio)
                    new_q = q * r
                    whole = new_q.to_integral_value(rounding=ROUND_FLOOR)
                    frac = new_q - whole
                    prev = mkt.last_close(sym, i) or ZERO
                    in_lieu = money(frac * prev / r)
                    con.execute("UPDATE positions SET quantity=?, updated_at=? WHERE account_id=? AND symbol=?",
                                (str(whole), utcnow(), aid, sym))
                    if in_lieu > 0:
                        con.execute("INSERT INTO cash_ledger(id, account_id, event_ref, currency, amount, event_type, "
                                    "session_date, recorded_at) VALUES (?,?,?,?,?,?,?,?)",
                                    (new_id(), aid, f"{aid}:{sym}:{d}:split_cash", "USD", str(in_lieu),
                                     "SPLIT_CASH_IN_LIEU", d, utcnow()))
                    summary["splits"].append({"symbol": sym, "ratio": float(r), "qty_after": str(whole)})
                    q = whole
                if b.dividend and b.dividend > 0 and q > 0:
                    amt = money(q * D(b.dividend))
                    con.execute("INSERT OR IGNORE INTO cash_ledger(id, account_id, event_ref, currency, amount, "
                                "event_type, session_date, recorded_at) VALUES (?,?,?,?,?,?,?,?)",
                                (new_id(), aid, f"{aid}:{sym}:{d}:dividend", "USD", str(amt), "DIVIDEND", d, utcnow()))
                    summary["dividends"].append({"symbol": sym, "amount": str(amt)})

            # 2) fill working orders at the open (sells first)
            rows = con.execute(
                "SELECT id FROM orders WHERE account_id=? AND status IN (?,?) "
                "ORDER BY CASE side WHEN 'SELL' THEN 0 ELSE 1 END, created_at, id",
                (aid, *[s.value for s in OPEN_ORDER_STATUSES])).fetchall()
            for (oid,) in rows:
                o = self._order(con, oid)
                b = mkt.bar(o["symbol"], j)
                if b is None:
                    summary["notes"].append(f"{o['symbol']}: no bar on {d}; order {oid[:8]} keeps working")
                    continue
                rem = D(o["quantity"]) - D(o["filled_quantity"])
                cap = (D(b.volume) * D(cfg["participation_rate"])).to_integral_value(rounding=ROUND_FLOOR)
                q = min(rem, cap)
                px = self._fill_price(cfg, D(b.open).quantize(PRICE_Q), o["side"])
                if o["side"] == "SELL":
                    held = self._positions(con, aid).get(o["symbol"], ZERO)
                    q = min(q, held)
                else:
                    cash = self._cash(con, aid)
                    while q > 0 and money(q * px) + self._commission(cfg, q * px) > cash:
                        afford = (cash / (px * (1 + D(cfg["commission_bps"]) / Decimal(10_000))))
                        q = min(q - 1, afford.to_integral_value(rounding=ROUND_FLOOR))
                if q <= 0:
                    reason = "INSUFFICIENT_CASH_AT_FILL" if o["side"] == "BUY" else "NO_POSITION_AT_FILL"
                    if cap <= 0:
                        summary["notes"].append(f"{o['symbol']}: zero volume capacity on {d}")
                        continue
                    con.execute("UPDATE orders SET status=?, reason=?, updated_at=? WHERE id=?",
                                (OrderStatus.CANCELED.value, reason, utcnow(), oid))
                    self._risk_event(con, aid, reason, "ORDER_CANCELED", {"order_id": oid})
                    continue
                fee = self._commission(cfg, money(q * px))
                n_prev = con.execute("SELECT COUNT(*) FROM executions WHERE order_id=?", (oid,)).fetchone()[0]
                ref = deterministic_id("exec", oid, d, n_prev)
                self.apply_execution(oid, ref, q, px, fee, d, con)
                summary["fills"].append({"order_id": oid, "symbol": o["symbol"], "side": o["side"], "quantity": str(q),
                                         "price": str(px), "fee": str(fee)})
                o2 = self._order(con, oid)
                cash_limited = o["side"] == "BUY" and q < min(rem, cap)
                if cash_limited and D(o2["remaining_quantity"]) > 0:
                    con.execute("UPDATE orders SET status=?, reason=?, updated_at=? WHERE id=?",
                                (OrderStatus.CANCELED.value, "REMAINDER_CANCELED_INSUFFICIENT_CASH", utcnow(), oid))

            # 3) advance the clock and value the portfolio at the close
            con.execute("UPDATE accounts SET current_session=? WHERE id=?", (d, aid))
            self._snapshot_portfolio(con, aid, mkt, j)
            if self._cash(con, aid) < 0:
                raise AccountingInvariantError("negative cash after session processing")
            self.db.audit("system", "paper.session.advanced", "account", aid, {"session": d,
                          "fills": len(summary["fills"])}, con=con)
            return summary

    def _snapshot_portfolio(self, con, aid: str, mkt: _Market, i: int) -> None:
        cash = self._cash(con, aid)
        pos = self._positions(con, aid)
        holdings = {}
        mv = ZERO
        for s, q in pos.items():
            px = mkt.last_close(s, i) or ZERO
            mv += q * px
            holdings[s] = {"quantity": str(q), "close": str(px)}
        nav = cash + mv
        con.execute("INSERT OR REPLACE INTO portfolio_snapshots(id, account_id, session_date, nav, cash_value, "
                    "gross_exposure, holdings) VALUES (?,?,?,?,?,?,?)",
                    (deterministic_id("ps", aid, mkt.sessions[i]), aid, mkt.sessions[i].strftime("%Y-%m-%d"),
                     str(money(nav)), str(money(cash)), str(money(mv)), dumps(holdings)))

    # ------------------------------------------------------------------ kill switch & audit
    def set_kill_switch(self, account_id: str, active: bool, reason: str = "") -> dict[str, Any]:
        with self.db.transaction() as con:
            acc = self._account_row(account_id, con)
            con.execute("UPDATE accounts SET kill_switch=? WHERE id=?", (int(active), acc["id"]))
            canceled = 0
            if active:
                cur = con.execute("UPDATE orders SET status=?, reason=?, updated_at=? WHERE account_id=? AND status IN (?,?)",
                                  (OrderStatus.CANCELED.value, "KILL_SWITCH", utcnow(), acc["id"],
                                   *[s.value for s in OPEN_ORDER_STATUSES]))
                canceled = cur.rowcount
            self._risk_event(con, acc["id"], "KILL_SWITCH", "ACTIVATED" if active else "DEACTIVATED",
                             {"reason": reason, "orders_canceled": canceled}, severity="CRITICAL" if active else "INFO")
            self.db.audit("user", "paper.kill_switch", "account", acc["id"], {"active": active, "reason": reason}, con=con)
        return {**self.account_state(account_id), "orders_canceled": canceled}

    def executions(self, account_id: str) -> list[dict[str, Any]]:
        aid = self._account_row(account_id)["id"]
        return self.db.query("SELECT e.*, o.symbol, o.side FROM executions e JOIN orders o ON o.id=e.order_id "
                             "WHERE o.account_id=? ORDER BY e.session_date, e.executed_at", (aid,))

    def ledger(self, account_id: str) -> list[dict[str, Any]]:
        aid = self._account_row(account_id)["id"]
        return self.db.query("SELECT * FROM cash_ledger WHERE account_id=? ORDER BY recorded_at, id", (aid,))

    def nav_history(self, account_id: str) -> pd.DataFrame:
        aid = self._account_row(account_id)["id"]
        rows = self.db.query("SELECT session_date, nav, cash_value, gross_exposure FROM portfolio_snapshots "
                             "WHERE account_id=? ORDER BY session_date", (aid,))
        df = pd.DataFrame(rows, columns=["session_date", "nav", "cash_value", "gross_exposure"])
        for c in ("nav", "cash_value", "gross_exposure"):
            df[c] = df[c].astype(float)
        return df

    def risk_events(self, account_id: str | None = None) -> list[dict[str, Any]]:
        if account_id:
            aid = self._account_row(account_id)["id"]
            rows = self.db.query("SELECT * FROM risk_events WHERE account_id=? ORDER BY created_at DESC", (aid,))
        else:
            rows = self.db.query("SELECT * FROM risk_events ORDER BY created_at DESC")
        for r in rows:
            r["details"] = loads(r["details"])
        return rows

    def reconcile(self, account_id: str) -> dict[str, Any]:
        """Recompute positions from executions (+splits) and cash from the ledger; compare to stored state."""
        acc = self._account_row(account_id)
        aid = acc["id"]
        with self.db.connect() as con:
            stored = self._positions(con, aid)
            ledger_cash = self._cash(con, aid)
            ex = con.execute("SELECT o.symbol, o.side, e.quantity, e.session_date FROM executions e JOIN orders o ON "
                             "o.id=e.order_id WHERE o.account_id=? ORDER BY e.session_date", (aid,)).fetchall()
            trades = sum((D(r[0]) for r in con.execute(
                "SELECT amount FROM cash_ledger WHERE account_id=? AND event_type LIKE 'TRADE_%'", (aid,)).fetchall()),
                ZERO)
            exec_cash = ZERO
            for r in con.execute("SELECT o.side, e.quantity, e.price FROM executions e JOIN orders o ON o.id=e.order_id "
                                 "WHERE o.account_id=?", (aid,)).fetchall():
                n = money(D(r[1]) * D(r[2]))
                exec_cash += -n if r[0] == "BUY" else n
        mkt = self._market(acc["snapshot_id"])
        derived: dict[str, Decimal] = {}
        split_events = []
        if mkt.price_basis == PriceBasis.UNADJUSTED:
            for (sym, d), b in mkt.bars.items():
                if b.split_ratio not in (0, 1.0):
                    split_events.append((d.strftime("%Y-%m-%d"), sym, D(b.split_ratio)))
        events = [(r[3], 1, r[0], r[1], D(r[2])) for r in ex] + [(d, 0, s, "SPLIT", r) for d, s, r in split_events
                                                                  if d <= acc["current_session"]]
        for _d, _, sym, kind, val in sorted(events):
            if kind == "SPLIT":
                if sym in derived:
                    derived[sym] = (derived[sym] * val).to_integral_value(rounding=ROUND_FLOOR)
            else:
                derived[sym] = derived.get(sym, ZERO) + (val if kind == "BUY" else -val)
        derived = {k: v for k, v in derived.items() if v != 0}
        ok = derived == stored and trades == exec_cash and ledger_cash >= 0
        return {"account_id": aid, "ok": ok, "positions_stored": {k: str(v) for k, v in stored.items()},
                "positions_from_executions": {k: str(v) for k, v in derived.items()},
                "ledger_cash": str(ledger_cash), "trade_cash_ledger": str(trades), "trade_cash_executions": str(exec_cash)}
