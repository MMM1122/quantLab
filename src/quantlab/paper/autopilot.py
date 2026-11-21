"""Strategy autopilot for paper accounts (historical replay only).

After the close of each replay session the autopilot asks a deterministic strategy for target
weights — computed from data up to and including that session only — and submits the difference
as MARKET/DAY paper orders through ``PaperBroker.submit_order``. Every order therefore passes the
same pre-trade risk rules as manual orders (kill switch, stale data, no shorting, cash, 25% cap).

* Targets above the account's ``max_position_weight`` are capped (with a 0.5% buffer so price drift
  between decision and check does not trigger rejections); capping is reported.
* Like the backtest engine, orders are generated only when a symbol's target changes, or when an
  earlier attempt was rejected/expired (retry), e.g. a buy that needs cash from a pending sell.
* Unlike the engine, a holding that has drifted above the account cap through price moves is trimmed
  back at the next decision, so the 25% limit holds continuously (up to one session of drift).
* ``client_order_id = auto:<session>:<symbol>`` makes a repeated step idempotent.
* The AI assistant has no access to this module.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.market import build_panel
from quantlab.context import AppContext
from quantlab.core.errors import InputValidationError, NotFoundError
from quantlab.core.ids import validate_symbol
from quantlab.paper.broker import PaperBroker, ReplayFinished
from quantlab.storage.db import dumps, loads, utcnow
from quantlab.strategies.registry import create_strategy, normalize_strategy_id

CAP_BUFFER = 0.995


class StrategyAutopilot:
    def __init__(self, ctx: AppContext, broker: PaperBroker | None = None):
        self.ctx = ctx
        self.broker = broker or PaperBroker(ctx)
        self._signals: dict[tuple, pd.DataFrame] = {}

    # ------------------------------------------------------------------ config
    def attach(self, account_id: str, strategy: str, symbols: list[str],
               params: dict[str, Any] | None = None) -> dict[str, Any]:
        acc = self.broker.account_state(account_id)
        sid = normalize_strategy_id(strategy)
        create_strategy(sid, params)  # validate parameters now
        syms = sorted({validate_symbol(s) for s in symbols})
        snap_syms = self.ctx.snapshots.get(acc["snapshot_id"]).symbols
        missing = [s for s in syms if s not in snap_syms]
        if missing:
            raise InputValidationError(f"symbols {missing} not in the account's replay snapshot")
        now = utcnow()
        self.ctx.db.execute(
            "INSERT INTO autopilots(account_id, strategy, params, symbols, enabled, state, created_at, updated_at) "
            "VALUES (?,?,?,?,1,'{}',?,?) ON CONFLICT(account_id) DO UPDATE SET strategy=excluded.strategy, "
            "params=excluded.params, symbols=excluded.symbols, enabled=1, state='{}', updated_at=excluded.updated_at",
            (acc["id"], sid, dumps(params or {}), dumps(syms), now, now),
        )
        self.ctx.db.audit("user", "paper.autopilot.attached", "account", acc["id"],
                          {"strategy": sid, "params": params or {}, "symbols": syms})
        return self.get(acc["id"])

    def get(self, account_id: str) -> dict[str, Any]:
        r = self.ctx.db.query_one("SELECT * FROM autopilots WHERE account_id=?", (account_id,))
        if r is None:
            raise NotFoundError(f"no autopilot attached to account {account_id}")
        for k in ("params", "symbols", "state"):
            r[k] = loads(r[k])
        r["enabled"] = bool(r["enabled"])
        return r

    def set_enabled(self, account_id: str, enabled: bool) -> dict[str, Any]:
        self.get(account_id)
        self.ctx.db.execute("UPDATE autopilots SET enabled=?, updated_at=? WHERE account_id=?",
                            (int(enabled), utcnow(), account_id))
        return self.get(account_id)

    # ------------------------------------------------------------------ signals
    def _targets_at(self, ap: dict[str, Any], snapshot_id: str, session: pd.Timestamp) -> pd.Series:
        key = (snapshot_id, ap["strategy"], dumps(ap["params"]), tuple(ap["symbols"]))
        if key not in self._signals:
            snap = self.ctx.snapshots.get(snapshot_id)
            panel = build_panel(self.ctx.snapshots.load(snapshot_id), snap.price_basis, symbols=ap["symbols"])
            self._signals[key] = panel.signal_price
        sig = self._signals[key]
        i = sig.index.get_loc(session)
        # only rows <= decision session are passed in (belt and braces on top of strategy causality)
        return create_strategy(ap["strategy"], ap["params"]).target_weights(sig.iloc[: i + 1]).iloc[-1]

    # ------------------------------------------------------------------ one decision
    def step(self, account_id: str) -> dict[str, Any]:
        ap = self.get(account_id)
        if not ap["enabled"]:
            return {"skipped": "autopilot disabled", "orders": []}
        acc = self.broker.account_state(account_id)
        session = acc["replay_session"]
        state = ap["state"] or {}
        if state.get("last_step_session") == session:
            return {"session": session, "skipped": "already stepped this session", "orders": []}
        cap = float(acc["risk_limits"]["max_position_weight"]) * CAP_BUFFER
        raw = self._targets_at(ap, acc["snapshot_id"], pd.Timestamp(session))
        last = state.get("last_targets", {})
        retry = set(state.get("retry", []))
        # orders from the previous decision that ended unfilled (DAY expiry, cash, volume cap) -> retry
        prev = {o["id"]: o for o in self.broker.list_orders(account_id) if o["id"] in set(state.get("last_order_ids", []))}
        for o in prev.values():
            if o["status"] == "CANCELED" and Decimal(o["remaining_quantity"]) > 0:
                retry.add(o["symbol"])
        capped = []
        changed = []
        for sym in ap["symbols"]:
            w = raw.get(sym)
            if w is None or not np.isfinite(w):
                continue  # no opinion (warm-up / missing data): keep position
            w = float(w)
            if w > cap:
                capped.append(sym)
                w = math.floor(cap * 1e6) / 1e6
            if sym not in last or abs(last[sym] - w) > 1e-12 or sym in retry:
                changed.append((sym, w))
            last[sym] = w
        # drift control: a holding whose weight has risen above the account cap since the last trade is
        # trimmed back to its (capped) target even if the signal did not change
        hard_cap = float(acc["risk_limits"]["max_position_weight"])
        trimmed = []
        changed_syms = {c[0] for c in changed}
        for h in acc["holdings"]:
            if h["weight"] is not None and h["weight"] > hard_cap and h["symbol"] in last \
                    and h["symbol"] not in changed_syms:
                changed.append((h["symbol"], min(last[h["symbol"]], math.floor(cap * 1e6) / 1e6)))
                trimmed.append(h["symbol"])
        nav = Decimal(acc["nav"])
        held = {h["symbol"]: Decimal(h["quantity"]) for h in acc["holdings"]}
        mkt = self.broker._market(acc["snapshot_id"])
        i = mkt.index(pd.Timestamp(session))
        open_orders = [o for o in self.broker.list_orders(account_id) if o["status"] in ("SUBMITTED", "PARTIALLY_FILLED")]
        orders, new_retry = [], set()
        intents = []
        for sym, w in changed:
            px = mkt.last_close(sym, i)
            if px is None or px <= 0:
                new_retry.add(sym)
                continue
            desired = (Decimal(str(w)) * nav / px).to_integral_value(rounding="ROUND_FLOOR")
            pending = sum((Decimal(o["remaining_quantity"]) * (1 if o["side"] == "BUY" else -1)
                           for o in open_orders if o["symbol"] == sym), Decimal(0))
            delta = desired - held.get(sym, Decimal(0)) - pending
            if delta != 0:
                intents.append((sym, "BUY" if delta > 0 else "SELL", abs(delta)))
        intents.sort(key=lambda x: 0 if x[1] == "SELL" else 1)  # free cash first
        for sym, side, qty in intents:
            o = self.broker.submit_order(account_id, sym, side, qty, client_order_id=f"auto:{session}:{sym}",
                                         time_in_force="DAY", actor=f"strategy:{ap['strategy']}")
            orders.append({k: o[k] for k in ("id", "symbol", "side", "quantity", "status", "reason")})
            if o["status"] == "REJECTED":
                new_retry.add(sym)
        state = {"last_targets": last, "retry": sorted(new_retry), "last_step_session": session,
                 "trim_count": int(state.get("trim_count", 0)) + len(trimmed),
                 "cap_count": int(state.get("cap_count", 0)) + len(capped),
                 "last_order_ids": [o["id"] for o in orders if o["status"] != "REJECTED"]}
        self.ctx.db.execute("UPDATE autopilots SET state=?, updated_at=? WHERE account_id=?",
                            (dumps(state), utcnow(), account_id))
        return {"session": session, "targets": {k: v for k, v in last.items()}, "capped": capped, "trimmed": trimmed,
                "orders": orders, "retry": sorted(new_retry)}

    # ------------------------------------------------------------------ loop
    def run(self, account_id: str, sessions: int) -> dict[str, Any]:
        """Decide at the close, advance the replay clock (fills at next open), repeat."""
        if sessions < 1 or sessions > 5000:
            raise InputValidationError("sessions must be between 1 and 5000")
        steps, n_orders, n_rejected, finished = 0, 0, 0, False
        for _ in range(sessions):
            out = self.step(account_id)
            n_orders += len(out["orders"])
            n_rejected += sum(1 for o in out["orders"] if o["status"] == "REJECTED")
            try:
                self.broker.advance_session(account_id)
            except ReplayFinished:
                finished = True
                break
            steps += 1
        return {"sessions_advanced": steps, "orders_submitted": n_orders, "orders_rejected": n_rejected,
                "replay_finished": finished, "account": self.broker.account_state(account_id)}
