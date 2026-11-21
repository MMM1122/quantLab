"""`quantlab paper ...` sub-commands (local paper broker on historical replay)."""

from __future__ import annotations

import sys

from quantlab.cli.main import _banner, _ctx, _print, _resolve_snapshot
from quantlab.core.labels import REPLAY_LABEL


def _broker(args):
    from quantlab.paper.broker import PaperBroker

    return PaperBroker(_ctx(args))


def _show(state):
    print(f"*** {REPLAY_LABEL} ***", file=sys.stderr)
    synth = state.get("is_synthetic") or (state.get("account") or {}).get("is_synthetic") or any(
        a.get("is_synthetic") for a in state.get("accounts", []))
    _banner(bool(synth))
    _print(state)


def cmd_create(args) -> int:
    b = _broker(args)
    sid = _resolve_snapshot(b.ctx, args, ["SPY"]) if not args.snapshot else args.snapshot
    _show(b.create_account(args.name, sid, args.cash, args.start))
    return 0


def cmd_status(args) -> int:
    b = _broker(args)
    _show(b.account_state(args.account) if args.account else {"accounts": b.list_accounts()})
    return 0


def cmd_order(args) -> int:
    b = _broker(args)
    o = b.submit_order(args.account, args.symbol, args.side, args.quantity, args.client_order_id,
                       order_type=args.type, actor="cli", limit_price=args.limit_price, time_in_force=args.tif)
    _print(o)
    return 0 if o["status"] != "REJECTED" else 3


def cmd_cancel(args) -> int:
    _print(_broker(args).cancel_order(args.account, args.order))
    return 0


def cmd_advance(args) -> int:
    b = _broker(args)
    out = b.advance_session(args.account, args.sessions)
    _show({"sessions": out, "account": b.account_state(args.account)})
    return 0


def cmd_orders(args) -> int:
    _print(_broker(args).list_orders(args.account))
    return 0


def cmd_kill(args) -> int:
    _show(_broker(args).set_kill_switch(args.account, args.state == "on", args.reason))
    return 0


def cmd_reconcile(args) -> int:
    r = _broker(args).reconcile(args.account)
    _print(r)
    return 0 if r["ok"] else 4


def register(sub) -> None:
    p = sub.add_parser("paper", help="paper trading on historical replay").add_subparsers(dest="cmd", required=True)
    c = p.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--snapshot", default=None)
    c.add_argument("--fixture", choices=["synthetic"], default=None)
    c.add_argument("--cash", default="10000")
    c.add_argument("--start", default=None, help="replay start session (YYYY-MM-DD)")
    c.set_defaults(func=cmd_create)
    s = p.add_parser("status")
    s.add_argument("--account", default=None)
    s.set_defaults(func=cmd_status)
    o = p.add_parser("order")
    o.add_argument("--account", required=True)
    o.add_argument("--symbol", required=True)
    o.add_argument("--side", choices=["BUY", "SELL", "buy", "sell"], required=True)
    o.add_argument("--quantity", required=True)
    o.add_argument("--client-order-id", default=None)
    o.add_argument("--type", choices=["MARKET", "LIMIT"], default="MARKET")
    o.add_argument("--limit-price", default=None)
    o.add_argument("--tif", choices=["DAY", "GTC"], default="GTC", help="time in force")
    o.set_defaults(func=cmd_order)
    x = p.add_parser("cancel")
    x.add_argument("--account", required=True)
    x.add_argument("--order", required=True)
    x.set_defaults(func=cmd_cancel)
    a = p.add_parser("advance")
    a.add_argument("--account", required=True)
    a.add_argument("--sessions", type=int, default=1)
    a.set_defaults(func=cmd_advance)
    ol = p.add_parser("orders")
    ol.add_argument("--account", required=True)
    ol.set_defaults(func=cmd_orders)
    k = p.add_parser("kill-switch")
    k.add_argument("--account", required=True)
    k.add_argument("state", choices=["on", "off"])
    k.add_argument("--reason", default="")
    k.set_defaults(func=cmd_kill)
    r = p.add_parser("reconcile")
    r.add_argument("--account", required=True)
    r.set_defaults(func=cmd_reconcile)
