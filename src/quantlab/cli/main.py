"""`quantlab` command-line interface.

Examples
  quantlab data ingest --symbols SPY QQQ IWM GLD TLT --start 2015-01-01
  quantlab data ingest --symbols SPY QQQ IWM GLD TLT --start 2015-01-01 --fixture synthetic
  quantlab data check --snapshot <id>
  quantlab backtest run --strategy ma-crossover --symbol SPY
  quantlab experiment run --id dq-001
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from quantlab.core.errors import QuantLabError
from quantlab.core.labels import SYNTHETIC_LABEL, data_label


def _ctx(args):
    from quantlab.context import build_context

    return build_context(args.data_dir)


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _banner(is_synthetic: bool) -> None:
    if is_synthetic:
        print(f"*** {SYNTHETIC_LABEL} ***", file=sys.stderr)


# ------------------------------------------------------------------ data
def cmd_data_ingest(args) -> int:
    from quantlab.data.ingestion import ingest
    from quantlab.data.providers import get_provider

    ctx = _ctx(args)
    provider = "synthetic" if args.fixture == "synthetic" else args.provider
    res = ingest(args.symbols, args.start, args.end, get_provider(provider, args.seed), ctx.snapshots, ctx.db)
    _banner(res.snapshot.is_synthetic)
    _print({
        "snapshot_id": res.snapshot.id,
        "created": res.created,
        "provider": res.snapshot.provider,
        "data_label": data_label(res.snapshot.is_synthetic),
        "dataset_hash": res.snapshot.dataset_hash,
        "symbols": res.snapshot.symbols,
        "range": [res.snapshot.start, res.snapshot.end],
        "rows": res.snapshot.n_rows,
        "quality": {"status": res.quality.status, "counts": res.quality.counts},
        "warnings": res.warnings,
    })
    return 0


def cmd_data_check(args) -> int:
    from quantlab.data.validation import run_quality_checks

    ctx = _ctx(args)
    snap = ctx.snapshots.get(args.snapshot)
    bars = ctx.snapshots.load(snap.id)  # verifies hash
    rep = run_quality_checks(bars, snap.price_basis, snap.parameters.get("start"), snap.parameters.get("end"),
                             expected_source=bars["source"].iloc[0])
    if snap.parent_id is None:
        ctx.snapshots.record_quality(snap.id, rep)
    _banner(snap.is_synthetic)
    _print({"snapshot_id": snap.id, "hash_verified": True, **rep.to_dict()})
    return 1 if rep.blocking else 0


def cmd_data_refresh(args) -> int:
    from quantlab.data.providers import YFinanceProvider
    from quantlab.data.refresh import refresh_snapshot

    ctx = _ctx(args)
    res = refresh_snapshot(ctx, args.snapshot, YFinanceProvider(), args.end)
    _print({"base_snapshot_id": res.base_snapshot_id, "new_snapshot_id": res.new_snapshot_id,
            "created": res.ingest.created, "quality": res.ingest.quality.status, "status": res.diff["status"],
            "totals": res.diff["totals"], "range": [res.ingest.snapshot.start, res.ingest.snapshot.end]})
    return 5 if res.diff["status"] == "REVISIONS_DETECTED" else 0


def cmd_data_list(args) -> int:
    ctx = _ctx(args)
    _print([{"id": s.id, "provider": s.provider, "synthetic": s.is_synthetic, "symbols": s.symbols,
             "range": [s.start, s.end], "rows": s.n_rows, "parent_id": s.parent_id,
             "quality": ctx.snapshots.quality_status(s.id)} for s in ctx.snapshots.list()])
    return 0


def _resolve_snapshot(ctx, args, symbols: list[str]) -> str:
    if args.snapshot:
        return args.snapshot
    if getattr(args, "fixture", None) == "synthetic":
        from quantlab.data.ingestion import ingest
        from quantlab.data.providers import SyntheticProvider

        universe = sorted(set(ctx.settings.base["universe"]) | set(symbols))
        res = ingest(universe, "2015-01-01", "2025-12-31", SyntheticProvider(seed=7), ctx.snapshots, ctx.db)
        return res.snapshot.id
    for s in ctx.snapshots.list(include_derived=False):
        if set(symbols) <= set(s.symbols) and ctx.snapshots.quality_status(s.id) != "FAILED":
            print(f"using most recent snapshot {s.id} ({data_label(s.is_synthetic)})", file=sys.stderr)
            return s.id
    raise SystemExit("no snapshot contains these symbols; run `quantlab data ingest` or pass --fixture synthetic")


# ------------------------------------------------------------------ backtest
def cmd_backtest_run(args) -> int:
    from quantlab.backtest.engine import BacktestConfig
    from quantlab.backtest.service import run_and_store

    ctx = _ctx(args)
    symbols = args.symbols or [args.symbol]
    sid = _resolve_snapshot(ctx, args, symbols)
    cfg = BacktestConfig(initial_cash=args.cash, commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
                         min_commission=args.min_commission, price_mode=args.price_mode,
                         risk_free_rate=args.risk_free_rate, start=args.start, end=args.end, seed=args.seed)
    out = run_and_store(ctx, args.strategy, sid, symbols, _parse_params(args.param), cfg, args.benchmark)
    s = out.summary
    _banner(s["is_synthetic"])
    keys = ["total_return", "cagr", "annual_volatility", "sharpe", "max_drawdown", "turnover_annual",
            "time_in_market", "n_fills", "total_commission", "final_nav"]
    _print({
        "run_id": out.run_id, "label": s["label"], "data_label": s["data_label"], "strategy": s["strategy"]["id"],
        "symbols": s["symbols"], "snapshot_id": sid, "window": [s["metrics"]["net"]["start"], s["metrics"]["net"]["end"]],
        "net": {k: s["metrics"]["net"].get(k) for k in keys},
        "gross": {k: s["metrics"]["gross"].get(k) for k in keys},
        "benchmark": {k: (s["metrics"]["benchmark"] or {}).get(k) for k in keys},
        "warnings": s["warnings"],
    })
    return 0


def cmd_backtest_list(args) -> int:
    from quantlab.backtest.service import list_runs

    ctx = _ctx(args)
    _print([{"id": r["id"], "strategy": r["strategy_name"], "status": r["status"], "synthetic": bool(r["is_synthetic"]),
             "created_at": r["created_at"],
             "cagr": ((r["metrics"] or {}).get("net") or {}).get("cagr"),
             "sharpe": ((r["metrics"] or {}).get("net") or {}).get("sharpe")} for r in list_runs(ctx)])
    return 0


def cmd_backtest_compare(args) -> int:
    from quantlab.backtest.service import get_run

    ctx = _ctx(args)
    rows = []
    for rid in args.runs:
        r = get_run(ctx, rid)
        n = (r.get("metrics") or {}).get("net") or {}
        rows.append({"run_id": rid, "strategy": r["strategy_name"], "synthetic": bool(r["is_synthetic"]),
                     **{k: n.get(k) for k in ("total_return", "cagr", "sharpe", "sharpe_ci95_low", "sharpe_ci95_high",
                                               "max_drawdown", "turnover_annual", "n_fills")}})
    if len({r["synthetic"] for r in rows}) > 1:
        print("WARNING: comparing runs on synthetic and real data", file=sys.stderr)
    _print(rows)
    return 0


def cmd_report_export(args) -> int:
    from quantlab.reports.export import export_backtest

    ctx = _ctx(args)
    paths = export_backtest(ctx, args.run, args.format, args.out)
    _print({"files": [str(p) for p in paths]})
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantlab", description="QuantLab research/backtest/paper CLI (no live trading)")
    p.add_argument("--data-dir", default=None, help="override QUANTLAB_DATA_DIR")
    sub = p.add_subparsers(dest="group", required=True)

    data = sub.add_parser("data", help="market data").add_subparsers(dest="cmd", required=True)
    di = data.add_parser("ingest", help="download/generate bars and create a snapshot")
    di.add_argument("--symbols", nargs="+", required=True)
    di.add_argument("--start", required=True)
    di.add_argument("--end", default=None)
    di.add_argument("--provider", choices=["yfinance", "synthetic"], default="yfinance")
    di.add_argument("--fixture", choices=["synthetic"], default=None, help="offline seeded synthetic data")
    di.add_argument("--seed", type=int, default=7)
    di.set_defaults(func=cmd_data_ingest)
    dc = data.add_parser("check", help="re-verify hash and run quality checks")
    dc.add_argument("--snapshot", required=True)
    dc.set_defaults(func=cmd_data_check)
    data.add_parser("list", help="list snapshots").set_defaults(func=cmd_data_list)
    dr = data.add_parser("refresh", help="re-download a real snapshot into a new one and report provider revisions")
    dr.add_argument("--snapshot", default="latest", help="snapshot id or 'latest' (most recent real snapshot)")
    dr.add_argument("--end", default=None, help="default: last completed XNYS session")
    dr.set_defaults(func=cmd_data_refresh)

    bt = sub.add_parser("backtest", help="backtests").add_subparsers(dest="cmd", required=True)
    br = bt.add_parser("run")
    br.add_argument("--strategy", required=True, help="buy-hold | sma-crossover (ma-crossover) | rsi-mean-reversion | momentum")
    br.add_argument("--symbol", default="SPY")
    br.add_argument("--symbols", nargs="+", default=None)
    br.add_argument("--snapshot", default=None)
    br.add_argument("--fixture", choices=["synthetic"], default=None)
    br.add_argument("--start", default=None)
    br.add_argument("--end", default=None)
    br.add_argument("--cash", type=float, default=10_000.0)
    br.add_argument("--commission-bps", type=float, default=1.0)
    br.add_argument("--slippage-bps", type=float, default=2.0)
    br.add_argument("--min-commission", type=float, default=0.0)
    br.add_argument("--price-mode", choices=["raw_with_dividends", "adjusted_total_return"], default="raw_with_dividends")
    br.add_argument("--risk-free-rate", type=float, default=0.0)
    br.add_argument("--benchmark", default=None)
    br.add_argument("--seed", type=int, default=42)
    br.add_argument("--param", action="append", help="strategy parameter key=value (repeatable)")
    br.set_defaults(func=cmd_backtest_run)
    bt.add_parser("list").set_defaults(func=cmd_backtest_list)
    bc = bt.add_parser("compare")
    bc.add_argument("--runs", nargs="+", required=True)
    bc.set_defaults(func=cmd_backtest_compare)

    rp = sub.add_parser("report", help="exports").add_subparsers(dest="cmd", required=True)
    re_ = rp.add_parser("export")
    re_.add_argument("--run", required=True)
    re_.add_argument("--format", choices=["json", "csv", "md", "html"], default="md")
    re_.add_argument("--out", default=None)
    re_.set_defaults(func=cmd_report_export)

    _register_extensions(sub)
    return p


def _register_extensions(sub) -> None:
    """Later phases (experiments, paper, ai) register their sub-commands here."""
    for mod in ("quantlab.cli.experiments", "quantlab.cli.paper", "quantlab.cli.ai"):
        try:
            m = __import__(mod, fromlist=["register"])
        except ModuleNotFoundError as exc:
            if exc.name != mod:
                raise
            continue
        m.register(sub)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except QuantLabError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
