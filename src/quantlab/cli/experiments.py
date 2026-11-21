"""`quantlab experiment ...` sub-commands."""

from __future__ import annotations

import sys

from quantlab.cli.main import _banner, _ctx, _print, _resolve_snapshot


def cmd_experiment_run(args) -> int:
    from quantlab.research.dq_001 import DQ001Config, run_dq001
    from quantlab.research.perturbations import DEFAULT_SCENARIOS

    if args.id.lower() == "walk-forward":
        return _run_walk_forward(args)
    if args.id.lower() != "dq-001":
        raise SystemExit(f"unknown experiment {args.id!r}; available: dq-001, walk-forward")
    ctx = _ctx(args)
    sid = _resolve_snapshot(ctx, args, [args.symbol])
    cfg = DQ001Config(
        snapshot_id=sid, symbol=args.symbol, strategies=args.strategies, seeds=args.seeds,
        eval_start=args.start, eval_end=args.end,
        scenarios=[s for s in DEFAULT_SCENARIOS if args.scenarios is None or s.name in args.scenarios or s.is_clean],
    )

    def progress(frac, msg):
        print(f"[{frac:5.0%}] {msg}", file=sys.stderr)

    res = run_dq001(ctx, cfg, progress=progress)
    _banner(res["is_synthetic"])
    _print({"experiment_id": res["experiment_id"], "data_label": res["data_label"], "n_runs": res["n_runs"],
            "evaluation_window": res["evaluation_window"],
            "source_snapshot_verified_unchanged": res["source_snapshot_verified_unchanged"],
            "findings": res["findings"]})
    return 0


def _run_walk_forward(args) -> int:
    import json

    from quantlab.research.walk_forward import WalkForwardConfig, run_walk_forward

    ctx = _ctx(args)
    symbols = args.symbols or [args.symbol]
    sid = _resolve_snapshot(ctx, args, symbols)
    try:
        grid = json.loads(args.grid) if args.grid else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--grid must be JSON, e.g. '{{\"fast\": [20, 50], \"slow\": [100, 200]}}': {exc}") from exc
    cfg = WalkForwardConfig(snapshot_id=sid, strategy=args.strategy or "sma-crossover", symbols=symbols,
                            param_grid=grid, train_sessions=args.train, test_sessions=args.test,
                            embargo_sessions=args.embargo, expanding=not args.rolling, objective=args.objective)
    res = run_walk_forward(ctx, cfg, progress=lambda f, m: print(f"[{f:5.0%}] {m}", file=sys.stderr))
    _banner(res["is_synthetic"])
    _print({"experiment_id": res["experiment_id"], "data_label": res["data_label"],
            "evaluation_window": res["evaluation_window"], "folds": len(res["folds"]),
            "chosen_params": [f["chosen_params"] for f in res["folds"]], "findings": res["findings"]})
    return 0


def cmd_experiment_list(args) -> int:
    from quantlab.research.dq_001 import list_experiments

    _print(list_experiments(_ctx(args)))
    return 0


def cmd_experiment_export(args) -> int:
    from quantlab.reports.export import export_experiment

    paths = export_experiment(_ctx(args), args.experiment, args.format, args.out)
    _print({"files": [str(p) for p in paths]})
    return 0


def register(sub) -> None:
    ex = sub.add_parser("experiment", help="research experiments").add_subparsers(dest="cmd", required=True)
    r = ex.add_parser("run", help="run an experiment (dq-001 | walk-forward)")
    r.add_argument("--id", required=True)
    r.add_argument("--snapshot", default=None)
    r.add_argument("--fixture", choices=["synthetic"], default=None)
    r.add_argument("--symbol", default="SPY")
    r.add_argument("--strategies", nargs="+", default=["sma-crossover", "rsi-mean-reversion"])
    r.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 41, 53])
    r.add_argument("--scenarios", nargs="+", default=None, help="subset of scenario names (clean always included)")
    r.add_argument("--symbols", nargs="+", default=None, help="walk-forward: symbols to trade")
    r.add_argument("--strategy", default=None, help="walk-forward: strategy id")
    r.add_argument("--grid", default=None, help="walk-forward: JSON parameter grid")
    r.add_argument("--train", type=int, default=756, help="walk-forward: train sessions")
    r.add_argument("--test", type=int, default=252, help="walk-forward: test sessions")
    r.add_argument("--embargo", type=int, default=0)
    r.add_argument("--rolling", action="store_true", help="walk-forward: rolling instead of expanding train window")
    r.add_argument("--objective", default="sharpe", choices=["sharpe", "cagr", "total_return", "calmar"])
    r.add_argument("--start", default=None, help="evaluation window start")
    r.add_argument("--end", default=None, help="evaluation window end")
    r.set_defaults(func=cmd_experiment_run)
    ex.add_parser("list").set_defaults(func=cmd_experiment_list)
    e = ex.add_parser("export")
    e.add_argument("--experiment", required=True)
    e.add_argument("--format", choices=["json", "csv", "md", "html"], default="md")
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_experiment_export)
