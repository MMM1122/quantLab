"""`quantlab experiment ...` sub-commands."""

from __future__ import annotations

import sys

from quantlab.cli.main import _banner, _ctx, _print, _resolve_snapshot


def cmd_experiment_run(args) -> int:
    from quantlab.research.dq_001 import DQ001Config, run_dq001
    from quantlab.research.perturbations import DEFAULT_SCENARIOS

    if args.id.lower() != "dq-001":
        raise SystemExit(f"unknown experiment {args.id!r}; available: dq-001")
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
    r = ex.add_parser("run", help="run an experiment (dq-001)")
    r.add_argument("--id", required=True)
    r.add_argument("--snapshot", default=None)
    r.add_argument("--fixture", choices=["synthetic"], default=None)
    r.add_argument("--symbol", default="SPY")
    r.add_argument("--strategies", nargs="+", default=["sma-crossover", "rsi-mean-reversion"])
    r.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 41, 53])
    r.add_argument("--scenarios", nargs="+", default=None, help="subset of scenario names (clean always included)")
    r.add_argument("--start", default=None, help="evaluation window start")
    r.add_argument("--end", default=None, help="evaluation window end")
    r.set_defaults(func=cmd_experiment_run)
    ex.add_parser("list").set_defaults(func=cmd_experiment_list)
    e = ex.add_parser("export")
    e.add_argument("--experiment", required=True)
    e.add_argument("--format", choices=["json", "csv", "md", "html"], default="md")
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_experiment_export)
