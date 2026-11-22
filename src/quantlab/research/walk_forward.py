"""Walk-forward (out-of-sample) validation with parameter selection on training folds only.

For each fold:
  1. every parameter set in the grid is backtested on the TRAIN window;
  2. the set with the best objective (ties -> first in grid order) is chosen;
  3. the chosen set is backtested on the following TEST window (never seen during selection);
  4. all sets are also evaluated on the TEST window — only to report where the chosen set ranks
     out of sample (a selection-overfitting diagnostic), never to choose.

The stitched test-window equity curve is the out-of-sample result. Each test fold starts in cash
with the previous fold's ending NAV (documented boundary cost). Because strategies are causal
and the engine only trades inside its window, nothing after a fold's train end can influence that
fold's selection (tested by perturbing post-train data).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.engine import BacktestConfig, run_backtest
from quantlab.backtest.market import build_panel
from quantlab.backtest.metrics import clean_for_json, performance_metrics
from quantlab.context import AppContext, git_commit
from quantlab.core.errors import InputValidationError
from quantlab.core.ids import new_id, validate_symbol, validate_uuid
from quantlab.core.labels import BACKTEST_LABEL, data_label
from quantlab.data.ingestion import require_tradable_snapshot
from quantlab.domain.models import RunStatus
from quantlab.research.splits import walk_forward
from quantlab.storage.db import dumps, utcnow
from quantlab.strategies.buy_hold import BuyAndHold
from quantlab.strategies.registry import create_strategy, normalize_strategy_id

CODE = "WALK-FORWARD"
OBJECTIVES = ("sharpe", "cagr", "total_return", "calmar")
MAX_CONFIGS = 200


@dataclass
class WalkForwardConfig:
    snapshot_id: str
    strategy: str
    symbols: list[str] = field(default_factory=lambda: ["SPY"])
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    train_sessions: int = 756
    test_sessions: int = 252
    step_sessions: int | None = None
    embargo_sessions: int = 0
    expanding: bool = True
    objective: str = "sharpe"
    initial_cash: float = 10_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    benchmark: str | None = "SPY"

    def validate(self) -> list[dict[str, Any]]:
        validate_uuid(self.snapshot_id, "snapshot_id")
        self.strategy = normalize_strategy_id(self.strategy)
        self.symbols = [validate_symbol(s) for s in self.symbols]
        if self.objective not in OBJECTIVES:
            raise InputValidationError(f"objective must be one of {OBJECTIVES}")
        if self.train_sessions < 60 or self.test_sessions < 20:
            raise InputValidationError("train_sessions >= 60 and test_sessions >= 20 required")
        keys = sorted(self.param_grid)
        for k in keys:
            if not isinstance(self.param_grid[k], list) or not self.param_grid[k]:
                raise InputValidationError(f"param_grid[{k!r}] must be a non-empty list")
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(self.param_grid[k] for k in keys))] or [{}]
        if len(combos) > MAX_CONFIGS:
            raise InputValidationError(f"grid has {len(combos)} combinations; max {MAX_CONFIGS}")
        valid = []
        for c in combos:
            try:
                create_strategy(self.strategy, c)
                valid.append(c)
            except InputValidationError:
                continue
        if not valid:
            raise InputValidationError("no valid parameter combination in grid")
        return valid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _objective(m: dict[str, Any], name: str) -> float:
    v = m.get(name)
    return float(v) if v is not None and np.isfinite(v) else float("-inf")


def select_params(panel, strategy_id: str, configs: list[dict], train_start: str, train_end: str,
                  base: BacktestConfig, objective: str) -> tuple[int, list[dict[str, Any]]]:
    """Evaluate every config on the train window only; return (chosen index, table)."""
    rows = []
    for k, params in enumerate(configs):
        cfg = BacktestConfig(**{**base.to_dict(), "start": train_start, "end": train_end})
        res = run_backtest(panel, create_strategy(strategy_id, params), cfg)
        m = performance_metrics(res.equity, res.trades, res.orders)
        rows.append({"config_index": k, "params": params, "train_objective": _objective(m, objective),
                     "train_total_return": m["total_return"], "train_sharpe": m["sharpe"]})
    best = max(range(len(rows)), key=lambda k: (rows[k]["train_objective"], -k))
    return best, rows


def run_walk_forward(ctx: AppContext, cfg: WalkForwardConfig, experiment_id: str | None = None,
                     progress=None) -> dict[str, Any]:
    configs = cfg.validate()
    snap = require_tradable_snapshot(ctx.snapshots, cfg.snapshot_id)
    missing = [s for s in cfg.symbols if s not in snap.symbols]
    if missing:
        raise InputValidationError(f"symbols {missing} not in snapshot")
    eid = experiment_id or new_id()
    design = {"config": cfg.to_dict(), "configs": configs, "n_configs": len(configs),
              "snapshot": {"id": snap.id, "dataset_hash": snap.dataset_hash, "is_synthetic": snap.is_synthetic},
              "code_version": git_commit(),
              "selection_rule": f"max train-window {cfg.objective}; ties -> first config in grid order"}
    hypothesis = (f"Parameters for {cfg.strategy} selected on past data only retain their in-sample quality "
                  "out of sample.")
    ctx.db.execute("INSERT INTO experiments(id, code, hypothesis, design, snapshot_id, status, created_at) "
                   "VALUES (?,?,?,?,?,?,?)", (eid, CODE, hypothesis, dumps(design), snap.id,
                                               RunStatus.RUNNING.value, utcnow()))
    t0 = time.perf_counter()
    try:
        result = _execute(ctx, cfg, configs, snap, eid, hypothesis, design, progress)
        result["computation_seconds"] = round(time.perf_counter() - t0, 3)
        folder = ctx.settings.experiments_dir / eid
        folder.mkdir(parents=True, exist_ok=True)
        result.pop("_oos_equity").reset_index().to_parquet(folder / "oos_equity.parquet", index=False)
        bench = result.pop("_bench_equity")
        if bench is not None:
            bench.reset_index().to_parquet(folder / "benchmark_equity.parquet", index=False)
        pd.DataFrame(result.pop("_grid")).to_csv(folder / "grid_results.csv", index=False)
        pd.DataFrame(result["folds"]).to_csv(folder / "folds.csv", index=False)
        result = clean_for_json(result)
        (folder / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        ctx.db.execute("UPDATE experiments SET status=?, results=?, artifact_uri=?, completed_at=? WHERE id=?",
                       (RunStatus.COMPLETED.value, dumps(result), str(folder), utcnow(), eid))
        ctx.db.audit("system", "experiment.completed", "experiment", eid, {"code": CODE})
        return result
    except Exception as exc:
        ctx.db.execute("UPDATE experiments SET status=?, failure_reason=?, completed_at=? WHERE id=?",
                       (RunStatus.FAILED.value, f"{type(exc).__name__}: {exc}", utcnow(), eid))
        raise


def _execute(ctx, cfg: WalkForwardConfig, configs, snap, eid, hypothesis, design, progress) -> dict[str, Any]:
    bars = ctx.snapshots.load(snap.id)
    panel = build_panel(bars, snap.price_basis, symbols=cfg.symbols)
    try:
        folds = walk_forward(panel.sessions, cfg.train_sessions, cfg.test_sessions, cfg.step_sessions,
                             cfg.embargo_sessions, cfg.expanding)
    except InputValidationError as exc:
        raise InputValidationError(f"{exc} ({len(panel.sessions)} sessions available)") from exc
    base = BacktestConfig(initial_cash=cfg.initial_cash, commission_bps=cfg.commission_bps,
                          slippage_bps=cfg.slippage_bps)
    fold_rows, grid_rows, eq_parts, trade_parts, order_parts = [], [], [], [], []
    nav = cfg.initial_cash
    for f, (train, test) in enumerate(folds):
        if progress:
            progress(f / len(folds), f"fold {f + 1}/{len(folds)}")
        tr_s, tr_e = train.start.strftime("%Y-%m-%d"), train.end.strftime("%Y-%m-%d")
        te_s, te_e = test.start.strftime("%Y-%m-%d"), test.end.strftime("%Y-%m-%d")
        best, table = select_params(panel, cfg.strategy, configs, tr_s, tr_e, base, cfg.objective)
        oos_all = []
        for k, params in enumerate(configs):
            tcfg = BacktestConfig(**{**base.to_dict(), "initial_cash": nav, "start": te_s, "end": te_e})
            res = run_backtest(panel, create_strategy(cfg.strategy, params), tcfg)
            m = performance_metrics(res.equity, res.trades, res.orders)
            oos_all.append((_objective(m, cfg.objective), m, res))
            grid_rows.append({"fold": f, **table[k], "params": json.dumps(params, sort_keys=True),
                              "test_objective": _objective(m, cfg.objective), "test_total_return": m["total_return"],
                              "chosen": k == best})
        chosen_obj, chosen_m, chosen_res = oos_all[best]
        objs = np.array([o[0] for o in oos_all])
        # share of configs the chosen one beat (or tied) out of sample: 1.0 = best, 0.0 = worst
        pct_rank = float((objs <= chosen_obj).sum() - 1) / max(len(objs) - 1, 1) if len(objs) > 1 else None
        eq = chosen_res.equity.copy()
        eq["fold"] = f
        eq_parts.append(eq)
        trade_parts.append(chosen_res.trades)
        order_parts.append(chosen_res.orders)
        fold_rows.append({
            "fold": f, "train_start": tr_s, "train_end": tr_e, "test_start": te_s, "test_end": te_e,
            "train_sessions": train.n_sessions, "test_sessions": test.n_sessions,
            "chosen_params": configs[best], "train_objective": table[best]["train_objective"],
            "test_objective": chosen_obj, "test_total_return": chosen_m["total_return"],
            "test_max_drawdown": chosen_m["max_drawdown"], "oos_percentile_rank": pct_rank,
            "start_nav": nav, "end_nav": chosen_m["final_nav"],
        })
        nav = chosen_m["final_nav"]
    if progress:
        progress(1.0, "done")
    oos_eq = pd.concat(eq_parts)
    oos_eq = oos_eq[~oos_eq.index.duplicated(keep="last")]
    trades = pd.concat(trade_parts, ignore_index=True)
    orders = pd.concat(order_parts, ignore_index=True)
    oos_m = performance_metrics(oos_eq, trades, orders)
    bench_eq = None
    bench_m = None
    if cfg.benchmark and cfg.benchmark in snap.symbols:
        bpanel = build_panel(bars, snap.price_basis, symbols=[cfg.benchmark])
        bcfg = BacktestConfig(**{**base.to_dict(), "start": fold_rows[0]["test_start"], "end": fold_rows[-1]["test_end"]})
        bres = run_backtest(bpanel, BuyAndHold(), bcfg)
        bench_eq, bench_m = bres.equity, performance_metrics(bres.equity, bres.trades, bres.orders)
    is_vals = [r["train_objective"] for r in fold_rows if np.isfinite(r["train_objective"])]
    oos_vals = [r["test_objective"] for r in fold_rows if np.isfinite(r["test_objective"])]
    ranks = [r["oos_percentile_rank"] for r in fold_rows if r["oos_percentile_rank"] is not None]
    chosen_sets = {json.dumps(r["chosen_params"], sort_keys=True) for r in fold_rows}
    findings = [
        f"{len(folds)} folds, {len(configs)} parameter sets tried per fold ({len(folds) * len(configs) * 2} backtests); "
        f"selection by train-window {cfg.objective}.",
        f"Out-of-sample (stitched test windows {fold_rows[0]['test_start']} → {fold_rows[-1]['test_end']}): "
        f"total return {oos_m['total_return'] * 100:.2f}%, CAGR {oos_m['cagr'] * 100:.2f}%, Sharpe "
        f"{oos_m['sharpe']:.2f} (95% CI {oos_m['sharpe_ci95_low']:.2f} to {oos_m['sharpe_ci95_high']:.2f}), "
        f"max drawdown {oos_m['max_drawdown'] * 100:.2f}%.",
    ]
    if is_vals and oos_vals:
        findings.append(f"Mean train {cfg.objective} {np.mean(is_vals):.3f} vs mean test {cfg.objective} "
                        f"{np.mean(oos_vals):.3f} (difference {np.mean(is_vals) - np.mean(oos_vals):.3f}; a large "
                        "positive gap indicates in-sample overfitting).")
    if ranks:
        findings.append(f"Average out-of-sample percentile rank of the chosen parameters: {np.mean(ranks):.2f} "
                        "(1.00 = best in grid, 0.00 = worst, about 0.50 = no better than a random pick).")
    findings.append(f"{len(chosen_sets)} distinct parameter set(s) chosen across folds "
                    f"({'stable' if len(chosen_sets) == 1 else 'selection varies over time'}).")
    if bench_m:
        findings.append(f"{cfg.benchmark} buy & hold over the same span: total return "
                        f"{bench_m['total_return'] * 100:.2f}%, Sharpe {bench_m['sharpe']:.2f}.")
    return {
        "experiment_id": eid, "code": CODE, "hypothesis": hypothesis, "label": BACKTEST_LABEL,
        "data_label": data_label(snap.is_synthetic), "is_synthetic": snap.is_synthetic, "design": design,
        "strategy": cfg.strategy, "symbols": cfg.symbols, "objective": cfg.objective,
        "evaluation_window": [fold_rows[0]["test_start"], fold_rows[-1]["test_end"]],
        "folds": fold_rows, "oos_metrics": oos_m, "benchmark_metrics": bench_m,
        "mean_oos_percentile_rank": float(np.mean(ranks)) if ranks else None,
        "n_configs": len(configs), "findings": findings,
        "assumptions": ["Each test fold starts in cash with the previous fold's ending NAV (re-entry costs included).",
                        "Selection uses only the train window; test-window results of other configs are reported "
                        "for diagnostics only.", "Same cost model as ordinary backtests.",
                        "Sessions after the last complete test fold are not evaluated."],
        "_oos_equity": oos_eq, "_bench_equity": bench_eq, "_grid": grid_rows,
    }


__all__ = ["WalkForwardConfig", "run_walk_forward", "select_params"]
