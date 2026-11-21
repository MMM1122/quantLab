"""Backtest orchestration + persistence (one run = one immutable artifact folder + DB row)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quantlab.backtest.market import build_panel
from quantlab.backtest.metrics import clean_for_json, performance_metrics, relative_metrics
from quantlab.context import AppContext, git_commit
from quantlab.core.errors import InputValidationError, NotFoundError
from quantlab.core.hashing import config_hash
from quantlab.core.ids import deterministic_id, new_id, validate_symbol, validate_uuid
from quantlab.core.labels import BACKTEST_LABEL, NOT_ADVICE, data_label
from quantlab.data.ingestion import require_tradable_snapshot
from quantlab.domain.models import PriceMode, RunStatus
from quantlab.storage.db import dumps, loads, utcnow
from quantlab.strategies.buy_hold import BuyAndHold
from quantlab.strategies.registry import create_strategy

ASSUMPTIONS = [
    "Signals are computed after the close of session t and executed at the open of session t+1.",
    "Fill price = next-session open adjusted adversely by slippage_bps; commission = max(min, bps x notional).",
    "Integer shares, long-only, no leverage, no shorting; buys are reduced to what cash can pay for.",
    "raw_with_dividends: trades as-reported prices and credits cash dividends on the ex-date for shares "
    "held at the prior close; adjusted_total_return: trades dividend-adjusted prices and credits no dividends.",
    "Splits rescale holdings only when snapshot prices are unadjusted (Yahoo prices are already split-adjusted).",
    "Strategies observe a causal total-return index (no future corporate actions leak into past signals).",
    "Sharpe uses daily NAV returns, 252 sessions/yr, and the configured annual risk-free rate.",
    "Benchmark = buy-and-hold of the benchmark symbol over the same window with the same costs.",
]


@dataclass
class RunOutput:
    run_id: str
    result: BacktestResult
    benchmark: BacktestResult | None
    summary: dict[str, Any]


def _bench_symbol(ctx: AppContext, available: list[str], requested: str | None) -> str | None:
    b = requested or ctx.settings.base.get("benchmark", "SPY")
    return b if b in available else None


def run_and_store(
    ctx: AppContext,
    strategy: str,
    snapshot_id: str,
    symbols: list[str],
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    benchmark: str | None = None,
    run_id: str | None = None,
) -> RunOutput:
    cfg = config or BacktestConfig()
    strat = create_strategy(strategy, params)
    sid = validate_uuid(snapshot_id, "snapshot_id")
    syms = [validate_symbol(s) for s in symbols]
    if not syms:
        raise InputValidationError("at least one symbol is required")
    snap = require_tradable_snapshot(ctx.snapshots, sid)
    if (snap.parameters or {}).get("asset_class") == "fx":
        raise InputValidationError("FX snapshots are for currency reporting only, not for backtests")
    missing = [s for s in syms if s not in snap.symbols]
    if missing:
        raise InputValidationError(f"symbols {missing} not in snapshot {snap.id} ({snap.symbols})")

    full_cfg = {"strategy": strat.describe(), "symbols": syms, "snapshot_id": snap.id,
                "dataset_hash": snap.dataset_hash, "backtest": cfg.to_dict(), "benchmark": benchmark}
    chash = config_hash(full_cfg)
    rid = run_id or new_id()
    strategy_row_id = deterministic_id("strategy", strat.id, strat.version, dumps(strat.params.to_dict()))
    now = utcnow()
    with ctx.db.transaction() as con:
        con.execute(
            "INSERT OR IGNORE INTO strategies(id, name, version, parameters, code_reference, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (strategy_row_id, strat.id, strat.version, dumps(strat.params.to_dict()),
             f"quantlab.strategies:{type(strat).__name__}", now),
        )
        con.execute(
            "INSERT INTO backtest_runs(id, strategy_id, strategy_name, snapshot_id, config, config_hash, git_commit, "
            "random_seed, is_synthetic, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rid, strategy_row_id, strat.id, snap.id, dumps(full_cfg), chash, git_commit(), cfg.seed,
             int(snap.is_synthetic), RunStatus.RUNNING.value, now),
        )
    t0 = time.perf_counter()
    try:
        bars = ctx.snapshots.load(snap.id)
        bench_sym = _bench_symbol(ctx, snap.symbols, benchmark)
        panel = build_panel(bars, snap.price_basis, cfg.price_mode, syms)
        res = run_backtest(panel, strat, cfg)
        gross_cfg = BacktestConfig(**{**cfg.to_dict(), "commission_bps": 0.0, "slippage_bps": 0.0, "min_commission": 0.0})
        gross = run_backtest(panel, strat, gross_cfg)
        bres = None
        warnings = list(res.warnings)
        if bench_sym:
            bpanel = build_panel(bars, snap.price_basis, cfg.price_mode, [bench_sym])
            bcfg = BacktestConfig(**{**cfg.to_dict(), "start": res.equity.index[0].strftime("%Y-%m-%d"),
                                     "end": res.equity.index[-1].strftime("%Y-%m-%d")})
            bres = run_backtest(bpanel, BuyAndHold(), bcfg)
        else:
            warnings.append("benchmark symbol not in snapshot; no benchmark comparison")
        net_m = performance_metrics(res.equity, res.trades, res.orders, cfg.risk_free_rate)
        gross_m = performance_metrics(gross.equity, gross.trades, gross.orders, cfg.risk_free_rate)
        rel = relative_metrics(res.equity["nav"], bres.equity["nav"]) if bres is not None else {}
        bench_m = performance_metrics(bres.equity, bres.trades, bres.orders, cfg.risk_free_rate) if bres is not None else None
        elapsed = time.perf_counter() - t0
        summary = clean_for_json({
            "run_id": rid,
            "mode": "BACKTEST",
            "label": BACKTEST_LABEL,
            "data_label": data_label(snap.is_synthetic),
            "is_synthetic": snap.is_synthetic,
            "disclaimer": NOT_ADVICE,
            "strategy": strat.describe(),
            "symbols": syms,
            "benchmark_symbol": bench_sym,
            "snapshot": {"id": snap.id, "provider": snap.provider, "dataset_hash": snap.dataset_hash,
                         "price_basis": snap.price_basis, "retrieved_at": snap.retrieved_at},
            "config": cfg.to_dict(),
            "config_hash": chash,
            "code_version": git_commit(),
            "seed": cfg.seed,
            "assumptions": ASSUMPTIONS,
            "metrics": {"net": net_m, "gross": gross_m, "benchmark": bench_m, "relative": rel},
            "warnings": warnings,
            "computation_seconds": round(elapsed, 4),
            "completed_at": utcnow(),
        })
        folder = ctx.settings.runs_dir / rid
        folder.mkdir(parents=True, exist_ok=True)
        res.equity.reset_index().to_parquet(folder / "equity.parquet", index=False)
        res.trades.to_parquet(folder / "trades.parquet", index=False)
        res.orders.to_parquet(folder / "orders.parquet", index=False)
        res.events.astype({"session": "datetime64[ns]"}).to_parquet(folder / "events.parquet", index=False)
        res.positions.reset_index().to_parquet(folder / "positions.parquet", index=False)
        gross.equity.reset_index().to_parquet(folder / "equity_gross.parquet", index=False)
        if bres is not None:
            bres.equity.reset_index().to_parquet(folder / "benchmark_equity.parquet", index=False)
        (folder / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        with ctx.db.transaction() as con:
            con.execute(
                "UPDATE backtest_runs SET status=?, metrics=?, artifact_uri=?, warnings=?, completed_at=? WHERE id=?",
                (RunStatus.COMPLETED.value, dumps(summary["metrics"]), str(folder), dumps(warnings), utcnow(), rid),
            )
            ctx.db.audit("system", "backtest.completed", "backtest_run", rid, {"strategy": strat.id}, con=con)
        return RunOutput(rid, res, bres, summary)
    except Exception as exc:
        with ctx.db.transaction() as con:
            con.execute("UPDATE backtest_runs SET status=?, failure_reason=?, completed_at=? WHERE id=?",
                        (RunStatus.FAILED.value, f"{type(exc).__name__}: {exc}", utcnow(), rid))
        raise


# ------------------------------------------------------------------ reads
def list_runs(ctx: AppContext, limit: int = 200) -> list[dict[str, Any]]:
    rows = ctx.db.query(
        "SELECT id, strategy_name, snapshot_id, status, is_synthetic, created_at, completed_at, failure_reason, "
        "config_hash, metrics, config FROM backtest_runs ORDER BY created_at DESC LIMIT ?", (limit,))
    for r in rows:
        r["metrics"] = loads(r["metrics"])
        r["config"] = loads(r["config"])
    return rows


def get_run(ctx: AppContext, run_id: str) -> dict[str, Any]:
    rid = validate_uuid(run_id, "run_id")
    r = ctx.db.query_one("SELECT * FROM backtest_runs WHERE id=?", (rid,))
    if r is None:
        raise NotFoundError(f"backtest run {rid} not found")
    out = dict(r)
    for k in ("config", "metrics", "warnings"):
        out[k] = loads(out[k])
    if out["artifact_uri"] and (Path(out["artifact_uri"]) / "result.json").exists():
        out["result"] = json.loads((Path(out["artifact_uri"]) / "result.json").read_text())
    return out


def load_artifact(ctx: AppContext, run_id: str, name: str) -> pd.DataFrame:
    run = get_run(ctx, run_id)
    if run["status"] != RunStatus.COMPLETED.value:
        raise NotFoundError(f"run {run_id} is {run['status']}; no artifacts")
    p = Path(run["artifact_uri"]) / f"{name}.parquet"
    if not p.exists():
        raise NotFoundError(f"artifact {name} not found for run {run_id}")
    return pd.read_parquet(p)


RETURN_ONLY_KEYS = ("start", "end", "n_sessions", "years", "initial_nav", "final_nav", "total_return", "cagr",
                    "annual_volatility", "sharpe", "sharpe_se", "sharpe_ci95_low", "sharpe_ci95_high", "sortino",
                    "max_drawdown", "max_drawdown_peak", "max_drawdown_trough", "max_drawdown_duration_sessions",
                    "calmar", "time_in_market", "risk_free_rate", "periods_per_year")


def metrics_in_currency(ctx: AppContext, run_id: str, currency: str = "USD") -> dict[str, Any]:
    """Return-based metrics of a stored run in USD or CAD (unhedged CAD investor view).

    Cost, turnover and dividend fields are USD cash amounts and are only reported in USD."""
    from quantlab.data.fx import fx_series, latest_fx_snapshot, to_cad

    cur = currency.upper()
    if cur not in ("USD", "CAD"):
        raise InputValidationError("currency must be USD or CAD")
    run = get_run(ctx, run_id)
    if run["status"] != RunStatus.COMPLETED.value:
        raise InputValidationError(f"run {run_id} is {run['status']}")
    res = run["result"]
    if cur == "USD":
        return {"currency": "USD", "metrics": res["metrics"], "fx": None}
    net = res["metrics"]["net"]
    fx_snap = latest_fx_snapshot(ctx, synthetic=bool(res["is_synthetic"]), start=net["start"], end=net["end"])
    rate = fx_series(ctx, fx_snap.id)
    rf = res["config"]["risk_free_rate"]
    out: dict[str, Any] = {}
    curves: dict[str, pd.Series] = {}
    info = {}
    for name, artifact in (("net", "equity"), ("gross", "equity_gross"), ("benchmark", "benchmark_equity")):
        try:
            eq = load_artifact(ctx, run_id, artifact).set_index("session")
        except NotFoundError:
            continue
        cad, info = to_cad(eq["nav"], rate)
        curves[name] = cad
        m = performance_metrics(eq.assign(nav=cad), None, None, rf)
        out[name] = {k: m[k] for k in RETURN_ONLY_KEYS}
    if "benchmark" in curves:
        out["relative"] = relative_metrics(curves["net"], curves["benchmark"])
    return clean_for_json({"currency": "CAD", "metrics": out,
                           "fx": {"snapshot_id": fx_snap.id, "dataset_hash": fx_snap.dataset_hash,
                                  "provider": fx_snap.provider, "pair": "USDCAD (CAD per USD)", **info},
                           "note": "Unhedged CAD view: NAV_CAD = NAV_USD x USDCAD. Trading and costs remain in USD.",
                           "_curves": curves})


def compare_runs(ctx: AppContext, run_ids: list[str], n_trials: int | None = None, seed: int = 0,
                 n_boot: int = 2000) -> dict[str, Any]:
    """Statistical comparison of stored runs (first run = reference). Reads artifacts only."""
    from quantlab.research.stat_tests import compare_strategies

    if len(run_ids) < 2:
        raise InputValidationError("compare needs at least two run ids")
    runs = [get_run(ctx, r) for r in run_ids]
    navs, meta = {}, {}
    for r in runs:
        if r["status"] != RunStatus.COMPLETED.value:
            raise InputValidationError(f"run {r['id']} is {r['status']}")
        key = f"{r['id'][:8]} {r['strategy_name']}"
        navs[key] = load_artifact(ctx, r["id"], "equity").set_index("session")["nav"]
        meta[key] = {"run_id": r["id"], "is_synthetic": bool(r["is_synthetic"]),
                     "symbols": r["config"]["symbols"], "snapshot_id": r["snapshot_id"]}
    out = compare_strategies(navs, n_trials=n_trials, seed=seed, n_boot=n_boot)
    warnings = []
    if len({m["is_synthetic"] for m in meta.values()}) > 1:
        warnings.append("mixing synthetic and real-data runs: the comparison is not meaningful")
    if len({m["snapshot_id"] for m in meta.values()}) > 1:
        warnings.append("runs use different snapshots")
    out.update({"runs": meta, "warnings": warnings, "label": BACKTEST_LABEL,
                "data_label": data_label(any(m["is_synthetic"] for m in meta.values()))})
    return clean_for_json(out)


__all__ = ["ASSUMPTIONS", "BacktestConfig", "PriceMode", "run_and_store", "list_runs", "get_run", "load_artifact"]
