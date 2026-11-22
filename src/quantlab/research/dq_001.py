"""Experiment DQ-001: how much do missing, delayed and corrupted daily observations hurt
rule-based ETF strategies, and how much do treatment policies help?

Design (all of it is recorded in the experiment row and result.json):
* Source: one validated snapshot, one symbol. The source is never modified; each
  (scenario, seed) perturbation is persisted as a derived snapshot with parent_id + event log.
* Decisions use the *observed* (perturbed, policy-treated) data; fills and valuation use the
  pristine data (the market traded even if our feed was wrong). This isolates the cost of bad
  data from the market path itself.
* Fixed evaluation window, identical costs and capital for every run.
* Baseline: the ordinary backtest on clean data (no policy). Every run reports deltas vs. that
  baseline for the same strategy, plus signal-fidelity measures.
* Multiple fixed seeds per scenario -> mean and Student-t 95% CI across seeds; paired policy
  differences per seed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.backtest.engine import BacktestConfig, run_backtest
from quantlab.backtest.market import build_panel
from quantlab.backtest.metrics import clean_for_json, performance_metrics
from quantlab.context import AppContext, git_commit
from quantlab.core.errors import DataIntegrityError, InputValidationError, NotFoundError
from quantlab.core.hashing import dataframe_sha256
from quantlab.core.ids import deterministic_id, new_id, validate_symbol, validate_uuid
from quantlab.core.labels import BACKTEST_LABEL, RESEARCH_ONLY_LABEL, data_label
from quantlab.data.adjustments import total_return_index
from quantlab.data.ingestion import require_tradable_snapshot
from quantlab.data.providers import canonicalize_bars
from quantlab.data.validation import run_quality_checks
from quantlab.domain.models import PriceMode, RunStatus
from quantlab.research.perturbations import DEFAULT_SCENARIOS, Scenario, perturb
from quantlab.research.policies import POLICIES, apply_policy, observed_from_events
from quantlab.research.statistics import block_bootstrap_ci, mean_ci
from quantlab.storage.db import dumps, loads, utcnow
from quantlab.strategies.registry import create_strategy, normalize_strategy_id

CODE = "DQ-001"
HYPOTHESIS = (
    "Missing, delayed and corrupted daily observations degrade the net performance and signal fidelity of "
    "rule-based ETF strategies; timestamp-aware strict/bounded treatment policies limit the damage relative to "
    "a naive forward-fill policy that trusts every delivered value."
)
MISSING_DATA_RULES = [
    "A bar is 'fresh' only if its as-of session equals the decision session; stale (delayed) bars are rejected "
    "by strict_skip and bounded_ffill.",
    "Observations jumping more than 10% from the last accepted value are held back until a second observation "
    "confirms the new level (within 2%).",
    "strict_skip: no imputation; the strategy expresses no opinion and the engine keeps the current position.",
    "bounded_ffill: carry the last accepted historical value forward for at most 3 sessions.",
    "naive_ffill (research-only counterfactual): trust every delivered value and forward-fill without limit.",
    "No policy ever uses future observations (no backward fill, no interpolation).",
    "Corporate actions (dividends/splits) come from the pristine feed in all scenarios.",
    "Orders always fill and are valued on pristine prices; only decisions see perturbed data.",
]
KEY_METRICS = ["delta_total_return", "delta_sharpe", "delta_max_drawdown", "total_return", "sharpe", "max_drawdown",
               "n_trades", "signal_agreement", "missed_signals", "spurious_signals", "refused_decisions"]


@dataclass
class DQ001Config:
    snapshot_id: str
    symbol: str = "SPY"
    strategies: list[str] = field(default_factory=lambda: ["sma-crossover", "rsi-mean-reversion"])
    strategy_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    scenarios: list[Scenario] = field(default_factory=lambda: list(DEFAULT_SCENARIOS))
    policies: list[str] = field(default_factory=lambda: list(POLICIES))
    seeds: list[int] = field(default_factory=lambda: [11, 23, 37, 41, 53])
    eval_start: str | None = None  # default: 252 sessions after the first bar (warm-up)
    eval_end: str | None = None
    initial_cash: float = 10_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    persist_derived: bool = True

    def validate(self) -> None:
        validate_uuid(self.snapshot_id, "snapshot_id")
        self.symbol = validate_symbol(self.symbol)
        self.strategies = [normalize_strategy_id(s) for s in self.strategies]
        bad = [p for p in self.policies if p not in POLICIES]
        if bad:
            raise InputValidationError(f"unknown policies {bad}; available {sorted(POLICIES)}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise InputValidationError("seeds must be a non-empty list of distinct integers")
        if not any(s.is_clean for s in self.scenarios):
            self.scenarios = [Scenario("clean"), *self.scenarios]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scenarios"] = [s.to_dict() for s in self.scenarios]
        return d


def _positions(targets: pd.Series) -> pd.Series:
    """Intended position implied by targets (NaN = keep previous intent; start flat)."""
    return targets.ffill().fillna(0.0)


def signal_fidelity(clean_t: pd.Series, test_t: pd.Series) -> dict[str, float]:
    pc, pt = _positions(clean_t), _positions(test_t)
    c_change = pc.diff().fillna(pc.iloc[0]).abs() > 1e-12
    t_change = pt.diff().fillna(pt.iloc[0]).abs() > 1e-12
    agree = (pc - pt).abs() <= 1e-12
    return {
        "signal_agreement": float(agree.mean()),
        "clean_signal_events": int(c_change.sum()),
        "missed_signals": int((c_change & ~agree).sum()),
        "spurious_signals": int((t_change & ~agree).sum()),
        "refused_decisions": int((test_t.isna() & clean_t.notna()).sum()),
    }


def run_dq001(
    ctx: AppContext,
    cfg: DQ001Config,
    experiment_id: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    cfg.validate()
    snap = require_tradable_snapshot(ctx.snapshots, cfg.snapshot_id)
    if cfg.symbol not in snap.symbols:
        raise InputValidationError(f"{cfg.symbol} not in snapshot {snap.id}")
    eid = experiment_id or new_id()
    design = {"config": cfg.to_dict(), "missing_data_rules": MISSING_DATA_RULES,
              "policies": {p: POLICIES[p].to_dict() for p in cfg.policies},
              "snapshot": {"id": snap.id, "dataset_hash": snap.dataset_hash, "provider": snap.provider,
                           "is_synthetic": snap.is_synthetic},
              "code_version": git_commit(), "price_mode": PriceMode.RAW_WITH_DIVIDENDS.value}
    ctx.db.execute(
        "INSERT INTO experiments(id, code, hypothesis, design, snapshot_id, status, created_at) VALUES (?,?,?,?,?,?,?)",
        (eid, CODE, HYPOTHESIS, dumps(design), snap.id, RunStatus.RUNNING.value, utcnow()),
    )
    t0 = time.perf_counter()
    try:
        result = _execute(ctx, cfg, snap, eid, design, progress)
        result["computation_seconds"] = round(time.perf_counter() - t0, 3)
        folder = ctx.settings.experiments_dir / eid
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.pop("_runs")).to_csv(folder / "runs.csv", index=False)
        pd.DataFrame(result["aggregate"]).to_csv(folder / "aggregate.csv", index=False)
        pd.DataFrame(result["policy_comparison"]).to_csv(folder / "policy_comparison.csv", index=False)
        result = clean_for_json(result)
        (folder / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        ctx.db.execute(
            "UPDATE experiments SET status=?, results=?, artifact_uri=?, completed_at=? WHERE id=?",
            (RunStatus.COMPLETED.value, dumps(result), str(folder), utcnow(), eid),
        )
        ctx.db.audit("system", "experiment.completed", "experiment", eid, {"code": CODE})
        return result
    except Exception as exc:
        ctx.db.execute("UPDATE experiments SET status=?, failure_reason=?, completed_at=? WHERE id=?",
                       (RunStatus.FAILED.value, f"{type(exc).__name__}: {exc}", utcnow(), eid))
        raise


def _execute(ctx, cfg: DQ001Config, snap, eid: str, design: dict, progress) -> dict[str, Any]:
    all_bars = ctx.snapshots.load(snap.id)
    bars = all_bars[all_bars["symbol"] == cfg.symbol].reset_index(drop=True)
    source_hash = dataframe_sha256(canonicalize_bars(all_bars))
    panel = build_panel(bars, snap.price_basis, PriceMode.RAW_WITH_DIVIDENDS, [cfg.symbol])
    grid = panel.sessions
    if len(grid) < 300:
        raise InputValidationError("DQ-001 needs at least 300 sessions of data")
    ws = pd.Timestamp(cfg.eval_start) if cfg.eval_start else grid[min(252, len(grid) - 50)]
    we = pd.Timestamp(cfg.eval_end) if cfg.eval_end else grid[-1]
    if ws >= we:
        raise InputValidationError("eval_start must be before eval_end")
    bt_cfg = BacktestConfig(initial_cash=cfg.initial_cash, commission_bps=cfg.commission_bps,
                            slippage_bps=cfg.slippage_bps, start=ws.strftime("%Y-%m-%d"),
                            end=we.strftime("%Y-%m-%d"))
    strategies = {s: create_strategy(s, cfg.strategy_params.get(s)) for s in cfg.strategies}

    # ---- baselines: ordinary clean backtest
    baseline: dict[str, dict] = {}
    for sid, strat in strategies.items():
        res = run_backtest(panel, strat, bt_cfg)
        m = performance_metrics(res.equity, res.trades, res.orders)
        rets = res.equity["nav"].pct_change().dropna().to_numpy()
        lo, hi = block_bootstrap_ci(rets, seed=cfg.seeds[0])
        baseline[sid] = {"metrics": m, "targets": res.targets[cfg.symbol], "n_trades": len(res.trades),
                         "sharpe_bootstrap_ci95": [lo, hi]}

    div = panel.dividend[cfg.symbol].to_numpy()
    spl = panel.split[cfg.symbol].to_numpy()
    runs: list[dict[str, Any]] = []
    derived_ids: dict[str, str] = {}
    detection: list[dict[str, Any]] = []
    jobs = [(sc, seed) for sc in cfg.scenarios for seed in ([cfg.seeds[0]] if sc.is_clean else cfg.seeds)]
    total = len(jobs)
    for k, (sc, seed) in enumerate(jobs):
        if progress:
            progress(k / total, f"{sc.name} seed={seed}")
        pert, events = perturb(bars, sc, seed, ws, we)
        derived_id = snap.id
        if not sc.is_clean:
            counts = events["kind"].value_counts().to_dict() if len(events) else {}
            if cfg.persist_derived:
                ev_records = events.assign(session_date=events["session_date"].dt.strftime("%Y-%m-%d"))
                dsnap, _ = ctx.snapshots.create(
                    pert, provider=f"{snap.provider}+dq001", price_basis=snap.price_basis,
                    is_synthetic=snap.is_synthetic,
                    parameters={"experiment": CODE, "experiment_id": eid, "scenario": sc.to_dict(), "seed": seed,
                                "symbol": cfg.symbol, "window": [str(ws.date()), str(we.date())]},
                    parent_id=snap.id,
                    manifest={"transformation": "dq001.perturb", "scenario": sc.name, "seed": seed,
                              "event_counts": counts,
                              "events": json.loads(ev_records.to_json(orient="records"))},
                )
                derived_id = dsnap.id
                qrep = run_quality_checks(pert, snap.price_basis, str(ws.date()), str(we.date()))
                ctx.snapshots.record_quality(dsnap.id, qrep)
                miss_issue = [i for i in qrep.issues if i.issue_type == "MISSING_SESSIONS"]
                extreme = [i for i in qrep.issues if i.issue_type == "EXTREME_RETURN"]
                detection.append({
                    "scenario": sc.name, "seed": seed,
                    "injected_missing": int(counts.get("missing", 0)),
                    "detected_missing": int(sum(i.row_count for i in miss_issue)),
                    "injected_corrupted": int(counts.get("corrupted", 0)),
                    "flagged_extreme_returns": int(sum(i.row_count for i in extreme)),
                    "injected_delayed": int(counts.get("delayed", 0)),
                    "quality_status": qrep.status,
                })
            derived_ids[f"{sc.name}:{seed}"] = derived_id
        for pname in cfg.policies:
            pol = POLICIES[pname]
            obs = observed_from_events(grid, pert, events, cfg.symbol)
            seen, diag = apply_policy(obs, pol)
            sig = total_return_index(seen, div, spl, snap.price_basis)
            sig_df = pd.DataFrame({cfg.symbol: sig}, index=grid)
            for sid, strat in strategies.items():
                res = run_backtest(panel, strat, bt_cfg, signal_prices=sig_df)
                m = performance_metrics(res.equity, res.trades, res.orders)
                b = baseline[sid]["metrics"]
                fid = signal_fidelity(baseline[sid]["targets"], res.targets[cfg.symbol])
                row = {
                    "experiment_id": eid, "strategy": sid, "scenario": sc.name, "policy": pname, "seed": seed,
                    "research_only": pol.research_only, "derived_snapshot_id": derived_id,
                    "total_return": m["total_return"], "cagr": m["cagr"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"], "n_trades": int(len(res.trades)),
                    "turnover_annual": m["turnover_annual"], "final_nav": m["final_nav"],
                    "delta_total_return": m["total_return"] - b["total_return"],
                    "delta_sharpe": m["sharpe"] - b["sharpe"],
                    "delta_max_drawdown": m["max_drawdown"] - b["max_drawdown"],
                    **fid, **{f"policy_{k2}": v for k2, v in diag.items()},
                }
                runs.append(row)
                ctx.db.execute(
                    "INSERT INTO experiment_runs(id, experiment_id, scenario, policy, strategy, seed, "
                    "derived_snapshot_id, outputs) VALUES (?,?,?,?,?,?,?,?)",
                    (deterministic_id(eid, sc.name, pname, sid, seed), eid, sc.name, pname, sid, seed,
                     derived_id, dumps(clean_for_json(row))),
                )

    # ---- verify the source is untouched
    reloaded = ctx.snapshots.load(snap.id)  # raises on hash mismatch
    if dataframe_sha256(canonicalize_bars(reloaded)) != source_hash:
        raise DataIntegrityError("source snapshot changed during DQ-001")

    df = pd.DataFrame(runs)
    agg_rows = []
    for (sid, scn, pol), g in df.groupby(["strategy", "scenario", "policy"], sort=False):
        row = {"strategy": sid, "scenario": scn, "policy": pol, "research_only": bool(g["research_only"].iloc[0]),
               "n_seeds": int(g["seed"].nunique())}
        for mname in KEY_METRICS:
            ci = mean_ci(g[mname].to_numpy(dtype=float))
            row[f"{mname}_mean"] = ci["mean"]
            row[f"{mname}_std"] = ci["std"]
            row[f"{mname}_ci_low"] = ci["ci_low"]
            row[f"{mname}_ci_high"] = ci["ci_high"]
        agg_rows.append(row)
    comp_rows = []
    ref = "strict_skip" if "strict_skip" in cfg.policies else cfg.policies[0]
    for (sid, scn), g in df.groupby(["strategy", "scenario"], sort=False):
        base = g[g.policy == ref].set_index("seed")
        for pol_name in cfg.policies:
            if pol_name == ref:
                continue
            other = g[g.policy == pol_name].set_index("seed")
            common = base.index.intersection(other.index)
            for mname in ("delta_total_return", "delta_sharpe", "missed_signals"):
                diff = (other.loc[common, mname] - base.loc[common, mname]).to_numpy(dtype=float)
                ci = mean_ci(diff)
                comp_rows.append({"strategy": sid, "scenario": scn, "metric": mname, "policy": pol_name,
                                  "minus": ref, "n_pairs": ci["n"], "mean_diff": ci["mean"],
                                  "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    findings = _findings(agg_rows, comp_rows)
    if progress:
        progress(1.0, "done")
    return {
        "experiment_id": eid,
        "code": CODE,
        "hypothesis": HYPOTHESIS,
        "label": BACKTEST_LABEL,
        "data_label": data_label(snap.is_synthetic),
        "is_synthetic": snap.is_synthetic,
        "research_only_policies": {p: RESEARCH_ONLY_LABEL for p in cfg.policies if POLICIES[p].research_only},
        "design": design,
        "evaluation_window": [str(ws.date()), str(we.date())],
        "source_snapshot_verified_unchanged": True,
        "baseline": {sid: {"metrics": v["metrics"], "n_trades": v["n_trades"],
                           "sharpe_bootstrap_ci95": v["sharpe_bootstrap_ci95"]} for sid, v in baseline.items()},
        "derived_snapshots": derived_ids,
        "detection": detection,
        "aggregate": agg_rows,
        "policy_comparison": comp_rows,
        "findings": findings,
        "n_runs": len(runs),
        "_runs": runs,
    }


def _fmt_ci(mean, lo, hi, pct=True) -> str:
    if mean is None:
        return "n/a"
    f = (lambda x: f"{x * 100:+.2f}pp") if pct else (lambda x: f"{x:+.3f}")
    if lo is None:
        return f"{f(mean)} (single run, no CI)"
    return f"{f(mean)} (95% CI {f(lo)} … {f(hi)})"


def _findings(agg_rows: list[dict], comp_rows: list[dict]) -> list[str]:
    """Plain statements generated only from the aggregate numbers."""
    out = []
    for r in agg_rows:
        if r["scenario"] == "clean":
            continue
        tag = " [research-only policy]" if r["research_only"] else ""
        sig = ""
        lo, hi = r["delta_total_return_ci_low"], r["delta_total_return_ci_high"]
        if lo is not None and hi is not None:
            sig = " — not distinguishable from zero" if lo <= 0 <= hi else " — CI excludes zero"
        out.append(
            f"{r['strategy']} · {r['scenario']} · {r['policy']}{tag}: Δ total return vs clean baseline "
            f"{_fmt_ci(r['delta_total_return_mean'], lo, hi)}{sig}; mean missed signals "
            f"{r['missed_signals_mean']:.1f}, signal agreement {r['signal_agreement_mean'] * 100:.1f}% "
            f"over {r['n_seeds']} seeds."
        )
    for c in comp_rows:
        if c["metric"] != "delta_total_return" or c["mean_diff"] is None:
            continue
        lo, hi = c["ci_low"], c["ci_high"]
        verdict = "inconclusive" if lo is None or lo <= 0 <= hi else ("better" if lo > 0 else "worse")
        out.append(f"{c['strategy']} · {c['scenario']}: {c['policy']} minus {c['minus']} total return "
                   f"{_fmt_ci(c['mean_diff'], lo, hi)} → {verdict} (paired by seed, n={c['n_pairs']}).")
    return out


# ------------------------------------------------------------------ reads
def list_experiments(ctx: AppContext) -> list[dict[str, Any]]:
    rows = ctx.db.query("SELECT id, code, status, snapshot_id, created_at, completed_at, failure_reason "
                        "FROM experiments ORDER BY created_at DESC")
    return rows


def get_experiment(ctx: AppContext, experiment_id: str) -> dict[str, Any]:
    eid = validate_uuid(experiment_id, "experiment_id")
    r = ctx.db.query_one("SELECT * FROM experiments WHERE id=?", (eid,))
    if r is None:
        raise NotFoundError(f"experiment {eid} not found")
    r["design"] = loads(r["design"])
    r["results"] = loads(r["results"])
    return r


def experiment_runs(ctx: AppContext, experiment_id: str) -> pd.DataFrame:
    eid = validate_uuid(experiment_id, "experiment_id")
    rows = ctx.db.query("SELECT outputs FROM experiment_runs WHERE experiment_id=? ORDER BY strategy, scenario, "
                        "policy, seed", (eid,))
    return pd.DataFrame([loads(r["outputs"]) for r in rows])


def artifact_dir(ctx: AppContext, experiment_id: str) -> Path:
    r = get_experiment(ctx, experiment_id)
    if not r["artifact_uri"]:
        raise NotFoundError("experiment has no artifacts")
    return Path(r["artifact_uri"])


__all__ = ["DQ001Config", "run_dq001", "get_experiment", "list_experiments", "experiment_runs"]
