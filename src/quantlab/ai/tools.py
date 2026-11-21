"""Read-only research tools for the AI assistant.

Structural guarantees:
* The SQLite connection is opened with ``mode=ro`` — writes fail at the database level.
* This package never imports ``quantlab.paper`` or any order/execution code
  (enforced by tests/unit/test_ai.py). There is no tool that can submit orders.
* Every fact returned carries its source (table/artifact + id) so claims can be traced.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from quantlab.core.errors import InputValidationError, NotFoundError
from quantlab.core.ids import validate_uuid

TOOL_NAMES = ("get_backtest_facts", "compare_backtests", "compare_statistics", "get_experiment_facts",
              "list_recent_backtests")
NET_KEYS = ["total_return", "cagr", "annual_volatility", "sharpe", "sharpe_ci95_low", "sharpe_ci95_high", "sortino",
            "max_drawdown", "max_drawdown_duration_sessions", "turnover_annual", "time_in_market", "n_fills",
            "total_commission", "total_slippage_cost", "total_dividends", "final_nav", "initial_nav", "n_sessions",
            "years"]


class ReadOnlyResearchTools:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con

    def _one(self, sql: str, params: tuple) -> dict[str, Any] | None:
        with closing(self._conn()) as con:
            r = con.execute(sql, params).fetchone()
            return dict(r) if r else None

    def list_recent_backtests(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._conn()) as con:
            rows = con.execute("SELECT id, strategy_name, status, is_synthetic, created_at FROM backtest_runs "
                               "WHERE status='COMPLETED' ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_backtest_facts(self, run_id: str) -> dict[str, Any]:
        rid = validate_uuid(run_id, "run_id")
        r = self._one("SELECT id, status, artifact_uri FROM backtest_runs WHERE id=?", (rid,))
        if r is None:
            raise NotFoundError(f"backtest {rid} not found")
        if r["status"] != "COMPLETED":
            raise InputValidationError(f"backtest {rid} is {r['status']}; no stored results to explain")
        res = json.loads((Path(r["artifact_uri"]) / "result.json").read_text())
        m = res["metrics"]
        facts: dict[str, Any] = {}
        for section in ("net", "gross", "benchmark"):
            for k in NET_KEYS:
                v = (m.get(section) or {}).get(k)
                if v is not None:
                    facts[f"{section}.{k}"] = v
        for k, v in (m.get("relative") or {}).items():
            if v is not None:
                facts[f"relative.{k}"] = v
        return {
            "kind": "backtest",
            "run_id": rid,
            "source": f"backtest_runs/{rid} + {r['artifact_uri']}/result.json",
            "strategy": res["strategy"]["name"],
            "strategy_id": res["strategy"]["id"],
            "parameters": res["strategy"]["parameters"],
            "symbols": res["symbols"],
            "benchmark_symbol": res.get("benchmark_symbol"),
            "window": [m["net"]["start"], m["net"]["end"]],
            "is_synthetic": res["is_synthetic"],
            "data_label": res["data_label"],
            "label": res["label"],
            "costs": {"commission_bps": res["config"]["commission_bps"], "slippage_bps": res["config"]["slippage_bps"]},
            "risk_free_rate": res["config"]["risk_free_rate"],
            "warnings": res["warnings"],
            "facts": facts,
        }

    def compare_backtests(self, run_ids: list[str]) -> list[dict[str, Any]]:
        if len(run_ids) < 2:
            raise InputValidationError("compare needs at least two run ids")
        return [self.get_backtest_facts(r) for r in run_ids]

    def compare_statistics(self, run_ids: list[str], n_trials: int | None = None) -> dict[str, Any]:
        """Read stored equity curves and run the statistical tests (read-only)."""
        import pandas as pd

        from quantlab.research.stat_tests import compare_strategies

        navs = {}
        for rid in run_ids:
            r = self._one("SELECT id, strategy_name, artifact_uri FROM backtest_runs WHERE id=?", (validate_uuid(rid),))
            if r is None:
                raise NotFoundError(f"backtest {rid} not found")
            eq = pd.read_parquet(Path(r["artifact_uri"]) / "equity.parquet").set_index("session")["nav"]
            navs[f"{r['id'][:8]} {r['strategy_name']}"] = eq
        return compare_strategies(navs, n_trials=n_trials, seed=0, n_boot=1000)

    def get_experiment_facts(self, experiment_id: str) -> dict[str, Any]:
        eid = validate_uuid(experiment_id, "experiment_id")
        r = self._one("SELECT id, code, status, results FROM experiments WHERE id=?", (eid,))
        if r is None:
            raise NotFoundError(f"experiment {eid} not found")
        if r["status"] != "COMPLETED":
            raise InputValidationError(f"experiment {eid} is {r['status']}")
        res = json.loads(r["results"])
        if r["code"] == "WALK-FORWARD":
            return self._walkforward_facts(eid, res)
        facts: dict[str, Any] = {}
        for a in res["aggregate"]:
            key = f"{a['strategy']}|{a['scenario']}|{a['policy']}"
            for m in ("delta_total_return_mean", "delta_total_return_ci_low", "delta_total_return_ci_high",
                      "missed_signals_mean", "signal_agreement_mean", "delta_sharpe_mean", "n_seeds"):
                if a.get(m) is not None:
                    facts[f"{key}.{m}"] = a[m]
        for sid, b in res["baseline"].items():
            for k in ("total_return", "cagr", "sharpe", "max_drawdown"):
                if b["metrics"].get(k) is not None:
                    facts[f"baseline|{sid}.{k}"] = b["metrics"][k]
        return {
            "kind": "experiment", "experiment_id": eid, "code": r["code"],
            "source": f"experiments/{eid}.results", "hypothesis": res["hypothesis"],
            "window": res["evaluation_window"], "is_synthetic": res["is_synthetic"], "data_label": res["data_label"],
            "label": res["label"], "seeds": res["design"]["config"]["seeds"], "symbol": res["design"]["config"]["symbol"],
            "aggregate": res["aggregate"], "findings": res["findings"], "facts": facts,
            "research_only_policies": res.get("research_only_policies", {}),
        }

    @staticmethod
    def _walkforward_facts(eid: str, res: dict[str, Any]) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        for k in NET_KEYS:
            if res["oos_metrics"].get(k) is not None:
                facts[f"oos.{k}"] = res["oos_metrics"][k]
            if (res.get("benchmark_metrics") or {}).get(k) is not None:
                facts[f"benchmark.{k}"] = res["benchmark_metrics"][k]
        for f in res["folds"]:
            for k in ("train_objective", "test_objective", "test_total_return", "oos_percentile_rank"):
                if f.get(k) is not None:
                    facts[f"fold{f['fold']}.{k}"] = f[k]
        if res.get("mean_oos_percentile_rank") is not None:
            facts["mean_oos_percentile_rank"] = res["mean_oos_percentile_rank"]
        return {
            "kind": "walkforward", "experiment_id": eid, "code": res["code"], "source": f"experiments/{eid}.results",
            "strategy": res["strategy"], "symbols": res["symbols"], "objective": res["objective"],
            "window": res["evaluation_window"], "n_folds": len(res["folds"]), "n_configs": res["n_configs"],
            "chosen_params": [f["chosen_params"] for f in res["folds"]], "is_synthetic": res["is_synthetic"],
            "data_label": res["data_label"], "label": res["label"], "findings": res["findings"], "facts": facts,
        }
