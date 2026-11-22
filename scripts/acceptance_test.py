"""Final acceptance test — runs the full QuantLab workflow end to end and writes an evidence report.

    python scripts/acceptance_test.py [--data-dir DIR] [--offline]

Steps (each recorded as PASS/FAIL with evidence):
 1. Download validated historical SPY (+ universe) data via yfinance. If the network/provider is
    unavailable, the failure is recorded and the labelled SYNTHETIC fixture is used instead.
 2. Validate the snapshot (hash re-verified, quality report).
 3. Run Buy & Hold, SMA crossover and RSI mean reversion on SPY; compare metrics.
 4. Run DQ-001.
 5. Create a paper account, process simulated orders, verify cash and positions independently.
 6. Generate an evidence-based (deterministic, number-verified) summary.
 7. Export reports.
 8. Restart: new processes (Python + API server) read the same data dir; state must be identical.
Exit code 0 only if every step passes.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
STEPS: list[dict] = []


def step(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                ev = fn(*a, **k)
                STEPS.append({"step": name, "status": "PASS", "seconds": round(time.perf_counter() - t0, 2),
                              "evidence": ev})
                print(f"[PASS] {name}")
                return ev
            except Exception as exc:
                STEPS.append({"step": name, "status": "FAIL", "seconds": round(time.perf_counter() - t0, 2),
                              "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)})
                print(f"[FAIL] {name}: {exc}")
                return None
        return run
    return deco


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--offline", action="store_true", help="skip the yfinance attempt")
    ap.add_argument("--require-real", action="store_true",
                    help="fail instead of falling back to the synthetic fixture when real data is unavailable")
    args = ap.parse_args()
    data_dir = Path(args.data_dir or tempfile.mkdtemp(prefix="quantlab_acceptance_")).resolve()
    os.environ["QUANTLAB_DATA_DIR"] = str(data_dir)
    os.environ["LIVE_TRADING_ENABLED"] = "false"

    from quantlab.ai.assistant import ResearchAssistant
    from quantlab.backtest.engine import BacktestConfig
    from quantlab.backtest.service import get_run, run_and_store
    from quantlab.context import build_context
    from quantlab.core.errors import DataUnavailableError
    from quantlab.data.ingestion import ingest
    from quantlab.data.providers import SyntheticProvider, YFinanceProvider
    from quantlab.data.validation import run_quality_checks
    from quantlab.paper.broker import PaperBroker
    from quantlab.reports.export import export_backtest, export_experiment
    from quantlab.research.dq_001 import DQ001Config, run_dq001

    ctx = build_context(data_dir)
    universe = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    state: dict = {"data_dir": str(data_dir), "limitations": []}

    @step("1. Download or load validated historical SPY data")
    def s1():
        if not args.offline:
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = ingest(universe, "2015-01-01", str(date.today()), YFinanceProvider(), ctx.snapshots, ctx.db)
                state["snapshot"] = r.snapshot
                return {"source": "yfinance (REAL)", "snapshot_id": r.snapshot.id, "rows": r.snapshot.n_rows,
                        "range": [r.snapshot.start, r.snapshot.end], "quality": r.quality.status,
                        "hash": r.snapshot.dataset_hash}
            except DataUnavailableError as exc:
                if args.require_real:
                    raise
                state["limitations"].append(f"Real data unavailable ({str(exc)[:160]}); used SYNTHETIC fixture.")
        r = ingest(universe, "2015-01-01", "2025-12-31", SyntheticProvider(seed=7), ctx.snapshots, ctx.db)
        state["snapshot"] = r.snapshot
        return {"source": "SYNTHETIC fixture — NOT REAL MARKET PERFORMANCE", "snapshot_id": r.snapshot.id,
                "rows": r.snapshot.n_rows, "range": [r.snapshot.start, r.snapshot.end], "quality": r.quality.status,
                "hash": r.snapshot.dataset_hash}

    @step("2. Validate snapshot (hash re-verified + quality checks)")
    def s2():
        snap = state["snapshot"]
        bars = ctx.snapshots.load(snap.id, verify=True)
        rep = run_quality_checks(bars, snap.price_basis, "2015-01-01", snap.end, expected_source=bars["source"].iloc[0])
        assert not rep.blocking, rep.to_dict()
        return {"hash_verified": True, "status": rep.status, "counts": rep.counts,
                "warn_types": sorted({i.issue_type for i in rep.issues})}

    @step("3. Run all three backtests on SPY and compare metrics")
    def s3():
        snap = state["snapshot"]
        out = {}
        state["runs"] = {}
        for sid in ("buy-hold", "sma-crossover", "rsi-mean-reversion"):
            o = run_and_store(ctx, sid, snap.id, ["SPY"], config=BacktestConfig(start="2016-01-04"))
            state["runs"][sid] = o.run_id
            n = o.summary["metrics"]["net"]
            assert o.summary["metrics"]["net"]["final_nav"] > 0
            out[sid] = {"run_id": o.run_id, **{k: n[k] for k in ("total_return", "cagr", "sharpe", "sharpe_ci95_low",
                                                               "sharpe_ci95_high", "max_drawdown", "turnover_annual",
                                                               "n_fills")}}
        bh = out["buy-hold"]
        bench = get_run(ctx, bh["run_id"])["result"]["metrics"]["benchmark"]
        assert abs(bh["total_return"] - bench["total_return"]) < 1e-9  # B&H SPY == SPY benchmark
        return out

    @step("4. Perform DQ-001 research")
    def s4():
        res = run_dq001(ctx, DQ001Config(snapshot_id=state["snapshot"].id, seeds=[11, 23, 37]))
        state["experiment_id"] = res["experiment_id"]
        assert res["source_snapshot_verified_unchanged"]
        return {"experiment_id": res["experiment_id"], "n_runs": res["n_runs"],
                "window": res["evaluation_window"], "findings_sample": res["findings"][:4]}

    @step("5. Create paper account, process simulated orders, verify cash and positions")
    def s5():
        b = PaperBroker(ctx)
        acc = b.create_account("acceptance", state["snapshot"].id, "10000", start_session="2024-01-02")
        aid = acc["id"]
        state["account_id"] = aid
        o1 = b.submit_order(aid, "SPY", "BUY", 5, client_order_id="acc-1")
        o2 = b.submit_order(aid, "TLT", "BUY", 10, client_order_id="acc-2")
        dup = b.submit_order(aid, "SPY", "BUY", 5, client_order_id="acc-1")
        short = b.submit_order(aid, "GLD", "SELL", 1)
        too_big = b.submit_order(aid, "QQQ", "BUY", 1000)
        over_25 = b.submit_order(aid, "SPY", "BUY", 8)  # 5 pending + 8 SPY > 25% of NAV
        assert o1["status"] == o2["status"] == "SUBMITTED" and dup["id"] == o1["id"]
        assert short["status"] == "REJECTED" and too_big["status"] == "REJECTED"
        assert over_25["status"] == "REJECTED" and over_25["reason"].startswith("MAX_POSITION_WEIGHT")
        b.advance_session(aid, 3)
        b.submit_order(aid, "SPY", "SELL", 2, client_order_id="acc-3")
        b.advance_session(aid, 2)
        st = b.account_state(aid)
        # independent check: cash = deposit + Σ ledger; NAV = cash + Σ qty x close
        ledger = sum(Decimal(r["amount"]) for r in b.ledger(aid))
        assert Decimal(st["cash"]) == ledger.quantize(Decimal("0.01"))
        mv = sum(Decimal(h["quantity"]) * Decimal(h["last_close"]) for h in st["holdings"])
        assert abs(Decimal(st["nav"]) - (Decimal(st["cash"]) + mv)) <= Decimal("0.01")
        assert {h["symbol"]: h["quantity"] for h in st["holdings"]} == {"SPY": "3", "TLT": "10"}
        assert Decimal(st["cash"]) >= 0 and b.reconcile(aid)["ok"]
        state["paper_state"] = st
        return {"account_id": aid, "cash": st["cash"], "nav": st["nav"], "holdings": st["holdings"],
                "rejections": [short["reason"], too_big["reason"], over_25["reason"]], "replay_session": st["replay_session"]}

    @step("6. Generate evidence-based summary")
    def s6():
        a = ResearchAssistant(ctx.settings.db_path, provider=None)
        c = a.compare(list(state["runs"].values()))
        e = a.summarize_experiment(state["experiment_id"])
        assert c.verification["verified"] and e.verification["verified"]
        state["summary_md"] = c.text + "\n\n" + e.text
        return {"mode": c.mode, "numbers_checked": c.verification["numbers_checked"] + e.verification["numbers_checked"],
                "all_numbers_traced_to_stored_data": True}

    @step("7. Export reports")
    def s7():
        files = []
        for rid in state["runs"].values():
            for f in ("md", "html", "csv", "json"):
                files += [str(p) for p in export_backtest(ctx, rid, f)]
        for f in ("md", "html", "csv", "json"):
            files += [str(p) for p in export_experiment(ctx, state["experiment_id"], f)]
        assert all(Path(p).stat().st_size > 0 for p in files)
        return {"n_files": len(files), "examples": files[:3]}

    @step("8. Restart the app and verify persistence")
    def s8():
        before = state["paper_state"]
        ctx.db.close()
        code = (
            "import json,sys; from quantlab.context import build_context; from quantlab.paper.broker import PaperBroker;"
            "from quantlab.backtest.service import list_runs; from quantlab.research.dq_001 import list_experiments;"
            f"c=build_context({str(data_dir)!r}); b=PaperBroker(c);"
            f"print(json.dumps({{'acc': b.account_state({state['account_id']!r}), 'runs': len(list_runs(c)),"
            "'exps': len(list_experiments(c)), 'snaps': len(c.snapshots.list())}))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
        after = json.loads(out.stdout.strip().splitlines()[-1])
        assert after["acc"] == before, "paper account state differs after restart"
        assert after["runs"] >= 3 and after["exps"] >= 1
        # API server process restart
        port = _free_port()
        env = {**os.environ, "QUANTLAB_DATA_DIR": str(data_dir)}
        api_states = []
        for _ in range(2):
            proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "quantlab.api.app:app", "--port", str(port)],
                                    env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                _wait(f"http://127.0.0.1:{port}/health")
                hdr = {"Authorization": f"Bearer {os.environ['QUANTLAB_API_TOKEN']}"} if os.environ.get(
                    "QUANTLAB_API_TOKEN") else {}
                r = httpx.get(f"http://127.0.0.1:{port}/paper/account", params={"account_id": state["account_id"]},
                              headers=hdr)
                api_states.append(r.json()["data"])
            finally:
                proc.terminate()
                proc.wait(10)
        assert api_states[0] == api_states[1] == before
        # Streamlit server boots against the same data dir
        sport = _free_port()
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(REPO / "apps/dashboard/streamlit_app.py"),
                                 "--server.headless", "true", "--server.port", str(sport)], env=env, cwd=REPO,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            _wait(f"http://127.0.0.1:{sport}/_stcore/health")
        finally:
            proc.terminate()
            proc.wait(10)
        return {"python_restart_identical": True, "api_restarts_identical": True, "streamlit_health": "ok",
                "runs": after["runs"], "experiments": after["exps"], "snapshots": after["snaps"]}

    for fn in (s1, s2, s3, s4, s5, s6, s7, s8):
        fn()

    ok = all(s["status"] == "PASS" for s in STEPS)
    report = {"overall": "PASS" if ok else "FAIL", "data_dir": str(data_dir), "limitations": state["limitations"],
              "steps": STEPS}
    out_dir = data_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "acceptance_report.json").write_text(json.dumps(report, indent=2, default=str))
    md = [f"# QuantLab acceptance test — {report['overall']}", "", f"Data dir: `{data_dir}`", ""]
    md += [f"- Limitation: {x}" for x in state["limitations"]]
    for s in STEPS:
        md.append(f"\n## {s['step']} — {s['status']} ({s['seconds']}s)\n")
        md.append("```json\n" + json.dumps(s.get("evidence") or s.get("error"), indent=2, default=str) + "\n```")
    if state.get("summary_md"):
        md += ["", "## Evidence-based summary", "", state["summary_md"]]
    (out_dir / "acceptance_report.md").write_text("\n".join(md))
    print(f"\nOVERALL: {report['overall']}  — report: {out_dir / 'acceptance_report.md'}")
    return 0 if ok else 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{url} did not become healthy")


if __name__ == "__main__":
    raise SystemExit(main())
