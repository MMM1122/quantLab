"""FastAPI backend (local only). Long-running work returns a job id; poll GET /jobs/{id}.

Every response is wrapped as {"data": ..., "meta": {...}} where meta carries mode, source,
computation time, labels and assumptions. There is no live-order endpoint; a request that
claims mode=LIVE is rejected.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import quantlab
from quantlab.backtest.engine import BacktestConfig
from quantlab.backtest.service import ASSUMPTIONS, get_run, list_runs, load_artifact, run_and_store
from quantlab.context import AppContext, build_context
from quantlab.core.errors import (
    DataIntegrityError,
    DataUnavailableError,
    InputValidationError,
    LiveTradingDisabledError,
    NotFoundError,
    QualityGateError,
    QuantLabError,
    RiskRejection,
)
from quantlab.core.labels import BACKTEST_LABEL, REPLAY_LABEL, data_label
from quantlab.data.ingestion import ingest, validate_date_range
from quantlab.data.providers import get_provider
from quantlab.jobs import JobRunner
from quantlab.strategies.registry import list_strategies, normalize_strategy_id


def _meta(**kw) -> dict[str, Any]:
    return {"service": "quantlab", "version": quantlab.__version__, "live_trading_enabled": False, **kw}


def envelope(data: Any, t0: float | None = None, **meta) -> dict[str, Any]:
    m = _meta(**meta)
    if t0 is not None:
        m["computation_seconds"] = round(time.perf_counter() - t0, 4)
    return {"data": data, "meta": m}


class IngestRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    start: str
    end: str | None = None
    provider: Literal["yfinance", "synthetic"] = "synthetic"
    seed: int = 7


class DQ001Request(BaseModel):
    snapshot_id: str
    symbol: str = "SPY"
    strategies: list[str] = Field(default_factory=lambda: ["sma-crossover", "rsi-mean-reversion"])
    seeds: list[int] = Field(default_factory=lambda: [11, 23, 37, 41, 53], min_length=1, max_length=20)
    scenarios: list[str] | None = None
    eval_start: str | None = None
    eval_end: str | None = None
    commission_bps: float = Field(1.0, ge=0)
    slippage_bps: float = Field(2.0, ge=0)


class PaperAccountRequest(BaseModel):
    name: str
    snapshot_id: str
    initial_cash: float = Field(10_000.0, gt=0)
    start_session: str | None = None
    mode: str = "PAPER"


class PaperOrderRequest(BaseModel):
    account_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int = Field(gt=0)
    client_order_id: str | None = Field(None, max_length=64)
    order_type: str = "MARKET"
    mode: str = "PAPER"


class PaperAdvanceRequest(BaseModel):
    account_id: str
    sessions: int = Field(1, ge=1, le=2520)


class KillSwitchRequest(BaseModel):
    account_id: str
    active: bool
    reason: str = ""


class WalkForwardRequest(BaseModel):
    snapshot_id: str
    strategy: str
    symbols: list[str] = Field(default_factory=lambda: ["SPY"], min_length=1)
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    train_sessions: int = Field(756, ge=60)
    test_sessions: int = Field(252, ge=20)
    embargo_sessions: int = Field(0, ge=0)
    expanding: bool = True
    objective: Literal["sharpe", "cagr", "total_return", "calmar"] = "sharpe"
    commission_bps: float = Field(1.0, ge=0)
    slippage_bps: float = Field(2.0, ge=0)


class AIRequest(BaseModel):
    task: Literal["explain", "compare", "summarize_experiment"]
    ids: list[str] = Field(min_length=1)
    question: str | None = Field(None, max_length=2000)


class BacktestRequest(BaseModel):
    strategy: str
    snapshot_id: str
    symbols: list[str] = Field(min_length=1)
    start: str | None = None
    end: str | None = None
    capital: float = Field(10_000.0, gt=0)
    commission_bps: float = Field(1.0, ge=0)
    slippage_bps: float = Field(2.0, ge=0)
    min_commission: float = Field(0.0, ge=0)
    price_mode: Literal["raw_with_dividends", "adjusted_total_return"] = "raw_with_dividends"
    risk_free_rate: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)
    benchmark: str | None = None
    mode: str = "BACKTEST"
    seed: int = 42


def reject_live(mode: str) -> None:
    if str(mode).upper() == "LIVE":
        raise LiveTradingDisabledError("mode=LIVE is not supported; QuantLab V1 has no live broker")
    if str(mode).upper() not in ("BACKTEST", "PAPER", "RESEARCH"):
        raise InputValidationError(f"unknown mode {mode!r}")


def create_app(data_dir: str | None = None, ctx: AppContext | None = None) -> FastAPI:
    ctx = ctx or build_context(data_dir)
    jobs = JobRunner(ctx)
    app = FastAPI(title="QuantLab API", version=quantlab.__version__,
                  description="Local research/backtest/paper API. No live trading.")
    app.state.ctx = ctx
    app.state.jobs = jobs

    status_map = {
        InputValidationError: 422,
        NotFoundError: 404,
        QualityGateError: 409,
        DataUnavailableError: 503,
        DataIntegrityError: 500,
        LiveTradingDisabledError: 403,
        RiskRejection: 422,
    }

    @app.exception_handler(QuantLabError)
    async def _qerr(_: Request, exc: QuantLabError):
        code = next((c for t, c in status_map.items() if isinstance(exc, t)), 400)
        return JSONResponse(status_code=code, content={"error": type(exc).__name__, "detail": str(exc),
                                                       "meta": _meta()})

    @app.get("/health")
    def health():
        ctx.db.query_one("SELECT 1 AS ok")
        return envelope({"status": "ok", "db": str(ctx.settings.db_path)}, mode="SYSTEM")

    @app.get("/assets")
    def assets():
        snaps = ctx.snapshots.list(include_derived=False)
        out: dict[str, dict] = {}
        for s in snaps:
            for sym in s.symbols:
                e = out.setdefault(sym, {"symbol": sym, "snapshots": [], "currency": "USD", "asset_type": "ETF"})
                e["snapshots"].append({"id": s.id, "is_synthetic": s.is_synthetic, "provider": s.provider})
        return envelope({"universe": ctx.settings.base.get("universe"), "assets": list(out.values())}, mode="SYSTEM")

    @app.post("/data/ingest", status_code=202)
    def data_ingest(req: IngestRequest):
        validate_date_range(req.start, req.end, allow_future=req.provider == "synthetic")
        prov = get_provider(req.provider, seed=req.seed)

        def work():
            res = ingest(req.symbols, req.start, req.end, prov, ctx.snapshots, ctx.db)
            return res.snapshot.id

        jid = jobs.submit("data.ingest", req.model_dump(), work)
        return envelope({"job_id": jid, "status": "PENDING"}, mode="RESEARCH", source=req.provider)

    @app.get("/data/snapshots")
    def snapshots():
        data = []
        for s in ctx.snapshots.list():
            d = s.to_dict()
            d["quality_status"] = ctx.snapshots.quality_status(s.id)
            d["data_label"] = data_label(s.is_synthetic)
            data.append(d)
        return envelope(data, mode="RESEARCH")

    @app.get("/data/quality/{snapshot_id}")
    def quality(snapshot_id: str):
        s = ctx.snapshots.get(snapshot_id)
        return envelope({"snapshot_id": s.id, "status": ctx.snapshots.quality_status(s.id),
                         "issues": ctx.snapshots.quality_issues(s.id)},
                        mode="RESEARCH", source=s.provider, data_label=data_label(s.is_synthetic))

    @app.get("/strategies")
    def strategies():
        return envelope(list_strategies(), mode="RESEARCH")

    @app.post("/backtests", status_code=202)
    def create_backtest(req: BacktestRequest):
        reject_live(req.mode)
        if req.mode.upper() != "BACKTEST":
            raise InputValidationError("POST /backtests only accepts mode=BACKTEST")
        normalize_strategy_id(req.strategy)
        snap = ctx.snapshots.get(req.snapshot_id)  # validates UUID & existence before queueing
        cfg = BacktestConfig(initial_cash=req.capital, commission_bps=req.commission_bps,
                             slippage_bps=req.slippage_bps, min_commission=req.min_commission,
                             price_mode=req.price_mode, risk_free_rate=req.risk_free_rate,
                             start=req.start, end=req.end, seed=req.seed)
        if req.end:
            validate_date_range(req.start or "1995-01-01", req.end)

        def work():
            return run_and_store(ctx, req.strategy, snap.id, req.symbols, req.params, cfg, req.benchmark).run_id

        jid = jobs.submit("backtest.run", req.model_dump(), work)
        return envelope({"job_id": jid, "status": "PENDING"}, mode="BACKTEST", label=BACKTEST_LABEL,
                        data_label=data_label(snap.is_synthetic), assumptions=ASSUMPTIONS)

    @app.get("/backtests")
    def backtests():
        rows = [{k: r[k] for k in ("id", "strategy_name", "snapshot_id", "status", "is_synthetic", "created_at",
                                   "failure_reason")} for r in list_runs(ctx)]
        return envelope(rows, mode="BACKTEST")

    @app.get("/backtests/{run_id}")
    def backtest(run_id: str):
        r = get_run(ctx, run_id)
        res = r.get("result") or {}
        return envelope(
            {"id": r["id"], "status": r["status"], "failure_reason": r["failure_reason"], "result": res or None},
            mode="BACKTEST", run_id=r["id"], label=BACKTEST_LABEL,
            data_label=res.get("data_label"), assumptions=res.get("assumptions"),
        )

    @app.get("/backtests/{run_id}/trades")
    def backtest_trades(run_id: str):
        t0 = time.perf_counter()
        r = get_run(ctx, run_id)
        df = load_artifact(ctx, run_id, "trades")
        df = df.assign(session=df["session"].astype(str), decision_session=df["decision_session"].astype(str))
        return envelope(df.to_dict(orient="records"), t0, mode="BACKTEST", run_id=r["id"], label=BACKTEST_LABEL,
                        data_label=(r.get("result") or {}).get("data_label"))

    @app.post("/experiments/dq-001", status_code=202)
    def create_dq001(req: DQ001Request):
        from quantlab.research.dq_001 import DQ001Config, run_dq001
        from quantlab.research.perturbations import DEFAULT_SCENARIOS

        snap = ctx.snapshots.get(req.snapshot_id)
        scen = [sc for sc in DEFAULT_SCENARIOS if req.scenarios is None or sc.name in req.scenarios or sc.is_clean]
        cfg = DQ001Config(snapshot_id=snap.id, symbol=req.symbol, strategies=req.strategies, seeds=req.seeds,
                          scenarios=scen, eval_start=req.eval_start, eval_end=req.eval_end,
                          commission_bps=req.commission_bps, slippage_bps=req.slippage_bps)
        cfg.validate()
        from quantlab.core.ids import new_id

        eid = new_id()

        def work():
            run_dq001(ctx, cfg, experiment_id=eid)
            return eid

        jid = jobs.submit("experiment.dq001", req.model_dump(), work)
        return envelope({"job_id": jid, "experiment_id": eid, "status": "PENDING"}, mode="RESEARCH",
                        data_label=data_label(snap.is_synthetic))

    @app.post("/experiments/walk-forward", status_code=202)
    def create_walk_forward(req: WalkForwardRequest):
        from quantlab.core.ids import new_id
        from quantlab.research.walk_forward import WalkForwardConfig, run_walk_forward

        snap = ctx.snapshots.get(req.snapshot_id)
        cfg = WalkForwardConfig(snapshot_id=snap.id, **req.model_dump(exclude={"snapshot_id"}))
        cfg.validate()
        eid = new_id()

        def work():
            run_walk_forward(ctx, cfg, experiment_id=eid)
            return eid

        jid = jobs.submit("experiment.walk_forward", req.model_dump(), work)
        return envelope({"job_id": jid, "experiment_id": eid, "status": "PENDING"}, mode="RESEARCH",
                        data_label=data_label(snap.is_synthetic))

    @app.get("/experiments")
    def experiments():
        from quantlab.research.dq_001 import list_experiments

        return envelope(list_experiments(ctx), mode="RESEARCH")

    @app.get("/experiments/{experiment_id}")
    def experiment(experiment_id: str):
        from quantlab.research.dq_001 import get_experiment

        e = get_experiment(ctx, experiment_id)
        res = e.get("results") or {}
        return envelope({"id": e["id"], "code": e["code"], "status": e["status"], "hypothesis": e["hypothesis"],
                         "failure_reason": e["failure_reason"], "design": e["design"], "results": res or None},
                        mode="RESEARCH", run_id=e["id"], label=res.get("label"), data_label=res.get("data_label"))

    from quantlab.paper.broker import PaperBroker

    broker = PaperBroker(ctx)
    app.state.broker = broker

    @app.get("/paper/account")
    def paper_account(account_id: str | None = None):
        data = broker.account_state(account_id) if account_id else broker.list_accounts()
        return envelope(data, mode="PAPER", label=REPLAY_LABEL)

    @app.post("/paper/accounts", status_code=201)
    def paper_create(req: PaperAccountRequest):
        reject_live(req.mode)
        return envelope(broker.create_account(req.name, req.snapshot_id, req.initial_cash, req.start_session,
                                              req.mode), mode="PAPER", label=REPLAY_LABEL)

    @app.post("/paper/orders", status_code=201)
    def paper_order(req: PaperOrderRequest):
        reject_live(req.mode)
        o = broker.submit_order(req.account_id, req.symbol, req.side, req.quantity, req.client_order_id,
                                req.order_type, actor="api", mode=req.mode)
        return envelope(o, mode="PAPER", label=REPLAY_LABEL)

    @app.get("/paper/orders")
    def paper_orders(account_id: str):
        return envelope(broker.list_orders(account_id), mode="PAPER", label=REPLAY_LABEL)

    @app.post("/paper/advance")
    def paper_advance(req: PaperAdvanceRequest):
        t0 = time.perf_counter()
        out = broker.advance_session(req.account_id, req.sessions)
        return envelope({"sessions": out, "account": broker.account_state(req.account_id)}, t0, mode="PAPER",
                        label=REPLAY_LABEL)

    @app.post("/paper/kill-switch")
    def paper_kill(req: KillSwitchRequest):
        return envelope(broker.set_kill_switch(req.account_id, req.active, req.reason), mode="PAPER")

    @app.get("/risk/events")
    def risk_events(account_id: str | None = None):
        return envelope(broker.risk_events(account_id), mode="PAPER")

    @app.post("/ai/ask")
    def ai_ask(req: AIRequest):
        from quantlab.ai.assistant import ResearchAssistant

        t0 = time.perf_counter()
        a = ResearchAssistant(ctx.settings.db_path)
        if req.task == "explain":
            r = a.explain(req.ids[0], req.question)
        elif req.task == "compare":
            r = a.compare(req.ids, req.question)
        else:
            r = a.summarize_experiment(req.ids[0], req.question)
        d = r.to_dict()
        d.pop("facts")
        return envelope(d, t0, mode="RESEARCH", source="stored results (read-only)")

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        return envelope(jobs.get(job_id), mode="SYSTEM")

    @app.post("/orders/live")
    def live_order():
        raise LiveTradingDisabledError("live order submission does not exist in QuantLab V1")

    return app


_APP: FastAPI | None = None


def __getattr__(name: str):
    """`uvicorn quantlab.api.app:app` builds the app lazily so importing this module has no side effects."""
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)
