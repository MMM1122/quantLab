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
from quantlab.core.labels import BACKTEST_LABEL, data_label
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
