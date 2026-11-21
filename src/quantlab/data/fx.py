"""USD→CAD reporting (trading stays in USD).

QuantLab accounts and backtests are denominated in USD. For a Canadian investor holding USD assets
unhedged, the CAD value is ``NAV_CAD(t) = NAV_USD(t) × USDCAD(t)`` where USDCAD is CAD per 1 USD.
This module ingests a daily USDCAD series as an immutable FX snapshot and converts NAV series.

Rules
* FX bars are aligned to XNYS sessions (FX trades on US holidays too; those rows are dropped).
* Conversion uses the rate of the same session, or the most recent earlier rate for at most
  ``max_stale`` sessions (as-of, never a future rate). Otherwise conversion fails loudly.
* Real-data results are never converted with synthetic FX (and vice versa).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from quantlab.context import AppContext
from quantlab.core.errors import DataUnavailableError, InputValidationError, NotFoundError
from quantlab.data import calendar
from quantlab.data.ingestion import validate_date_range
from quantlab.data.providers import canonicalize_bars
from quantlab.domain.models import DataQualityIssue, DataSnapshot, QualityReport, Severity

FX_SYMBOL = "USDCAD"
YAHOO_TICKER = "CAD=X"
ASSET_CLASS = "fx"


def is_fx_snapshot(s: DataSnapshot) -> bool:
    return (s.parameters or {}).get("asset_class") == ASSET_CLASS


def _bars_from_rates(sess: pd.DatetimeIndex, o, h, lo, c, source: str) -> pd.DataFrame:
    return pd.DataFrame({"symbol": FX_SYMBOL, "session_date": sess, "open": o, "high": h, "low": lo, "close": c,
                         "adj_close": c, "volume": 0.0, "dividend": 0.0, "split_ratio": 1.0, "currency": "CAD",
                         "source": source})


def synthetic_fx(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    sess = calendar.sessions(start, end)
    rng = np.random.default_rng([seed, 4242])
    r = rng.normal(0, 0.07 / np.sqrt(252), len(sess))
    c = np.round(1.30 * np.exp(np.cumsum(r)), 5)
    o = np.round(np.r_[c[0], c[:-1]], 5)
    return _bars_from_rates(sess, o, np.maximum(o, c), np.minimum(o, c), c, "synthetic")


def _yf_download(ticker: str, start: str, end_exclusive: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).history(start=start, end=end_exclusive, auto_adjust=False, actions=False)


def yahoo_fx(start: str, end: str, downloader: Callable[[str, str, str], pd.DataFrame] | None = None) -> pd.DataFrame:
    dl = downloader or _yf_download
    try:
        raw = dl(YAHOO_TICKER, start, (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    except Exception as exc:
        raise DataUnavailableError(f"USDCAD download failed: {type(exc).__name__}: {exc}") from exc
    if raw is None or raw.empty:
        raise DataUnavailableError("USDCAD download returned no rows")
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    idx = idx.normalize()
    df = pd.DataFrame({"o": raw["Open"].to_numpy(float), "h": raw["High"].to_numpy(float),
                       "l": raw["Low"].to_numpy(float), "c": raw["Close"].to_numpy(float)}, index=idx)
    df = df[~df.index.duplicated(keep="last")]
    sess = calendar.sessions(start, end)
    df = df[df.index.isin(sess)]  # align to XNYS sessions
    return _bars_from_rates(pd.DatetimeIndex(df.index), df["o"], df["h"], df["l"], df["c"], "yfinance")


def fx_quality(bars: pd.DataFrame, start: str, end: str) -> QualityReport:
    """FX-specific checks (volume and corporate-action checks do not apply)."""
    issues: list[DataQualityIssue] = []
    c = bars["close"].astype(float)
    if bars["session_date"].duplicated().any():
        issues.append(DataQualityIssue("DUPLICATE_BAR", Severity.ERROR, "duplicate FX sessions", FX_SYMBOL))
    bad = ~np.isfinite(c) | (c <= 0)
    if bad.any():
        issues.append(DataQualityIssue("NONPOSITIVE_PRICE", Severity.ERROR, "non-finite or non-positive FX rate",
                                       FX_SYMBOL, row_count=int(bad.sum())))
    expected = calendar.sessions(max(pd.Timestamp(start), bars["session_date"].min()), end)
    missing = expected.difference(pd.DatetimeIndex(bars["session_date"]))
    if len(missing):
        frac = len(missing) / max(len(expected), 1)
        issues.append(DataQualityIssue("MISSING_SESSIONS", Severity.ERROR if frac > 0.02 else Severity.WARN,
                                       f"{len(missing)} XNYS sessions without an FX rate ({frac:.2%}); conversion "
                                       "uses the previous rate for up to 5 sessions", FX_SYMBOL,
                                       row_count=len(missing),
                                       details={"dates": [d.strftime("%Y-%m-%d") for d in missing[:25]]}))
    jumps = c.pct_change().abs() > 0.05
    if jumps.any():
        issues.append(DataQualityIssue("EXTREME_RETURN", Severity.WARN, "daily USDCAD move > 5% (flagged)",
                                       FX_SYMBOL, row_count=int(jumps.sum())))
    return QualityReport(issues, len(bars), [FX_SYMBOL],
                         ["duplicates", "positive_rate", "missing_sessions", "extreme_moves"])


def ingest_fx(ctx: AppContext, start: str, end: str | None = None, synthetic: bool = False, seed: int = 7,
              downloader: Callable | None = None) -> tuple[DataSnapshot, QualityReport, bool]:
    s, e = validate_date_range(start, end, allow_future=synthetic)
    bars = synthetic_fx(s, e, seed) if synthetic else yahoo_fx(s, e, downloader)
    bars = canonicalize_bars(bars)
    params: dict[str, Any] = {"asset_class": ASSET_CLASS, "pair": FX_SYMBOL, "quote": "CAD per 1 USD",
                              "start": s, "end": e, "source_ticker": None if synthetic else YAHOO_TICKER}
    if synthetic:
        params["seed"] = seed
    snap, created = ctx.snapshots.create(bars, provider="synthetic-fx" if synthetic else "yfinance-fx",
                                         price_basis="unadjusted", is_synthetic=synthetic, parameters=params,
                                         retrieved_at=datetime.now(UTC).isoformat())
    rep = fx_quality(bars, s, e)
    ctx.snapshots.record_quality(snap.id, rep)
    return snap, rep, created


def latest_fx_snapshot(ctx: AppContext, synthetic: bool, start: str | None = None,
                       end: str | None = None) -> DataSnapshot:
    """Most recent valid FX snapshot of the same realness that covers [start, end]."""
    for s in ctx.snapshots.list(include_derived=False):
        if not (is_fx_snapshot(s) and s.is_synthetic == synthetic):
            continue
        if ctx.snapshots.quality_status(s.id) == "FAILED":
            continue
        if (start and pd.Timestamp(s.start) > pd.Timestamp(start)) or (end and pd.Timestamp(s.end) < pd.Timestamp(end)):
            continue
        return s
    kind = "synthetic" if synthetic else "real"
    hint = " --fixture synthetic" if synthetic else ""
    span = f" covering {start} → {end}" if start and end else ""
    raise NotFoundError(f"no {kind} USDCAD snapshot{span}; run `quantlab data ingest-fx --start <date>{hint}`")


def fx_series(ctx: AppContext, snapshot_id: str) -> pd.Series:
    bars = ctx.snapshots.load(snapshot_id)
    return bars.set_index("session_date")["close"].astype(float).sort_index()


def to_cad(nav_usd: pd.Series, usdcad: pd.Series, max_stale: int = 5) -> tuple[pd.Series, dict[str, Any]]:
    idx = pd.DatetimeIndex(nav_usd.index)
    exact = usdcad.reindex(idx)
    union = usdcad.index.union(idx)
    asof = usdcad.reindex(union).ffill(limit=max_stale).reindex(idx)
    if asof.isna().any():
        first = idx[asof.isna()][0]
        raise InputValidationError(f"no USDCAD rate on or within {max_stale} sessions before {first:%Y-%m-%d}; "
                                   "ingest FX covering the whole period")
    return nav_usd.astype(float) * asof.to_numpy(), {"sessions": len(idx), "stale_rate_sessions": int(exact.isna().sum()),
                                                      "max_stale_sessions": max_stale}
