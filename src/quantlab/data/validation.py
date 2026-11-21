"""Data-quality checks. Issues are flagged with severity; data is never silently modified.

Severity policy
* BLOCK – structurally unusable (missing columns / empty).
* ERROR – blocks backtests and paper trading on the snapshot (duplicates, invalid prices,
  OHLC violations, non-session rows, >2% missing sessions, inconsistent corporate actions,
  missing provenance).
* WARN  – usable but must be reported (few missing sessions, stale tail, extreme returns,
  zero volume, flatlines).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.data import calendar
from quantlab.domain.models import (
    BAR_COLUMNS,
    DataQualityIssue,
    PriceBasis,
    QualityReport,
    Severity,
)

DEFAULT_QUALITY = {
    "missing_fraction_error": 0.02,
    "extreme_return_threshold": 0.25,
    "stale_sessions_warn": 5,
    "flatline_sessions_warn": 5,
}


def _dates(s: pd.Series, limit: int = 10) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in list(s)[:limit]]


def run_quality_checks(
    bars: pd.DataFrame,
    price_basis: str,
    requested_start: str | None = None,
    requested_end: str | None = None,
    expected_source: str | None = None,
    params: dict[str, Any] | None = None,
) -> QualityReport:
    p = {**DEFAULT_QUALITY, **(params or {})}
    issues: list[DataQualityIssue] = []
    checks: list[str] = []

    def add(issue_type, severity, message, symbol=None, session_date=None, row_count=0, **details):
        issues.append(
            DataQualityIssue(issue_type, severity, message, symbol, session_date, int(row_count), details)
        )

    checks.append("schema")
    missing_cols = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing_cols or len(bars) == 0:
        add("SCHEMA", Severity.BLOCK, f"missing columns {missing_cols}" if missing_cols else "no rows")
        return QualityReport(issues, len(bars), [], checks)

    symbols = sorted(bars["symbol"].unique().tolist())
    df = bars.copy()
    df["session_date"] = pd.to_datetime(df["session_date"])

    checks.append("provenance")
    no_src = df["source"].isna() | (df["source"].astype(str).str.len() == 0)
    if no_src.any():
        add("PROVENANCE_MISSING", Severity.ERROR, "rows without a source", row_count=no_src.sum())
    if expected_source is not None:
        bad = df["source"].astype(str) != expected_source
        if bad.any():
            add(
                "PROVENANCE_MISMATCH",
                Severity.ERROR,
                f"rows whose source differs from snapshot provider {expected_source!r}",
                row_count=bad.sum(),
                sources=sorted(df.loc[bad, "source"].astype(str).unique().tolist()),
            )

    checks.append("duplicates")
    dup = df.duplicated(["symbol", "session_date"], keep=False)
    if dup.any():
        for sym, g in df[dup].groupby("symbol"):
            add("DUPLICATE_BAR", Severity.ERROR, "duplicate (symbol, session) rows", sym,
                row_count=len(g), dates=_dates(g["session_date"].drop_duplicates()))

    checks.append("non_finite")
    num_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    nonfinite = ~np.isfinite(df[num_cols].to_numpy(dtype=float)).all(axis=1)
    if nonfinite.any():
        for sym, g in df[nonfinite].groupby("symbol"):
            add("NON_FINITE", Severity.ERROR, "NaN/inf in OHLCV", sym, row_count=len(g),
                dates=_dates(g["session_date"]))

    checks.append("nonpositive_price")
    nonpos = (df[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1)
    if nonpos.any():
        for sym, g in df[nonpos].groupby("symbol"):
            add("NONPOSITIVE_PRICE", Severity.ERROR, "price <= 0", sym, row_count=len(g),
                dates=_dates(g["session_date"]))

    checks.append("ohlc_bounds")
    tol = 1e-9
    bad_ohlc = (
        (df["low"] > df[["open", "close"]].min(axis=1) + tol)
        | (df["high"] < df[["open", "close"]].max(axis=1) - tol)
        | (df["low"] > df["high"] + tol)
    )
    if bad_ohlc.any():
        for sym, g in df[bad_ohlc].groupby("symbol"):
            add("OHLC_BOUNDS", Severity.ERROR, "violates low <= open/close <= high", sym,
                row_count=len(g), dates=_dates(g["session_date"]))

    checks.append("volume")
    neg_vol = df["volume"] < 0
    if neg_vol.any():
        add("NEGATIVE_VOLUME", Severity.ERROR, "volume < 0", row_count=neg_vol.sum())
    zero_vol = df["volume"] == 0
    if zero_vol.any():
        for sym, g in df[zero_vol].groupby("symbol"):
            add("ZERO_VOLUME", Severity.WARN, "volume == 0", sym, row_count=len(g),
                dates=_dates(g["session_date"]))

    checks.append("corporate_actions_sanity")
    if (df["dividend"] < 0).any():
        add("NEGATIVE_DIVIDEND", Severity.ERROR, "dividend < 0", row_count=(df["dividend"] < 0).sum())
    if (df["split_ratio"] <= 0).any():
        add("INVALID_SPLIT", Severity.ERROR, "split_ratio <= 0", row_count=(df["split_ratio"] <= 0).sum())

    checks.append("calendar_sessions")
    checks.append("missing_sessions")
    checks.append("stale_data")
    checks.append("extreme_returns")
    checks.append("flatline")
    checks.append("split_consistency")
    checks.append("adjustment_consistency")
    for sym, g in df.groupby("symbol"):
        g = g.drop_duplicates("session_date").sort_values("session_date")
        dates = pd.DatetimeIndex(g["session_date"])
        first = pd.Timestamp(requested_start) if requested_start else dates.min()
        last = pd.Timestamp(requested_end) if requested_end else dates.max()
        expected = calendar.sessions(first, last)

        non_sess = dates.difference(expected)
        non_sess = non_sess[(non_sess >= first) & (non_sess <= last)]
        if len(non_sess):
            add("NON_SESSION_ROW", Severity.ERROR, "rows on non-trading days (weekend/holiday)", sym,
                row_count=len(non_sess), dates=_dates(non_sess))

        # Missing sessions are measured between the first available bar (or requested start,
        # whichever is later) and the requested end — inception before the start is not an error.
        lo = max(dates.min(), first)
        exp_inside = expected[(expected >= lo) & (expected <= last)]
        missing = exp_inside.difference(dates)
        if len(missing):
            frac = len(missing) / max(len(exp_inside), 1)
            sev = Severity.ERROR if frac > p["missing_fraction_error"] else Severity.WARN
            add("MISSING_SESSIONS", sev, f"{len(missing)} expected XNYS sessions missing ({frac:.2%})", sym,
                row_count=len(missing), missing_fraction=round(frac, 6), dates=_dates(missing, 25))

        if requested_end is not None and len(exp_inside):
            tail = exp_inside[exp_inside > dates.max()]
            if len(tail) > p["stale_sessions_warn"]:
                add("STALE_DATA", Severity.WARN,
                    f"last bar {dates.max():%Y-%m-%d} is {len(tail)} sessions before requested end", sym,
                    row_count=len(tail))

        valid = g[(g["adj_close"] > 0) & np.isfinite(g["adj_close"])]
        rets = valid["adj_close"].pct_change()
        extreme = rets.abs() > p["extreme_return_threshold"]
        if extreme.any():
            add("EXTREME_RETURN", Severity.WARN,
                f"|daily total return| > {p['extreme_return_threshold']:.0%} (flagged, not removed)", sym,
                row_count=int(extreme.sum()), dates=_dates(valid.loc[extreme, "session_date"]),
                returns=[round(float(x), 4) for x in rets[extreme].head(10)])

        c = g["close"].to_numpy()
        run = 1
        longest = 1
        for i in range(1, len(c)):
            run = run + 1 if c[i] == c[i - 1] else 1
            longest = max(longest, run)
        if longest >= p["flatline_sessions_warn"]:
            add("FLATLINE", Severity.WARN, f"{longest} identical consecutive closes", sym, row_count=longest)

        # Split consistency: compare the raw close jump on split ex-dates with the ratio.
        closes = g["close"].to_numpy()
        ratios = g["split_ratio"].to_numpy()
        for i in np.where(ratios != 1.0)[0]:
            if i == 0 or not (closes[i - 1] > 0 and closes[i] > 0):
                continue
            jump = closes[i] / closes[i - 1]
            looks_unadjusted = abs(jump * ratios[i] - 1) < 0.15
            d = g["session_date"].iloc[i].strftime("%Y-%m-%d")
            if price_basis == PriceBasis.SPLIT_ADJUSTED and looks_unadjusted and ratios[i] >= 1.5:
                add("SPLIT_INCONSISTENT", Severity.ERROR,
                    "split-adjusted basis but price jump matches the split ratio (double adjustment risk)",
                    sym, d, 1, ratio=float(ratios[i]), jump=round(float(jump), 4))
            if price_basis == PriceBasis.UNADJUSTED and not looks_unadjusted:
                add("SPLIT_INCONSISTENT", Severity.ERROR,
                    "unadjusted basis but no price jump at recorded split", sym, d, 1,
                    ratio=float(ratios[i]), jump=round(float(jump), 4))

        # Adjustment consistency: adj_close / close factor changes should be explained by
        # recorded dividends/splits (tolerance absorbs provider rounding).
        f = (g["adj_close"] / g["close"]).to_numpy()
        divs = g["dividend"].to_numpy()
        unexplained = []
        for i in range(1, len(f)):
            if not (np.isfinite(f[i]) and np.isfinite(f[i - 1]) and f[i] > 0):
                continue
            step = f[i - 1] / f[i]  # factor at i-1 relative to i
            expected_step = 1.0
            if divs[i] > 0 and closes[i - 1] > 0:
                expected_step *= 1 - divs[i] / closes[i - 1]
            if price_basis == PriceBasis.UNADJUSTED and ratios[i] != 1.0:
                expected_step *= 1 / ratios[i]
            if abs(step / expected_step - 1) > 0.005:
                unexplained.append(g["session_date"].iloc[i])
        if unexplained:
            add("ADJUSTMENT_INCONSISTENT", Severity.WARN,
                "adj_close/close factor changes not explained by recorded dividends/splits", sym,
                row_count=len(unexplained), dates=_dates(pd.Series(unexplained)))

    return QualityReport(issues, len(bars), symbols, checks)
