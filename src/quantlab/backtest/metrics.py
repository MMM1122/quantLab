"""Performance metrics computed ONLY from stored equity curves and trade ledgers.

Conventions: daily sampling, 252 sessions/year, arithmetic daily returns of NAV.
Sharpe = mean(daily excess) / std(daily excess, ddof=1) * sqrt(252), with the annual
risk-free rate converted geometrically to daily. Sharpe standard error uses Lo (2002) iid
approximation: SE(SR_daily) = sqrt((1 + SR_daily^2 / 2) / n).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

PERIODS = 252


def drawdown_series(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0


def _max_dd_duration(dd: pd.Series) -> int:
    longest = cur = 0
    for v in dd.to_numpy():
        cur = cur + 1 if v < -1e-12 else 0
        longest = max(longest, cur)
    return int(longest)


def performance_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    nav = equity["nav"].astype(float)
    n = len(nav)
    rets = nav.pct_change().dropna()
    years = (n - 1) / PERIODS if n > 1 else float("nan")
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years and years > 0 and nav.iloc[-1] > 0 else float("nan")
    rf_d = (1 + risk_free_rate) ** (1 / PERIODS) - 1
    ex = rets - rf_d
    sd = ex.std(ddof=1) if len(ex) > 1 else float("nan")
    sharpe_d = ex.mean() / sd if sd and sd > 0 else float("nan")
    sharpe = sharpe_d * math.sqrt(PERIODS) if np.isfinite(sharpe_d) else float("nan")
    se = math.sqrt((1 + sharpe_d**2 / 2) / len(ex)) * math.sqrt(PERIODS) if np.isfinite(sharpe_d) and len(ex) else float("nan")
    downside = ex[ex < 0]
    dsd = math.sqrt((downside**2).sum() / len(ex)) if len(ex) else float("nan")
    sortino = ex.mean() / dsd * math.sqrt(PERIODS) if dsd and dsd > 0 else float("nan")
    dd = drawdown_series(nav)
    mdd = float(dd.min())
    trough = dd.idxmin()
    peak_date = nav.loc[:trough].idxmax()
    vol = rets.std(ddof=1) * math.sqrt(PERIODS) if len(rets) > 1 else float("nan")

    traded = float(trades["notional"].abs().sum()) if trades is not None and len(trades) else 0.0
    fees = float(trades["commission"].sum()) if trades is not None and len(trades) else 0.0
    slip = float(trades["slippage_cost"].sum()) if trades is not None and len(trades) else 0.0
    avg_nav = float(nav.mean())
    turnover = traded / avg_nav / years if years and years > 0 and avg_nav > 0 else float("nan")
    tim = float((equity["gross_exposure"] > 1e-9).mean()) if "gross_exposure" in equity else float("nan")
    out = {
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "n_sessions": int(n),
        "years": years,
        "initial_nav": float(nav.iloc[0]),
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(total),
        "cagr": float(cagr),
        "annual_volatility": float(vol),
        "sharpe": float(sharpe),
        "sharpe_se": float(se),
        "sharpe_ci95_low": float(sharpe - 1.96 * se) if np.isfinite(se) else float("nan"),
        "sharpe_ci95_high": float(sharpe + 1.96 * se) if np.isfinite(se) else float("nan"),
        "sortino": float(sortino),
        "max_drawdown": mdd,
        "max_drawdown_peak": peak_date.strftime("%Y-%m-%d"),
        "max_drawdown_trough": trough.strftime("%Y-%m-%d"),
        "max_drawdown_duration_sessions": _max_dd_duration(dd),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan"),
        "turnover_annual": float(turnover),
        "time_in_market": tim,
        "traded_notional": traded,
        "total_commission": fees,
        "total_slippage_cost": slip,
        "total_dividends": float(equity["dividends"].sum()) if "dividends" in equity else 0.0,
        "n_fills": int(len(trades)) if trades is not None else 0,
        "risk_free_rate": risk_free_rate,
        "periods_per_year": PERIODS,
    }
    if orders is not None and len(orders):
        out["n_orders"] = int(len(orders))
        out["n_rejected_or_expired"] = int(orders["status"].isin(["REJECTED", "EXPIRED"]).sum())
    else:
        out["n_orders"] = 0
        out["n_rejected_or_expired"] = 0
    return out


def relative_metrics(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> dict[str, Any]:
    df = pd.concat([strategy_nav.rename("s"), benchmark_nav.rename("b")], axis=1).dropna()
    r = df.pct_change().dropna()
    if len(r) < 3:
        return {}
    cov = np.cov(r["s"], r["b"], ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else float("nan")
    active = r["s"] - r["b"]
    te = active.std(ddof=1) * math.sqrt(PERIODS)
    years = (len(df) - 1) / PERIODS
    s_cagr = (df["s"].iloc[-1] / df["s"].iloc[0]) ** (1 / years) - 1
    b_cagr = (df["b"].iloc[-1] / df["b"].iloc[0]) ** (1 / years) - 1
    return {
        "benchmark_total_return": float(df["b"].iloc[-1] / df["b"].iloc[0] - 1),
        "benchmark_cagr": float(b_cagr),
        "excess_cagr": float(s_cagr - b_cagr),
        "beta": float(beta),
        "correlation": float(r["s"].corr(r["b"])),
        "tracking_error": float(te),
        "information_ratio": float(active.mean() * PERIODS / te) if te > 0 else float("nan"),
    }


def clean_for_json(d: Any) -> Any:
    """Replace NaN/inf with None so stored JSON is valid and never shows fabricated numbers."""
    if isinstance(d, dict):
        return {k: clean_for_json(v) for k, v in d.items()}
    if isinstance(d, list | tuple):
        return [clean_for_json(v) for v in d]
    if isinstance(d, float | np.floating):
        return float(d) if np.isfinite(d) else None
    if isinstance(d, np.integer):
        return int(d)
    return d
