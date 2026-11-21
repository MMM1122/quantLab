"""Paper-vs-backtest consistency report for autopilot accounts.

Runs the same strategy through the backtest engine on the same snapshot, window, capital, costs
and weight cap, then compares the two NAV paths. Remaining differences are expected to come from
mechanics the backtest does not model; each is counted from stored records so the gap is
explained by evidence rather than guessed:

* volume participation cap (partial fills / expired remainders) — paper only;
* pre-trade risk rejections (e.g. cash before a pending sell settles) — paper only;
* Decimal cent rounding and commission rounded up to the cent — paper only;
* the 0.5% cap buffer used by the autopilot, and drift trims (the autopilot sells holdings that
  have risen above the cap; the backtest engine only rebalances on signal changes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.engine import BacktestConfig, run_backtest
from quantlab.backtest.market import build_panel
from quantlab.backtest.metrics import clean_for_json, performance_metrics
from quantlab.context import AppContext
from quantlab.core.errors import InputValidationError
from quantlab.core.labels import BACKTEST_LABEL, REPLAY_LABEL
from quantlab.paper.autopilot import CAP_BUFFER, StrategyAutopilot
from quantlab.strategies.registry import create_strategy


def paper_vs_backtest(ctx: AppContext, account_id: str) -> dict[str, Any]:
    ap_svc = StrategyAutopilot(ctx)
    ap = ap_svc.get(account_id)
    broker = ap_svc.broker
    acc = broker.account_state(account_id)
    paper = broker.nav_history(account_id)
    if len(paper) < 3:
        raise InputValidationError("advance the paper account at least a few sessions first")
    paper = paper.set_index(pd.to_datetime(paper["session_date"]))["nav"].astype(float)
    snap = ctx.snapshots.get(acc["snapshot_id"])
    cfg_risk = acc["risk_limits"]
    panel = build_panel(ctx.snapshots.load(snap.id), snap.price_basis, symbols=ap["symbols"])
    cfg = BacktestConfig(initial_cash=float(acc["initial_cash"]), commission_bps=float(cfg_risk["commission_bps"]),
                         slippage_bps=float(cfg_risk["slippage_bps"]), min_commission=float(cfg_risk["min_commission"]),
                         max_weight=min(1.0, float(cfg_risk["max_position_weight"]) * CAP_BUFFER),
                         start=paper.index[0].strftime("%Y-%m-%d"), end=paper.index[-1].strftime("%Y-%m-%d"))
    bt = run_backtest(panel, create_strategy(ap["strategy"], ap["params"]), cfg)
    bnav = bt.equity["nav"]
    df = pd.concat([paper.rename("paper"), bnav.rename("backtest")], axis=1).dropna()
    rel = df["paper"] / df["backtest"] - 1
    rp, rb = df["paper"].pct_change().dropna(), df["backtest"].pct_change().dropna()
    orders = pd.DataFrame(broker.list_orders(account_id))
    auto = orders[orders["client_order_id"].str.startswith("auto:")] if len(orders) else orders
    risk = pd.DataFrame(broker.risk_events(account_id))
    explain = {
        "paper_orders": int(len(auto)),
        "paper_rejected": int((auto["status"] == "REJECTED").sum()) if len(auto) else 0,
        "paper_canceled_unfilled": int(((auto["status"] == "CANCELED") & (auto["filled_quantity"] == "0")).sum())
        if len(auto) else 0,
        "paper_canceled_partially_filled": int(((auto["status"] == "CANCELED") & (auto["filled_quantity"] != "0")).sum())
        if len(auto) else 0,
        "autopilot_drift_trims": int((ap["state"] or {}).get("trim_count", 0)),
        "autopilot_targets_capped": int((ap["state"] or {}).get("cap_count", 0)),
        "risk_events_by_rule": risk["rule"].value_counts().to_dict() if len(risk) else {},
        "backtest_fills": int(len(bt.trades)),
        "backtest_rejected_or_expired": int(bt.orders["status"].isin(["REJECTED", "EXPIRED"]).sum())
        if len(bt.orders) else 0,
    }
    pm = performance_metrics(pd.DataFrame({"nav": df["paper"], "gross_exposure": np.nan}))
    bm = performance_metrics(bt.equity.loc[df.index])
    return clean_for_json({
        "account_id": account_id, "strategy": ap["strategy"], "params": ap["params"], "symbols": ap["symbols"],
        "labels": [REPLAY_LABEL, BACKTEST_LABEL], "is_synthetic": snap.is_synthetic,
        "window": [df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")], "n_sessions": len(df),
        "final_nav": {"paper": float(df["paper"].iloc[-1]), "backtest": float(df["backtest"].iloc[-1])},
        "final_relative_difference": float(rel.iloc[-1]),
        "max_abs_relative_difference": float(rel.abs().max()),
        "daily_return_correlation": float(rp.corr(rb)) if len(rp) > 2 else None,
        "total_return": {"paper": pm["total_return"], "backtest": bm["total_return"]},
        "max_drawdown": {"paper": pm["max_drawdown"], "backtest": bm["max_drawdown"]},
        "explanations": explain,
        "notes": ["Differences come from mechanics only the paper side models: volume participation cap, "
                  "pre-trade risk rejections, drift trims above the weight cap, Decimal cent rounding and "
                  "commission rounded up to the cent."],
        "_series": df,
    })
