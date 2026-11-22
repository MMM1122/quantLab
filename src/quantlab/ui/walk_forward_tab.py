"""Walk-forward validation tab for the Research Lab page (UI only)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from quantlab.context import AppContext
from quantlab.core.errors import QuantLabError
from quantlab.reports import charts
from quantlab.reports.export import load_walkforward_curves, walkforward_html, walkforward_markdown
from quantlab.research.dq_001 import get_experiment, list_experiments
from quantlab.research.walk_forward import OBJECTIVES, WalkForwardConfig, run_walk_forward
from quantlab.strategies.registry import STRATEGIES
from quantlab.ui.common import backtest_banner, fmt_num, fmt_pct, snapshot_selector

DEFAULT_GRIDS = {
    "sma-crossover": {"fast": [20, 50], "slow": [100, 200]},
    "rsi-mean-reversion": {"entry": [25, 30, 35], "exit": [50, 60]},
    "momentum": {"lookback": [126, 252], "top_k": [1, 2]},
    "buy-hold": {},
}


def walk_forward_tab(ctx: AppContext) -> None:
    st.markdown("Parameters are chosen on each **training** window only and then evaluated on the **next, unseen** "
                "window. The stitched test windows form the out-of-sample result.")
    snap = snapshot_selector(ctx, "wf_snap", require_tradable=True)
    if snap is not None:
        with st.form("wf_form"):
            c1, c2, c3 = st.columns(3)
            sid = c1.selectbox("Strategy", [s for s in STRATEGIES if s != "buy-hold"],
                               format_func=lambda s: STRATEGIES[s][0].name)
            default_syms = snap.symbols if sid == "momentum" else (["SPY"] if "SPY" in snap.symbols else snap.symbols[:1])
            syms = c2.multiselect("Symbols", snap.symbols, default=default_syms)
            obj = c3.selectbox("Objective", OBJECTIVES)
            grid = st.text_area("Parameter grid (JSON: name → list of values)", json.dumps(DEFAULT_GRIDS.get(sid, {})))
            c1, c2, c3, c4 = st.columns(4)
            train = c1.number_input("Train sessions", value=756, min_value=60, step=21)
            test = c2.number_input("Test sessions", value=252, min_value=20, step=21)
            emb = c3.number_input("Embargo sessions", value=0, min_value=0)
            rolling = c4.checkbox("Rolling train window (default expanding)", False)
            go = st.form_submit_button("Run walk-forward")
        if go:
            try:
                cfg = WalkForwardConfig(snapshot_id=snap.id, strategy=sid, symbols=syms, param_grid=json.loads(grid),
                                        train_sessions=int(train), test_sessions=int(test),
                                        embargo_sessions=int(emb), expanding=not rolling, objective=obj)
                bar = st.progress(0.0, "starting…")
                res = run_walk_forward(ctx, cfg, progress=lambda f, m: bar.progress(min(f, 1.0), m))
                st.session_state["wf_selected"] = res["experiment_id"]
                st.success(f"Walk-forward {res['experiment_id']} completed ({len(res['folds'])} folds).")
            except (QuantLabError, ValueError) as exc:
                st.error(f"Walk-forward failed: {exc}")

    exps = [e for e in list_experiments(ctx) if e["status"] == "COMPLETED" and e["code"] == "WALK-FORWARD"]
    if not exps:
        st.info("No completed walk-forward validations yet.")
        return
    labels = {f"{e['id'][:8]} · {e['completed_at'][:19]}": e["id"] for e in exps}
    keys = list(labels)
    sel = st.session_state.get("wf_selected")
    idx = next((i for i, k in enumerate(keys) if labels[k] == sel), 0)
    exp = get_experiment(ctx, labels[st.selectbox("Walk-forward run", keys, index=idx, key="wf_pick")])
    r = exp["results"]
    backtest_banner(r["is_synthetic"])
    m, b = r["oos_metrics"], r.get("benchmark_metrics") or {}
    c = st.columns(5)
    c[0].metric("OOS total return", fmt_pct(m["total_return"]), delta=f"bench {fmt_pct(b.get('total_return'))}",
                delta_color="off")
    c[1].metric("OOS CAGR", fmt_pct(m["cagr"]))
    c[2].metric("OOS Sharpe", fmt_num(m["sharpe"]),
                delta=f"95% CI {fmt_num(m['sharpe_ci95_low'])}…{fmt_num(m['sharpe_ci95_high'])}", delta_color="off")
    c[3].metric("OOS max drawdown", fmt_pct(m["max_drawdown"]))
    c[4].metric("Mean OOS rank of chosen params", fmt_num(r.get("mean_oos_percentile_rank")))
    oos, bench = load_walkforward_curves(exp)
    curves = {"out-of-sample (chosen params)": oos}
    if bench is not None:
        curves["benchmark buy & hold"] = bench
    st.plotly_chart(charts.equity_and_drawdown(curves, r["is_synthetic"], f"Walk-forward {r['strategy']}"),
                    width="stretch")
    folds = pd.DataFrame(r["folds"])
    folds["chosen_params"] = folds["chosen_params"].apply(json.dumps)
    st.dataframe(folds, hide_index=True, width="stretch")
    st.markdown("\n".join(f"- {f}" for f in r["findings"]))
    dl = st.columns(2)
    dl[0].download_button("Report (Markdown)", walkforward_markdown(exp), f"walkforward_{exp['id'][:8]}.md", key="wf_md")
    dl[1].download_button("Report (HTML)", walkforward_html(exp, oos, bench), f"walkforward_{exp['id'][:8]}.html",
                         key="wf_html")
