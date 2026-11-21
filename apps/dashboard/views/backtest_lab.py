import dataclasses
import json

import pandas as pd
import streamlit as st

from quantlab.backtest.engine import BacktestConfig
from quantlab.backtest.service import compare_runs, get_run, list_runs, load_artifact, run_and_store
from quantlab.core.errors import QuantLabError
from quantlab.reports import charts
from quantlab.reports.export import backtest_html, backtest_markdown, metrics_table
from quantlab.strategies.registry import STRATEGIES
from quantlab.ui.common import (
    backtest_banner,
    df_download,
    fmt_num,
    fmt_pct,
    get_ctx,
    safety_banner,
    snapshot_selector,
)

ctx = get_ctx()
st.title("Backtest Lab")
safety_banner()

tab_run, tab_results, tab_compare = st.tabs(["Run", "Results", "Compare"])

with tab_run:
    snap = snapshot_selector(ctx, "bl_snap", require_tradable=True)
    if snap is not None:
        with st.form("bt_form"):
            c1, c2 = st.columns(2)
            sid = c1.selectbox("Strategy", list(STRATEGIES), format_func=lambda s: STRATEGIES[s][0].name)
            default = snap.symbols if sid == "momentum" else ["SPY"] if "SPY" in snap.symbols else snap.symbols[:1]
            syms = c2.multiselect("Symbols", snap.symbols, default=default)
            c1, c2, c3, c4 = st.columns(4)
            start = c1.text_input("Start (blank = first session)", "")
            end = c2.text_input("End (blank = last session)", "")
            cash = c3.number_input("Initial cash (USD)", value=10_000.0, min_value=1.0, step=1000.0)
            mode = c4.selectbox("Price mode", ["raw_with_dividends", "adjusted_total_return"])
            c1, c2, c3, c4 = st.columns(4)
            comm = c1.number_input("Commission (bps)", value=1.0, min_value=0.0)
            slip = c2.number_input("Slippage (bps)", value=2.0, min_value=0.0)
            minc = c3.number_input("Min commission (USD)", value=0.0, min_value=0.0)
            rf = c4.number_input("Risk-free rate (annual)", value=0.0, format="%.4f")
            pjson = st.text_area("Strategy parameters (JSON; blank = defaults)",
                                 json.dumps({f.name: getattr(STRATEGIES[sid][1](), f.name)
                                             for f in dataclasses.fields(STRATEGIES[sid][1]) if f.name != "weights"}))
            go = st.form_submit_button("Run backtest")
        if go:
            try:
                params = json.loads(pjson) if pjson.strip() else {}
                cfg = BacktestConfig(initial_cash=cash, commission_bps=comm, slippage_bps=slip, min_commission=minc,
                                     price_mode=mode, risk_free_rate=rf, start=start or None, end=end or None)
                with st.spinner("Running backtest (net, gross and benchmark)…"):
                    out = run_and_store(ctx, sid, snap.id, syms, params, cfg)
                st.session_state["bl_selected"] = out.run_id
                st.success(f"Run {out.run_id} completed. See the Results tab.")
            except (QuantLabError, json.JSONDecodeError, ValueError) as exc:
                st.error(f"Backtest failed: {exc}")

runs = [r for r in list_runs(ctx) if r["status"] == "COMPLETED"]
labels = {f"{r['id'][:8]} · {r['strategy_name']} · {', '.join(r['config']['symbols'])} · "
          f"{'SYNTHETIC' if r['is_synthetic'] else 'REAL'} · {r['created_at'][:19]}": r["id"] for r in runs}

with tab_results:
    if not runs:
        st.info("No completed backtests yet.")
    else:
        keys = list(labels)
        sel = st.session_state.get("bl_selected")
        idx = next((i for i, k in enumerate(keys) if labels[k] == sel), 0)
        rid = labels[st.selectbox("Run", keys, index=idx, key="bl_run")]
        run = get_run(ctx, rid)
        res = run["result"]
        backtest_banner(res["is_synthetic"])
        n, g = res["metrics"]["net"], res["metrics"]["gross"]
        b = res["metrics"].get("benchmark") or {}
        c = st.columns(6)
        c[0].metric("Total return (net)", fmt_pct(n["total_return"]), delta=f"gross {fmt_pct(g['total_return'])}",
                    delta_color="off")
        c[1].metric("CAGR", fmt_pct(n["cagr"]), delta=f"bench {fmt_pct(b.get('cagr'))}", delta_color="off")
        c[2].metric("Sharpe", fmt_num(n["sharpe"]),
                    delta=f"95% CI {fmt_num(n['sharpe_ci95_low'])}…{fmt_num(n['sharpe_ci95_high'])}", delta_color="off")
        c[3].metric("Max drawdown", fmt_pct(n["max_drawdown"]))
        c[4].metric("Turnover (x/yr)", fmt_num(n["turnover_annual"]))
        c[5].metric("Time in market", fmt_pct(n["time_in_market"]))

        eq = load_artifact(ctx, rid, "equity").set_index("session")
        curves = {f"{res['strategy']['name']} (net)": eq["nav"],
                  "gross (no costs)": load_artifact(ctx, rid, "equity_gross").set_index("session")["nav"]}
        if res.get("benchmark_symbol"):
            curves[f"{res['benchmark_symbol']} buy & hold"] = load_artifact(ctx, rid, "benchmark_equity").set_index("session")["nav"]
        st.plotly_chart(charts.equity_and_drawdown(curves, res["is_synthetic"]), width="stretch")

        st.subheader("Metrics")
        st.dataframe(metrics_table(res["metrics"]), hide_index=True, width="stretch")
        if res["metrics"].get("relative"):
            st.json(res["metrics"]["relative"])
        trades = load_artifact(ctx, rid, "trades")
        st.subheader(f"Trade ledger ({len(trades)} fills)")
        st.dataframe(trades, hide_index=True, width="stretch")
        orders = load_artifact(ctx, rid, "orders")
        bad = orders[orders.status.isin(["REJECTED", "EXPIRED"])]
        if len(bad):
            st.warning(f"{len(bad)} orders rejected/expired")
            st.dataframe(bad, hide_index=True)
        if res["warnings"]:
            st.warning("\n".join(f"- {w}" for w in res["warnings"]))
        with st.expander("Assumptions & provenance"):
            st.markdown("\n".join(f"- {a}" for a in res["assumptions"]))
            st.json({"snapshot": res["snapshot"], "config": res["config"], "config_hash": res["config_hash"],
                     "code_version": res["code_version"], "seed": res["seed"]})
        st.subheader("Export")
        hdr = res["label"] + (" | " + res["data_label"] if res["is_synthetic"] else "")
        c = st.columns(4)
        with c[0]:
            df_download(eq.reset_index(), "Equity CSV", f"equity_{rid[:8]}.csv", "dl_eq", hdr)
        with c[1]:
            df_download(trades, "Trades CSV", f"trades_{rid[:8]}.csv", "dl_tr", hdr)
        c[2].download_button("Report (Markdown)", backtest_markdown(run), f"report_{rid[:8]}.md", key="dl_md")
        c[3].download_button("Report (HTML)", backtest_html(ctx, run), f"report_{rid[:8]}.html", key="dl_html")

with tab_compare:
    if len(runs) < 2:
        st.info("Run at least two backtests to compare.")
    else:
        chosen = st.multiselect("Runs", list(labels), default=list(labels)[: min(3, len(labels))], key="bl_cmp")
        if chosen:
            rows, curves, synth = [], {}, set()
            for k in chosen:
                r = get_run(ctx, labels[k])
                res = r["result"]
                synth.add(res["is_synthetic"])
                n = res["metrics"]["net"]
                rows.append({"run": k.split(" · ")[0], "strategy": res["strategy"]["name"],
                             "symbols": ", ".join(res["symbols"]), "window": f"{n['start']}→{n['end']}",
                             "total return": fmt_pct(n["total_return"]), "CAGR": fmt_pct(n["cagr"]),
                             "Sharpe": fmt_num(n["sharpe"]),
                             "Sharpe 95% CI": f"{fmt_num(n['sharpe_ci95_low'])}…{fmt_num(n['sharpe_ci95_high'])}",
                             "max DD": fmt_pct(n["max_drawdown"]), "turnover": fmt_num(n["turnover_annual"]),
                             "fills": n["n_fills"], "costs USD": fmt_num(n["total_commission"] + n["total_slippage_cost"])})
                curves[f"{k.split(' · ')[0]} {res['strategy']['name']}"] = load_artifact(ctx, labels[k], "equity").set_index("session")["nav"]
            if len(synth) > 1:
                st.error("You are comparing runs on SYNTHETIC and REAL data — not comparable.")
            backtest_banner(True in synth)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Overlapping Sharpe confidence intervals mean the difference is not statistically distinguishable.")
            st.subheader("Statistical tests (first selected run = reference)")
            trials = st.number_input("Configurations really tried (for the deflated Sharpe)", min_value=len(chosen),
                                     value=len(chosen), step=1, key="bl_trials")
            try:
                stt = compare_runs(ctx, [labels[k] for k in chosen], int(trials))
                pw = pd.DataFrame(stt["pairwise"])
                st.dataframe(pw, hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame(stt["deflated_sharpe"]).T.reset_index(names="run"), hide_index=True,
                             width="stretch")
                st.caption(" ".join(stt["notes"]) + (" " + " ".join(stt["warnings"]) if stt["warnings"] else ""))
            except QuantLabError as exc:
                st.info(f"Statistical tests unavailable: {exc}")
            st.plotly_chart(charts.equity_and_drawdown(curves, True in synth, "Comparison"), width="stretch")
