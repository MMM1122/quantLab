import pandas as pd
import streamlit as st

from quantlab.core.errors import QuantLabError
from quantlab.core.labels import RESEARCH_ONLY_LABEL
from quantlab.reports import charts
from quantlab.reports.export import experiment_html, experiment_markdown, experiment_table
from quantlab.research.dq_001 import (
    HYPOTHESIS,
    MISSING_DATA_RULES,
    DQ001Config,
    experiment_runs,
    get_experiment,
    list_experiments,
    run_dq001,
)
from quantlab.research.perturbations import DEFAULT_SCENARIOS
from quantlab.research.policies import POLICIES
from quantlab.strategies.registry import STRATEGIES
from quantlab.ui.common import backtest_banner, df_download, get_ctx, safety_banner, snapshot_selector

ctx = get_ctx()
st.title("Research Lab — DQ-001")
safety_banner()
st.markdown(f"**Hypothesis.** {HYPOTHESIS}")
with st.expander("Missing-data rules and policies"):
    st.markdown("\n".join(f"- {r}" for r in MISSING_DATA_RULES))
    st.caption(f"`naive_ffill` is a {RESEARCH_ONLY_LABEL}.")

tab_run, tab_res = st.tabs(["Run experiment", "Results"])
with tab_run:
    snap = snapshot_selector(ctx, "rl_snap", require_tradable=True)
    if snap is not None:
        with st.form("dq_form"):
            c1, c2 = st.columns(2)
            sym = c1.selectbox("Symbol", snap.symbols, index=snap.symbols.index("SPY") if "SPY" in snap.symbols else 0)
            strats = c2.multiselect("Strategies", [s for s in STRATEGIES if s != "momentum"],
                                    default=["sma-crossover", "rsi-mean-reversion"])
            scen = st.multiselect("Scenarios (clean baseline always included)",
                                  [s.name for s in DEFAULT_SCENARIOS if not s.is_clean],
                                  default=[s.name for s in DEFAULT_SCENARIOS if not s.is_clean])
            pols = st.multiselect("Policies", list(POLICIES), default=list(POLICIES))
            c1, c2, c3 = st.columns(3)
            seeds = c1.text_input("Seeds (comma-separated)", "11,23,37,41,53")
            ws = c2.text_input("Evaluation start (blank = 252 sessions after first bar)", "")
            we = c3.text_input("Evaluation end (blank = last session)", "")
            go = st.form_submit_button("Run DQ-001")
        if go:
            try:
                cfg = DQ001Config(snapshot_id=snap.id, symbol=sym, strategies=strats, policies=pols,
                                  seeds=[int(x) for x in seeds.split(",") if x.strip()],
                                  scenarios=[s for s in DEFAULT_SCENARIOS if s.is_clean or s.name in scen],
                                  eval_start=ws or None, eval_end=we or None)
                bar = st.progress(0.0, "starting…")
                res = run_dq001(ctx, cfg, progress=lambda f, m: bar.progress(min(f, 1.0), m))
                st.session_state["rl_selected"] = res["experiment_id"]
                st.success(f"Experiment {res['experiment_id']} completed ({res['n_runs']} backtests). See Results.")
            except (QuantLabError, ValueError) as exc:
                st.error(f"Experiment failed: {exc}")

with tab_res:
    exps = [e for e in list_experiments(ctx) if e["status"] == "COMPLETED"]
    failed = [e for e in list_experiments(ctx) if e["status"] == "FAILED"]
    if failed:
        st.caption(f"{len(failed)} failed experiment(s): " + "; ".join(f"{e['id'][:8]}: {e['failure_reason']}" for e in failed[:3]))
    if not exps:
        st.info("No completed experiments yet.")
        st.stop()
    labels = {f"{e['id'][:8]} · {e['code']} · {e['completed_at'][:19]}": e["id"] for e in exps}
    sel = st.session_state.get("rl_selected")
    keys = list(labels)
    idx = next((i for i, k in enumerate(keys) if labels[k] == sel), 0)
    eid = labels[st.selectbox("Experiment", keys, index=idx)]
    exp = get_experiment(ctx, eid)
    r = exp["results"]
    backtest_banner(r["is_synthetic"])
    st.write(f"Window **{r['evaluation_window'][0]} → {r['evaluation_window'][1]}** · {r['n_runs']} backtests · "
             f"source snapshot verified unchanged: **{r['source_snapshot_verified_unchanged']}**")

    st.subheader("Clean baselines")
    st.dataframe(pd.DataFrame([{"strategy": k, **{m: v["metrics"][m] for m in ("total_return", "cagr", "sharpe", "max_drawdown")},
                                "sharpe_bootstrap_ci95": v["sharpe_bootstrap_ci95"], "trades": v["n_trades"]}
                               for k, v in r["baseline"].items()]), hide_index=True)

    agg = pd.DataFrame(r["aggregate"])
    nonclean = agg[agg.scenario != "clean"]
    for sid, g in nonclean.groupby("strategy", sort=False):
        g = g.assign(y=g["delta_total_return_mean"] * 100,
                     err=(g["delta_total_return_ci_high"] - g["delta_total_return_mean"]).fillna(0) * 100)
        st.plotly_chart(charts.bar_compare(g, "scenario", "y", "policy",
                                           f"{sid}: Δ total return vs clean baseline (pp; mean ± 95% CI across seeds)",
                                           r["is_synthetic"], error_y="err"), width="stretch")
    st.subheader("Aggregate results")
    st.dataframe(experiment_table(r["aggregate"]), hide_index=True, width="stretch")
    if r.get("detection"):
        st.subheader("Did validation detect the injected problems?")
        st.dataframe(pd.DataFrame(r["detection"]).groupby("scenario").sum(numeric_only=True).drop(columns=["seed"]),
                     width="stretch")
    st.subheader("Findings (generated from the numbers)")
    st.markdown("\n".join(f"- {f}" for f in r["findings"]))
    with st.expander("Per-run results"):
        runs = experiment_runs(ctx, eid)
        st.dataframe(runs, hide_index=True, width="stretch")
    st.subheader("Export")
    c = st.columns(3)
    hdr = r["label"] + (" | " + r["data_label"] if r["is_synthetic"] else "")
    with c[0]:
        df_download(runs, "Runs CSV", f"dq001_runs_{eid[:8]}.csv", "rl_csv", hdr)
    c[1].download_button("Report (Markdown)", experiment_markdown(exp), f"dq001_{eid[:8]}.md", key="rl_md")
    c[2].download_button("Report (HTML)", experiment_html(exp), f"dq001_{eid[:8]}.html", key="rl_html")
