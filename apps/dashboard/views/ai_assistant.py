import streamlit as st

from quantlab.ai.assistant import ResearchAssistant
from quantlab.backtest.service import list_runs
from quantlab.core.errors import QuantLabError
from quantlab.research.dq_001 import list_experiments
from quantlab.ui.common import get_ctx, safety_banner

ctx = get_ctx()
st.title("AI Research Assistant")
safety_banner()
assistant = ResearchAssistant(ctx.settings.db_path)
if assistant.provider is None:
    st.info("No LLM API key configured → **deterministic analytics mode**: explanations are generated from stored "
            "numbers by templates. Set `QUANTLAB_LLM_PROVIDER` and a key in `.env` to enable LLM phrasing.")
else:
    st.info(f"LLM provider: **{assistant.provider.name}**. Every number in an LLM answer is checked against stored "
            "data; unverifiable answers are discarded.")
st.caption("Read-only: the assistant reads stored backtests and experiments. It has no tool to place or submit orders.")

runs = [r for r in list_runs(ctx) if r["status"] == "COMPLETED"]
exps = [e for e in list_experiments(ctx) if e["status"] == "COMPLETED"]
task = st.radio("Task", ["Explain a backtest", "Compare backtests", "Summarize DQ-001 experiment"], horizontal=True)
rlabels = {f"{r['id'][:8]} · {r['strategy_name']} · {', '.join(r['config']['symbols'])} · "
           f"{'SYNTHETIC' if r['is_synthetic'] else 'REAL'}": r["id"] for r in runs}
question = st.text_input("Optional question", "")

resp = None
try:
    if task == "Explain a backtest":
        if not runs:
            st.info("No completed backtests yet.")
            st.stop()
        rid = rlabels[st.selectbox("Backtest", list(rlabels))]
        if st.button("Explain"):
            resp = assistant.explain(rid, question or None)
    elif task == "Compare backtests":
        if len(runs) < 2:
            st.info("Need at least two completed backtests.")
            st.stop()
        chosen = st.multiselect("Backtests", list(rlabels), default=list(rlabels)[:2])
        if st.button("Compare") and len(chosen) >= 2:
            resp = assistant.compare([rlabels[c] for c in chosen], question or None)
    else:
        if not exps:
            st.info("No completed experiments yet.")
            st.stop()
        elabels = {f"{e['id'][:8]} · {e['code']} · {e['completed_at'][:19]}": e["id"] for e in exps}
        eid = elabels[st.selectbox("Experiment", list(elabels))]
        if st.button("Summarize"):
            resp = assistant.summarize_experiment(eid, question or None)
except QuantLabError as exc:
    st.error(str(exc))

if resp is not None:
    badge = {"deterministic": "🧮 deterministic analytics", "llm": "🤖 LLM (numbers verified)", "refused": "⛔ refused"}
    st.caption(f"{badge.get(resp.mode, resp.mode)} · numbers checked: {resp.verification.get('numbers_checked')} · "
               f"all verified: {resp.verification.get('verified')}")
    st.markdown(resp.text)
    with st.expander("Sources and facts used"):
        st.write(resp.sources)
        st.json(resp.facts)
