import dataclasses

import pandas as pd
import streamlit as st

from quantlab.backtest.market import build_panel
from quantlab.core.errors import QuantLabError
from quantlab.reports import charts
from quantlab.strategies.registry import STRATEGIES, create_strategy
from quantlab.ui.common import get_ctx, safety_banner, snapshot_selector, synthetic_banner

ctx = get_ctx()
st.title("Strategy Studio")
safety_banner()
st.markdown("Inspect strategy definitions and preview **causal** signals (target weights) on a snapshot. "
            "No trades are simulated here — use Backtest Lab for that.")

sid = st.selectbox("Strategy", list(STRATEGIES), format_func=lambda s: STRATEGIES[s][0].name)
cls, pcls = STRATEGIES[sid]
st.markdown(f"**{cls.name}** `v{cls.version}` — {cls.description}")

params = {}
fields = [f for f in dataclasses.fields(pcls) if f.name != "weights"]
cols = st.columns(max(1, len(fields)))
defaults = pcls()
for col, f in zip(cols, fields):
    v = getattr(defaults, f.name)
    if isinstance(v, bool):
        params[f.name] = col.checkbox(f.name, v, key=f"ss_{sid}_{f.name}")
    elif isinstance(v, int):
        params[f.name] = int(col.number_input(f.name, value=v, step=1, key=f"ss_{sid}_{f.name}"))
    elif isinstance(v, float):
        params[f.name] = float(col.number_input(f.name, value=v, key=f"ss_{sid}_{f.name}"))

try:
    strat = create_strategy(sid, params)
except QuantLabError as exc:
    st.error(str(exc))
    st.stop()
st.json(strat.describe())

st.subheader("Signal preview")
snap = snapshot_selector(ctx, "ss_snap", require_tradable=True)
if snap is None:
    st.stop()
synthetic_banner(snap.is_synthetic)
default_syms = snap.symbols if sid == "momentum" else snap.symbols[:1]
syms = st.multiselect("Symbols", snap.symbols, default=default_syms, key="ss_syms")
if not syms:
    st.stop()
panel = build_panel(ctx.snapshots.load(snap.id), snap.price_basis, symbols=syms)
w = strat.target_weights(panel.signal_price)
show = st.selectbox("Chart symbol", syms, key="ss_chart_sym")
st.plotly_chart(charts.signal_chart(panel.signal_price[show], w[show], show, snap.is_synthetic,
                                    f"{strat.name}: {show} signal (decided after close; executes next open)"),
                width="stretch")
st.caption("Blank weight = no opinion (warm-up or missing data). Weights shown are targets, not holdings.")
changes = w.ffill().diff().abs().sum(axis=1)
st.write(f"Signal changes in sample: **{int((changes > 1e-12).sum())}** · first valid signal: "
         f"**{w.dropna(how='all').index.min().date() if w.notna().any().any() else 'none'}**")
st.dataframe(pd.DataFrame(w.dropna(how="all").tail(20)), width="stretch")
