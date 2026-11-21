import json
import uuid
from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.core.errors import QuantLabError
from quantlab.core.labels import REPLAY_LABEL
from quantlab.paper.autopilot import StrategyAutopilot
from quantlab.paper.broker import PaperBroker
from quantlab.paper.compare import paper_vs_backtest
from quantlab.strategies.registry import STRATEGIES
from quantlab.ui.common import get_ctx, safety_banner, snapshot_selector, synthetic_banner

ctx = get_ctx()
broker = PaperBroker(ctx)
st.title("Paper Trading")
safety_banner()
st.error(f"**{REPLAY_LABEL}** — virtual account stepping through stored historical sessions. "
         "No broker connection, no real orders, no live prices.", icon="🕰️")

with st.expander("Create paper account", expanded=not broker.list_accounts()):
    snap = snapshot_selector(ctx, "pt_snap", require_tradable=True)
    if snap is not None:
        with st.form("pt_create"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name", "paper-1")
            cash = c2.number_input("Initial cash (USD)", value=10_000.0, min_value=1.0, step=1000.0)
            start = c3.text_input("Replay start session (blank = first)", "")
            ok = st.form_submit_button("Create account")
        if ok:
            try:
                acc = broker.create_account(name, snap.id, Decimal(str(cash)), start or None)
                st.session_state["pt_acc"] = acc["id"]
                st.success(f"Account {acc['id']} created at replay session {acc['replay_session']}.")
            except QuantLabError as exc:
                st.error(str(exc))

accounts = broker.list_accounts()
if not accounts:
    st.info("No paper accounts yet.")
    st.stop()
labels = {f"{a['name']} · {a['id'][:8]} · {'SYNTHETIC' if a['is_synthetic'] else 'REAL'} data": a["id"] for a in accounts}
keys = list(labels)
sel = st.session_state.get("pt_acc")
idx = next((i for i, k in enumerate(keys) if labels[k] == sel), 0)
aid = labels[st.selectbox("Account", keys, index=idx)]
state = broker.account_state(aid)
synthetic_banner(state["is_synthetic"])

c = st.columns(6)
c[0].metric("Replay session", state["replay_session"])
c[1].metric("Sessions left", state["replay_sessions_remaining"])
c[2].metric("NAV (USD)", f"{Decimal(state['nav']):,.2f}")
c[3].metric("Cash (USD)", f"{Decimal(state['cash']):,.2f}")
c[4].metric("Positions (USD)", f"{Decimal(state['positions_value']):,.2f}")
c[5].metric("Kill switch", "ON" if state["kill_switch"] else "off")
st.caption(f"Risk limits: {state['risk_limits']} · no shorting · no margin · stale-data orders rejected")

left, right = st.columns([1, 1])
with left:
    st.subheader("Order ticket (executable from the next replay session)")
    with st.form("pt_order"):
        snap_syms = ctx.snapshots.get(state["snapshot_id"]).symbols
        c1, c2, c3 = st.columns(3)
        sym = c1.selectbox("Symbol", snap_syms)
        side = c2.radio("Side", ["BUY", "SELL"], horizontal=True)
        qty = c3.number_input("Shares", min_value=1, value=1, step=1)
        c1, c2, c3 = st.columns(3)
        otype = c1.radio("Type", ["MARKET", "LIMIT"], horizontal=True)
        lpx = c2.text_input("Limit price (LIMIT only)", "")
        tif = c3.radio("Time in force", ["GTC", "DAY"], horizontal=True)
        coid = st.text_input("Client order id (idempotency key)", str(uuid.uuid4()))
        submit = st.form_submit_button("Submit paper order")
    if submit:
        try:
            o = broker.submit_order(aid, sym, side, int(qty), coid, order_type=otype, actor="dashboard",
                                    limit_price=(lpx.strip() or None) if otype == "LIMIT" else None,
                                    time_in_force=tif)
            if o["status"] == "REJECTED":
                st.error(f"Rejected by risk checks: {o['reason']}")
            elif o.get("idempotent_replay"):
                st.warning("Same client order id already used — returning the existing order; nothing new created.")
            else:
                st.success(f"Order {o['id'][:8]} {o['status']}")
        except QuantLabError as exc:
            st.error(str(exc))
with right:
    st.subheader("Replay controls")
    n = st.number_input("Advance sessions", min_value=1, max_value=250, value=1)
    if st.button("Advance replay clock"):
        try:
            out = broker.advance_session(aid, int(n))
            fills = sum(len(s["fills"]) for s in out)
            st.success(f"Advanced to {out[-1]['session']}; {fills} fill(s).")
        except QuantLabError as exc:
            st.error(str(exc))
    reason = st.text_input("Kill-switch reason", "")
    k1, k2 = st.columns(2)
    if k1.button("Activate kill switch", type="primary"):
        r = broker.set_kill_switch(aid, True, reason)
        st.warning(f"Kill switch ON; {r['orders_canceled']} open order(s) canceled.")
    if k2.button("Deactivate kill switch"):
        broker.set_kill_switch(aid, False, reason)
        st.info("Kill switch off.")

with st.expander("Strategy autopilot (deterministic strategy places the paper orders; AI never does)"):
    ap_svc = StrategyAutopilot(ctx, broker)
    try:
        ap_cfg = ap_svc.get(aid)
    except QuantLabError:
        ap_cfg = None
    snap_syms = ctx.snapshots.get(state["snapshot_id"]).symbols
    with st.form("ap_form"):
        c1, c2 = st.columns(2)
        ap_strat = c1.selectbox("Strategy", list(STRATEGIES),
                                index=list(STRATEGIES).index(ap_cfg["strategy"]) if ap_cfg else 1)
        ap_syms = c2.multiselect("Symbols", snap_syms, default=ap_cfg["symbols"] if ap_cfg else snap_syms[:1])
        ap_params = st.text_input("Parameters (JSON)", json.dumps(ap_cfg["params"]) if ap_cfg else "{}")
        if st.form_submit_button("Attach / replace autopilot"):
            try:
                ap_cfg = ap_svc.attach(aid, ap_strat, ap_syms, json.loads(ap_params or "{}"))
                st.success("Autopilot attached.")
            except (QuantLabError, ValueError) as exc:
                st.error(str(exc))
    if ap_cfg:
        st.caption(f"Attached: {ap_cfg['strategy']} {ap_cfg['params']} on {', '.join(ap_cfg['symbols'])} · "
                   f"enabled={ap_cfg['enabled']} · weights capped at {state['risk_limits']['max_position_weight']}")
        n_ap = st.number_input("Sessions to run", min_value=1, max_value=2000, value=20, key="ap_n")
        if st.button("Run autopilot"):
            try:
                out = ap_svc.run(aid, int(n_ap))
                st.success(f"Advanced {out['sessions_advanced']} sessions · {out['orders_submitted']} orders "
                           f"({out['orders_rejected']} rejected by risk checks)")
            except QuantLabError as exc:
                st.error(str(exc))
        if st.button("Compare with backtest"):
            try:
                cmp_ = paper_vs_backtest(ctx, aid)
                ser = cmp_.pop("_series")
                fig = go.Figure([go.Scatter(x=ser.index, y=ser["paper"], name="paper (replay)"),
                                 go.Scatter(x=ser.index, y=ser["backtest"], name="backtest", line={"dash": "dot"})])
                fig.update_layout(title="Paper vs backtest NAV", template="plotly_white", height=300)
                st.plotly_chart(fig, width="stretch")
                st.json(cmp_)
            except QuantLabError as exc:
                st.info(str(exc))

state = broker.account_state(aid)
st.subheader("Holdings")
st.dataframe(pd.DataFrame(state["holdings"]), hide_index=True, width="stretch")
hist = broker.nav_history(aid)
if len(hist) > 1:
    fig = go.Figure(go.Scatter(x=hist["session_date"], y=hist["nav"], name="NAV", line={"color": "#2F5D8A"}))
    fig.update_layout(title=f"Paper NAV<br><sup>{REPLAY_LABEL}</sup>", template="plotly_white", height=320)
    st.plotly_chart(fig, width="stretch")
tabs = st.tabs(["Orders", "Executions", "Cash ledger", "Risk events", "Reconciliation"])
with tabs[0]:
    st.dataframe(pd.DataFrame(broker.list_orders(aid)), hide_index=True, width="stretch")
with tabs[1]:
    st.dataframe(pd.DataFrame(broker.executions(aid)), hide_index=True, width="stretch")
with tabs[2]:
    st.dataframe(pd.DataFrame(broker.ledger(aid)), hide_index=True, width="stretch")
with tabs[3]:
    ev = pd.DataFrame(broker.risk_events(aid))
    if len(ev):
        ev["details"] = ev["details"].astype(str)
    st.dataframe(ev, hide_index=True, width="stretch")
with tabs[4]:
    st.json(broker.reconcile(aid))
