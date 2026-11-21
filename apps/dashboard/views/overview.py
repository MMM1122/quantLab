import pandas as pd
import streamlit as st

from quantlab.backtest.service import list_runs
from quantlab.ui.common import fmt_num, fmt_pct, get_ctx, safety_banner

ctx = get_ctx()
st.title("QuantLab")
st.markdown("Local-first research, backtesting and **paper** trading for US-listed ETFs · daily bars · USD · long-only")
safety_banner()

c = st.columns(5)
snaps = ctx.snapshots.list()
real = [s for s in snaps if not s.is_synthetic and s.parent_id is None]
synth = [s for s in snaps if s.is_synthetic and s.parent_id is None]
runs = list_runs(ctx)
n_exp = ctx.db.query_one("SELECT COUNT(*) AS n FROM experiments")["n"]
n_acc = ctx.db.query_one("SELECT COUNT(*) AS n FROM accounts")["n"]
c[0].metric("Real-data snapshots", len(real))
c[1].metric("Synthetic snapshots", len(synth))
c[2].metric("Backtest runs", len(runs))
c[3].metric("Experiments", n_exp)
c[4].metric("Paper accounts", n_acc)

st.subheader("Mode")
st.markdown(
    "- **Backtest / Research** — historical simulation on immutable snapshots.\n"
    "- **Paper trading** — virtual account on *historical session replay*, not live market conditions.\n"
    "- **Live trading** — not available in V1 (`LIVE_TRADING_ENABLED=false`)."
)

st.subheader("Recent backtests")
if not runs:
    st.info("No backtests yet. 1) Ingest data on **Market Explorer**. 2) Run a strategy in **Backtest Lab**.")
else:
    rows = []
    for r in runs[:15]:
        n = ((r.get("metrics") or {}).get("net")) or {}
        rows.append({
            "run": r["id"][:8], "strategy": r["strategy_name"], "status": r["status"],
            "data": "SYNTHETIC" if r["is_synthetic"] else "REAL",
            "symbols": ", ".join((r.get("config") or {}).get("symbols", [])),
            "CAGR": fmt_pct(n.get("cagr")), "Sharpe": fmt_num(n.get("sharpe")),
            "Max DD": fmt_pct(n.get("max_drawdown")), "created": r["created_at"][:19],
            "failure": r["failure_reason"] or "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("Workflow")
st.markdown("`Ingest → Validate → Research → Backtest → Compare → Paper Trade → Monitor`")
