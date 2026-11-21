"""QuantLab Streamlit dashboard entry point.

Run:  streamlit run apps/dashboard/streamlit_app.py
"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="QuantLab", page_icon="📈", layout="wide")

VIEWS = Path(__file__).parent / "views"
pages = [
    st.Page(VIEWS / "overview.py", title="Overview", icon="🏛️", default=True),
    st.Page(VIEWS / "market_explorer.py", title="Market Explorer", icon="📊"),
    st.Page(VIEWS / "strategy_studio.py", title="Strategy Studio", icon="🧭"),
    st.Page(VIEWS / "backtest_lab.py", title="Backtest Lab", icon="🧪"),
    st.Page(VIEWS / "research_lab.py", title="Research Lab", icon="🔬"),
    st.Page(VIEWS / "paper_trading.py", title="Paper Trading", icon="📝"),
    st.Page(VIEWS / "ai_assistant.py", title="AI Assistant", icon="💬"),
]
st.navigation(pages).run()
