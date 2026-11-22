"""QuantLab Streamlit dashboard entry point.

Run:  streamlit run apps/dashboard/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Hosted platforms (e.g. Streamlit Community Cloud) may install only requirements.txt without the
# package itself; make the repo's src/ importable in that case.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    try:
        import quantlab  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(_SRC))

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
