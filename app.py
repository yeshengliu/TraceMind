"""TraceMind Streamlit entry point.

Launch with:

    streamlit run app.py
"""

import streamlit as st

from src.ui.dashboard import render_dashboard
from src.ui.xray_tab import render_xray_tab


if __name__ == "__main__":
    st.set_page_config(
        page_title="TraceMind · Local Agent Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    agent_tab, xray_tab = st.tabs(["◈ Agent Studio", "🔬 LLM X-Ray Lab"])
    with agent_tab:
        render_dashboard(configure_page=False)
    with xray_tab:
        render_xray_tab()
