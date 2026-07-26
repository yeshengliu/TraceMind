"""TraceMind Streamlit entry point.

Launch with:

    streamlit run app.py
"""

from src.ui.dashboard import render_dashboard


if __name__ == "__main__":
    render_dashboard()
