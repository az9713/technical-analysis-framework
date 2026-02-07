"""Streamlit entry point for the Technical Analysis Framework."""

import streamlit as st

st.set_page_config(
    page_title="TA Framework",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ta_framework.pages import dashboard, backtester, compare, analysis


def main() -> None:
    st.sidebar.title("TA Framework")
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Backtester", "Compare", "Analysis"],
        key="nav",
    )

    if page == "Dashboard":
        dashboard.render()
    elif page == "Backtester":
        backtester.render()
    elif page == "Compare":
        compare.render()
    elif page == "Analysis":
        analysis.render()


if __name__ == "__main__":
    main()
