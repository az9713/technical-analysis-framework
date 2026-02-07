"""Dashboard page: data loading, indicator selection, interactive charting."""

from __future__ import annotations

import streamlit as st

from ta_framework.indicators.engine import IndicatorEngine
from ta_framework.indicators.catalog import INDICATOR_CATALOG
from ta_framework.viz.charts import multi_panel_chart
from ta_framework.pages.components import (
    date_range_input,
    fetch_data,
    indicator_multiselect,
    symbol_input,
    timeframe_select,
)


def render() -> None:
    st.header("Dashboard")

    # --- Sidebar controls ---
    with st.sidebar:
        symbol = symbol_input("dash_symbol")
        timeframe = timeframe_select("dash_tf")
        start, end = date_range_input("dash")
        st.divider()
        indicators = indicator_multiselect("dash_ind")

    # --- Fetch data ---
    if not symbol:
        st.info("Enter a symbol to get started.")
        return

    try:
        df = fetch_data(symbol, timeframe.value, start, end)
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return

    if df.empty:
        st.warning(f"No data returned for {symbol}.")
        return

    # --- Compute indicators ---
    engine = IndicatorEngine()
    for ind in indicators:
        try:
            df = engine.compute(df, ind["name"], **ind.get("params", {}))
        except Exception as e:
            st.warning(f"Could not compute {ind['name']}: {e}")

    # --- Classify columns as overlays vs oscillators ---
    base_cols = {"open", "high", "low", "close", "volume"}
    indicator_cols = [c for c in df.columns if c not in base_cols]

    # Heuristic: if values are in a small bounded range, treat as oscillator
    overlays = []
    oscillators = []
    for col in indicator_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        col_range = series.max() - series.min()
        price_range = df["close"].max() - df["close"].min()
        if col_range < price_range * 0.3 and series.max() <= 200:
            oscillators.append(col)
        else:
            overlays.append(col)

    # --- Chart ---
    fig = multi_panel_chart(
        df,
        overlays=overlays if overlays else None,
        oscillators=oscillators if oscillators else None,
        show_volume=True,
    )
    fig.update_layout(title=f"{symbol} - {timeframe.value}")
    st.plotly_chart(fig, use_container_width=True)

    # --- Data table ---
    with st.expander("Raw Data"):
        st.dataframe(df.tail(50), use_container_width=True)

    # --- Summary stats ---
    with st.expander("Summary Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Close", f"${df['close'].iloc[-1]:.2f}")
        col2.metric("High", f"${df['high'].max():.2f}")
        col3.metric("Low", f"${df['low'].min():.2f}")
        daily_ret = df["close"].pct_change().dropna()
        col4.metric("Avg Daily Return", f"{daily_ret.mean()*100:.3f}%")
