"""Shared Streamlit widgets used across pages."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from ta_framework.core.types import Timeframe
from ta_framework.core.types import IndicatorTier
from ta_framework.data.yfinance_provider import YFinanceProvider
from ta_framework.indicators.catalog import INDICATOR_CATALOG


def symbol_input(key: str = "symbol") -> str:
    """Render a symbol text input and return the entered symbol."""
    return st.text_input("Symbol", value="AAPL", key=key).strip().upper()


def timeframe_select(key: str = "timeframe") -> Timeframe:
    """Render a timeframe selector and return the chosen Timeframe."""
    options = {tf.value: tf for tf in Timeframe}
    label = st.selectbox("Timeframe", list(options.keys()), index=6, key=key)
    return options[label]


def date_range_input(key_prefix: str = "date") -> tuple[str, str]:
    """Render start/end date inputs and return (start, end) as strings."""
    col1, col2 = st.columns(2)
    default_start = date.today() - timedelta(days=5 * 365)
    with col1:
        start = st.date_input("Start Date", value=default_start, key=f"{key_prefix}_start")
    with col2:
        end = st.date_input("End Date", value=date.today(), key=f"{key_prefix}_end")
    return str(start), str(end)


def indicator_multiselect(key: str = "indicators") -> list[dict]:
    """Render a multi-select for indicators grouped by category."""
    categories: dict[str, list[str]] = {}
    for name, config in INDICATOR_CATALOG.items():
        if config.tier.value <= IndicatorTier.TIER1.value:
            cat = config.category.value
            categories.setdefault(cat, []).append(name)

    selected = []
    for cat, names in sorted(categories.items()):
        choices = st.multiselect(f"{cat.title()} Indicators", names, key=f"{key}_{cat}")
        for name in choices:
            selected.append({"name": name, "params": {}})
    return selected


def strategy_select(key: str = "strategy") -> str:
    """Render a strategy selector and return the strategy key."""
    strategies = ["ema_cross", "rsi", "macd", "bbands", "supertrend", "ttm_squeeze"]
    labels = {
        "ema_cross": "EMA Crossover",
        "rsi": "RSI Threshold",
        "macd": "MACD Signal Line",
        "bbands": "Bollinger Bands",
        "supertrend": "Supertrend",
        "ttm_squeeze": "TTM Squeeze",
    }
    display = [labels.get(s, s) for s in strategies]
    idx = st.selectbox("Strategy", display, key=key)
    return strategies[display.index(idx)]


def strategy_params(strategy_name: str, key_prefix: str = "param") -> dict:
    """Render parameter inputs for a given strategy."""
    params: dict = {}
    if strategy_name == "ema_cross":
        c1, c2 = st.columns(2)
        with c1:
            params["fast_period"] = st.number_input("Fast Period", 2, 100, 12, key=f"{key_prefix}_fast")
        with c2:
            params["slow_period"] = st.number_input("Slow Period", 2, 200, 26, key=f"{key_prefix}_slow")
    elif strategy_name == "rsi":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["period"] = st.number_input("Period", 2, 100, 14, key=f"{key_prefix}_p")
        with c2:
            params["overbought"] = st.number_input("Overbought", 50, 100, 70, key=f"{key_prefix}_ob")
        with c3:
            params["oversold"] = st.number_input("Oversold", 0, 50, 30, key=f"{key_prefix}_os")
    elif strategy_name == "macd":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["fast"] = st.number_input("Fast", 2, 100, 12, key=f"{key_prefix}_f")
        with c2:
            params["slow"] = st.number_input("Slow", 2, 200, 26, key=f"{key_prefix}_s")
        with c3:
            params["signal"] = st.number_input("Signal", 2, 50, 9, key=f"{key_prefix}_sig")
    elif strategy_name == "bbands":
        c1, c2 = st.columns(2)
        with c1:
            params["period"] = st.number_input("Period", 2, 100, 20, key=f"{key_prefix}_p")
        with c2:
            params["std_dev"] = st.number_input("Std Dev", 0.5, 5.0, 2.0, step=0.5, key=f"{key_prefix}_std")
    elif strategy_name == "supertrend":
        c1, c2 = st.columns(2)
        with c1:
            params["period"] = st.number_input("Period", 2, 100, 10, key=f"{key_prefix}_p")
        with c2:
            params["multiplier"] = st.number_input("Multiplier", 0.5, 10.0, 3.0, step=0.5, key=f"{key_prefix}_m")
    elif strategy_name == "ttm_squeeze":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["bb_period"] = st.number_input("BB Period", 2, 100, 20, key=f"{key_prefix}_bb")
        with c2:
            params["kc_period"] = st.number_input("KC Period", 2, 100, 20, key=f"{key_prefix}_kc")
        with c3:
            params["kc_mult"] = st.number_input("KC Mult", 0.5, 5.0, 1.5, step=0.5, key=f"{key_prefix}_km")
    return params


def backtest_config_panel(key_prefix: str = "bt") -> dict:
    """Render backtest configuration inputs and return a config dict."""
    st.subheader("Backtest Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        capital = st.number_input(
            "Initial Capital ($)", 1_000, 10_000_000, 100_000, step=10_000, key=f"{key_prefix}_cap"
        )
    with c2:
        commission = st.number_input("Commission (%)", 0.0, 5.0, 0.1, step=0.01, key=f"{key_prefix}_comm")
    with c3:
        slippage = st.number_input("Slippage (%)", 0.0, 5.0, 0.05, step=0.01, key=f"{key_prefix}_slip")

    c4, c5 = st.columns(2)
    with c4:
        position_size = st.slider("Position Size (%)", 10, 100, 100, step=10, key=f"{key_prefix}_pos")
    with c5:
        allow_short = st.checkbox("Allow Short", value=True, key=f"{key_prefix}_short")

    return {
        "initial_capital": capital,
        "commission_pct": commission / 100.0,
        "slippage_pct": slippage / 100.0,
        "position_size_pct": position_size / 100.0,
        "allow_short": allow_short,
    }


@st.cache_data(ttl=300)
def fetch_data(symbol: str, timeframe: str, start: str, end: str):
    """Cached data fetch via yfinance."""
    provider = YFinanceProvider()
    tf = Timeframe(timeframe)
    return provider.fetch(symbol, tf, start, end)
