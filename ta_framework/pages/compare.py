"""Compare page: multi-strategy and multi-asset comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.core.registry import strategy_registry
from ta_framework.core.types import BacktestConfig
from ta_framework.indicators.engine import IndicatorEngine
from ta_framework.viz.heatmaps import correlation_heatmap
from ta_framework.pages.components import (
    backtest_config_panel,
    date_range_input,
    fetch_data,
    timeframe_select,
)


def render() -> None:
    st.header("Compare")

    tab1, tab2 = st.tabs(["Multi-Asset", "Multi-Strategy"])

    with tab1:
        _multi_asset_view()

    with tab2:
        _multi_strategy_view()


def _multi_asset_view() -> None:
    """Compare returns across multiple assets."""
    with st.sidebar:
        timeframe = timeframe_select("cmp_tf")
        start, end = date_range_input("cmp")

    symbols_input = st.text_input(
        "Symbols (comma-separated)",
        value="AAPL, MSFT, GOOGL, BTC-USD",
        key="cmp_symbols",
    )
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if not symbols or len(symbols) < 2:
        st.info("Enter at least 2 symbols to compare.")
        return

    if not st.button("Compare Assets", key="cmp_asset_btn"):
        return

    # Fetch all
    returns_dict: dict[str, pd.Series] = {}
    equity_dict: dict[str, pd.Series] = {}

    for sym in symbols:
        with st.spinner(f"Fetching {sym}..."):
            try:
                df = fetch_data(sym, timeframe.value, start, end)
                if df.empty:
                    st.warning(f"No data for {sym}, skipping.")
                    continue
                ret = df["close"].pct_change().dropna()
                returns_dict[sym] = ret
                equity_dict[sym] = (1 + ret).cumprod()
            except Exception as e:
                st.warning(f"Failed to fetch {sym}: {e}")

    if len(returns_dict) < 2:
        st.error("Need at least 2 valid symbols.")
        return

    # Normalized equity curves
    st.subheader("Normalized Performance (Base = 1.0)")
    equity_df = pd.DataFrame(equity_dict)
    st.line_chart(equity_df, use_container_width=True)

    # Summary table
    st.subheader("Return Summary")
    summary_rows = []
    for sym, ret in returns_dict.items():
        summary_rows.append({
            "Symbol": sym,
            "Total Return": f"{(equity_dict[sym].iloc[-1] - 1) * 100:.2f}%",
            "Ann. Volatility": f"{ret.std() * (252**0.5) * 100:.2f}%",
            "Sharpe": f"{ret.mean() / ret.std() * (252**0.5):.2f}" if ret.std() > 0 else "N/A",
            "Max DD": f"{((equity_dict[sym] / equity_dict[sym].cummax() - 1).min()) * 100:.2f}%",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    corr_fig = correlation_heatmap(returns_dict)
    st.plotly_chart(corr_fig, use_container_width=True)


def _multi_strategy_view() -> None:
    """Compare multiple strategies on the same asset."""
    with st.sidebar:
        pass  # Uses same sidebar from multi-asset

    symbol = st.text_input("Symbol", value="AAPL", key="cmp_strat_sym")
    timeframe = st.selectbox(
        "Timeframe", [tf.value for tf in __import__("ta_framework.core.types", fromlist=["Timeframe"]).Timeframe],
        index=6,
        key="cmp_strat_tf",
    )
    start = st.date_input("Start", value=pd.Timestamp.now() - pd.Timedelta(days=5 * 365), key="cmp_strat_start")
    end = st.date_input("End", value=pd.Timestamp.now(), key="cmp_strat_end")

    strategies = st.multiselect(
        "Strategies to compare",
        ["ema_cross", "rsi", "macd", "bbands", "supertrend"],
        default=["ema_cross", "rsi"],
        key="cmp_strats",
    )

    if not strategies or not symbol:
        st.info("Select at least one strategy.")
        return

    bt_cfg = backtest_config_panel("cmp_bt")

    if not st.button("Compare Strategies", key="cmp_strat_btn"):
        return

    # Fetch data once
    with st.spinner("Fetching data..."):
        try:
            from ta_framework.core.types import Timeframe as TF
            df_base = fetch_data(symbol.upper(), timeframe, str(start), str(end))
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

    if df_base.empty:
        st.warning("No data returned.")
        return

    engine = IndicatorEngine()
    config = BacktestConfig(**bt_cfg)
    backtester = VectorizedBacktester(config)

    results = {}
    for strat_name in strategies:
        try:
            strategy_cls = strategy_registry.get(strat_name)
            strategy = strategy_cls()  # default params
            df = df_base.copy()

            # Compute required indicators
            for ind_spec in strategy.required_indicators:
                spec = ind_spec.copy()
                name = spec.pop("name")
                df = engine.compute(df, name, **spec)

            df = strategy.generate(df)
            result = backtester.run(df)
            results[strat_name] = result
        except Exception as e:
            st.warning(f"Strategy '{strat_name}' failed: {e}")

    if not results:
        st.error("All strategies failed.")
        return

    # Equity curves
    st.subheader("Equity Curves")
    eq_df = pd.DataFrame({name: r.equity_curve for name, r in results.items()})
    st.line_chart(eq_df, use_container_width=True)

    # Comparison table
    st.subheader("Strategy Comparison")
    rows = []
    for name, r in results.items():
        s = r.summary()
        rows.append({
            "Strategy": name,
            "Total Return": f"{s['total_return']*100:.2f}%",
            "Max Drawdown": f"{s['max_drawdown']*100:.2f}%",
            "Sharpe": f"{s['sharpe_ratio']:.2f}",
            "Win Rate": f"{s['win_rate']*100:.1f}%",
            "Trades": s["num_trades"],
            "Profit Factor": f"{s['profit_factor']:.2f}" if s["profit_factor"] != float("inf") else "inf",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
