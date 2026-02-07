"""Backtester page: strategy selection, parameter tuning, backtest execution."""

from __future__ import annotations

import streamlit as st

from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.core.registry import strategy_registry
from ta_framework.core.types import BacktestConfig
from ta_framework.indicators.engine import IndicatorEngine
from ta_framework.viz.charts import candlestick_chart
from ta_framework.viz.drawdown import drawdown_chart
from ta_framework.viz.trade_plots import pnl_chart, trade_markers
from ta_framework.pages.components import (
    backtest_config_panel,
    date_range_input,
    fetch_data,
    strategy_params,
    strategy_select,
    symbol_input,
    timeframe_select,
)


def render() -> None:
    st.header("Strategy Backtester")

    # --- Sidebar ---
    with st.sidebar:
        symbol = symbol_input("bt_symbol")
        timeframe = timeframe_select("bt_tf")
        start, end = date_range_input("bt")
        st.divider()
        strat_name = strategy_select("bt_strat")
        params = strategy_params(strat_name, "bt_param")
        st.divider()
        bt_cfg = backtest_config_panel("bt")

    if not symbol:
        st.info("Enter a symbol to get started.")
        return

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
    if not run_btn:
        st.info("Configure your strategy and click **Run Backtest**.")
        return

    # --- Fetch data ---
    with st.spinner("Fetching data..."):
        try:
            df = fetch_data(symbol, timeframe.value, start, end)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

    if df.empty:
        st.warning("No data returned.")
        return

    # --- Build strategy and compute indicators ---
    with st.spinner("Computing indicators..."):
        strategy_cls = strategy_registry.get(strat_name)
        strategy = strategy_cls(**params)

        engine = IndicatorEngine()
        for ind_spec in strategy.required_indicators:
            name = ind_spec.pop("name")
            try:
                df = engine.compute(df, name, **ind_spec)
            except Exception as e:
                st.error(f"Indicator '{name}' failed: {e}")
                return

    # --- Generate signals ---
    with st.spinner("Generating signals..."):
        try:
            df = strategy.generate(df)
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            return

    # --- Run backtest ---
    with st.spinner("Running backtest..."):
        config = BacktestConfig(**bt_cfg)
        backtester = VectorizedBacktester(config)
        try:
            result = backtester.run(df)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            return

    # --- Display results ---
    st.subheader("Results")

    # Key metrics
    summary = result.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{summary['total_return']*100:.2f}%")
    c2.metric("Max Drawdown", f"{summary['max_drawdown']*100:.2f}%")
    c3.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    c4.metric("Win Rate", f"{summary['win_rate']*100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Trades", summary["num_trades"])
    c6.metric("Profit Factor", f"{summary['profit_factor']:.2f}" if summary["profit_factor"] != float("inf") else "inf")
    c7.metric("Final Equity", f"${summary['final_equity']:,.0f}")
    c8.metric("Initial Capital", f"${summary['initial_capital']:,.0f}")

    # Equity curve
    st.subheader("Equity Curve")
    st.line_chart(result.equity_curve, use_container_width=True)

    # Drawdown
    st.subheader("Drawdown")
    dd_fig = drawdown_chart(result.equity_curve)
    st.plotly_chart(dd_fig, use_container_width=True)

    # Trade markers on chart
    if result.trades:
        st.subheader("Trade Signals")
        trades_fig = trade_markers(df, result.trades)
        st.plotly_chart(trades_fig, use_container_width=True)

        # P&L
        st.subheader("P&L Analysis")
        pnl_fig = pnl_chart(result.trades)
        st.plotly_chart(pnl_fig, use_container_width=True)

    # Trades table
    with st.expander("Trade Log"):
        trades_df = result.to_dataframe()
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades executed.")
