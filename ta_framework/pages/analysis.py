"""Analysis page: tearsheet, regime detection, risk analytics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ta_framework.analytics.regime import RegimeDetector
from ta_framework.analytics.tearsheet import TearSheet
from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.core.registry import strategy_registry
from ta_framework.core.types import BacktestConfig
from ta_framework.indicators.engine import IndicatorEngine
from ta_framework.risk.var import cvar, historical_var, parametric_var
from ta_framework.viz.distribution import returns_histogram
from ta_framework.viz.drawdown import drawdown_chart, drawdown_periods
from ta_framework.viz.heatmaps import monthly_returns_heatmap
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
    st.header("Deep Analysis")

    # --- Sidebar ---
    with st.sidebar:
        symbol = symbol_input("an_symbol")
        timeframe = timeframe_select("an_tf")
        start, end = date_range_input("an")
        st.divider()
        strat_name = strategy_select("an_strat")
        params = strategy_params(strat_name, "an_param")
        st.divider()
        bt_cfg = backtest_config_panel("an")

    if not symbol:
        st.info("Enter a symbol to get started.")
        return

    if not st.button("Run Analysis", type="primary", key="an_run", use_container_width=True):
        st.info("Configure settings and click **Run Analysis**.")
        return

    # --- Fetch + backtest ---
    with st.spinner("Fetching data & running backtest..."):
        try:
            df = fetch_data(symbol, timeframe.value, start, end)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

        if df.empty:
            st.warning("No data returned.")
            return

        strategy_cls = strategy_registry.get(strat_name)
        strategy = strategy_cls(**params)
        engine = IndicatorEngine()

        for ind_spec in strategy.required_indicators:
            spec = ind_spec.copy()
            name = spec.pop("name")
            try:
                df = engine.compute(df, name, **spec)
            except Exception as e:
                st.error(f"Indicator '{name}' failed: {e}")
                return

        try:
            df = strategy.generate(df)
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            return

        config = BacktestConfig(**bt_cfg)
        backtester = VectorizedBacktester(config)
        try:
            result = backtester.run(df)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            return

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Tear Sheet", "Risk Analysis", "Regime Detection", "Distributions"])

    # --- Tear Sheet ---
    with tab1:
        ts = TearSheet(result.equity_curve, result.trades)
        report = ts.generate()

        for category, metrics in report.items():
            st.subheader(category.title())
            if isinstance(metrics, dict):
                cols = st.columns(min(len(metrics), 4))
                for i, (key, val) in enumerate(metrics.items()):
                    with cols[i % len(cols)]:
                        if isinstance(val, float):
                            if abs(val) < 1:
                                st.metric(key.replace("_", " ").title(), f"{val:.4f}")
                            else:
                                st.metric(key.replace("_", " ").title(), f"{val:.2f}")
                        else:
                            st.metric(key.replace("_", " ").title(), str(val))

        # Monthly returns
        st.subheader("Monthly Returns")
        monthly = ts.monthly_returns()
        if not monthly.empty:
            heatmap_fig = monthly_returns_heatmap(result.equity_curve)
            st.plotly_chart(heatmap_fig, use_container_width=True)

        # Drawdown analysis
        st.subheader("Drawdown Analysis")
        dd_analysis = ts.drawdown_analysis()
        st.metric("Max Drawdown", f"{dd_analysis['max_drawdown']*100:.2f}%")
        if dd_analysis["drawdowns"]:
            dd_df = pd.DataFrame(dd_analysis["drawdowns"])
            dd_df["depth"] = dd_df["depth"].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(dd_df, use_container_width=True)

        dd_fig = drawdown_periods(result.equity_curve, top_n=5)
        st.plotly_chart(dd_fig, use_container_width=True)

    # --- Risk Analysis ---
    with tab2:
        returns = result.equity_curve.pct_change().dropna()

        st.subheader("Value at Risk")
        c1, c2, c3 = st.columns(3)
        c1.metric("Parametric VaR (95%)", f"{parametric_var(returns)*100:.3f}%")
        c2.metric("Historical VaR (95%)", f"{historical_var(returns)*100:.3f}%")
        c3.metric("CVaR (95%)", f"{cvar(returns)*100:.3f}%")

        st.subheader("Drawdown")
        dd_fig = drawdown_chart(result.equity_curve)
        st.plotly_chart(dd_fig, use_container_width=True)

    # --- Regime Detection ---
    with tab3:
        st.subheader("Market Regime Detection (KMeans)")
        returns = df["close"].pct_change().dropna()

        detector = RegimeDetector()
        try:
            regimes = detector.detect_kmeans(returns, n_regimes=3)
            regime_stats = detector.regime_statistics(returns, regimes)

            st.dataframe(regime_stats, use_container_width=True)

            # Plot regimes on price
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close"))

            colors = {0: "red", 1: "gray", 2: "green"}
            labels = {0: "Bear", 1: "Neutral", 2: "Bull"}
            for regime_id in regimes.dropna().unique():
                mask = regimes == regime_id
                fig.add_trace(go.Scatter(
                    x=df.index[mask],
                    y=df["close"][mask],
                    mode="markers",
                    marker=dict(color=colors.get(int(regime_id), "blue"), size=4),
                    name=labels.get(int(regime_id), f"Regime {regime_id}"),
                ))
            fig.update_layout(template="plotly_white", height=400, title="Price with Regime Overlay")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Regime detection failed: {e}")

    # --- Distributions ---
    with tab4:
        st.subheader("Return Distribution")
        returns = result.equity_curve.pct_change().dropna()
        hist_fig = returns_histogram(returns)
        st.plotly_chart(hist_fig, use_container_width=True)

        st.subheader("Key Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{returns.mean()*100:.4f}%")
        c2.metric("Std Dev", f"{returns.std()*100:.4f}%")
        c3.metric("Skewness", f"{returns.skew():.3f}")
        c4.metric("Kurtosis", f"{returns.kurtosis():.3f}")
