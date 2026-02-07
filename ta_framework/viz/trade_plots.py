"""Trade visualization: markers on charts and P&L plots."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta_framework.core.types import SignalDirection


def trade_markers(df: pd.DataFrame, trades: list) -> go.Figure:
    """Candlestick chart with buy/sell trade markers.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    trades : list
        List of Trade objects with entry_time, exit_time, direction, entry_price, exit_price.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )

    buy_times, buy_prices = [], []
    sell_times, sell_prices = [], []

    for trade in trades:
        entry_time = getattr(trade, "entry_time", None)
        exit_time = getattr(trade, "exit_time", None)
        direction = getattr(trade, "direction", SignalDirection.LONG)
        entry_price = getattr(trade, "entry_price", 0)
        exit_price = getattr(trade, "exit_price", 0)

        if direction in (SignalDirection.LONG, "long"):
            if entry_time is not None:
                buy_times.append(entry_time)
                buy_prices.append(entry_price)
            if exit_time is not None:
                sell_times.append(exit_time)
                sell_prices.append(exit_price)
        else:
            if entry_time is not None:
                sell_times.append(entry_time)
                sell_prices.append(entry_price)
            if exit_time is not None:
                buy_times.append(exit_time)
                buy_prices.append(exit_price)

    if buy_times:
        fig.add_trace(
            go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="green",
                    line=dict(width=1, color="darkgreen"),
                ),
            )
        )

    if sell_times:
        fig.add_trace(
            go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="red",
                    line=dict(width=1, color="darkred"),
                ),
            )
        )

    fig.update_layout(
        title="Trades",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
    )

    return fig


def pnl_chart(trades: list) -> go.Figure:
    """Cumulative P&L line chart and per-trade bar chart.

    Parameters
    ----------
    trades : list
        List of Trade objects.

    Returns
    -------
    go.Figure
    """
    if not trades:
        return go.Figure()

    pnls = []
    labels = []
    for i, trade in enumerate(trades):
        pnl = getattr(trade, "net_pnl", None)
        if pnl is None:
            pnl = getattr(trade, "pnl", 0.0)
        pnls.append(pnl)
        exit_t = getattr(trade, "exit_time", None)
        labels.append(str(exit_t) if exit_t else f"Trade {i + 1}")

    pnl_series = pd.Series(pnls, index=labels, dtype=float)
    cum_pnl = pnl_series.cumsum()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=["Cumulative P&L", "Per-Trade P&L"],
        row_heights=[0.6, 0.4],
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cum_pnl) + 1)),
            y=cum_pnl.values,
            mode="lines+markers",
            name="Cumulative P&L",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    colors = ["green" if p >= 0 else "red" for p in pnl_series]
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(pnl_series) + 1)),
            y=pnl_series.values,
            marker_color=colors,
            name="Trade P&L",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=600,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Trade #", row=2, col=1)
    fig.update_yaxes(title_text="P&L", row=1, col=1)
    fig.update_yaxes(title_text="P&L", row=2, col=1)

    return fig
