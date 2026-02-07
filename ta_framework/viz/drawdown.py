"""Drawdown visualization charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def drawdown_chart(equity_curve: pd.Series) -> go.Figure:
    """Underwater chart showing drawdown depth over time.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve.

    Returns
    -------
    go.Figure
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="red", width=1),
            name="Drawdown",
        )
    )

    fig.update_layout(
        title="Underwater Chart",
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        template="plotly_white",
        height=400,
    )

    return fig


def drawdown_periods(
    equity_curve: pd.Series, top_n: int = 5
) -> go.Figure:
    """Highlight top N drawdown periods on the equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve.
    top_n : int
        Number of worst drawdown periods to highlight.

    Returns
    -------
    go.Figure
    """
    running_max = equity_curve.cummax()
    dd_series = (equity_curve - running_max) / running_max

    # Identify drawdown periods
    periods = []
    in_dd = False
    start_idx = None

    for i in range(len(dd_series)):
        if dd_series.iloc[i] < 0 and not in_dd:
            in_dd = True
            start_idx = i - 1 if i > 0 else 0
        elif dd_series.iloc[i] >= 0 and in_dd:
            in_dd = False
            segment = dd_series.iloc[start_idx: i + 1]
            periods.append(
                {
                    "start": dd_series.index[start_idx],
                    "end": dd_series.index[i],
                    "depth": abs(segment.min()),
                }
            )

    if in_dd and start_idx is not None:
        segment = dd_series.iloc[start_idx:]
        periods.append(
            {
                "start": dd_series.index[start_idx],
                "end": dd_series.index[-1],
                "depth": abs(segment.min()),
            }
        )

    periods.sort(key=lambda p: p["depth"], reverse=True)
    top = periods[:top_n]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="Equity",
            line=dict(color="blue"),
        )
    )

    colors = [
        "rgba(255,0,0,0.2)",
        "rgba(255,128,0,0.2)",
        "rgba(255,200,0,0.2)",
        "rgba(200,200,0,0.2)",
        "rgba(180,180,0,0.2)",
    ]

    for i, period in enumerate(top):
        color = colors[i % len(colors)]
        fig.add_vrect(
            x0=period["start"],
            x1=period["end"],
            fillcolor=color,
            opacity=0.5,
            layer="below",
            line_width=0,
            annotation_text=f"DD #{i + 1}: {period['depth']:.1%}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"Top {top_n} Drawdown Periods",
        yaxis_title="Equity",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
    )

    return fig
