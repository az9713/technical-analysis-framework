"""Heatmap visualizations for correlation and monthly returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def correlation_heatmap(returns_dict: dict[str, pd.Series]) -> go.Figure:
    """Correlation matrix heatmap for multiple assets.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of asset name to return series.

    Returns
    -------
    go.Figure
    """
    df = pd.DataFrame(returns_dict).dropna()
    corr = df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=12),
        )
    )

    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_white",
        height=500,
        width=600,
    )

    return fig


def monthly_returns_heatmap(equity_curve: pd.Series) -> go.Figure:
    """Year x month heatmap of returns.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve (indexed by date).

    Returns
    -------
    go.Figure
    """
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return go.Figure()

    returns.index = pd.to_datetime(returns.index)
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    table = pd.DataFrame(
        {
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        }
    )
    pivot = table.pivot_table(
        values="Return", index="Year", columns="Month", aggfunc="sum"
    )

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    # Ensure all 12 months present
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[range(1, 13)]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=pivot.index.astype(str).tolist(),
            colorscale="RdYlGn",
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            textfont=dict(size=11),
        )
    )

    fig.update_layout(
        title="Monthly Returns (%)",
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
        height=max(300, 50 * len(pivot)),
    )

    return fig
