"""Return distribution visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def returns_histogram(
    returns: pd.Series, bins: int = 50
) -> go.Figure:
    """Histogram of returns with normal overlay and VaR lines.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    bins : int
        Number of histogram bins.

    Returns
    -------
    go.Figure
    """
    clean = returns.dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=clean.values,
            nbinsx=bins,
            name="Returns",
            opacity=0.7,
            marker_color="steelblue",
            histnorm="probability density",
        )
    )

    # Normal distribution overlay
    mu, sigma = clean.mean(), clean.std()
    if sigma > 0:
        x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        y_normal = stats.norm.pdf(x_range, mu, sigma)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode="lines",
                name="Normal",
                line=dict(color="red", width=2),
            )
        )

    # VaR lines
    var_95 = np.percentile(clean, 5)
    var_99 = np.percentile(clean, 1)

    fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"VaR 95%: {var_95:.4f}",
    )
    fig.add_vline(
        x=var_99,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 99%: {var_99:.4f}",
    )

    fig.update_layout(
        title="Return Distribution",
        xaxis_title="Return",
        yaxis_title="Density",
        template="plotly_white",
        height=450,
    )

    return fig


def qq_plot(returns: pd.Series) -> go.Figure:
    """Quantile-quantile plot against normal distribution.

    Parameters
    ----------
    returns : pd.Series
        Return series.

    Returns
    -------
    go.Figure
    """
    clean = returns.dropna().sort_values()
    n = len(clean)

    if n < 2:
        return go.Figure()

    theoretical = stats.norm.ppf(
        (np.arange(1, n + 1) - 0.5) / n,
        loc=clean.mean(),
        scale=clean.std(),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=clean.values,
            mode="markers",
            name="Q-Q",
            marker=dict(color="steelblue", size=4),
        )
    )

    # 45-degree reference line
    min_val = min(theoretical.min(), clean.min())
    max_val = max(theoretical.max(), clean.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Normal Reference",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.update_layout(
        title="Q-Q Plot (vs. Normal)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_white",
        height=450,
    )

    return fig
