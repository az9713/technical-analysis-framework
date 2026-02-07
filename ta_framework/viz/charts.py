"""Core chart types: candlestick, indicator panels, multi-panel."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def candlestick_chart(
    df: pd.DataFrame,
    overlays: list[str] | None = None,
    title: str = "",
) -> go.Figure:
    """Candlestick chart with optional overlay indicators and volume panel.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame. May contain additional columns for overlays.
    overlays : list[str] | None
        Column names to plot on the price axis (e.g. SMA, EMA, BB).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    has_volume = "volume" in df.columns and df["volume"].sum() > 0
    rows = 2 if has_volume else 1
    row_heights = [0.75, 0.25] if has_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    if overlays:
        for col_name in overlays:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col_name],
                        mode="lines",
                        name=col_name,
                    ),
                    row=1,
                    col=1,
                )

    if has_volume:
        colors = [
            "green" if c >= o else "red"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=colors,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)

    return fig


def indicator_panel(
    df: pd.DataFrame, indicators: list[str], title: str = ""
) -> go.Figure:
    """Oscillator-style panel chart with each indicator in its own subplot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing indicator columns.
    indicators : list[str]
        Column names to plot.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    valid = [ind for ind in indicators if ind in df.columns]
    n = max(len(valid), 1)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=valid or ["(no data)"],
    )

    for i, col_name in enumerate(valid, start=1):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_name],
                mode="lines",
                name=col_name,
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=250 * n,
        template="plotly_white",
        showlegend=True,
    )

    return fig


def multi_panel_chart(
    df: pd.DataFrame,
    overlays: list[str] | None = None,
    oscillators: list[str] | None = None,
    show_volume: bool = True,
) -> go.Figure:
    """Combined multi-panel chart: candlestick + overlays, volume, oscillators.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with indicator columns.
    overlays : list[str] | None
        Column names for price overlays.
    oscillators : list[str] | None
        Column names for oscillator subplots.
    show_volume : bool
        Whether to show the volume panel.

    Returns
    -------
    go.Figure
    """
    has_volume = show_volume and "volume" in df.columns and df["volume"].sum() > 0
    osc_list = [o for o in (oscillators or []) if o in df.columns]
    n_osc = len(osc_list)

    total_rows = 1 + (1 if has_volume else 0) + n_osc
    heights = [0.5]
    titles = ["Price"]

    if has_volume:
        heights.append(0.15)
        titles.append("Volume")

    for osc in osc_list:
        heights.append(0.35 / max(n_osc, 1))
        titles.append(osc)

    # Normalize heights to sum to 1
    total_h = sum(heights)
    heights = [h / total_h for h in heights]

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=titles,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # Overlays
    for col_name in (overlays or []):
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    mode="lines",
                    name=col_name,
                ),
                row=1,
                col=1,
            )

    current_row = 2

    # Volume
    if has_volume:
        colors = [
            "green" if c >= o else "red"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=colors,
                name="Volume",
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        current_row += 1

    # Oscillators
    for osc in osc_list:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[osc],
                mode="lines",
                name=osc,
            ),
            row=current_row,
            col=1,
        )
        current_row += 1

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=200 + 200 * total_rows,
        template="plotly_white",
    )

    return fig
