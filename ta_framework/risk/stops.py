"""Stop-loss strategies returning stop prices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_stop(entry_price: float, stop_pct: float) -> float:
    """Fixed percentage stop-loss.

    Parameters
    ----------
    entry_price : float
        Trade entry price.
    stop_pct : float
        Stop-loss percentage as a decimal (e.g. 0.02 for 2%).

    Returns
    -------
    float
        Stop price.
    """
    if entry_price <= 0 or stop_pct <= 0:
        return entry_price
    return entry_price * (1.0 - stop_pct)


def atr_stop(
    df: pd.DataFrame, multiplier: float = 2.0, period: int = 14
) -> pd.Series:
    """ATR-based stop-loss (below close by ATR * multiplier).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'high', 'low', 'close' columns.
    multiplier : float
        ATR multiplier.
    period : int
        ATR lookback period.

    Returns
    -------
    pd.Series
        Stop price series.
    """
    if df.empty:
        return pd.Series(dtype=float)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    return close - atr * multiplier


def trailing_stop(close: pd.Series, trail_pct: float) -> pd.Series:
    """Trailing stop-loss that follows the price upward.

    Parameters
    ----------
    close : pd.Series
        Close price series.
    trail_pct : float
        Trail percentage as a decimal (e.g. 0.05 for 5%).

    Returns
    -------
    pd.Series
        Trailing stop price series.
    """
    if close.empty or trail_pct <= 0:
        return close.copy() if not close.empty else pd.Series(dtype=float)

    stops = np.empty(len(close))
    running_max = close.iloc[0]

    for i in range(len(close)):
        price = close.iloc[i]
        if price > running_max:
            running_max = price
        stops[i] = running_max * (1.0 - trail_pct)

    return pd.Series(stops, index=close.index, name="trailing_stop")


def chandelier_stop(
    df: pd.DataFrame, period: int = 22, multiplier: float = 3.0
) -> pd.Series:
    """Chandelier exit: highest high minus ATR * multiplier.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    period : int
        Lookback period for highest high and ATR.
    multiplier : float
        ATR multiplier.

    Returns
    -------
    pd.Series
        Chandelier stop price series.
    """
    if df.empty:
        return pd.Series(dtype=float)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    highest_high = high.rolling(window=period, min_periods=1).max()

    return highest_high - atr * multiplier
