"""Reusable signal rule functions.

Each function accepts pandas Series and returns a Series of 1 / -1 / 0
matching the framework signal convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Detect crossover events between *fast* and *slow* Series.

    Returns
    -------
    pd.Series
        1 when *fast* crosses above *slow*, -1 when below, 0 otherwise.
    """
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)

    cross_up = (fast > slow) & (prev_fast <= prev_slow)
    cross_down = (fast < slow) & (prev_fast >= prev_slow)

    signal = pd.Series(0, index=fast.index, dtype=np.int8)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def threshold(
    series: pd.Series,
    upper: float,
    lower: float,
) -> pd.Series:
    """Generate signals based on threshold levels.

    Returns
    -------
    pd.Series
        1 when *series* drops below *lower* (oversold / buy),
        -1 when *series* rises above *upper* (overbought / sell),
        0 otherwise.
    """
    signal = pd.Series(0, index=series.index, dtype=np.int8)
    signal[series < lower] = 1
    signal[series > upper] = -1
    return signal


def divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 14,
) -> pd.Series:
    """Detect bullish / bearish divergence between *price* and *indicator*.

    Bullish divergence (1): price makes a lower low while the indicator
    makes a higher low.  Bearish divergence (-1): price makes a higher
    high while the indicator makes a lower high.

    Parameters
    ----------
    price : pd.Series
    indicator : pd.Series
    lookback : int
        Rolling window for local extrema comparison.

    Returns
    -------
    pd.Series
        1 for bullish divergence, -1 for bearish, 0 otherwise.
    """
    signal = pd.Series(0, index=price.index, dtype=np.int8)
    if len(price) < lookback * 2:
        return signal

    price_low = price.rolling(lookback).min()
    price_high = price.rolling(lookback).max()
    ind_low = indicator.rolling(lookback).min()
    ind_high = indicator.rolling(lookback).max()

    prev_price_low = price_low.shift(lookback)
    prev_ind_low = ind_low.shift(lookback)
    prev_price_high = price_high.shift(lookback)
    prev_ind_high = ind_high.shift(lookback)

    # Bullish: price lower low, indicator higher low
    bullish = (price_low < prev_price_low) & (ind_low > prev_ind_low)
    # Bearish: price higher high, indicator lower high
    bearish = (price_high > prev_price_high) & (ind_high < prev_ind_high)

    signal[bullish] = 1
    signal[bearish] = -1
    return signal


def breakout(
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
) -> pd.Series:
    """Generate signals when *close* breaks out of a band.

    Returns
    -------
    pd.Series
        1 when *close* breaks below *lower_band* (mean-reversion buy),
        -1 when *close* breaks above *upper_band* (mean-reversion sell),
        0 otherwise.
    """
    signal = pd.Series(0, index=close.index, dtype=np.int8)
    signal[close < lower_band] = 1
    signal[close > upper_band] = -1
    return signal
