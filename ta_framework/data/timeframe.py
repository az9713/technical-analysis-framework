"""Multi-timeframe utilities: resampling and alignment."""

from __future__ import annotations

import pandas as pd

from ta_framework.core.exceptions import DataError
from ta_framework.core.types import Timeframe


def resample(df: pd.DataFrame, target_timeframe: Timeframe) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a coarser timeframe.

    Uses OHLCV-aware aggregation:
    * open  -> first
    * high  -> max
    * low   -> min
    * close -> last
    * volume -> sum

    Parameters
    ----------
    df : DataFrame
        Source OHLCV data with ``DatetimeIndex``.
    target_timeframe : Timeframe
        The target timeframe to resample to.  Must be equal to or coarser
        than the source frequency.

    Returns
    -------
    DataFrame
        Resampled OHLCV data.
    """
    if df.empty:
        return df.copy()

    freq = target_timeframe.to_pandas_freq()

    resampled = df.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )

    # Drop rows where open is NaN (periods with no data)
    resampled = resampled.dropna(subset=["open"])

    resampled.index.name = "date"
    return resampled


def align_timeframes(
    dfs: dict[Timeframe, pd.DataFrame],
) -> dict[Timeframe, pd.DataFrame]:
    """Align multiple timeframe DataFrames to a common date range.

    The common range is the intersection of all DataFrames' date ranges
    (max of all starts, min of all ends).

    Parameters
    ----------
    dfs : dict[Timeframe, DataFrame]
        Mapping of timeframe to its OHLCV DataFrame.

    Returns
    -------
    dict[Timeframe, DataFrame]
        Trimmed DataFrames sharing the same start/end boundaries.
    """
    if not dfs:
        return {}

    # Determine the overlapping date range
    starts = [df.index.min() for df in dfs.values() if not df.empty]
    ends = [df.index.max() for df in dfs.values() if not df.empty]

    if not starts or not ends:
        raise DataError("Cannot align empty DataFrames.")

    common_start = max(starts)
    common_end = min(ends)

    if common_start > common_end:
        raise DataError(
            f"No overlapping date range found. "
            f"Latest start={common_start}, earliest end={common_end}."
        )

    aligned: dict[Timeframe, pd.DataFrame] = {}
    for tf, df in dfs.items():
        trimmed = df[(df.index >= common_start) & (df.index <= common_end)].copy()
        aligned[tf] = trimmed

    return aligned
