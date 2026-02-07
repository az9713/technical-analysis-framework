"""Data quality checks and cleaning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta_framework.core.exceptions import DataQualityError


class DataQualityChecker:
    """Analyse and clean OHLCV DataFrames.

    All ``check_*`` methods return a report dict that can be inspected
    programmatically.  The :meth:`clean` method returns a sanitised copy.
    """

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def check_gaps(df: pd.DataFrame, max_gap_pct: float = 0.05) -> dict:
        """Detect missing-data gaps in the time series.

        Parameters
        ----------
        df : DataFrame
            OHLCV DataFrame with ``DatetimeIndex``.
        max_gap_pct : float
            Maximum tolerable fraction of NaN rows (0 - 1).

        Returns
        -------
        dict
            ``{"total_rows", "nan_rows", "nan_pct", "passed", "gap_locations"}``
        """
        total = len(df)
        if total == 0:
            return {"total_rows": 0, "nan_rows": 0, "nan_pct": 0.0, "passed": True, "gap_locations": []}

        nan_mask = df[["open", "high", "low", "close"]].isna().any(axis=1)
        nan_rows = int(nan_mask.sum())
        nan_pct = nan_rows / total

        gap_locations = list(df.index[nan_mask])

        return {
            "total_rows": total,
            "nan_rows": nan_rows,
            "nan_pct": round(nan_pct, 6),
            "passed": nan_pct <= max_gap_pct,
            "gap_locations": gap_locations,
        }

    @staticmethod
    def check_outliers(df: pd.DataFrame, z_threshold: float = 5.0) -> dict:
        """Flag rows where log-returns exceed *z_threshold* standard deviations.

        Parameters
        ----------
        df : DataFrame
            OHLCV DataFrame.
        z_threshold : float
            Number of standard deviations to consider an outlier.

        Returns
        -------
        dict
            ``{"outlier_count", "outlier_pct", "passed", "outlier_locations"}``
        """
        total = len(df)
        if total < 2:
            return {"outlier_count": 0, "outlier_pct": 0.0, "passed": True, "outlier_locations": []}

        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        if log_ret.std() == 0:
            return {"outlier_count": 0, "outlier_pct": 0.0, "passed": True, "outlier_locations": []}

        z_scores = (log_ret - log_ret.mean()) / log_ret.std()
        outlier_mask = z_scores.abs() > z_threshold
        outlier_count = int(outlier_mask.sum())

        return {
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / total, 6),
            "passed": outlier_count == 0,
            "outlier_locations": list(log_ret.index[outlier_mask]),
        }

    @staticmethod
    def check_ohlc_consistency(df: pd.DataFrame) -> dict:
        """Verify OHLC relationships and non-negative volume.

        Rules checked:
        * ``high >= max(open, close, low)``
        * ``low  <= min(open, close, high)``
        * ``volume >= 0``

        Returns
        -------
        dict
            ``{"high_violations", "low_violations", "volume_violations", "passed"}``
        """
        if df.empty:
            return {"high_violations": 0, "low_violations": 0, "volume_violations": 0, "passed": True}

        high_bad = (
            (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
        )
        low_bad = (
            (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
        )
        vol_bad = df["volume"] < 0

        h = int(high_bad.sum())
        l_ = int(low_bad.sum())
        v = int(vol_bad.sum())

        return {
            "high_violations": h,
            "low_violations": l_,
            "volume_violations": v,
            "passed": (h + l_ + v) == 0,
        }

    def full_check(self, df: pd.DataFrame) -> dict:
        """Run all checks and return a combined report."""
        return {
            "gaps": self.check_gaps(df),
            "outliers": self.check_outliers(df),
            "ohlc_consistency": self.check_ohlc_consistency(df),
        }

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean(
        df: pd.DataFrame,
        fill_method: str = "ffill",
        remove_outliers: bool = False,
        z_threshold: float = 5.0,
    ) -> pd.DataFrame:
        """Return a cleaned copy of the DataFrame.

        Parameters
        ----------
        df : DataFrame
            Raw OHLCV data.
        fill_method : str
            Method for filling NaN gaps (``"ffill"``, ``"bfill"``, ``"interpolate"``).
        remove_outliers : bool
            If True, rows whose close log-returns exceed *z_threshold*
            standard deviations are removed.
        z_threshold : float
            Outlier threshold (only used when *remove_outliers* is True).
        """
        df = df.copy()

        # Fill NaN gaps
        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        elif fill_method == "interpolate":
            df = df.interpolate(method="time")
        else:
            df = df.ffill()

        # Fill any remaining leading NaN with bfill
        df = df.bfill()

        # Remove outliers
        if remove_outliers and len(df) > 1:
            log_ret = np.log(df["close"] / df["close"].shift(1))
            std = log_ret.std()
            if std > 0:
                z_scores = (log_ret - log_ret.mean()) / std
                df = df[z_scores.abs().fillna(0) <= z_threshold]

        # Ensure non-negative volume
        df["volume"] = df["volume"].clip(lower=0)

        # Fix OHLC consistency: clamp open/close within [low, high]
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        return df
