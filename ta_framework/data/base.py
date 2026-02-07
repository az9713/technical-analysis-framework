"""Abstract base class for data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import pandas as pd

from ta_framework.core.config import FrameworkConfig
from ta_framework.core.exceptions import DataError
from ta_framework.core.types import AssetClass, Timeframe

DateLike = Union[str, datetime]

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    """Base class that every data provider must implement.

    Subclasses provide the actual data-fetching logic while this class
    supplies common validation and normalisation helpers.
    """

    def __init__(self) -> None:
        self.config = FrameworkConfig.get()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: DateLike,
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for *symbol* over the requested range.

        Returns a DataFrame with a ``DatetimeIndex`` named ``"date"`` and
        columns ``open, high, low, close, volume`` (all lowercase, float64).
        """

    @abstractmethod
    def supported_assets(self) -> list[AssetClass]:
        """Return the asset classes this provider can serve."""

    @abstractmethod
    def search_symbols(self, query: str) -> list[dict]:
        """Search for symbols matching *query*.

        Each dict should contain at least ``{"symbol": ..., "name": ...}``.
        """

    # ------------------------------------------------------------------
    # Validation / normalisation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has the expected OHLCV schema.

        * Columns are lowercased.
        * Required columns are present.
        * Index is a ``DatetimeIndex`` named ``"date"``.
        * Rows are sorted chronologically.
        * Numeric columns are cast to ``float64``.
        """
        if df.empty:
            raise DataError("Received an empty DataFrame.")

        # Lowercase columns
        df.columns = [c.lower().strip() for c in df.columns]

        # If 'date' is a regular column, promote it to the index
        if "date" in df.columns:
            df = df.set_index("date")

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                raise DataError(f"Cannot convert index to DatetimeIndex: {exc}") from exc

        df.index.name = "date"

        # Check required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataError(f"Missing required columns: {missing}")

        # Keep only OHLCV (drop extra columns that providers may return)
        df = df[REQUIRED_COLUMNS].copy()

        # Numeric types
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort ascending by date
        df = df.sort_index()

        # Drop duplicate index entries
        df = df[~df.index.duplicated(keep="last")]

        return df
