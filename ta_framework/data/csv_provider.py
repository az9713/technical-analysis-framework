"""File-based data provider for CSV and Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ta_framework.core.exceptions import DataError
from ta_framework.core.registry import provider_registry
from ta_framework.core.types import AssetClass, Timeframe
from ta_framework.data.base import DataProvider, DateLike


@provider_registry.register("csv")
class CSVProvider(DataProvider):
    """Load OHLCV data from a local CSV or Parquet file.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV or Parquet file.  Format is auto-detected from the
        file extension (``.csv`` or ``.parquet`` / ``.pq``).
    """

    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise DataError(f"File not found: {self.file_path}")

    def fetch(
        self,
        symbol: str = "",
        timeframe: Timeframe = Timeframe.D1,
        start: DateLike = "1900-01-01",
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        ext = self.file_path.suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(self.file_path, parse_dates=True)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(self.file_path)
        else:
            raise DataError(f"Unsupported file extension '{ext}'. Use .csv or .parquet.")

        # Normalise column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Promote a date-like column to the index if the index isn't already dates
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = None
            for candidate in ("date", "datetime", "timestamp", "time"):
                if candidate in df.columns:
                    date_col = candidate
                    break
            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            else:
                # Try converting the existing index
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as exc:
                    raise DataError(
                        "Cannot determine date column or convert index to DatetimeIndex."
                    ) from exc

        df.index.name = "date"

        # Filter by date range
        start_ts = pd.Timestamp(start)
        df = df[df.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            df = df[df.index <= end_ts]

        return self.validate(df)

    def supported_assets(self) -> list[AssetClass]:
        # File-based provider is asset-agnostic
        return list(AssetClass)

    def search_symbols(self, query: str) -> list[dict]:
        # Not applicable for file-based provider
        return [{"symbol": self.file_path.stem, "name": str(self.file_path)}]
