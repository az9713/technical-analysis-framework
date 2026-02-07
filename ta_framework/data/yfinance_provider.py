"""Yahoo Finance data provider backed by the *yfinance* library."""

from __future__ import annotations

from datetime import datetime
from typing import Union

import pandas as pd

from ta_framework.core.exceptions import DataError, InvalidSymbolError
from ta_framework.core.registry import provider_registry
from ta_framework.core.types import AssetClass, Timeframe
from ta_framework.data.base import DataProvider, DateLike

# Map Timeframe enum values to yfinance interval strings.
_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",      # yfinance doesn't support 4h natively
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


@provider_registry.register("yfinance")
class YFinanceProvider(DataProvider):
    """Fetch data from Yahoo Finance via the ``yfinance`` package."""

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.D1,
        start: DateLike = "2020-01-01",
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataError(
                "yfinance is not installed. Install it with: pip install yfinance"
            ) from exc

        interval = _TF_MAP.get(timeframe.value, "1d")

        # yfinance doesn't natively support 4h; fetch 1h and resample
        needs_resample = timeframe == Timeframe.H4

        ticker = yf.Ticker(symbol)

        try:
            df: pd.DataFrame = ticker.history(
                start=str(start),
                end=str(end) if end else None,
                interval="1h" if needs_resample else interval,
                auto_adjust=True,
            )
        except Exception as exc:
            raise InvalidSymbolError(f"Failed to fetch data for '{symbol}': {exc}") from exc

        if df.empty:
            raise InvalidSymbolError(
                f"No data returned for symbol '{symbol}'. "
                "Check that the ticker is valid and the date range is correct."
            )

        # yfinance returns columns in Title Case; lowercase them
        df.columns = [c.lower().strip() for c in df.columns]

        # Resample to 4h if needed
        if needs_resample:
            df = df.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["open"])

        return self.validate(df)

    def supported_assets(self) -> list[AssetClass]:
        return [AssetClass.EQUITY, AssetClass.ETF, AssetClass.INDEX, AssetClass.CRYPTO, AssetClass.FOREX]

    def search_symbols(self, query: str) -> list[dict]:
        try:
            import yfinance as yf
        except ImportError:
            return []

        try:
            results = yf.Tickers(query)
            return [{"symbol": t, "name": t} for t in results.tickers]
        except Exception:
            return [{"symbol": query, "name": query}]
