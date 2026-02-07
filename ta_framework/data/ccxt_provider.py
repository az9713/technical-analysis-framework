"""Cryptocurrency data provider backed by the *ccxt* library (optional)."""

from __future__ import annotations

from datetime import datetime
from typing import Union

import pandas as pd

from ta_framework.core.exceptions import DataError, InvalidSymbolError
from ta_framework.core.registry import provider_registry
from ta_framework.core.types import AssetClass, Timeframe
from ta_framework.data.base import DataProvider, DateLike

# Map Timeframe enum values to ccxt timeframe strings.
_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1w",
    "1mo": "1M",
}

_SUPPORTED_EXCHANGES = ("binance", "coinbase")


def _has_ccxt() -> bool:
    try:
        import ccxt  # noqa: F401
        return True
    except ImportError:
        return False


@provider_registry.register("ccxt")
class CCXTProvider(DataProvider):
    """Fetch cryptocurrency OHLCV data via ccxt.

    If ``ccxt`` is not installed, the class can still be imported but
    :meth:`fetch` will raise a helpful error.
    """

    def __init__(self, exchange: str = "binance") -> None:
        super().__init__()
        self.exchange_id = exchange.lower()
        if self.exchange_id not in _SUPPORTED_EXCHANGES:
            raise DataError(
                f"Unsupported exchange '{exchange}'. "
                f"Supported: {', '.join(_SUPPORTED_EXCHANGES)}"
            )
        self._exchange = None

    def _get_exchange(self):
        if not _has_ccxt():
            raise DataError(
                "ccxt is not installed. Install it with: pip install ccxt"
            )
        if self._exchange is None:
            import ccxt
            exchange_cls = getattr(ccxt, self.exchange_id, None)
            if exchange_cls is None:
                raise DataError(f"Exchange '{self.exchange_id}' not found in ccxt.")
            self._exchange = exchange_cls({"enableRateLimit": True})
        return self._exchange

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.D1,
        start: DateLike = "2020-01-01",
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        exchange = self._get_exchange()

        tf_str = _TF_MAP.get(timeframe.value, "1d")

        # Convert start to millisecond timestamp
        if isinstance(start, str):
            start_dt = pd.Timestamp(start)
        else:
            start_dt = pd.Timestamp(start)
        since_ms = int(start_dt.timestamp() * 1000)

        if end is not None:
            if isinstance(end, str):
                end_dt = pd.Timestamp(end)
            else:
                end_dt = pd.Timestamp(end)
            end_ms = int(end_dt.timestamp() * 1000)
        else:
            end_ms = None

        all_ohlcv: list[list] = []
        limit = 1000

        try:
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, tf_str, since=since_ms, limit=limit)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                # Move forward past the last candle
                since_ms = ohlcv[-1][0] + 1
                if end_ms is not None and since_ms >= end_ms:
                    break
                if len(ohlcv) < limit:
                    break
        except Exception as exc:
            raise InvalidSymbolError(
                f"Failed to fetch '{symbol}' from {self.exchange_id}: {exc}"
            ) from exc

        if not all_ohlcv:
            raise InvalidSymbolError(f"No data returned for '{symbol}' on {self.exchange_id}.")

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("date").drop(columns=["timestamp"])

        # Trim to end date if provided
        if end_ms is not None:
            df = df[df.index <= end_dt]

        return self.validate(df)

    def supported_assets(self) -> list[AssetClass]:
        return [AssetClass.CRYPTO]

    def search_symbols(self, query: str) -> list[dict]:
        try:
            exchange = self._get_exchange()
            exchange.load_markets()
            matches = [
                {"symbol": s, "name": s}
                for s in exchange.symbols
                if query.upper() in s.upper()
            ]
            return matches[:50]
        except Exception:
            return []
