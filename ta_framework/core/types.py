"""Core enums and dataclasses used throughout the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    ETF = "etf"
    INDEX = "index"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1wk"
    MN1 = "1mo"

    def to_pandas_freq(self) -> str:
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1wk": "1W",
            "1mo": "1ME",
        }
        return mapping[self.value]

    def to_minutes(self) -> int:
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1wk": 10080,
            "1mo": 43200,
        }
        return mapping[self.value]


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    NEUTRAL = "neutral"


class IndicatorTier(int, Enum):
    TIER1 = 1  # Essential (~40 indicators)
    TIER2 = 2  # Extended (~80 indicators)
    TIER3 = 3  # Full catalog (150+)


class IndicatorCategory(str, Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OVERLAP = "overlap"
    STATISTICS = "statistics"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    timestamp: datetime
    direction: SignalDirection
    strength: float = 0.0       # 0.0 to 1.0
    confidence: float = 0.0     # 0.0 to 1.0
    source: str = ""            # strategy name that generated it
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    direction: SignalDirection = SignalDirection.LONG
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    holding_period: int = 0     # bars
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.commission - self.slippage

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001       # 0.1% per trade
    slippage_pct: float = 0.0005        # 0.05% per trade
    position_size_pct: float = 1.0      # fraction of capital per trade
    max_positions: int = 1
    risk_free_rate: float = 0.04        # annual, for Sharpe etc.
    benchmark_symbol: str | None = None
    allow_short: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorConfig:
    name: str
    category: IndicatorCategory
    tier: IndicatorTier = IndicatorTier.TIER1
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
