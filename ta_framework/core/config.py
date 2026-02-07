"""Framework-wide configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ta_framework.core.types import Timeframe


@dataclass
class FrameworkConfig:
    """Global configuration for the TA framework."""

    default_timeframe: Timeframe = Timeframe.D1
    default_lookback_bars: int = 500
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    max_workers: int = 4
    log_level: str = "WARNING"
    default_benchmark: str = "SPY"
    ohlcv_columns: list[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )
    extra: dict[str, Any] = field(default_factory=dict)

    _instance: FrameworkConfig | None = None

    @classmethod
    def get(cls) -> FrameworkConfig:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
