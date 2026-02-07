"""Technical Analysis Framework: indicators, signals, backtesting, risk management, and analytics."""

__version__ = "0.1.0"

from ta_framework.core.types import (
    AssetClass,
    Timeframe,
    SignalDirection,
    Signal,
    Trade,
    BacktestConfig,
)
from ta_framework.core.config import FrameworkConfig
from ta_framework.core.registry import Registry
