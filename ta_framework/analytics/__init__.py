"""Performance analytics, metrics, and regime detection."""

from ta_framework.analytics.metrics import (
    calculate_all,
    calmar_ratio,
    cvar_95,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    var_95,
    volatility,
    win_rate,
    profit_factor,
)
from ta_framework.analytics.tearsheet import TearSheet
from ta_framework.analytics.benchmark import (
    alpha_beta,
    information_ratio,
    tracking_error,
    up_capture,
    down_capture,
    active_return,
)
from ta_framework.analytics.regime import RegimeDetector

__all__ = [
    # Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "volatility",
    "var_95",
    "cvar_95",
    "win_rate",
    "profit_factor",
    "calculate_all",
    # TearSheet
    "TearSheet",
    # Benchmark
    "alpha_beta",
    "information_ratio",
    "tracking_error",
    "up_capture",
    "down_capture",
    "active_return",
    # Regime
    "RegimeDetector",
]
