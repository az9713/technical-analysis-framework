"""Risk management: position sizing, VaR, stop-loss."""

from ta_framework.risk.position_sizing import (
    FixedFractional,
    KellyCriterion,
    PositionSizer,
    RiskParity,
    VolatilityBased,
)
from ta_framework.risk.var import cvar, historical_var, monte_carlo_var, parametric_var
from ta_framework.risk.stops import atr_stop, chandelier_stop, fixed_stop, trailing_stop
from ta_framework.risk.portfolio import (
    calmar_ratio,
    max_drawdown_duration,
    portfolio_volatility,
    risk_contribution,
    ulcer_index,
)

__all__ = [
    # Position sizing
    "PositionSizer",
    "FixedFractional",
    "KellyCriterion",
    "VolatilityBased",
    "RiskParity",
    # VaR
    "parametric_var",
    "historical_var",
    "monte_carlo_var",
    "cvar",
    # Stops
    "fixed_stop",
    "atr_stop",
    "trailing_stop",
    "chandelier_stop",
    # Portfolio
    "portfolio_volatility",
    "max_drawdown_duration",
    "calmar_ratio",
    "ulcer_index",
    "risk_contribution",
]
