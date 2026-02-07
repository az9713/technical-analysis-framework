"""Signal generation and strategies."""

from ta_framework.signals.base import SignalGenerator
from ta_framework.signals.composite import CompositeSignal, CombineMode
from ta_framework.signals.rules import breakout, crossover, divergence, threshold
from ta_framework.signals.strategies import (
    BollingerBandStrategy,
    EMACrossStrategy,
    MACDStrategy,
    RSIStrategy,
    SupertrendStrategy,
    TTMSqueezeStrategy,
)

__all__ = [
    "SignalGenerator",
    "CompositeSignal",
    "CombineMode",
    "breakout",
    "crossover",
    "divergence",
    "threshold",
    "BollingerBandStrategy",
    "EMACrossStrategy",
    "MACDStrategy",
    "RSIStrategy",
    "SupertrendStrategy",
    "TTMSqueezeStrategy",
]
