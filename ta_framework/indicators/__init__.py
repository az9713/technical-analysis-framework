"""Indicator engine and catalog."""

from ta_framework.indicators.catalog import INDICATOR_CATALOG
from ta_framework.indicators.composite import CompositeIndicator
from ta_framework.indicators.custom import CustomIndicator, register_indicator
from ta_framework.indicators.engine import IndicatorEngine
from ta_framework.indicators.wrappers import WRAPPER_MAP

__all__ = [
    "INDICATOR_CATALOG",
    "CompositeIndicator",
    "CustomIndicator",
    "IndicatorEngine",
    "WRAPPER_MAP",
    "register_indicator",
]
