"""Base class and registration decorator for user-defined indicators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import pandas as pd

from ta_framework.core.registry import indicator_registry
from ta_framework.core.types import IndicatorCategory, IndicatorTier

T = TypeVar("T", bound="CustomIndicator")


class CustomIndicator(ABC):
    """Abstract base for user-defined technical indicators.

    Subclasses must set :attr:`name` and :attr:`category` and implement
    :meth:`compute`.

    Example
    -------
    ::

        @register_indicator(tier=IndicatorTier.TIER1)
        class MyRSI(CustomIndicator):
            name = "my_rsi"
            category = IndicatorCategory.MOMENTUM

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out["MY_RSI"] = ...
                return out

            @property
            def output_columns(self) -> list[str]:
                return ["MY_RSI"]
    """

    name: str = ""
    category: IndicatorCategory = IndicatorCategory.CUSTOM

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the indicator and return *df* with new column(s) appended."""

    @property
    def output_columns(self) -> list[str]:
        """Column names produced by :meth:`compute`."""
        return []


def register_indicator(
    tier: IndicatorTier = IndicatorTier.TIER1,
) -> Callable[[type[T]], type[T]]:
    """Class decorator that adds a :class:`CustomIndicator` subclass to the
    global :data:`indicator_registry`.

    The registered callable wraps the class so that the engine can call it
    with the standard ``func(df, **params) -> DataFrame`` signature.
    """

    def decorator(cls: type[T]) -> type[T]:
        if not issubclass(cls, CustomIndicator):
            raise TypeError(f"{cls.__name__} must subclass CustomIndicator.")

        instance = cls()
        name = instance.name or cls.__name__.lower()

        def _wrapper(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
            return instance.compute(df)

        indicator_registry.register(
            name,
            category=instance.category.value,
            tier=tier.value,
        )(_wrapper)

        # Also store the wrapper on the class for introspection
        cls._registry_wrapper = _wrapper
        return cls

    return decorator
