"""Indicator computation engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ta_framework.core.exceptions import (
    IndicatorError,
    InsufficientDataError,
    InvalidParameterError,
)
from ta_framework.core.registry import indicator_registry
from ta_framework.core.types import IndicatorCategory, IndicatorTier
from ta_framework.indicators.catalog import INDICATOR_CATALOG
from ta_framework.indicators.wrappers import WRAPPER_MAP


class IndicatorEngine:
    """Central engine for computing technical indicators on OHLCV data.

    Uses :data:`WRAPPER_MAP` for built-in indicators and falls back to
    :data:`indicator_registry` for user-registered custom indicators.
    """

    def __init__(self) -> None:
        # Seed the registry with catalog wrappers so custom indicators
        # added later via `register()` are discoverable alongside built-ins.
        for name, func in WRAPPER_MAP.items():
            if name not in indicator_registry:
                meta = {}
                cat_entry = INDICATOR_CATALOG.get(name)
                if cat_entry is not None:
                    meta = {"category": cat_entry.category.value, "tier": cat_entry.tier.value}
                indicator_registry.register(name, **meta)(func)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        **params: Any,
    ) -> pd.DataFrame:
        """Compute a single indicator and return the augmented DataFrame.

        Parameters
        ----------
        df : DataFrame
            OHLCV data.
        indicator_name : str
            Name of the indicator (must exist in the wrapper map or registry).
        **params
            Keyword arguments forwarded to the underlying wrapper function.

        Raises
        ------
        IndicatorError
            If the indicator name is unknown.
        InsufficientDataError
            If *df* is too short for the indicator's look-back period.
        InvalidParameterError
            If the wrapper rejects a parameter.
        """
        func = self._resolve(indicator_name)

        if df.empty:
            raise InsufficientDataError(
                f"Cannot compute '{indicator_name}' on an empty DataFrame."
            )

        try:
            return func(df, **params)
        except TypeError as exc:
            raise InvalidParameterError(
                f"Invalid parameters for '{indicator_name}': {exc}"
            ) from exc
        except Exception as exc:
            raise IndicatorError(
                f"Error computing '{indicator_name}': {exc}"
            ) from exc

    def compute_batch(
        self,
        df: pd.DataFrame,
        indicators: list[dict],
    ) -> pd.DataFrame:
        """Compute multiple indicators in sequence.

        Parameters
        ----------
        df : DataFrame
            OHLCV data.
        indicators : list[dict]
            Each dict must have ``"name"`` and may have ``"params"``::

                [
                    {"name": "sma", "params": {"length": 20}},
                    {"name": "rsi", "params": {"length": 14}},
                ]
        """
        result = df.copy()
        for spec in indicators:
            name = spec["name"]
            params = spec.get("params", {})
            result = self.compute(result, name, **params)
        return result

    def register(
        self,
        name: str,
        func: callable,
        category: IndicatorCategory = IndicatorCategory.CUSTOM,
        tier: IndicatorTier = IndicatorTier.TIER1,
    ) -> None:
        """Register a custom indicator function at runtime.

        The function signature must be ``func(df, **params) -> DataFrame``.
        """
        WRAPPER_MAP[name] = func
        indicator_registry.register(name, category=category.value, tier=tier.value)(func)

    def available(self) -> list[str]:
        """Return sorted list of all registered indicator names."""
        names = set(WRAPPER_MAP.keys()) | set(indicator_registry.keys())
        return sorted(names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> callable:
        """Look up indicator function by *name*."""
        func = WRAPPER_MAP.get(name)
        if func is not None:
            return func
        if name in indicator_registry:
            return indicator_registry.get(name)
        raise IndicatorError(
            f"Unknown indicator '{name}'. "
            f"Available: {', '.join(self.available()[:20])}..."
        )
