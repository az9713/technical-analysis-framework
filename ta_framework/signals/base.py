"""Abstract base class for signal generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ta_framework.core.exceptions import SignalError
from ta_framework.core.registry import strategy_registry


class SignalGenerator(ABC):
    """Base class for all signal generation strategies.

    Subclasses must implement ``generate`` and ``required_indicators``.
    The ``generate`` method receives a DataFrame that already contains the
    indicator columns listed in ``required_indicators`` and must add at
    minimum a ``signal`` column with values 1 (long), -1 (short), or 0
    (neutral).  It may also add ``signal_strength`` and
    ``signal_confidence`` columns.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add signal columns to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with pre-computed indicator columns.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with at least a ``signal`` column added.

        Raises
        ------
        SignalError
            If required indicator columns are missing.
        """

    @property
    @abstractmethod
    def required_indicators(self) -> list[dict]:
        """Indicator configurations this strategy depends on.

        Each dict must have a ``name`` key and optional parameter keys.
        Example::

            [{'name': 'ema', 'length': 20}, {'name': 'ema', 'length': 50}]
        """

    def validate_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        """Raise ``SignalError`` if any *columns* are missing from *df*."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise SignalError(
                f"{self.name}: missing required columns {missing}. "
                f"Available: {list(df.columns)}"
            )
