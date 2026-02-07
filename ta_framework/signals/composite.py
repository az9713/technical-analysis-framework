"""Composite signal generator that combines multiple SignalGenerators."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ta_framework.core.exceptions import SignalError
from ta_framework.signals.base import SignalGenerator


class CombineMode(str, Enum):
    VOTING = "voting"
    WEIGHTED = "weighted"
    CONFIRMATION = "confirmation"


class CompositeSignal(SignalGenerator):
    """Combine multiple :class:`SignalGenerator` instances.

    Modes
    -----
    VOTING
        Majority vote across generators.  Signal is the sign of the sum.
    WEIGHTED
        Weighted sum of signals.  Signal is the sign of the weighted sum.
    CONFIRMATION
        All generators must agree for a signal to fire.
    """

    name: str = "composite"

    def __init__(self, mode: CombineMode = CombineMode.VOTING) -> None:
        self.mode = mode
        self._generators: list[tuple[SignalGenerator, float]] = []

    def add_generator(self, gen: SignalGenerator, weight: float = 1.0) -> None:
        """Register a generator with an optional weight."""
        self._generators.append((gen, weight))

    @property
    def required_indicators(self) -> list[dict]:
        """Union of all child generators' required indicators."""
        seen: list[dict] = []
        for gen, _ in self._generators:
            for ind in gen.required_indicators:
                if ind not in seen:
                    seen.append(ind)
        return seen

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine signals from all child generators.

        Each child's ``generate`` is called, producing a temporary signal
        column.  The signals are then merged according to *self.mode*.
        """
        if not self._generators:
            raise SignalError("CompositeSignal has no generators added.")

        child_signals: list[pd.Series] = []
        weights: list[float] = []

        for gen, w in self._generators:
            child_df = gen.generate(df.copy())
            child_signals.append(child_df["signal"].astype(float))
            weights.append(w)

        signals_matrix = pd.concat(child_signals, axis=1)

        if self.mode == CombineMode.VOTING:
            combined = signals_matrix.sum(axis=1)
            df["signal"] = np.sign(combined).astype(int)

        elif self.mode == CombineMode.WEIGHTED:
            weight_arr = np.array(weights)
            weighted = signals_matrix.mul(weight_arr, axis=1).sum(axis=1)
            df["signal"] = np.sign(weighted).astype(int)

        elif self.mode == CombineMode.CONFIRMATION:
            # All must agree (all positive or all negative)
            all_long = (signals_matrix > 0).all(axis=1)
            all_short = (signals_matrix < 0).all(axis=1)
            df["signal"] = 0
            df.loc[all_long, "signal"] = 1
            df.loc[all_short, "signal"] = -1

        else:
            raise SignalError(f"Unknown combine mode: {self.mode}")

        return df
