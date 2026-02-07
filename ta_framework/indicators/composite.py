"""Composite indicators that chain multiple computations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ta_framework.core.exceptions import IndicatorError
from ta_framework.indicators.wrappers import WRAPPER_MAP


class CompositeIndicator:
    """Chain multiple indicators sequentially.

    Each step computes an indicator and the next step can reference the
    columns produced by the previous one.

    Parameters
    ----------
    steps : list[tuple[str, dict]]
        Ordered sequence of ``(indicator_name, params)`` tuples.
        Example::

            steps = [
                ("sma", {"length": 20, "column": "close"}),
                ("rsi", {"length": 14, "column": "SMA_20"}),
            ]

    name : str, optional
        Human-readable name for this composite (used for the final output
        column naming).

    Example
    -------
    ::

        comp = CompositeIndicator(
            steps=[("sma", {"length": 20}), ("rsi", {"length": 14, "column": "SMA_20"})],
            name="RSI_of_SMA",
        )
        df_out = comp.compute(df)
    """

    def __init__(
        self,
        steps: list[tuple[str, dict[str, Any]]],
        name: str | None = None,
    ) -> None:
        if not steps:
            raise IndicatorError("CompositeIndicator requires at least one step.")
        self.steps = steps
        self.name = name or "_then_".join(s[0] for s in steps)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the chain of indicators on *df*.

        Returns the DataFrame with all intermediate and final columns added.
        """
        result = df.copy()
        for indicator_name, params in self.steps:
            func = WRAPPER_MAP.get(indicator_name)
            if func is None:
                raise IndicatorError(
                    f"Unknown indicator '{indicator_name}' in composite chain. "
                    f"Available: {sorted(WRAPPER_MAP.keys())}"
                )
            try:
                result = func(result, **params)
            except Exception as exc:
                raise IndicatorError(
                    f"Error in composite step '{indicator_name}': {exc}"
                ) from exc
        return result

    def __repr__(self) -> str:
        step_strs = [f"{n}({p})" for n, p in self.steps]
        return f"CompositeIndicator({' -> '.join(step_strs)})"
