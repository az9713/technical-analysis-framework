"""Strategy parameter optimization."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Type

import pandas as pd

from ta_framework.core.exceptions import BacktestError
from ta_framework.core.types import BacktestConfig
from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.backtest.results import BacktestResult
from ta_framework.signals.base import SignalGenerator


@dataclass
class OptimizationResult:
    """Single parameter-combination result."""
    params: dict[str, Any]
    metrics: dict[str, float]
    result: BacktestResult | None = None


@dataclass
class OptimizationReport:
    """Collection of results sorted by a target metric."""
    results: list[OptimizationResult] = field(default_factory=list)
    metric: str = "total_return"

    @property
    def best(self) -> OptimizationResult | None:
        return self.results[0] if self.results else None

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            row = {**r.params, **r.metrics}
            rows.append(row)
        return pd.DataFrame(rows)


class GridSearchOptimizer:
    """Exhaustive grid search over parameter combinations.

    Parameters
    ----------
    strategy_cls : Type[SignalGenerator]
        Strategy class to instantiate for each combination.
    param_grid : dict[str, list]
        Mapping of constructor parameter names to lists of values.
    config : BacktestConfig, optional
        Backtest configuration.
    metric : str
        Metric from ``BacktestResult.summary()`` to rank by.
    """

    def __init__(
        self,
        strategy_cls: Type[SignalGenerator],
        param_grid: dict[str, list],
        config: BacktestConfig | None = None,
        metric: str = "total_return",
    ) -> None:
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.config = config or BacktestConfig()
        self.metric = metric

    def run(self, df: pd.DataFrame) -> OptimizationReport:
        """Run grid search over all parameter combinations.

        *df* must already contain all possible indicator columns that any
        combination might need (i.e. pre-compute a superset of indicators).
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))

        results: list[OptimizationResult] = []
        backtester = VectorizedBacktester(self.config)

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                strategy = self.strategy_cls(**params)
                sig_df = strategy.generate(df.copy())
                bt_result = backtester.run(sig_df)
                metrics = bt_result.summary()
                results.append(OptimizationResult(
                    params=params,
                    metrics=metrics,
                    result=bt_result,
                ))
            except Exception:
                # Skip invalid parameter combinations
                continue

        # Sort descending by target metric
        results.sort(key=lambda r: r.metrics.get(self.metric, float("-inf")), reverse=True)
        return OptimizationReport(results=results, metric=self.metric)


class WalkForwardOptimizer:
    """Rolling in-sample / out-of-sample optimisation.

    Parameters
    ----------
    strategy_cls : Type[SignalGenerator]
        Strategy class to instantiate.
    param_grid : dict[str, list]
        Parameter search space.
    in_sample_pct : float
        Fraction of each window used for in-sample optimisation.
    n_windows : int
        Number of rolling windows.
    config : BacktestConfig, optional
    metric : str
        Optimisation target metric.
    """

    def __init__(
        self,
        strategy_cls: Type[SignalGenerator],
        param_grid: dict[str, list],
        in_sample_pct: float = 0.7,
        n_windows: int = 4,
        config: BacktestConfig | None = None,
        metric: str = "total_return",
    ) -> None:
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.in_sample_pct = in_sample_pct
        self.n_windows = n_windows
        self.config = config or BacktestConfig()
        self.metric = metric

    def run(self, df: pd.DataFrame) -> OptimizationReport:
        """Run walk-forward analysis.

        Returns an :class:`OptimizationReport` with one entry per window
        (the out-of-sample result using the best in-sample parameters).
        """
        n = len(df)
        if n == 0:
            return OptimizationReport()

        window_size = n // self.n_windows
        if window_size < 2:
            raise BacktestError(
                f"Not enough data ({n} rows) for {self.n_windows} windows."
            )

        oos_results: list[OptimizationResult] = []
        backtester = VectorizedBacktester(self.config)

        for w in range(self.n_windows):
            start = w * window_size
            end = min(start + window_size, n)
            window_df = df.iloc[start:end]

            split = int(len(window_df) * self.in_sample_pct)
            is_df = window_df.iloc[:split]
            oos_df = window_df.iloc[split:]

            if is_df.empty or oos_df.empty:
                continue

            # In-sample optimisation
            grid_opt = GridSearchOptimizer(
                strategy_cls=self.strategy_cls,
                param_grid=self.param_grid,
                config=self.config,
                metric=self.metric,
            )
            is_report = grid_opt.run(is_df)
            if is_report.best is None:
                continue

            best_params = is_report.best.params

            # Out-of-sample evaluation
            try:
                strategy = self.strategy_cls(**best_params)
                sig_df = strategy.generate(oos_df.copy())
                bt_result = backtester.run(sig_df)
                metrics = bt_result.summary()
                oos_results.append(OptimizationResult(
                    params=best_params,
                    metrics=metrics,
                    result=bt_result,
                ))
            except Exception:
                continue

        oos_results.sort(
            key=lambda r: r.metrics.get(self.metric, float("-inf")),
            reverse=True,
        )
        return OptimizationReport(results=oos_results, metric=self.metric)
