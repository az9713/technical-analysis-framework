"""Monte Carlo simulation for backtest robustness analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ta_framework.core.types import Trade


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation outputs."""
    simulations: np.ndarray  # shape (n_simulations, n_periods)
    confidence_intervals: dict[float, float]


class MonteCarloSimulator:
    """Monte Carlo methods for backtest robustness testing."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def bootstrap_returns(
        self,
        equity_curve: pd.Series,
        n_simulations: int = 1000,
    ) -> np.ndarray:
        """Resample daily returns with replacement to build simulated equity paths.

        Parameters
        ----------
        equity_curve : pd.Series
            Original equity curve.
        n_simulations : int
            Number of simulated paths.

        Returns
        -------
        np.ndarray
            Shape ``(n_simulations, len(equity_curve))``.
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return np.array([]).reshape(0, 0)

        returns = equity_curve.pct_change().dropna().values
        n_periods = len(returns)
        initial = equity_curve.iloc[0]

        # Bootstrap: sample returns with replacement
        indices = self.rng.integers(0, n_periods, size=(n_simulations, n_periods))
        sampled_returns = returns[indices]  # (n_simulations, n_periods)

        # Build cumulative equity paths
        cum_returns = np.cumprod(1 + sampled_returns, axis=1)
        paths = initial * np.column_stack([
            np.ones(n_simulations),
            cum_returns,
        ])
        return paths

    def shuffle_trades(
        self,
        trades: list[Trade],
        n_simulations: int = 1000,
        initial_capital: float = 100_000.0,
    ) -> np.ndarray:
        """Shuffle trade order to produce distribution of final equity.

        Parameters
        ----------
        trades : list[Trade]
        n_simulations : int
        initial_capital : float

        Returns
        -------
        np.ndarray
            Shape ``(n_simulations,)`` of final equity values.
        """
        if not trades:
            return np.full(n_simulations, initial_capital)

        pnls = np.array([t.net_pnl for t in trades])
        final_equities = np.empty(n_simulations)

        for i in range(n_simulations):
            shuffled = self.rng.permutation(pnls)
            equity = initial_capital + np.cumsum(shuffled)
            final_equities[i] = equity[-1]

        return final_equities

    def confidence_intervals(
        self,
        simulations: np.ndarray,
        levels: list[float] | None = None,
    ) -> dict[float, float]:
        """Compute confidence intervals from simulation results.

        Parameters
        ----------
        simulations : np.ndarray
            Either 1-D (final equities) or 2-D (equity paths, uses last
            column).
        levels : list[float]
            Percentile levels as decimals (e.g. 0.05 for 5th percentile).

        Returns
        -------
        dict[float, float]
            Mapping of level -> value.
        """
        if levels is None:
            levels = [0.05, 0.25, 0.5, 0.75, 0.95]

        if simulations.size == 0:
            return {lvl: 0.0 for lvl in levels}

        # Use final values
        if simulations.ndim == 2:
            values = simulations[:, -1]
        else:
            values = simulations

        return {
            lvl: float(np.percentile(values, lvl * 100))
            for lvl in levels
        }
