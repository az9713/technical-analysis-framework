"""TearSheet: comprehensive strategy performance report."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ta_framework.analytics import metrics
from ta_framework.core.types import Trade


class TearSheet:
    """Generate a full performance tearsheet from backtest results.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve (indexed by date).
    trades : list[Trade]
        List of completed trades.
    benchmark : pd.Series | None
        Benchmark equity curve for comparison.
    risk_free_rate : float
        Annual risk-free rate.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: list[Trade],
        benchmark: pd.Series | None = None,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.equity_curve = equity_curve
        self.trades = trades
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self._returns = equity_curve.pct_change().dropna()

    def generate(self) -> dict[str, Any]:
        """Generate all metrics organized by category.

        Returns
        -------
        dict
            Metrics grouped into 'returns', 'risk', 'trades', 'distribution'.
        """
        ret = self._returns
        eq = self.equity_curve

        result: dict[str, Any] = {
            "returns": {
                "cagr": metrics.cagr(eq),
                "sharpe_ratio": metrics.sharpe_ratio(ret, self.risk_free_rate),
                "sortino_ratio": metrics.sortino_ratio(ret, self.risk_free_rate),
                "calmar_ratio": metrics.calmar_ratio(ret),
                "omega_ratio": metrics.omega_ratio(ret),
                "volatility": metrics.volatility(ret),
            },
            "risk": {
                "max_drawdown": metrics.max_drawdown(equity_curve=eq),
                "max_drawdown_duration": metrics.max_drawdown_duration(eq),
                "recovery_factor": metrics.recovery_factor(eq),
                "var_95": metrics.var_95(ret),
                "cvar_95": metrics.cvar_95(ret),
                "downside_deviation": metrics.downside_deviation(ret),
            },
            "distribution": {
                "skewness": metrics.skewness(ret),
                "kurtosis": metrics.kurtosis(ret),
                "tail_ratio": metrics.tail_ratio(ret),
                "common_sense_ratio": metrics.common_sense_ratio(ret),
            },
        }

        if self.trades:
            result["trades"] = {
                "total_trades": len(self.trades),
                "win_rate": metrics.win_rate(self.trades),
                "profit_factor": metrics.profit_factor(self.trades),
                "expectancy": metrics.expectancy(self.trades),
                "avg_win": metrics.avg_win(self.trades),
                "avg_loss": metrics.avg_loss(self.trades),
                "largest_win": metrics.largest_win(self.trades),
                "largest_loss": metrics.largest_loss(self.trades),
                "consecutive_wins": metrics.consecutive_wins(self.trades),
                "consecutive_losses": metrics.consecutive_losses(self.trades),
            }

        if self.benchmark is not None:
            from ta_framework.analytics.benchmark import (
                alpha_beta,
                information_ratio,
                tracking_error,
            )

            bench_ret = self.benchmark.pct_change().dropna()
            aligned = pd.concat([ret, bench_ret], axis=1).dropna()
            if len(aligned) > 1:
                r = aligned.iloc[:, 0]
                b = aligned.iloc[:, 1]
                alpha, beta = alpha_beta(r, b, self.risk_free_rate)
                result["benchmark"] = {
                    "alpha": alpha,
                    "beta": beta,
                    "information_ratio": information_ratio(r, b),
                    "tracking_error": tracking_error(r, b),
                }

        return result

    def summary_table(self) -> pd.DataFrame:
        """Flat summary table of all metrics for display.

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame: Metric, Value.
        """
        all_metrics = metrics.calculate_all(
            self._returns, self.equity_curve, self.trades or None
        )
        rows = [{"Metric": k, "Value": v} for k, v in all_metrics.items()]
        return pd.DataFrame(rows)

    def monthly_returns(self) -> pd.DataFrame:
        """Pivot table of monthly returns: year x month.

        Returns
        -------
        pd.DataFrame
            Rows = years, Columns = months (1..12), values = monthly return %.
        """
        if self._returns.empty:
            return pd.DataFrame()

        ret = self._returns.copy()
        ret.index = pd.to_datetime(ret.index)
        monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly = monthly * 100  # percent

        table = pd.DataFrame(
            {
                "Year": monthly.index.year,
                "Month": monthly.index.month,
                "Return": monthly.values,
            }
        )
        pivot = table.pivot_table(
            values="Return", index="Year", columns="Month", aggfunc="sum"
        )
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][: len(pivot.columns)]
        return pivot

    def drawdown_analysis(self) -> dict[str, Any]:
        """Analyze top 5 drawdown periods.

        Returns
        -------
        dict
            'max_drawdown': float,
            'drawdowns': list of dicts with 'start', 'trough', 'end',
                         'depth', 'duration', 'recovery'.
        """
        eq = self.equity_curve
        if eq.empty or len(eq) < 2:
            return {"max_drawdown": 0.0, "drawdowns": []}

        running_max = eq.cummax()
        dd_series = (eq - running_max) / running_max

        # Identify drawdown periods
        periods: list[dict[str, Any]] = []
        in_dd = False
        start_idx = None

        for i in range(len(dd_series)):
            if dd_series.iloc[i] < 0 and not in_dd:
                in_dd = True
                start_idx = i - 1 if i > 0 else 0
            elif dd_series.iloc[i] >= 0 and in_dd:
                in_dd = False
                segment = dd_series.iloc[start_idx: i + 1]
                trough_pos = segment.idxmin()
                periods.append(
                    {
                        "start": dd_series.index[start_idx],
                        "trough": trough_pos,
                        "end": dd_series.index[i],
                        "depth": abs(segment.min()),
                        "duration": i - start_idx,
                        "recovery": i - dd_series.index.get_loc(trough_pos),
                    }
                )

        # Handle ongoing drawdown
        if in_dd and start_idx is not None:
            segment = dd_series.iloc[start_idx:]
            trough_pos = segment.idxmin()
            periods.append(
                {
                    "start": dd_series.index[start_idx],
                    "trough": trough_pos,
                    "end": dd_series.index[-1],
                    "depth": abs(segment.min()),
                    "duration": len(dd_series) - start_idx,
                    "recovery": None,
                }
            )

        # Sort by depth and take top 5
        periods.sort(key=lambda p: p["depth"], reverse=True)
        return {
            "max_drawdown": float(abs(dd_series.min())),
            "drawdowns": periods[:5],
        }
