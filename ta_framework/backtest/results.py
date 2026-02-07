"""Backtest result container with performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ta_framework.core.types import BacktestConfig, Trade


@dataclass
class BacktestResult:
    """Stores the output of a backtest run."""

    equity_curve: pd.Series
    trades: list[Trade]
    signals_df: pd.DataFrame
    config: BacktestConfig

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_return(self) -> float:
        """Total return as a decimal (e.g. 0.12 = 12%)."""
        if self.equity_curve.empty:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1.0

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a positive decimal (e.g. 0.15 = 15%)."""
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        return abs(drawdown.min())

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Fraction of winning trades (0.0 -- 1.0)."""
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.is_winner)
        return winners / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss.  Returns inf if no losses."""
        gross_profit = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio using daily returns."""
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        excess = returns.mean() - self.config.risk_free_rate / 252
        return float(excess / returns.std() * np.sqrt(252))

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """All key metrics as a dictionary."""
        return {
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "initial_capital": self.config.initial_capital,
            "final_equity": float(self.equity_curve.iloc[-1]) if not self.equity_curve.empty else 0.0,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "direction": t.direction.value,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "commission": t.commission,
                "slippage": t.slippage,
                "net_pnl": t.net_pnl,
                "holding_period": t.holding_period,
            })
        return pd.DataFrame(records)
