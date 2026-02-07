"""Vectorized backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta_framework.core.exceptions import BacktestError
from ta_framework.core.types import BacktestConfig, SignalDirection, Trade
from ta_framework.backtest.results import BacktestResult


class VectorizedBacktester:
    """Fast vectorized backtester.

    Signals are integers: 1 = go long, -1 = go short, 0 = flat.
    Position changes are detected from signal transitions.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
    ) -> BacktestResult:
        """Run a vectorized backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain OHLCV columns and a *signal_col*.
        signal_col : str
            Column name holding 1 / -1 / 0 signals.

        Returns
        -------
        BacktestResult
        """
        if signal_col not in df.columns:
            raise BacktestError(f"Signal column '{signal_col}' not found in DataFrame.")
        if "close" not in df.columns:
            raise BacktestError("DataFrame must contain a 'close' column.")
        if df.empty:
            return BacktestResult(
                equity_curve=pd.Series(dtype=float),
                trades=[],
                signals_df=df,
                config=self.config,
            )

        cfg = self.config
        signals = df[signal_col].fillna(0).astype(int)
        close = df["close"].values
        index = df.index

        if not cfg.allow_short:
            signals = signals.clip(lower=0)

        position = signals.values.copy()
        # Forward-fill positions: hold until a new signal appears
        for i in range(1, len(position)):
            if position[i] == 0:
                position[i] = position[i - 1]
        position = pd.Series(position, index=index)

        # Detect trades from position changes
        pos_diff = position.diff().fillna(position.iloc[0] if len(position) > 0 else 0)
        trade_entries = pos_diff[pos_diff != 0].index

        trades: list[Trade] = []
        equity = np.full(len(close), cfg.initial_capital, dtype=float)

        # Track cumulative P&L bar-by-bar
        capital = cfg.initial_capital
        current_pos = 0  # units held (positive = long, negative = short)
        entry_price = 0.0
        entry_idx = 0
        entry_time = index[0]

        for i in range(len(close)):
            new_pos = position.iloc[i]
            price = close[i]

            if new_pos != current_pos:
                # Close existing position
                if current_pos != 0:
                    exit_price = price
                    trade_value = abs(current_pos) * entry_price
                    comm = trade_value * cfg.commission_pct
                    slip = trade_value * cfg.slippage_pct

                    if current_pos > 0:
                        pnl = current_pos * (exit_price - entry_price)
                    else:
                        pnl = abs(current_pos) * (entry_price - exit_price)

                    pnl_pct = pnl / trade_value if trade_value > 0 else 0.0

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=index[i],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=abs(current_pos),
                        direction=SignalDirection.LONG if current_pos > 0 else SignalDirection.SHORT,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=comm,
                        slippage=slip,
                        holding_period=i - entry_idx,
                    ))
                    capital += pnl - comm - slip

                # Open new position
                if new_pos != 0:
                    entry_price = price
                    entry_idx = i
                    entry_time = index[i]
                    alloc = capital * cfg.position_size_pct
                    current_pos = (alloc / price) * (1 if new_pos > 0 else -1)
                    # Entry costs
                    entry_value = abs(current_pos) * price
                    capital -= entry_value * cfg.commission_pct + entry_value * cfg.slippage_pct
                else:
                    current_pos = 0

            # Mark-to-market
            if current_pos > 0:
                equity[i] = capital + current_pos * (price - entry_price)
            elif current_pos < 0:
                equity[i] = capital + abs(current_pos) * (entry_price - price)
            else:
                equity[i] = capital

        equity_series = pd.Series(equity, index=index, name="equity")

        return BacktestResult(
            equity_curve=equity_series,
            trades=trades,
            signals_df=df,
            config=cfg,
        )
