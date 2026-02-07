"""Tests for the backtest module."""

import numpy as np
import pandas as pd
import pytest

from ta_framework.core.exceptions import BacktestError
from ta_framework.core.types import BacktestConfig
from ta_framework.backtest.costs import (
    FixedCommission,
    PercentageCommission,
    SlippageModel,
    TieredCommission,
)
from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.backtest.monte_carlo import MonteCarloSimulator
from ta_framework.backtest.optimization import GridSearchOptimizer
from ta_framework.backtest.results import BacktestResult
from ta_framework.signals.strategies import EMACrossStrategy


# ---------------------------------------------------------------------------
# VectorizedBacktester
# ---------------------------------------------------------------------------


class TestVectorizedBacktester:
    def test_run_with_sample_signals(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(sample_signals)
        assert result.num_trades > 0

    def test_all_neutral_signals(self, sample_ohlcv, default_backtest_config):
        df = sample_ohlcv.copy()
        df["signal"] = 0
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(df)

        assert result.num_trades == 0
        # Equity should remain at initial capital (no trades)
        assert result.equity_curve.iloc[-1] == pytest.approx(
            default_backtest_config.initial_capital, rel=1e-6
        )

    def test_missing_signal_column_raises(self, sample_ohlcv):
        bt = VectorizedBacktester()
        with pytest.raises(BacktestError):
            bt.run(sample_ohlcv)

    def test_missing_close_column_raises(self):
        df = pd.DataFrame({"signal": [1, 0, -1]})
        bt = VectorizedBacktester()
        with pytest.raises(BacktestError):
            bt.run(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "signal"])
        bt = VectorizedBacktester()
        result = bt.run(df)
        assert result.num_trades == 0
        assert result.equity_curve.empty

    def test_single_long_trade(self):
        """Buy at bar 0, sell at bar 4."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        df = pd.DataFrame(
            {
                "close": [100.0, 102.0, 104.0, 103.0, 105.0],
                "signal": [1, 0, 0, 0, -1],
            },
            index=dates,
        )
        config = BacktestConfig(
            initial_capital=10_000,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        bt = VectorizedBacktester(config)
        result = bt.run(df)

        assert result.num_trades >= 1
        # With zero costs, final equity should reflect the price move
        assert result.equity_curve.iloc[-1] > config.initial_capital

    def test_commission_reduces_equity(self, sample_signals):
        config_no_cost = BacktestConfig(commission_pct=0.0, slippage_pct=0.0)
        config_high_cost = BacktestConfig(commission_pct=0.01, slippage_pct=0.005)

        result_no = VectorizedBacktester(config_no_cost).run(sample_signals)
        result_hi = VectorizedBacktester(config_high_cost).run(sample_signals)

        # Higher costs should result in lower or equal final equity
        assert result_hi.equity_curve.iloc[-1] <= result_no.equity_curve.iloc[-1]

    def test_short_selling_disabled(self, sample_signals):
        config = BacktestConfig(allow_short=False, commission_pct=0.0, slippage_pct=0.0)
        bt = VectorizedBacktester(config)
        result = bt.run(sample_signals)

        # Should not have short trades
        from ta_framework.core.types import SignalDirection
        for trade in result.trades:
            assert trade.direction == SignalDirection.LONG


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_total_return(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)
        tr = result.total_return
        assert isinstance(tr, float)

    def test_max_drawdown_non_negative(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)
        assert result.max_drawdown >= 0.0

    def test_win_rate_in_range(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)
        assert 0.0 <= result.win_rate <= 1.0

    def test_profit_factor(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)
        pf = result.profit_factor
        assert pf >= 0.0

    def test_summary_dict(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)
        summary = result.summary()
        assert "total_return" in summary
        assert "max_drawdown" in summary
        assert "num_trades" in summary
        assert "win_rate" in summary
        assert "profit_factor" in summary

    def test_to_dataframe(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)
        trades_df = result.to_dataframe()
        if result.num_trades > 0:
            assert "entry_time" in trades_df.columns
            assert "pnl" in trades_df.columns
            assert len(trades_df) == result.num_trades

    def test_empty_result(self):
        result = BacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=[],
            signals_df=pd.DataFrame(),
            config=BacktestConfig(),
        )
        assert result.total_return == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0
        assert result.num_trades == 0


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


class TestCostModels:
    def test_fixed_commission(self):
        model = FixedCommission(5.0)
        assert model.apply(10_000) == 5.0
        assert model.apply(1_000_000) == 5.0

    def test_percentage_commission(self):
        model = PercentageCommission(0.001)
        assert model.apply(10_000) == pytest.approx(10.0)
        assert model.apply(-10_000) == pytest.approx(10.0)  # absolute value

    def test_tiered_commission(self):
        model = TieredCommission([
            (100_000, 0.002),
            (500_000, 0.001),
            (float("inf"), 0.0005),
        ])
        # Volume within first tier
        assert model.apply(10_000, volume=50_000) == pytest.approx(20.0)
        # Volume in second tier
        assert model.apply(10_000, volume=200_000) == pytest.approx(10.0)

    def test_slippage_model(self):
        model = SlippageModel(base_pct=0.001, volume_impact=0.0)
        assert model.apply(10_000) == pytest.approx(10.0)

    def test_slippage_with_volume_impact(self):
        model = SlippageModel(base_pct=0.001, volume_impact=100.0)
        cost = model.apply(10_000, volume=1_000_000)
        expected = 10_000 * 0.001 + 10_000 * 100.0 / 1_000_000
        assert cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# GridSearchOptimizer
# ---------------------------------------------------------------------------


class TestGridSearchOptimizer:
    def test_basic_grid_search(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        # Pre-compute multiple EMA columns
        for period in [5, 10, 12, 20, 26, 50]:
            df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        optimizer = GridSearchOptimizer(
            strategy_cls=EMACrossStrategy,
            param_grid={
                "fast_period": [5, 10, 12],
                "slow_period": [20, 26, 50],
            },
            metric="total_return",
        )
        report = optimizer.run(df)

        assert len(report.results) > 0
        assert report.best is not None
        assert "total_return" in report.best.metrics

    def test_results_sorted(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        for period in [10, 12, 20, 26]:
            df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        optimizer = GridSearchOptimizer(
            strategy_cls=EMACrossStrategy,
            param_grid={"fast_period": [10, 12], "slow_period": [20, 26]},
        )
        report = optimizer.run(df)

        returns = [r.metrics["total_return"] for r in report.results]
        assert returns == sorted(returns, reverse=True)

    def test_to_dataframe(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        for period in [10, 12, 20, 26]:
            df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        optimizer = GridSearchOptimizer(
            strategy_cls=EMACrossStrategy,
            param_grid={"fast_period": [10, 12], "slow_period": [20, 26]},
        )
        report = optimizer.run(df)
        report_df = report.to_dataframe()
        assert not report_df.empty
        assert "fast_period" in report_df.columns


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------


class TestMonteCarloSimulator:
    def test_bootstrap_returns_shape(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)

        mc = MonteCarloSimulator(seed=42)
        sims = mc.bootstrap_returns(result.equity_curve, n_simulations=100)

        assert sims.shape[0] == 100
        assert sims.shape[1] > 0

    def test_shuffle_trades_shape(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)

        mc = MonteCarloSimulator(seed=42)
        finals = mc.shuffle_trades(result.trades, n_simulations=50)

        assert finals.shape == (50,)

    def test_confidence_intervals(self, sample_signals, default_backtest_config):
        bt = VectorizedBacktester(default_backtest_config)
        result = bt.run(sample_signals)

        mc = MonteCarloSimulator(seed=42)
        sims = mc.bootstrap_returns(result.equity_curve, n_simulations=100)
        ci = mc.confidence_intervals(sims)

        assert 0.05 in ci
        assert 0.5 in ci
        assert 0.95 in ci
        assert ci[0.05] <= ci[0.5] <= ci[0.95]

    def test_empty_equity_curve(self):
        mc = MonteCarloSimulator(seed=42)
        sims = mc.bootstrap_returns(pd.Series(dtype=float))
        assert sims.size == 0

    def test_no_trades_shuffle(self):
        mc = MonteCarloSimulator(seed=42)
        finals = mc.shuffle_trades([], n_simulations=10, initial_capital=50_000)
        assert (finals == 50_000).all()
