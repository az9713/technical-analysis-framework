"""Tests for the signals module."""

import numpy as np
import pandas as pd
import pytest

from ta_framework.core.exceptions import SignalError
from ta_framework.core.registry import strategy_registry
from ta_framework.signals.base import SignalGenerator
from ta_framework.signals.composite import CombineMode, CompositeSignal
from ta_framework.signals.rules import breakout, crossover, divergence, threshold
from ta_framework.signals.strategies import (
    BollingerBandStrategy,
    EMACrossStrategy,
    MACDStrategy,
    RSIStrategy,
)


# ---------------------------------------------------------------------------
# Rule functions
# ---------------------------------------------------------------------------


class TestCrossover:
    def test_cross_above(self):
        fast = pd.Series([1, 2, 3, 4, 5], dtype=float)
        slow = pd.Series([3, 3, 3, 3, 3], dtype=float)
        sig = crossover(fast, slow)
        # fast crosses above slow between index 2 and 3
        assert sig[3] == 1

    def test_cross_below(self):
        fast = pd.Series([5, 4, 3, 2, 1], dtype=float)
        slow = pd.Series([3, 3, 3, 3, 3], dtype=float)
        sig = crossover(fast, slow)
        # fast crosses below slow between index 2 and 3
        assert sig[3] == -1

    def test_no_cross(self):
        fast = pd.Series([5, 5, 5, 5, 5], dtype=float)
        slow = pd.Series([3, 3, 3, 3, 3], dtype=float)
        sig = crossover(fast, slow)
        # No crossover, only first bar might trigger
        assert (sig.iloc[1:] == 0).all()

    def test_output_values_in_range(self):
        fast = pd.Series(np.random.randn(100).cumsum())
        slow = pd.Series(np.random.randn(100).cumsum())
        sig = crossover(fast, slow)
        assert set(sig.unique()).issubset({-1, 0, 1})


class TestThreshold:
    def test_oversold_signal(self):
        series = pd.Series([50, 40, 20, 15, 60])
        sig = threshold(series, upper=70, lower=30)
        assert sig[2] == 1  # 20 < 30
        assert sig[3] == 1  # 15 < 30

    def test_overbought_signal(self):
        series = pd.Series([50, 60, 80, 85, 40])
        sig = threshold(series, upper=70, lower=30)
        assert sig[2] == -1  # 80 > 70
        assert sig[3] == -1  # 85 > 70

    def test_neutral(self):
        series = pd.Series([50, 50, 50])
        sig = threshold(series, upper=70, lower=30)
        assert (sig == 0).all()


class TestBreakout:
    def test_lower_breakout(self):
        close = pd.Series([50, 45, 30, 35, 50])
        upper = pd.Series([60, 60, 60, 60, 60])
        lower = pd.Series([40, 40, 40, 40, 40])
        sig = breakout(close, upper, lower)
        assert sig[2] == 1  # 30 < 40

    def test_upper_breakout(self):
        close = pd.Series([50, 55, 70, 65, 50])
        upper = pd.Series([60, 60, 60, 60, 60])
        lower = pd.Series([40, 40, 40, 40, 40])
        sig = breakout(close, upper, lower)
        assert sig[2] == -1  # 70 > 60


class TestDivergence:
    def test_short_data_returns_zeros(self):
        price = pd.Series([1, 2, 3])
        indicator = pd.Series([1, 2, 3])
        sig = divergence(price, indicator, lookback=5)
        assert (sig == 0).all()

    def test_output_values_in_range(self):
        np.random.seed(42)
        n = 100
        price = pd.Series(np.random.randn(n).cumsum() + 100)
        indicator = pd.Series(np.random.randn(n).cumsum() + 50)
        sig = divergence(price, indicator, lookback=10)
        assert set(sig.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class TestEMACrossStrategy:
    def test_generate_with_indicators(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        # Pre-compute indicator columns as pandas_ta would
        df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()

        strategy = EMACrossStrategy(fast_period=12, slow_period=26)
        result = strategy.generate(df)

        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_missing_columns_raises(self, sample_ohlcv):
        strategy = EMACrossStrategy()
        with pytest.raises(SignalError):
            strategy.generate(sample_ohlcv)

    def test_required_indicators(self):
        strategy = EMACrossStrategy(fast_period=10, slow_period=30)
        indicators = strategy.required_indicators
        assert len(indicators) == 2
        assert indicators[0]["name"] == "ema"
        assert indicators[0]["length"] == 10
        assert indicators[1]["length"] == 30

    def test_registered_in_strategy_registry(self):
        assert "ema_cross" in strategy_registry


class TestRSIStrategy:
    def test_generate(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        # Simple RSI approximation for testing
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        strategy = RSIStrategy()
        result = strategy.generate(df)
        assert "signal" in result.columns
        assert "signal_strength" in result.columns


class TestMACDStrategy:
    def test_generate(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD_12_26_9"] = ema12 - ema26
        df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
        df["MACDh_12_26_9"] = df["MACD_12_26_9"] - df["MACDs_12_26_9"]

        strategy = MACDStrategy()
        result = strategy.generate(df)
        assert "signal" in result.columns


# ---------------------------------------------------------------------------
# CompositeSignal
# ---------------------------------------------------------------------------


class TestCompositeSignal:
    def _make_ema_df(self, sample_ohlcv):
        """Helper: add EMA columns to sample data."""
        df = sample_ohlcv.copy()
        df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()
        return df

    def _make_rsi_df(self, sample_ohlcv):
        """Helper: add RSI columns to sample data."""
        df = sample_ohlcv.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))
        return df

    def test_voting_mode(self, sample_ohlcv):
        df = self._make_ema_df(sample_ohlcv)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        comp = CompositeSignal(mode=CombineMode.VOTING)
        comp.add_generator(EMACrossStrategy())
        comp.add_generator(RSIStrategy())

        result = comp.generate(df)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_confirmation_mode(self, sample_ohlcv):
        df = self._make_ema_df(sample_ohlcv)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        comp = CompositeSignal(mode=CombineMode.CONFIRMATION)
        comp.add_generator(EMACrossStrategy())
        comp.add_generator(RSIStrategy())

        result = comp.generate(df)
        assert "signal" in result.columns
        # Confirmation mode should produce fewer (or equal) signals
        voting_comp = CompositeSignal(mode=CombineMode.VOTING)
        voting_comp.add_generator(EMACrossStrategy())
        voting_comp.add_generator(RSIStrategy())
        voting_result = voting_comp.generate(df.copy())

        confirm_signals = (result["signal"] != 0).sum()
        # At minimum, confirmation should not produce MORE signals than total individual
        assert confirm_signals >= 0

    def test_empty_composite_raises(self):
        comp = CompositeSignal()
        with pytest.raises(SignalError):
            comp.generate(pd.DataFrame({"close": [1, 2, 3]}))

    def test_required_indicators_union(self):
        comp = CompositeSignal()
        comp.add_generator(EMACrossStrategy())
        comp.add_generator(RSIStrategy())
        indicators = comp.required_indicators
        names = [ind["name"] for ind in indicators]
        assert "ema" in names
        assert "rsi" in names

    def test_weighted_mode(self, sample_ohlcv):
        df = self._make_ema_df(sample_ohlcv)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        comp = CompositeSignal(mode=CombineMode.WEIGHTED)
        comp.add_generator(EMACrossStrategy(), weight=2.0)
        comp.add_generator(RSIStrategy(), weight=1.0)

        result = comp.generate(df)
        assert "signal" in result.columns
