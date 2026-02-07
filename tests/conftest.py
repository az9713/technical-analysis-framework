"""Shared test fixtures: sample OHLCV data and test configurations."""

import numpy as np
import pandas as pd
import pytest

from ta_framework.core.types import BacktestConfig, Timeframe


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 252 days (~1 year) of realistic OHLCV data."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")

    # Random walk for close prices starting at 100
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Derive OHLV from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(1_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def short_ohlcv() -> pd.DataFrame:
    """Small 20-bar dataset for quick unit tests."""
    np.random.seed(7)
    n = 20
    dates = pd.bdate_range(start="2024-01-02", periods=n, freq="B")
    close = 50.0 + np.cumsum(np.random.normal(0, 0.5, n))
    high = close + np.abs(np.random.normal(0, 0.3, n))
    low = close - np.abs(np.random.normal(0, 0.3, n))
    open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n)
    volume = np.random.randint(100_000, 5_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def empty_ohlcv() -> pd.DataFrame:
    """Empty DataFrame with correct OHLCV columns."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def default_backtest_config() -> BacktestConfig:
    return BacktestConfig()


@pytest.fixture
def sample_signals(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Sample OHLCV with a basic signal column: 1=long, -1=short, 0=neutral."""
    df = sample_ohlcv.copy()
    # Simple alternating signals for testing
    df["signal"] = 0
    df.iloc[10::40, df.columns.get_loc("signal")] = 1   # buy every 40 bars
    df.iloc[30::40, df.columns.get_loc("signal")] = -1   # sell every 40 bars offset
    return df
