"""Tests for the analytics module: metrics, tearsheet, benchmark, regime."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ta_framework.analytics.metrics import (
    calculate_all,
    calmar_ratio,
    cvar_95,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    var_95,
    volatility,
    win_rate,
    profit_factor,
    expectancy,
    consecutive_wins,
    consecutive_losses,
)
from ta_framework.analytics.tearsheet import TearSheet
from ta_framework.analytics.benchmark import (
    active_return,
    alpha_beta,
    down_capture,
    information_ratio,
    tracking_error,
    up_capture,
)
from ta_framework.analytics.regime import RegimeDetector
from ta_framework.core.types import SignalDirection, Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns():
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0.0005, 0.02, 252),
        index=pd.bdate_range("2023-01-03", periods=252),
    )


@pytest.fixture
def equity_curve(daily_returns):
    return (1 + daily_returns).cumprod() * 100_000


@pytest.fixture
def sample_trades():
    return [
        Trade(
            entry_time=datetime(2023, 1, 10),
            exit_time=datetime(2023, 1, 20),
            entry_price=100,
            exit_price=110,
            quantity=10,
            pnl=100.0,
        ),
        Trade(
            entry_time=datetime(2023, 2, 1),
            exit_time=datetime(2023, 2, 10),
            entry_price=105,
            exit_price=100,
            quantity=10,
            pnl=-50.0,
        ),
        Trade(
            entry_time=datetime(2023, 3, 1),
            exit_time=datetime(2023, 3, 15),
            entry_price=102,
            exit_price=115,
            quantity=10,
            pnl=130.0,
        ),
        Trade(
            entry_time=datetime(2023, 4, 1),
            exit_time=datetime(2023, 4, 10),
            entry_price=110,
            exit_price=108,
            quantity=10,
            pnl=-20.0,
        ),
        Trade(
            entry_time=datetime(2023, 5, 1),
            exit_time=datetime(2023, 5, 15),
            entry_price=108,
            exit_price=120,
            quantity=10,
            pnl=120.0,
        ),
    ]


@pytest.fixture
def benchmark_returns(daily_returns):
    np.random.seed(99)
    return pd.Series(
        np.random.normal(0.0004, 0.015, 252),
        index=daily_returns.index,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_sharpe_ratio(self, daily_returns):
        sr = sharpe_ratio(daily_returns, risk_free_rate=0.04)
        assert isinstance(sr, float)
        # With positive drift, Sharpe should be plausible (not extreme)
        assert -5.0 < sr < 5.0

    def test_sortino_ratio(self, daily_returns):
        sor = sortino_ratio(daily_returns, risk_free_rate=0.04)
        assert isinstance(sor, float)

    def test_sharpe_empty(self):
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0

    def test_max_drawdown(self, daily_returns):
        mdd = max_drawdown(returns=daily_returns)
        assert 0.0 <= mdd <= 1.0

    def test_max_drawdown_equity(self, equity_curve):
        mdd = max_drawdown(equity_curve=equity_curve)
        assert 0.0 <= mdd <= 1.0

    def test_max_drawdown_duration(self, equity_curve):
        dur = max_drawdown_duration(equity_curve)
        assert dur >= 0

    def test_calmar_ratio(self, daily_returns):
        cr = calmar_ratio(daily_returns)
        assert isinstance(cr, float)

    def test_volatility(self, daily_returns):
        vol = volatility(daily_returns)
        assert vol > 0

    def test_var_95(self, daily_returns):
        v = var_95(daily_returns)
        assert v > 0

    def test_cvar_95(self, daily_returns):
        cv = cvar_95(daily_returns)
        assert cv > 0
        assert cv >= var_95(daily_returns)


class TestTradeMetrics:
    def test_win_rate(self, sample_trades):
        wr = win_rate(sample_trades)
        # 3 winners out of 5
        assert wr == pytest.approx(3 / 5)

    def test_profit_factor(self, sample_trades):
        pf = profit_factor(sample_trades)
        # gross profit = 100+130+120=350, gross loss = 50+20=70
        assert pf == pytest.approx(350 / 70)

    def test_expectancy(self, sample_trades):
        exp = expectancy(sample_trades)
        # mean of [100, -50, 130, -20, 120] = 56
        assert exp == pytest.approx(56.0)

    def test_consecutive_wins(self, sample_trades):
        cw = consecutive_wins(sample_trades)
        # Pattern: W, L, W, L, W -> max consecutive = 1
        assert cw == 1

    def test_consecutive_losses(self, sample_trades):
        cl = consecutive_losses(sample_trades)
        assert cl == 1

    def test_empty_trades(self):
        assert win_rate([]) == 0.0
        assert profit_factor([]) == 0.0


# ---------------------------------------------------------------------------
# TearSheet
# ---------------------------------------------------------------------------

class TestTearSheet:
    def test_generate_structure(self, equity_curve, sample_trades):
        ts = TearSheet(equity_curve, sample_trades)
        result = ts.generate()
        assert "returns" in result
        assert "risk" in result
        assert "trades" in result
        assert "distribution" in result
        assert "sharpe_ratio" in result["returns"]
        assert "max_drawdown" in result["risk"]
        assert "win_rate" in result["trades"]

    def test_generate_no_trades(self, equity_curve):
        ts = TearSheet(equity_curve, [])
        result = ts.generate()
        assert "trades" not in result

    def test_summary_table(self, equity_curve, sample_trades):
        ts = TearSheet(equity_curve, sample_trades)
        table = ts.summary_table()
        assert isinstance(table, pd.DataFrame)
        assert "Metric" in table.columns
        assert "Value" in table.columns
        assert len(table) > 0

    def test_monthly_returns_shape(self, equity_curve):
        ts = TearSheet(equity_curve, [])
        monthly = ts.monthly_returns()
        assert isinstance(monthly, pd.DataFrame)
        # Should have rows (years) and columns (months)
        assert monthly.shape[1] <= 12

    def test_drawdown_analysis(self, equity_curve):
        ts = TearSheet(equity_curve, [])
        dd = ts.drawdown_analysis()
        assert "max_drawdown" in dd
        assert "drawdowns" in dd
        assert dd["max_drawdown"] >= 0

    def test_with_benchmark(self, equity_curve, sample_trades, benchmark_returns):
        bench_eq = (1 + benchmark_returns).cumprod() * 100_000
        ts = TearSheet(equity_curve, sample_trades, benchmark=bench_eq)
        result = ts.generate()
        assert "benchmark" in result
        assert "alpha" in result["benchmark"]
        assert "beta" in result["benchmark"]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_alpha_beta(self, daily_returns, benchmark_returns):
        alpha, beta = alpha_beta(daily_returns, benchmark_returns)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)

    def test_alpha_beta_identical(self, daily_returns):
        alpha, beta = alpha_beta(daily_returns, daily_returns)
        assert beta == pytest.approx(1.0, rel=0.01)

    def test_information_ratio(self, daily_returns, benchmark_returns):
        ir = information_ratio(daily_returns, benchmark_returns)
        assert isinstance(ir, float)

    def test_tracking_error(self, daily_returns, benchmark_returns):
        te = tracking_error(daily_returns, benchmark_returns)
        assert te >= 0

    def test_up_capture(self, daily_returns, benchmark_returns):
        uc = up_capture(daily_returns, benchmark_returns)
        assert isinstance(uc, float)

    def test_down_capture(self, daily_returns, benchmark_returns):
        dc = down_capture(daily_returns, benchmark_returns)
        assert isinstance(dc, float)

    def test_active_return(self, daily_returns, benchmark_returns):
        ar = active_return(daily_returns, benchmark_returns)
        assert isinstance(ar, float)

    def test_empty_benchmark(self, daily_returns):
        empty = pd.Series(dtype=float)
        alpha, beta = alpha_beta(daily_returns, empty)
        assert alpha == 0.0
        assert beta == 0.0


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    def test_kmeans_labels(self, daily_returns):
        detector = RegimeDetector()
        regimes = detector.detect_kmeans(daily_returns, n_regimes=3)
        assert len(regimes) == len(daily_returns)
        unique = regimes.dropna().unique()
        assert len(unique) <= 3

    def test_kmeans_two_regimes(self, daily_returns):
        detector = RegimeDetector()
        regimes = detector.detect_kmeans(daily_returns, n_regimes=2)
        unique = regimes.dropna().unique()
        assert len(unique) <= 2

    def test_kmeans_short_series(self):
        detector = RegimeDetector()
        short = pd.Series([0.01, -0.01, 0.02])
        regimes = detector.detect_kmeans(short, n_regimes=3)
        assert len(regimes) == len(short)

    def test_regime_statistics(self, daily_returns):
        detector = RegimeDetector()
        regimes = detector.detect_kmeans(daily_returns, n_regimes=3)
        stats = detector.regime_statistics(daily_returns, regimes)
        assert isinstance(stats, pd.DataFrame)
        assert "mean_return" in stats.columns
        assert "volatility" in stats.columns
        assert "sharpe" in stats.columns

    def test_hmm_import_error(self, daily_returns):
        detector = RegimeDetector()
        # This may or may not raise depending on if hmmlearn is installed
        try:
            regimes = detector.detect_hmm(daily_returns, n_regimes=2)
            assert len(regimes) == len(daily_returns)
        except ImportError as e:
            assert "hmmlearn" in str(e)


# ---------------------------------------------------------------------------
# calculate_all
# ---------------------------------------------------------------------------

class TestCalculateAll:
    def test_returns_only(self, daily_returns):
        result = calculate_all(daily_returns)
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "volatility" in result
        assert "var_95" in result

    def test_with_trades(self, daily_returns, sample_trades):
        result = calculate_all(daily_returns, trades=sample_trades)
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "expectancy" in result
