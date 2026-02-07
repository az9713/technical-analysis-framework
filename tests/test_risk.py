"""Tests for the risk module: position sizing, VaR, stops, portfolio."""

import numpy as np
import pandas as pd
import pytest

from ta_framework.risk.position_sizing import (
    FixedFractional,
    KellyCriterion,
    RiskParity,
    VolatilityBased,
)
from ta_framework.risk.var import cvar, historical_var, monte_carlo_var, parametric_var
from ta_framework.risk.stops import atr_stop, chandelier_stop, fixed_stop, trailing_stop
from ta_framework.risk.portfolio import (
    calmar_ratio,
    max_drawdown_duration,
    portfolio_volatility,
    risk_contribution,
    ulcer_index,
)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestFixedFractional:
    def test_basic(self):
        sizer = FixedFractional()
        # 100k capital, $50 stock, risk 2% -> (100000 * 0.02) / 50 = 40 shares
        result = sizer.calculate(capital=100_000, price=50, risk_per_trade=0.02)
        assert result == pytest.approx(40.0)

    def test_zero_price(self):
        sizer = FixedFractional()
        assert sizer.calculate(capital=100_000, price=0, risk_per_trade=0.02) == 0.0

    def test_zero_capital(self):
        sizer = FixedFractional()
        assert sizer.calculate(capital=0, price=50, risk_per_trade=0.02) == 0.0

    def test_zero_risk(self):
        sizer = FixedFractional()
        assert sizer.calculate(capital=100_000, price=50, risk_per_trade=0) == 0.0


class TestKellyCriterion:
    def test_positive_edge(self):
        sizer = KellyCriterion()
        # win_rate=0.6, payoff=2.0 -> Kelly = 0.6 - 0.4/2.0 = 0.4, half-Kelly = 0.2
        result = sizer.calculate(
            capital=100_000,
            price=100,
            risk_per_trade=0.5,
            win_rate=0.6,
            payoff_ratio=2.0,
            fraction=0.5,
        )
        expected = (100_000 * 0.2) / 100  # 200 shares
        assert result == pytest.approx(expected)

    def test_no_edge(self):
        sizer = KellyCriterion()
        # win_rate=0.5, payoff=1.0 -> Kelly = 0.5 - 0.5/1.0 = 0 -> 0 shares
        result = sizer.calculate(
            capital=100_000,
            price=100,
            risk_per_trade=0.5,
            win_rate=0.5,
            payoff_ratio=1.0,
        )
        assert result == 0.0

    def test_negative_edge(self):
        sizer = KellyCriterion()
        # win_rate=0.3, payoff=1.0 -> Kelly = 0.3 - 0.7 = -0.4 -> clamped to 0
        result = sizer.calculate(
            capital=100_000,
            price=100,
            risk_per_trade=0.5,
            win_rate=0.3,
            payoff_ratio=1.0,
        )
        assert result == 0.0

    def test_capped_by_risk_per_trade(self):
        sizer = KellyCriterion()
        # Large Kelly but small risk_per_trade cap
        result = sizer.calculate(
            capital=100_000,
            price=100,
            risk_per_trade=0.01,
            win_rate=0.9,
            payoff_ratio=5.0,
            fraction=1.0,
        )
        # Kelly = 0.9 - 0.1/5.0 = 0.88, capped at 0.01
        expected = (100_000 * 0.01) / 100
        assert result == pytest.approx(expected)


class TestVolatilityBased:
    def test_basic(self):
        sizer = VolatilityBased()
        # capital=100k, risk=2%, ATR=2.0, mult=2.0 -> 100000*0.02 / (2*2) = 500
        result = sizer.calculate(
            capital=100_000, price=50, risk_per_trade=0.02, atr=2.0, multiplier=2.0
        )
        assert result == pytest.approx(500.0)

    def test_zero_atr(self):
        sizer = VolatilityBased()
        assert sizer.calculate(capital=100_000, price=50, risk_per_trade=0.02, atr=0) == 0.0


# ---------------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------------

class TestVaR:
    @pytest.fixture
    def returns(self):
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.02, 1000))

    def test_parametric_var_positive(self, returns):
        var = parametric_var(returns, confidence=0.95)
        assert var > 0

    def test_historical_var_positive(self, returns):
        var = historical_var(returns, confidence=0.95)
        assert var > 0

    def test_monte_carlo_var_positive(self, returns):
        np.random.seed(123)
        var = monte_carlo_var(returns, confidence=0.95, n_sims=5000)
        assert var > 0

    def test_cvar_greater_than_var(self, returns):
        var_val = historical_var(returns, confidence=0.95)
        cvar_val = cvar(returns, confidence=0.95)
        assert cvar_val >= var_val

    def test_empty_returns(self):
        empty = pd.Series(dtype=float)
        assert parametric_var(empty) == 0.0
        assert historical_var(empty) == 0.0
        assert cvar(empty) == 0.0

    def test_zero_volatility(self):
        flat = pd.Series([0.001] * 100)
        assert parametric_var(flat) == 0.0


# ---------------------------------------------------------------------------
# Stops
# ---------------------------------------------------------------------------

class TestStops:
    def test_fixed_stop(self):
        # $100 entry, 5% stop -> $95
        assert fixed_stop(100.0, 0.05) == pytest.approx(95.0)

    def test_fixed_stop_zero(self):
        assert fixed_stop(0.0, 0.05) == 0.0

    def test_atr_stop_length(self, sample_ohlcv):
        stops = atr_stop(sample_ohlcv, multiplier=2.0, period=14)
        assert len(stops) == len(sample_ohlcv)
        assert stops.notna().all()

    def test_atr_stop_below_close(self, sample_ohlcv):
        stops = atr_stop(sample_ohlcv, multiplier=2.0, period=14)
        # Stop should generally be below close for positive ATR
        pct_below = (stops < sample_ohlcv["close"]).mean()
        assert pct_below > 0.5

    def test_trailing_stop_monotonic_peak(self):
        prices = pd.Series([10, 11, 12, 11, 10, 13, 12])
        stops = trailing_stop(prices, trail_pct=0.1)
        # After peak of 12, stop is 12*0.9=10.8
        assert stops.iloc[3] == pytest.approx(12 * 0.9)
        # After new peak of 13, stop is 13*0.9=11.7
        assert stops.iloc[5] == pytest.approx(13 * 0.9)

    def test_trailing_stop_empty(self):
        result = trailing_stop(pd.Series(dtype=float), trail_pct=0.05)
        assert result.empty

    def test_chandelier_stop_length(self, sample_ohlcv):
        stops = chandelier_stop(sample_ohlcv, period=22, multiplier=3.0)
        assert len(stops) == len(sample_ohlcv)

    def test_atr_stop_empty(self):
        empty_df = pd.DataFrame(columns=["high", "low", "close"])
        result = atr_stop(empty_df)
        assert result.empty


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_portfolio_volatility(self):
        # Two uncorrelated assets with equal weight
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        weights = np.array([0.5, 0.5])
        vol = portfolio_volatility(weights, cov)
        # sqrt(0.25*0.04 + 0.25*0.04) = sqrt(0.02) ~ 0.1414
        assert vol == pytest.approx(np.sqrt(0.02), rel=1e-4)

    def test_portfolio_volatility_empty(self):
        assert portfolio_volatility(np.array([]), np.array([])) == 0.0

    def test_max_drawdown_duration(self):
        # Equity: rises, drops, stays down, recovers
        eq = pd.Series([100, 110, 105, 100, 95, 100, 110, 115])
        dur = max_drawdown_duration(eq)
        assert dur == 4  # bars 2-5 are in drawdown

    def test_calmar_ratio(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        ratio = calmar_ratio(returns, period=252)
        assert isinstance(ratio, float)

    def test_ulcer_index(self):
        eq = pd.Series([100, 110, 105, 100, 110, 120])
        ui = ulcer_index(eq)
        assert ui > 0

    def test_risk_contribution_sums(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = np.array([0.6, 0.4])
        rc = risk_contribution(weights, cov)
        port_vol = portfolio_volatility(weights, cov)
        assert np.sum(rc) == pytest.approx(port_vol, rel=1e-6)

    def test_risk_contribution_empty(self):
        rc = risk_contribution(np.array([]), np.array([]))
        assert len(rc) == 0
