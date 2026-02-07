"""Position sizing strategies for risk management."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class PositionSizer(ABC):
    """Base class for position sizing strategies."""

    @abstractmethod
    def calculate(
        self, capital: float, price: float, risk_per_trade: float, **kwargs
    ) -> float:
        """Calculate position size (number of shares/units).

        Parameters
        ----------
        capital : float
            Available trading capital.
        price : float
            Current asset price.
        risk_per_trade : float
            Fraction of capital to risk (0.0 to 1.0).

        Returns
        -------
        float
            Number of shares/units to trade.
        """


class FixedFractional(PositionSizer):
    """Risk a fixed percentage of capital per trade.

    Position size = (capital * risk_per_trade) / price
    """

    def calculate(
        self, capital: float, price: float, risk_per_trade: float, **kwargs
    ) -> float:
        if price <= 0 or capital <= 0 or risk_per_trade <= 0:
            return 0.0
        return (capital * risk_per_trade) / price


class KellyCriterion(PositionSizer):
    """Kelly criterion position sizing.

    Kelly fraction = win_rate - (1 - win_rate) / payoff_ratio
    Position size = (capital * kelly_fraction) / price

    Parameters via kwargs:
        win_rate : float -- historical win rate (0.0 to 1.0)
        payoff_ratio : float -- average win / average loss
        fraction : float -- Kelly fraction multiplier (default 0.5 for half-Kelly)
    """

    def calculate(
        self, capital: float, price: float, risk_per_trade: float, **kwargs
    ) -> float:
        if price <= 0 or capital <= 0:
            return 0.0

        win_rate = kwargs.get("win_rate", 0.5)
        payoff_ratio = kwargs.get("payoff_ratio", 1.0)
        fraction = kwargs.get("fraction", 0.5)

        if payoff_ratio <= 0:
            return 0.0

        kelly = win_rate - (1.0 - win_rate) / payoff_ratio
        kelly = max(kelly, 0.0) * fraction

        # Cap at risk_per_trade to limit exposure
        kelly = min(kelly, risk_per_trade)
        return (capital * kelly) / price


class VolatilityBased(PositionSizer):
    """Position size inversely proportional to asset volatility (ATR).

    Position size = (capital * risk_per_trade) / (atr * multiplier)

    Parameters via kwargs:
        atr : float -- current Average True Range value
        multiplier : float -- ATR multiplier for stop distance (default 2.0)
    """

    def calculate(
        self, capital: float, price: float, risk_per_trade: float, **kwargs
    ) -> float:
        if price <= 0 or capital <= 0 or risk_per_trade <= 0:
            return 0.0

        atr = kwargs.get("atr", 0.0)
        multiplier = kwargs.get("multiplier", 2.0)

        if atr <= 0 or multiplier <= 0:
            return 0.0

        dollar_risk = capital * risk_per_trade
        risk_per_share = atr * multiplier
        return dollar_risk / risk_per_share


class RiskParity(PositionSizer):
    """Equalize risk contribution across assets using covariance matrix.

    Each asset contributes equally to total portfolio volatility.

    Parameters via kwargs:
        covariance : np.ndarray -- covariance matrix of asset returns
        asset_index : int -- index of the current asset in the covariance matrix
        n_assets : int -- total number of assets (defaults to covariance shape)
    """

    def calculate(
        self, capital: float, price: float, risk_per_trade: float, **kwargs
    ) -> float:
        if price <= 0 or capital <= 0:
            return 0.0

        covariance = kwargs.get("covariance")
        asset_index = kwargs.get("asset_index", 0)

        if covariance is None:
            return 0.0

        cov = np.asarray(covariance, dtype=float)
        n = cov.shape[0]

        if n == 0 or asset_index >= n:
            return 0.0

        # Inverse-volatility weights as starting point for risk parity
        vols = np.sqrt(np.diag(cov))
        if np.any(vols <= 0):
            return 0.0

        inv_vol_weights = (1.0 / vols) / np.sum(1.0 / vols)
        weight = inv_vol_weights[asset_index]

        allocation = capital * risk_per_trade * weight
        return allocation / price
