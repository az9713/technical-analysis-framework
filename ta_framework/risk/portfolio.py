"""Portfolio-level risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Annualized portfolio volatility.

    Parameters
    ----------
    weights : np.ndarray
        Asset weight vector.
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.

    Returns
    -------
    float
        Portfolio volatility (standard deviation).
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    if w.size == 0 or cov.size == 0:
        return 0.0

    variance = w @ cov @ w
    return float(np.sqrt(max(variance, 0.0)))


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Maximum drawdown duration in bars.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve.

    Returns
    -------
    int
        Longest drawdown duration in number of bars.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0

    running_max = equity_curve.cummax()
    in_drawdown = equity_curve < running_max

    max_dur = 0
    current_dur = 0
    for dd in in_drawdown:
        if dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0

    return max_dur


def calmar_ratio(returns: pd.Series, period: int = 252) -> float:
    """Calmar ratio: annualized return / max drawdown.

    Parameters
    ----------
    returns : pd.Series
        Period return series.
    period : int
        Number of periods per year for annualization.

    Returns
    -------
    float
        Calmar ratio.
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    max_dd = abs(drawdowns.min())

    if max_dd == 0:
        return 0.0

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_periods = len(returns)
    ann_return = (1 + total_return) ** (period / n_periods) - 1

    return ann_return / max_dd


def ulcer_index(equity_curve: pd.Series) -> float:
    """Ulcer Index: root mean square of percentage drawdowns.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve.

    Returns
    -------
    float
        Ulcer Index value.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    running_max = equity_curve.cummax()
    pct_drawdowns = ((equity_curve - running_max) / running_max) * 100
    return float(np.sqrt((pct_drawdowns**2).mean()))


def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """Marginal risk contribution per asset.

    Parameters
    ----------
    weights : np.ndarray
        Asset weight vector.
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.

    Returns
    -------
    np.ndarray
        Risk contribution of each asset (sums to portfolio volatility).
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    if w.size == 0 or cov.size == 0:
        return np.array([])

    port_vol = portfolio_volatility(w, cov)
    if port_vol == 0:
        return np.zeros_like(w)

    marginal = cov @ w / port_vol
    return w * marginal
