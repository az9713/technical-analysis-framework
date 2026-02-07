"""Value at Risk (VaR) calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def parametric_var(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """Parametric (Gaussian) Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Historical return series.
    confidence : float
        Confidence level (e.g., 0.95 for 95%).
    horizon : int
        Time horizon in periods.

    Returns
    -------
    float
        VaR as a positive number representing potential loss.
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu * horizon + z * sigma * np.sqrt(horizon))
    return max(var, 0.0)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical simulation Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Historical return series.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        VaR as a positive number representing potential loss.
    """
    if returns.empty:
        return 0.0

    percentile = (1 - confidence) * 100
    var = -np.percentile(returns.dropna(), percentile)
    return max(var, 0.0)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10000,
    horizon: int = 1,
) -> float:
    """Monte Carlo simulation Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Historical return series.
    confidence : float
        Confidence level.
    n_sims : int
        Number of simulations.
    horizon : int
        Time horizon in periods.

    Returns
    -------
    float
        VaR as a positive number representing potential loss.
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    mu = returns.mean()
    sigma = returns.std()

    sim_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
    percentile = (1 - confidence) * 100
    var = -np.percentile(sim_returns, percentile)
    return max(var, 0.0)


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall).

    Average loss beyond the VaR threshold.

    Parameters
    ----------
    returns : pd.Series
        Historical return series.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        CVaR as a positive number.
    """
    if returns.empty:
        return 0.0

    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0

    percentile = (1 - confidence) * 100
    var_threshold = np.percentile(clean, percentile)
    tail_losses = clean[clean <= var_threshold]

    if len(tail_losses) == 0:
        return 0.0

    return -tail_losses.mean()
