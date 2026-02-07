"""Benchmark comparison metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> tuple[float, float]:
    """CAPM alpha and beta via OLS regression.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    benchmark_returns : pd.Series
        Benchmark returns.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    tuple[float, float]
        (annualized alpha, beta)
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 0.0

    rf_daily = risk_free_rate / 252
    r = aligned.iloc[:, 0].values - rf_daily
    b = aligned.iloc[:, 1].values - rf_daily

    cov_rb = np.cov(r, b, ddof=1)
    var_b = cov_rb[1, 1]
    if var_b == 0:
        return 0.0, 0.0

    beta = cov_rb[0, 1] / var_b
    alpha_daily = r.mean() - beta * b.mean()
    alpha_annual = alpha_daily * 252

    return float(alpha_annual), float(beta)


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Information ratio: active return / tracking error.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    benchmark_returns : pd.Series
        Benchmark returns.

    Returns
    -------
    float
        Annualized information ratio.
    """
    te = tracking_error(returns, benchmark_returns)
    if te == 0:
        return 0.0
    ar = active_return(returns, benchmark_returns)
    return float(ar / te)


def tracking_error(
    returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Annualized tracking error (std of active returns).

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    benchmark_returns : pd.Series
        Benchmark returns.

    Returns
    -------
    float
        Annualized tracking error.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std() * np.sqrt(252))


def up_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Up-capture ratio: strategy return in up markets / benchmark up return.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    benchmark_returns : pd.Series
        Benchmark returns.

    Returns
    -------
    float
        Up-capture ratio (1.0 = matches benchmark in up markets).
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0

    up_mask = aligned.iloc[:, 1] > 0
    if up_mask.sum() == 0:
        return 0.0

    strat_up = aligned.iloc[:, 0][up_mask].mean()
    bench_up = aligned.iloc[:, 1][up_mask].mean()
    if bench_up == 0:
        return 0.0
    return float(strat_up / bench_up)


def down_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Down-capture ratio: strategy return in down markets / benchmark down return.

    Lower is better (capturing less of the downside).

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    benchmark_returns : pd.Series
        Benchmark returns.

    Returns
    -------
    float
        Down-capture ratio.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0

    down_mask = aligned.iloc[:, 1] < 0
    if down_mask.sum() == 0:
        return 0.0

    strat_down = aligned.iloc[:, 0][down_mask].mean()
    bench_down = aligned.iloc[:, 1][down_mask].mean()
    if bench_down == 0:
        return 0.0
    return float(strat_down / bench_down)


def active_return(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized active return (strategy minus benchmark).

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    benchmark_returns : pd.Series
        Benchmark returns.

    Returns
    -------
    float
        Annualized active return.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.mean() * 252)
