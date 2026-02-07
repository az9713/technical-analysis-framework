"""Comprehensive performance metrics for strategy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Return-based metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.04, periods: int = 252
) -> float:
    """Annualized Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.04, periods: int = 252
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if returns.empty:
        return 0.0
    dd = downside_deviation(returns, threshold=risk_free_rate / periods)
    if dd == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / periods
    return float(excess / dd * np.sqrt(periods))


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    if returns.empty or len(returns) < 2:
        return 0.0
    equity = (1 + returns).cumprod()
    mdd = max_drawdown(equity_curve=equity)
    if mdd == 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n = len(returns)
    ann_return = (1 + total_return) ** (periods / n) - 1
    return float(ann_return / mdd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains over losses."""
    if returns.empty:
        return 0.0
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    sum_losses = losses.sum()
    if sum_losses == 0:
        return float("inf") if gains.sum() > 0 else 0.0
    return float(gains.sum() / sum_losses)


def max_drawdown(
    returns: pd.Series | None = None, equity_curve: pd.Series | None = None
) -> float:
    """Maximum drawdown as a positive fraction.

    Provide either returns or equity_curve.
    """
    if equity_curve is None:
        if returns is None or returns.empty:
            return 0.0
        equity_curve = (1 + returns).cumprod()

    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    return float(abs(drawdowns.min()))


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Maximum drawdown duration in bars."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0

    running_max = equity_curve.cummax()
    in_dd = equity_curve < running_max

    max_dur = 0
    cur = 0
    for dd in in_dd:
        if dd:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return max_dur


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    if start <= 0:
        return 0.0

    n_years = len(equity_curve) / 252
    if n_years <= 0:
        return 0.0

    return float((end / start) ** (1 / n_years) - 1)


def volatility(returns: pd.Series, periods: int = 252) -> float:
    """Annualized volatility."""
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods))


def downside_deviation(returns: pd.Series, threshold: float = 0.0) -> float:
    """Downside deviation below threshold."""
    if returns.empty:
        return 0.0
    diff = returns - threshold
    neg = diff[diff < 0]
    if neg.empty:
        return 0.0
    return float(np.sqrt((neg**2).mean()))


# ---------------------------------------------------------------------------
# Trade-based metrics
# ---------------------------------------------------------------------------

def _trade_pnls(trades) -> pd.Series:
    """Extract net P&L from a list of Trade objects or numeric series."""
    if isinstance(trades, pd.Series):
        return trades.dropna()
    pnls = []
    for t in trades:
        pnl = getattr(t, "net_pnl", None)
        if pnl is None:
            pnl = getattr(t, "pnl", 0.0)
        pnls.append(pnl)
    return pd.Series(pnls, dtype=float)


def win_rate(trades) -> float:
    """Fraction of winning trades."""
    pnls = _trade_pnls(trades)
    if pnls.empty:
        return 0.0
    return float((pnls > 0).sum() / len(pnls))


def profit_factor(trades) -> float:
    """Gross profit / gross loss."""
    pnls = _trade_pnls(trades)
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def expectancy(trades) -> float:
    """Expected value per trade."""
    pnls = _trade_pnls(trades)
    if pnls.empty:
        return 0.0
    return float(pnls.mean())


def avg_win(trades) -> float:
    """Average winning trade P&L."""
    pnls = _trade_pnls(trades)
    wins = pnls[pnls > 0]
    return float(wins.mean()) if not wins.empty else 0.0


def avg_loss(trades) -> float:
    """Average losing trade P&L (returned as negative)."""
    pnls = _trade_pnls(trades)
    losses = pnls[pnls < 0]
    return float(losses.mean()) if not losses.empty else 0.0


def largest_win(trades) -> float:
    """Largest single winning trade."""
    pnls = _trade_pnls(trades)
    wins = pnls[pnls > 0]
    return float(wins.max()) if not wins.empty else 0.0


def largest_loss(trades) -> float:
    """Largest single losing trade (returned as negative)."""
    pnls = _trade_pnls(trades)
    losses = pnls[pnls < 0]
    return float(losses.min()) if not losses.empty else 0.0


def consecutive_wins(trades) -> int:
    """Maximum consecutive winning trades."""
    pnls = _trade_pnls(trades)
    if pnls.empty:
        return 0
    max_streak = 0
    cur = 0
    for p in pnls:
        if p > 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def consecutive_losses(trades) -> int:
    """Maximum consecutive losing trades."""
    pnls = _trade_pnls(trades)
    if pnls.empty:
        return 0
    max_streak = 0
    cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def recovery_factor(equity_curve: pd.Series) -> float:
    """Net profit / max drawdown in absolute terms."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    net_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
    mdd = max_drawdown(equity_curve=equity_curve)
    peak_val = equity_curve.max()
    dd_abs = mdd * peak_val
    if dd_abs == 0:
        return 0.0
    return float(net_profit / dd_abs)


def tail_ratio(returns: pd.Series) -> float:
    """Ratio of the 95th percentile to the absolute 5th percentile."""
    if returns.empty:
        return 0.0
    right = np.percentile(returns.dropna(), 95)
    left = abs(np.percentile(returns.dropna(), 5))
    if left == 0:
        return 0.0
    return float(right / left)


def common_sense_ratio(returns: pd.Series) -> float:
    """Tail ratio * profit factor (using sign of returns as trades)."""
    if returns.empty:
        return 0.0
    tr = tail_ratio(returns)
    pf = profit_factor(returns)
    if pf == float("inf"):
        return 0.0
    return float(tr * pf)


def skewness(returns: pd.Series) -> float:
    """Return series skewness."""
    if returns.empty or len(returns) < 3:
        return 0.0
    return float(returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Return series excess kurtosis."""
    if returns.empty or len(returns) < 4:
        return 0.0
    return float(returns.kurtosis())


def var_95(returns: pd.Series) -> float:
    """95% historical VaR (positive number)."""
    if returns.empty:
        return 0.0
    return float(-np.percentile(returns.dropna(), 5))


def cvar_95(returns: pd.Series) -> float:
    """95% Conditional VaR / Expected Shortfall (positive number)."""
    if returns.empty:
        return 0.0
    clean = returns.dropna()
    threshold = np.percentile(clean, 5)
    tail = clean[clean <= threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def calculate_all(
    returns: pd.Series,
    equity_curve: pd.Series | None = None,
    trades: list | None = None,
) -> dict[str, Any]:
    """Calculate all available metrics and return as a dictionary."""
    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()

    result: dict[str, Any] = {
        # Return metrics
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "calmar_ratio": calmar_ratio(returns),
        "omega_ratio": omega_ratio(returns),
        "cagr": cagr(equity_curve),
        "volatility": volatility(returns),
        "max_drawdown": max_drawdown(equity_curve=equity_curve),
        "max_drawdown_duration": max_drawdown_duration(equity_curve),
        "recovery_factor": recovery_factor(equity_curve),
        # Distribution
        "skewness": skewness(returns),
        "kurtosis": kurtosis(returns),
        "tail_ratio": tail_ratio(returns),
        "common_sense_ratio": common_sense_ratio(returns),
        "var_95": var_95(returns),
        "cvar_95": cvar_95(returns),
        "downside_deviation": downside_deviation(returns),
    }

    if trades is not None and len(trades) > 0:
        result.update(
            {
                "win_rate": win_rate(trades),
                "profit_factor": profit_factor(trades),
                "expectancy": expectancy(trades),
                "avg_win": avg_win(trades),
                "avg_loss": avg_loss(trades),
                "largest_win": largest_win(trades),
                "largest_loss": largest_loss(trades),
                "consecutive_wins": consecutive_wins(trades),
                "consecutive_losses": consecutive_losses(trades),
            }
        )

    return result
