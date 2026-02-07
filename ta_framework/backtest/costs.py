"""Cost models for commissions and slippage."""

from __future__ import annotations

from abc import ABC, abstractmethod


class CostModel(ABC):
    """Base interface for cost models."""

    @abstractmethod
    def apply(self, trade_value: float, volume: float = 0) -> float:
        """Return the cost in currency units for a given *trade_value*."""


# ---------------------------------------------------------------------------
# Commission models
# ---------------------------------------------------------------------------

class FixedCommission(CostModel):
    """Flat fee per trade regardless of size."""

    def __init__(self, amount_per_trade: float) -> None:
        self.amount_per_trade = amount_per_trade

    def apply(self, trade_value: float, volume: float = 0) -> float:
        return self.amount_per_trade


class PercentageCommission(CostModel):
    """Commission as a percentage of trade value."""

    def __init__(self, pct: float) -> None:
        self.pct = pct

    def apply(self, trade_value: float, volume: float = 0) -> float:
        return abs(trade_value) * self.pct


class TieredCommission(CostModel):
    """Volume-tiered commission.

    Parameters
    ----------
    tiers : list[tuple[float, float]]
        Each tuple is ``(volume_threshold, pct)``.  Tiers must be sorted
        ascending by threshold.  The first tier whose threshold exceeds
        *volume* is used; if *volume* exceeds all thresholds the last
        tier applies.
    """

    def __init__(self, tiers: list[tuple[float, float]]) -> None:
        self.tiers = sorted(tiers, key=lambda t: t[0])

    def apply(self, trade_value: float, volume: float = 0) -> float:
        pct = self.tiers[-1][1]  # default: highest tier
        for threshold, tier_pct in self.tiers:
            if volume <= threshold:
                pct = tier_pct
                break
        return abs(trade_value) * pct


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------

class SlippageModel(CostModel):
    """Simple slippage model.

    Cost = ``trade_value * base_pct + trade_value * volume_impact / volume``
    (volume_impact term is skipped when *volume* is zero).
    """

    def __init__(self, base_pct: float, volume_impact: float = 0.0) -> None:
        self.base_pct = base_pct
        self.volume_impact = volume_impact

    def apply(self, trade_value: float, volume: float = 0) -> float:
        cost = abs(trade_value) * self.base_pct
        if self.volume_impact and volume > 0:
            cost += abs(trade_value) * self.volume_impact / volume
        return cost
