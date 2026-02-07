"""Indicator catalog: metadata for all Tier 1 indicators."""

from __future__ import annotations

from ta_framework.core.types import IndicatorCategory, IndicatorConfig, IndicatorTier

# ---------------------------------------------------------------------------
# Tier 1 Indicator Catalog (~40 essential indicators)
# ---------------------------------------------------------------------------

INDICATOR_CATALOG: dict[str, IndicatorConfig] = {
    # ---- Trend ----
    "sma": IndicatorConfig(
        name="sma", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 20, "column": "close"},
        description="Simple Moving Average",
    ),
    "ema": IndicatorConfig(
        name="ema", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 20, "column": "close"},
        description="Exponential Moving Average",
    ),
    "wma": IndicatorConfig(
        name="wma", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 20, "column": "close"},
        description="Weighted Moving Average",
    ),
    "dema": IndicatorConfig(
        name="dema", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 20, "column": "close"},
        description="Double Exponential Moving Average",
    ),
    "tema": IndicatorConfig(
        name="tema", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 20, "column": "close"},
        description="Triple Exponential Moving Average",
    ),
    "vwma": IndicatorConfig(
        name="vwma", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 20},
        description="Volume Weighted Moving Average",
    ),
    "hma": IndicatorConfig(
        name="hma", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 20},
        description="Hull Moving Average",
    ),
    "kama": IndicatorConfig(
        name="kama", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 10, "fast": 2, "slow": 30},
        description="Kaufman Adaptive Moving Average",
    ),
    "t3": IndicatorConfig(
        name="t3", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 5, "a": 0.7},
        description="Tillson T3 Moving Average",
    ),
    "supertrend": IndicatorConfig(
        name="supertrend", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 7, "multiplier": 3.0},
        description="Supertrend",
    ),
    "ichimoku": IndicatorConfig(
        name="ichimoku", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"tenkan": 9, "kijun": 26, "senkou": 52},
        description="Ichimoku Cloud",
    ),
    "adx": IndicatorConfig(
        name="adx", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 14},
        description="Average Directional Index",
    ),
    "aroon": IndicatorConfig(
        name="aroon", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"length": 25},
        description="Aroon Indicator",
    ),
    "psar": IndicatorConfig(
        name="psar", category=IndicatorCategory.TREND, tier=IndicatorTier.TIER1,
        params={"af0": 0.02, "af": 0.02, "max_af": 0.2},
        description="Parabolic SAR",
    ),

    # ---- Momentum ----
    "rsi": IndicatorConfig(
        name="rsi", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"length": 14},
        description="Relative Strength Index",
    ),
    "macd": IndicatorConfig(
        name="macd", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"fast": 12, "slow": 26, "signal": 9},
        description="Moving Average Convergence Divergence",
    ),
    "stoch": IndicatorConfig(
        name="stoch", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"k": 14, "d": 3, "smooth_k": 3},
        description="Stochastic Oscillator",
    ),
    "cci": IndicatorConfig(
        name="cci", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"length": 20},
        description="Commodity Channel Index",
    ),
    "willr": IndicatorConfig(
        name="willr", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"length": 14},
        description="Williams %R",
    ),
    "roc": IndicatorConfig(
        name="roc", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"length": 10},
        description="Rate of Change",
    ),
    "mfi": IndicatorConfig(
        name="mfi", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"length": 14},
        description="Money Flow Index",
    ),
    "tsi": IndicatorConfig(
        name="tsi", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"fast": 13, "slow": 25},
        description="True Strength Index",
    ),
    "uo": IndicatorConfig(
        name="uo", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"fast": 7, "medium": 14, "slow": 28},
        description="Ultimate Oscillator",
    ),
    "ao": IndicatorConfig(
        name="ao", category=IndicatorCategory.MOMENTUM, tier=IndicatorTier.TIER1,
        params={"fast": 5, "slow": 34},
        description="Awesome Oscillator",
    ),

    # ---- Volatility ----
    "bbands": IndicatorConfig(
        name="bbands", category=IndicatorCategory.VOLATILITY, tier=IndicatorTier.TIER1,
        params={"length": 20, "std": 2.0},
        description="Bollinger Bands",
    ),
    "atr": IndicatorConfig(
        name="atr", category=IndicatorCategory.VOLATILITY, tier=IndicatorTier.TIER1,
        params={"length": 14},
        description="Average True Range",
    ),
    "kc": IndicatorConfig(
        name="kc", category=IndicatorCategory.VOLATILITY, tier=IndicatorTier.TIER1,
        params={"length": 20, "scalar": 1.5},
        description="Keltner Channel",
    ),
    "donchian": IndicatorConfig(
        name="donchian", category=IndicatorCategory.VOLATILITY, tier=IndicatorTier.TIER1,
        params={"lower_length": 20, "upper_length": 20},
        description="Donchian Channel",
    ),
    "stdev": IndicatorConfig(
        name="stdev", category=IndicatorCategory.VOLATILITY, tier=IndicatorTier.TIER1,
        params={"length": 20},
        description="Standard Deviation",
    ),
    "true_range": IndicatorConfig(
        name="true_range", category=IndicatorCategory.VOLATILITY, tier=IndicatorTier.TIER1,
        params={},
        description="True Range",
    ),

    # ---- Volume ----
    "obv": IndicatorConfig(
        name="obv", category=IndicatorCategory.VOLUME, tier=IndicatorTier.TIER1,
        params={},
        description="On Balance Volume",
    ),
    "vwap": IndicatorConfig(
        name="vwap", category=IndicatorCategory.VOLUME, tier=IndicatorTier.TIER1,
        params={},
        description="Volume Weighted Average Price",
    ),
    "ad": IndicatorConfig(
        name="ad", category=IndicatorCategory.VOLUME, tier=IndicatorTier.TIER1,
        params={},
        description="Accumulation/Distribution Line",
    ),
    "cmf": IndicatorConfig(
        name="cmf", category=IndicatorCategory.VOLUME, tier=IndicatorTier.TIER1,
        params={"length": 20},
        description="Chaikin Money Flow",
    ),
    "fi": IndicatorConfig(
        name="fi", category=IndicatorCategory.VOLUME, tier=IndicatorTier.TIER1,
        params={"length": 13},
        description="Force Index",
    ),
    "volume_sma": IndicatorConfig(
        name="volume_sma", category=IndicatorCategory.VOLUME, tier=IndicatorTier.TIER1,
        params={"length": 20},
        description="Volume Simple Moving Average",
    ),

    # ---- Overlap ----
    "pivot": IndicatorConfig(
        name="pivot", category=IndicatorCategory.OVERLAP, tier=IndicatorTier.TIER1,
        params={"method": "traditional"},
        description="Pivot Points",
    ),
    "fib": IndicatorConfig(
        name="fib", category=IndicatorCategory.OVERLAP, tier=IndicatorTier.TIER1,
        params={"length": 20},
        description="Fibonacci Retracement Levels",
    ),
}


def get_catalog_by_category(category: IndicatorCategory) -> dict[str, IndicatorConfig]:
    """Return a subset of the catalog filtered by category."""
    return {k: v for k, v in INDICATOR_CATALOG.items() if v.category == category}


def get_catalog_by_tier(tier: IndicatorTier) -> dict[str, IndicatorConfig]:
    """Return a subset of the catalog filtered by tier."""
    return {k: v for k, v in INDICATOR_CATALOG.items() if v.tier == tier}
