"""Pre-built trading strategies.

Each strategy is a :class:`SignalGenerator` subclass that assumes the
required indicator columns are already present in the DataFrame (computed
by the IndicatorEngine).  Column names follow the pandas_ta convention,
e.g. ``EMA_12``, ``RSI_14``, ``MACD_12_26_9``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta_framework.core.registry import strategy_registry
from ta_framework.signals.base import SignalGenerator
from ta_framework.signals.rules import breakout, crossover, threshold


# ---------------------------------------------------------------------------
# EMA Crossover
# ---------------------------------------------------------------------------

@strategy_registry.register("ema_cross")
class EMACrossStrategy(SignalGenerator):
    """Classic EMA crossover strategy."""

    name = "ema_cross"

    def __init__(self, fast_period: int = 12, slow_period: int = 26) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def required_indicators(self) -> list[dict]:
        return [
            {"name": "ema", "length": self.fast_period},
            {"name": "ema", "length": self.slow_period},
        ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        fast_col = f"EMA_{self.fast_period}"
        slow_col = f"EMA_{self.slow_period}"
        self.validate_columns(df, [fast_col, slow_col])

        df["signal"] = crossover(df[fast_col], df[slow_col])

        # Signal strength: distance between the two EMAs normalised by price
        spread = (df[fast_col] - df[slow_col]).abs() / df["close"]
        df["signal_strength"] = spread / spread.max() if spread.max() > 0 else 0.0
        return df


# ---------------------------------------------------------------------------
# RSI Threshold
# ---------------------------------------------------------------------------

@strategy_registry.register("rsi")
class RSIStrategy(SignalGenerator):
    """RSI overbought / oversold threshold strategy."""

    name = "rsi"

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    @property
    def required_indicators(self) -> list[dict]:
        return [{"name": "rsi", "length": self.period}]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"RSI_{self.period}"
        self.validate_columns(df, [col])

        df["signal"] = threshold(df[col], upper=self.overbought, lower=self.oversold)

        # Strength: how far into the zone the RSI is
        strength = pd.Series(0.0, index=df.index)
        oversold_mask = df[col] < self.oversold
        overbought_mask = df[col] > self.overbought
        strength[oversold_mask] = (self.oversold - df[col][oversold_mask]) / self.oversold
        strength[overbought_mask] = (df[col][overbought_mask] - self.overbought) / (100 - self.overbought)
        df["signal_strength"] = strength.clip(0.0, 1.0)
        return df


# ---------------------------------------------------------------------------
# MACD Signal Line Cross
# ---------------------------------------------------------------------------

@strategy_registry.register("macd")
class MACDStrategy(SignalGenerator):
    """MACD / signal line crossover strategy."""

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def required_indicators(self) -> list[dict]:
        return [{"name": "macd", "fast": self.fast, "slow": self.slow, "signal": self.signal}]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        macd_col = f"MACD_{self.fast}_{self.slow}_{self.signal}"
        signal_col = f"MACDs_{self.fast}_{self.slow}_{self.signal}"
        self.validate_columns(df, [macd_col, signal_col])

        df["signal"] = crossover(df[macd_col], df[signal_col])

        hist_col = f"MACDh_{self.fast}_{self.slow}_{self.signal}"
        if hist_col in df.columns:
            hist_abs = df[hist_col].abs()
            max_hist = hist_abs.max()
            df["signal_strength"] = (hist_abs / max_hist) if max_hist > 0 else 0.0
        return df


# ---------------------------------------------------------------------------
# Bollinger Band Breakout / Mean Reversion
# ---------------------------------------------------------------------------

@strategy_registry.register("bbands")
class BollingerBandStrategy(SignalGenerator):
    """Bollinger Band mean-reversion strategy."""

    name = "bbands"

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        self.period = period
        self.std_dev = std_dev

    @property
    def required_indicators(self) -> list[dict]:
        return [{"name": "bbands", "length": self.period, "std": self.std_dev}]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        upper_col = f"BBU_{self.period}_{self.std_dev}"
        lower_col = f"BBL_{self.period}_{self.std_dev}"
        mid_col = f"BBM_{self.period}_{self.std_dev}"
        self.validate_columns(df, [upper_col, lower_col])

        df["signal"] = breakout(df["close"], df[upper_col], df[lower_col])

        if mid_col in df.columns:
            band_width = df[upper_col] - df[lower_col]
            dist_from_mid = (df["close"] - df[mid_col]).abs()
            df["signal_strength"] = np.where(
                band_width > 0, dist_from_mid / band_width, 0.0
            )
        return df


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------

@strategy_registry.register("supertrend")
class SupertrendStrategy(SignalGenerator):
    """Supertrend direction-change strategy."""

    name = "supertrend"

    def __init__(self, period: int = 10, multiplier: float = 3.0) -> None:
        self.period = period
        self.multiplier = multiplier

    @property
    def required_indicators(self) -> list[dict]:
        return [{"name": "supertrend", "length": self.period, "multiplier": self.multiplier}]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        direction_col = f"SUPERTd_{self.period}_{self.multiplier}"
        trend_col = f"SUPERT_{self.period}_{self.multiplier}"
        self.validate_columns(df, [direction_col])

        # Supertrend direction: 1 = uptrend, -1 = downtrend
        direction = df[direction_col]
        prev_direction = direction.shift(1)

        signal = pd.Series(0, index=df.index, dtype=np.int8)
        signal[(direction == 1) & (prev_direction == -1)] = 1   # switch to uptrend
        signal[(direction == -1) & (prev_direction == 1)] = -1  # switch to downtrend
        df["signal"] = signal

        if trend_col in df.columns:
            dist = (df["close"] - df[trend_col]).abs() / df["close"]
            max_dist = dist.max()
            df["signal_strength"] = (dist / max_dist) if max_dist > 0 else 0.0
        return df


# ---------------------------------------------------------------------------
# TTM Squeeze
# ---------------------------------------------------------------------------

@strategy_registry.register("ttm_squeeze")
class TTMSqueezeStrategy(SignalGenerator):
    """TTM Squeeze momentum strategy.

    Fires long when squeeze releases with positive momentum, short when
    squeeze releases with negative momentum.
    """

    name = "ttm_squeeze"

    def __init__(
        self,
        bb_period: int = 20,
        kc_period: int = 20,
        kc_mult: float = 1.5,
    ) -> None:
        self.bb_period = bb_period
        self.kc_period = kc_period
        self.kc_mult = kc_mult

    @property
    def required_indicators(self) -> list[dict]:
        return [
            {"name": "squeeze", "bb_length": self.bb_period, "kc_length": self.kc_period, "kc_scalar": self.kc_mult},
        ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # pandas_ta squeeze columns
        squeeze_on = f"SQZ_ON"
        squeeze_off = f"SQZ_OFF"
        squeeze_no = f"SQZ_NO"

        # Try to find momentum column - pandas_ta uses SQZ_20_2.0_20_1.5 pattern
        mom_col = None
        for col in df.columns:
            if col.startswith("SQZ_") and not col.startswith(("SQZ_ON", "SQZ_OFF", "SQZ_NO")):
                mom_col = col
                break

        # Fallback: if we can't find specific squeeze columns, look for generic ones
        has_squeeze_cols = squeeze_on in df.columns and squeeze_off in df.columns

        if mom_col is not None and has_squeeze_cols:
            momentum = df[mom_col]
            on = df[squeeze_on].astype(bool)
            off = df[squeeze_off].astype(bool)
            prev_on = on.shift(1).fillna(False)

            # Squeeze just released
            released = off & prev_on
            signal = pd.Series(0, index=df.index, dtype=np.int8)
            signal[released & (momentum > 0)] = 1
            signal[released & (momentum < 0)] = -1
            df["signal"] = signal

            mom_abs = momentum.abs()
            max_mom = mom_abs.max()
            df["signal_strength"] = (mom_abs / max_mom) if max_mom > 0 else 0.0
        else:
            # Graceful fallback: no signal
            df["signal"] = 0
            df["signal_strength"] = 0.0

        return df
