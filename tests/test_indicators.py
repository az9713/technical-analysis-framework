"""Tests for the indicator engine, wrappers, composite, and custom indicators."""

from __future__ import annotations

import pandas as pd
import pytest

from ta_framework.core.exceptions import (
    IndicatorError,
    InsufficientDataError,
    InvalidParameterError,
)
from ta_framework.core.types import IndicatorCategory, IndicatorTier
from ta_framework.indicators.composite import CompositeIndicator
from ta_framework.indicators.custom import CustomIndicator, register_indicator
from ta_framework.indicators.engine import IndicatorEngine


# ===================================================================
# IndicatorEngine.compute -- individual indicators
# ===================================================================

class TestEngineCompute:
    def test_sma(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "sma", length=20)
        assert "SMA_20" in result.columns
        # First 19 values should be NaN, rest computed
        assert result["SMA_20"].isna().sum() == 19

    def test_sma_custom_length(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "sma", length=50)
        assert "SMA_50" in result.columns

    def test_ema(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "ema", length=12)
        assert "EMA_12" in result.columns

    def test_rsi(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "rsi", length=14)
        assert "RSI_14" in result.columns
        # RSI should be bounded 0-100 (where not NaN)
        valid = result["RSI_14"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_macd(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "macd")
        macd_cols = [c for c in result.columns if "MACD" in c.upper()]
        assert len(macd_cols) >= 3  # MACD, MACDh, MACDs

    def test_bbands(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "bbands", length=20, std=2.0)
        bb_cols = [c for c in result.columns if "BB" in c.upper()]
        assert len(bb_cols) >= 3  # upper, mid, lower (and possibly bandwidth/percent)

    def test_atr(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "atr", length=14)
        assert "ATR_14" in result.columns
        valid = result["ATR_14"].dropna()
        assert (valid >= 0).all()

    def test_adx(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "adx", length=14)
        adx_cols = [c for c in result.columns if "ADX" in c.upper() or "DM" in c.upper()]
        assert len(adx_cols) >= 1

    def test_stoch(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "stoch")
        stoch_cols = [c for c in result.columns if "STOCH" in c.upper()]
        assert len(stoch_cols) >= 2

    def test_obv(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv, "obv")
        assert "OBV" in result.columns

    def test_unknown_indicator_raises(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        with pytest.raises(IndicatorError, match="Unknown indicator"):
            engine.compute(sample_ohlcv, "nonexistent_indicator_xyz")

    def test_empty_df_raises(self):
        engine = IndicatorEngine()
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(InsufficientDataError):
            engine.compute(empty, "sma")


# ===================================================================
# IndicatorEngine.compute_batch
# ===================================================================

class TestEngineBatch:
    def test_batch_multiple(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        indicators = [
            {"name": "sma", "params": {"length": 20}},
            {"name": "rsi", "params": {"length": 14}},
            {"name": "atr", "params": {"length": 14}},
        ]
        result = engine.compute_batch(sample_ohlcv, indicators)
        assert "SMA_20" in result.columns
        assert "RSI_14" in result.columns
        assert "ATR_14" in result.columns

    def test_batch_preserves_ohlcv(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        indicators = [
            {"name": "sma", "params": {"length": 10}},
            {"name": "ema", "params": {"length": 10}},
        ]
        result = engine.compute_batch(sample_ohlcv, indicators)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_batch_empty_list(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        result = engine.compute_batch(sample_ohlcv, [])
        assert result.equals(sample_ohlcv)


# ===================================================================
# IndicatorEngine.available
# ===================================================================

class TestEngineAvailable:
    def test_available_not_empty(self):
        engine = IndicatorEngine()
        names = engine.available()
        assert len(names) > 30  # We registered ~38 Tier 1 indicators

    def test_available_includes_core(self):
        engine = IndicatorEngine()
        names = engine.available()
        for expected in ("sma", "ema", "rsi", "macd", "bbands", "atr", "obv"):
            assert expected in names


# ===================================================================
# CompositeIndicator
# ===================================================================

class TestCompositeIndicator:
    def test_rsi_of_sma(self, sample_ohlcv: pd.DataFrame):
        comp = CompositeIndicator(
            steps=[
                ("sma", {"length": 20}),
                ("rsi", {"length": 14, "column": "SMA_20"}),
            ],
            name="RSI_of_SMA",
        )
        result = comp.compute(sample_ohlcv)
        assert "SMA_20" in result.columns
        assert "RSI_14" in result.columns

    def test_multiple_chain(self, sample_ohlcv: pd.DataFrame):
        comp = CompositeIndicator(
            steps=[
                ("sma", {"length": 10}),
                ("ema", {"length": 5, "column": "SMA_10"}),
            ],
        )
        result = comp.compute(sample_ohlcv)
        assert "SMA_10" in result.columns
        assert "EMA_5" in result.columns

    def test_unknown_step_raises(self, sample_ohlcv: pd.DataFrame):
        comp = CompositeIndicator(steps=[("fake_indicator", {})])
        with pytest.raises(IndicatorError, match="Unknown indicator"):
            comp.compute(sample_ohlcv)

    def test_empty_steps_raises(self):
        with pytest.raises(IndicatorError):
            CompositeIndicator(steps=[])

    def test_repr(self):
        comp = CompositeIndicator(
            steps=[("sma", {"length": 20}), ("rsi", {"length": 14})],
        )
        r = repr(comp)
        assert "sma" in r
        assert "rsi" in r


# ===================================================================
# CustomIndicator
# ===================================================================

class TestCustomIndicator:
    def test_register_and_compute(self, sample_ohlcv: pd.DataFrame):
        @register_indicator(tier=IndicatorTier.TIER1)
        class SpreadIndicator(CustomIndicator):
            name = "test_spread"
            category = IndicatorCategory.CUSTOM

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out["SPREAD"] = df["high"] - df["low"]
                return out

            @property
            def output_columns(self) -> list[str]:
                return ["SPREAD"]

        # It should now be discoverable via the engine
        engine = IndicatorEngine()
        assert "test_spread" in engine.available()

        result = engine.compute(sample_ohlcv, "test_spread")
        assert "SPREAD" in result.columns
        assert (result["SPREAD"] >= 0).all()

    def test_custom_output_columns(self):
        class DummyIndicator(CustomIndicator):
            name = "dummy"
            category = IndicatorCategory.CUSTOM

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

            @property
            def output_columns(self) -> list[str]:
                return ["DUMMY_COL"]

        ind = DummyIndicator()
        assert ind.output_columns == ["DUMMY_COL"]

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError):
            @register_indicator()
            class NotAnIndicator:
                pass


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_short_data_sma(self, short_ohlcv: pd.DataFrame):
        """SMA with length > data length should still return (all NaN)."""
        engine = IndicatorEngine()
        result = engine.compute(short_ohlcv, "sma", length=50)
        assert "SMA_50" in result.columns
        # All NaN because 20 bars < 50 length
        assert result["SMA_50"].isna().all()

    def test_invalid_parameter_type(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()
        with pytest.raises(InvalidParameterError):
            engine.compute(sample_ohlcv, "sma", nonexistent_param="bad")

    def test_engine_register_custom_function(self, sample_ohlcv: pd.DataFrame):
        engine = IndicatorEngine()

        def my_range(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["MY_RANGE"] = df["high"] - df["low"]
            return out

        engine.register("my_range", my_range)
        assert "my_range" in engine.available()

        result = engine.compute(sample_ohlcv, "my_range")
        assert "MY_RANGE" in result.columns
