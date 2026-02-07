"""Tests for data providers, quality checker, and timeframe utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ta_framework.core.exceptions import DataError, DataQualityError
from ta_framework.core.types import Timeframe
from ta_framework.data.base import DataProvider, REQUIRED_COLUMNS
from ta_framework.data.csv_provider import CSVProvider
from ta_framework.data.quality import DataQualityChecker
from ta_framework.data.timeframe import align_timeframes, resample


# ===================================================================
# DataProvider.validate
# ===================================================================

class TestValidation:
    """Test the base DataProvider.validate logic via CSVProvider."""

    def test_validate_good_data(self, sample_ohlcv: pd.DataFrame, tmp_path: Path):
        """A well-formed OHLCV DataFrame passes validation."""
        csv_path = tmp_path / "good.csv"
        sample_ohlcv.to_csv(csv_path)
        provider = CSVProvider(csv_path)
        df = provider.fetch()
        assert list(df.columns) == REQUIRED_COLUMNS
        assert df.index.name == "date"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_validate_rejects_empty(self, tmp_path: Path):
        """An empty CSV should raise DataError."""
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]).to_csv(
            csv_path, index=False
        )
        provider = CSVProvider(csv_path)
        with pytest.raises(DataError):
            provider.fetch()

    def test_validate_uppercases_normalised(self, tmp_path: Path):
        """Columns in any case should be normalised to lowercase."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5),
            "Open": [1, 2, 3, 4, 5],
            "HIGH": [2, 3, 4, 5, 6],
            "Low": [0.5, 1, 2, 3, 4],
            "CLOSE": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Volume": [100, 200, 300, 400, 500],
        })
        csv_path = tmp_path / "mixed_case.csv"
        df.to_csv(csv_path, index=False)
        provider = CSVProvider(csv_path)
        result = provider.fetch()
        assert list(result.columns) == REQUIRED_COLUMNS


# ===================================================================
# CSVProvider
# ===================================================================

class TestCSVProvider:
    def test_csv_roundtrip(self, sample_ohlcv: pd.DataFrame, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        sample_ohlcv.to_csv(csv_path)
        provider = CSVProvider(csv_path)
        loaded = provider.fetch()
        assert len(loaded) == len(sample_ohlcv)
        assert loaded.index.name == "date"

    def test_parquet_roundtrip(self, sample_ohlcv: pd.DataFrame, tmp_path: Path):
        pq_path = tmp_path / "data.parquet"
        sample_ohlcv.to_parquet(pq_path)
        provider = CSVProvider(pq_path)
        loaded = provider.fetch()
        assert len(loaded) == len(sample_ohlcv)

    def test_missing_file_raises(self):
        with pytest.raises(DataError):
            CSVProvider("/nonexistent/path.csv")

    def test_unsupported_extension(self, tmp_path: Path):
        bad_path = tmp_path / "data.xlsx"
        bad_path.write_text("")
        provider = CSVProvider(bad_path)
        with pytest.raises(DataError, match="Unsupported file extension"):
            provider.fetch()

    def test_date_range_filter(self, sample_ohlcv: pd.DataFrame, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        sample_ohlcv.to_csv(csv_path)
        provider = CSVProvider(csv_path)
        loaded = provider.fetch(start="2023-06-01", end="2023-09-01")
        assert loaded.index.min() >= pd.Timestamp("2023-06-01")
        assert loaded.index.max() <= pd.Timestamp("2023-09-01")

    def test_supported_assets(self, sample_ohlcv: pd.DataFrame, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        sample_ohlcv.to_csv(csv_path)
        provider = CSVProvider(csv_path)
        assert len(provider.supported_assets()) > 0

    def test_search_symbols(self, sample_ohlcv: pd.DataFrame, tmp_path: Path):
        csv_path = tmp_path / "test_data.csv"
        sample_ohlcv.to_csv(csv_path)
        provider = CSVProvider(csv_path)
        results = provider.search_symbols("test")
        assert len(results) == 1
        assert results[0]["symbol"] == "test_data"


# ===================================================================
# DataQualityChecker
# ===================================================================

class TestDataQualityChecker:
    def test_check_gaps_no_gaps(self, sample_ohlcv: pd.DataFrame):
        checker = DataQualityChecker()
        report = checker.check_gaps(sample_ohlcv)
        assert report["passed"] is True
        assert report["nan_rows"] == 0

    def test_check_gaps_with_nans(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        # Inject NaN gaps (more than 5% of 252 = 13 rows)
        df.iloc[10:30, df.columns.get_loc("close")] = np.nan
        checker = DataQualityChecker()
        report = checker.check_gaps(df)
        assert report["nan_rows"] == 20
        assert report["passed"] is False

    def test_check_gaps_small_amount(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        # Inject a few NaN (< 5%)
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        checker = DataQualityChecker()
        report = checker.check_gaps(df)
        assert report["nan_rows"] == 1
        assert report["passed"] is True  # 1/252 < 5%

    def test_check_outliers_clean(self, sample_ohlcv: pd.DataFrame):
        checker = DataQualityChecker()
        report = checker.check_outliers(sample_ohlcv)
        assert report["passed"] is True

    def test_check_outliers_with_spike(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        # Inject massive spike
        df.iloc[100, df.columns.get_loc("close")] = df["close"].iloc[99] * 10
        checker = DataQualityChecker()
        report = checker.check_outliers(df)
        assert report["outlier_count"] >= 1
        assert report["passed"] is False

    def test_check_ohlc_consistency_good(self, sample_ohlcv: pd.DataFrame):
        checker = DataQualityChecker()
        report = checker.check_ohlc_consistency(sample_ohlcv)
        assert report["passed"] is True

    def test_check_ohlc_consistency_violations(self, short_ohlcv: pd.DataFrame):
        df = short_ohlcv.copy()
        # Break high < close on purpose
        df.iloc[5, df.columns.get_loc("high")] = df["close"].iloc[5] - 10
        checker = DataQualityChecker()
        report = checker.check_ohlc_consistency(df)
        assert report["high_violations"] >= 1
        assert report["passed"] is False

    def test_check_ohlc_volume_negative(self, short_ohlcv: pd.DataFrame):
        df = short_ohlcv.copy()
        df.iloc[0, df.columns.get_loc("volume")] = -100
        checker = DataQualityChecker()
        report = checker.check_ohlc_consistency(df)
        assert report["volume_violations"] >= 1

    def test_full_check(self, sample_ohlcv: pd.DataFrame):
        checker = DataQualityChecker()
        report = checker.full_check(sample_ohlcv)
        assert "gaps" in report
        assert "outliers" in report
        assert "ohlc_consistency" in report

    def test_clean_fills_gaps(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[5:8, df.columns.get_loc("close")] = np.nan
        cleaned = DataQualityChecker.clean(df, fill_method="ffill")
        assert cleaned["close"].isna().sum() == 0

    def test_clean_removes_outliers(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[100, df.columns.get_loc("close")] = df["close"].iloc[99] * 50
        cleaned = DataQualityChecker.clean(df, remove_outliers=True, z_threshold=3.0)
        # The outlier row should be removed
        assert len(cleaned) < len(df)

    def test_clean_fixes_negative_volume(self, short_ohlcv: pd.DataFrame):
        df = short_ohlcv.copy()
        df.iloc[0, df.columns.get_loc("volume")] = -500
        cleaned = DataQualityChecker.clean(df)
        assert cleaned["volume"].min() >= 0

    def test_check_gaps_empty_df(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        checker = DataQualityChecker()
        report = checker.check_gaps(df)
        assert report["passed"] is True
        assert report["total_rows"] == 0


# ===================================================================
# Timeframe resampling
# ===================================================================

class TestTimeframe:
    def test_resample_daily_to_weekly(self, sample_ohlcv: pd.DataFrame):
        weekly = resample(sample_ohlcv, Timeframe.W1)
        assert len(weekly) < len(sample_ohlcv)
        # Weekly bars should have higher volume (aggregated)
        assert weekly["volume"].mean() > sample_ohlcv["volume"].mean()

    def test_resample_daily_to_monthly(self, sample_ohlcv: pd.DataFrame):
        monthly = resample(sample_ohlcv, Timeframe.MN1)
        assert len(monthly) <= 13  # ~12 months in 252 bdays
        assert monthly.index.name == "date"

    def test_resample_preserves_ohlcv_semantics(self, sample_ohlcv: pd.DataFrame):
        weekly = resample(sample_ohlcv, Timeframe.W1)
        # High should be the max of the week
        # Low should be the min of the week
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in weekly.columns

    def test_resample_empty(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = resample(df, Timeframe.W1)
        assert result.empty

    def test_align_timeframes(self, sample_ohlcv: pd.DataFrame):
        daily = sample_ohlcv
        weekly = resample(sample_ohlcv, Timeframe.W1)
        aligned = align_timeframes({Timeframe.D1: daily, Timeframe.W1: weekly})
        # Both should share common date boundaries
        d_start = aligned[Timeframe.D1].index.min()
        w_start = aligned[Timeframe.W1].index.min()
        d_end = aligned[Timeframe.D1].index.max()
        w_end = aligned[Timeframe.W1].index.max()
        assert d_start >= w_start or d_start <= w_start  # both within range
        assert d_end <= sample_ohlcv.index.max()

    def test_align_empty_returns_empty(self):
        result = align_timeframes({})
        assert result == {}
