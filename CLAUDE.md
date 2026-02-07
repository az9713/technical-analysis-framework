# CLAUDE.md -- Project Guide for Claude Code

## Project Overview

TA Framework is a Python-based technical analysis toolkit that combines real-time market data ingestion, 38+ technical indicators, rule-based signal generation, vectorized backtesting, risk management, and interactive Streamlit dashboards into a single importable package (`ta_framework`, v0.1.0). It is built on pandas DataFrames as the universal data container, uses pandas-ta-classic for cross-platform indicator computation (no TA-Lib/C dependency), and targets Python 3.10+ with pandas >= 2.0, < 3.0.

## Quick Commands

```bash
# Run the Streamlit app (opens in browser)
streamlit run app.py

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_indicators.py -v

# Run a single test
pytest tests/test_indicators.py::TestEngineCompute::test_sma -v

# Install runtime dependencies
pip install -r requirements.txt

# Install in dev/editable mode with test deps
pip install -e ".[dev]"

# Install all optional extras (ccxt, hmmlearn, numba)
pip install -e ".[all]"
```

## Project Structure

```
pandas_TA_neuralNine/
|-- app.py                                  # Streamlit entry point (page router)
|-- pyproject.toml                          # Package metadata, deps, pytest config
|-- requirements.txt                        # Runtime dependencies (pip freeze format)
|-- CLAUDE.md                               # This file
|
|-- ta_framework/
|   |-- __init__.py                         # Package root, version string
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   |-- types.py                        # Enums (AssetClass, Timeframe, SignalDirection, IndicatorTier, IndicatorCategory) + dataclasses (Signal, Trade, BacktestConfig, IndicatorConfig)
|   |   |-- registry.py                     # Generic decorator-based plugin Registry class + global registries (indicator_registry, strategy_registry, provider_registry)
|   |   |-- config.py                       # FrameworkConfig singleton (default_timeframe, cache settings, OHLCV column list)
|   |   |-- exceptions.py                   # Exception hierarchy: TAFrameworkError -> DataError, IndicatorError, SignalError, BacktestError, ConfigError, RegistryError
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- base.py                         # ABC DataProvider (fetch, supported_assets, search_symbols, validate)
|   |   |-- yfinance_provider.py            # YFinanceProvider -- Yahoo Finance via yfinance lib
|   |   |-- csv_provider.py                 # CSVProvider -- loads local CSV/Parquet files
|   |   |-- ccxt_provider.py                # CCXTProvider -- cryptocurrency data via ccxt (optional dep)
|   |   |-- quality.py                      # DataQualityChecker (gap detection, outlier detection, OHLC consistency, cleaning)
|   |   |-- timeframe.py                    # resample() and align_timeframes() utilities for multi-TF analysis
|   |
|   |-- indicators/
|   |   |-- __init__.py
|   |   |-- engine.py                       # IndicatorEngine (compute, compute_batch, register, available)
|   |   |-- catalog.py                      # INDICATOR_CATALOG dict: metadata for all 38 Tier 1 indicators
|   |   |-- wrappers.py                     # Typed wrapper functions around pandas_ta + WRAPPER_MAP dict
|   |   |-- custom.py                       # ABC CustomIndicator + @register_indicator decorator
|   |   |-- composite.py                    # CompositeIndicator: chain multiple indicators sequentially
|   |
|   |-- signals/
|   |   |-- __init__.py
|   |   |-- base.py                         # ABC SignalGenerator (generate, required_indicators, validate_columns)
|   |   |-- rules.py                        # Reusable rule functions: crossover, threshold, divergence, breakout
|   |   |-- strategies.py                   # 6 built-in strategies: EMACross, RSI, MACD, BollingerBand, Supertrend, TTMSqueeze
|   |   |-- composite.py                    # CompositeSignal: combine multiple generators (voting, weighted, confirmation)
|   |
|   |-- backtest/
|   |   |-- __init__.py
|   |   |-- engine.py                       # VectorizedBacktester.run() -- signal-driven vectorized backtest
|   |   |-- results.py                      # BacktestResult dataclass (equity_curve, trades, summary, to_dataframe)
|   |   |-- costs.py                        # Cost models: FixedCommission, PercentageCommission, TieredCommission, SlippageModel
|   |   |-- optimization.py                 # GridSearchOptimizer + WalkForwardOptimizer for parameter tuning
|   |   |-- monte_carlo.py                  # MonteCarloSimulator (bootstrap_returns, shuffle_trades, confidence_intervals)
|   |
|   |-- risk/
|   |   |-- __init__.py
|   |   |-- position_sizing.py              # FixedFractional, KellyCriterion, VolatilityBased, RiskParity sizers
|   |   |-- var.py                          # VaR calculations: parametric, historical, Monte Carlo, CVaR
|   |   |-- stops.py                        # Stop-loss functions: fixed_stop, atr_stop, trailing_stop, chandelier_stop
|   |   |-- portfolio.py                    # Portfolio risk: portfolio_volatility, calmar_ratio, ulcer_index, risk_contribution
|   |
|   |-- analytics/
|   |   |-- __init__.py
|   |   |-- metrics.py                      # 25+ performance metrics (Sharpe, Sortino, Calmar, Omega, VaR, win_rate, expectancy, etc.)
|   |   |-- tearsheet.py                    # TearSheet class: full performance report (returns, risk, trades, distribution, benchmark)
|   |   |-- benchmark.py                    # Benchmark comparison: alpha_beta, information_ratio, tracking_error, up/down capture
|   |   |-- regime.py                       # RegimeDetector: KMeans and HMM-based market regime detection
|   |
|   |-- viz/
|   |   |-- __init__.py
|   |   |-- charts.py                       # candlestick_chart, indicator_panel, multi_panel_chart (Plotly)
|   |   |-- drawdown.py                     # drawdown_chart (underwater), drawdown_periods (highlight top N)
|   |   |-- heatmaps.py                     # correlation_heatmap, monthly_returns_heatmap
|   |   |-- distribution.py                 # returns_histogram (with normal overlay + VaR lines), qq_plot
|   |   |-- trade_plots.py                  # trade_markers (buy/sell on candlestick), pnl_chart (cumulative + per-trade)
|   |
|   |-- pages/
|       |-- __init__.py
|       |-- components.py                   # Shared Streamlit widgets: symbol_input, timeframe_select, date_range_input, indicator_multiselect, strategy_select, strategy_params, backtest_config_panel, fetch_data (cached)
|       |-- dashboard.py                    # Dashboard page: data loading, indicator selection, multi-panel chart
|       |-- backtester.py                   # Backtester page: strategy config, run backtest, display results
|       |-- compare.py                      # Compare page: multi-asset and multi-strategy comparison
|       |-- analysis.py                     # Analysis page: tearsheet, VaR, regime detection, return distributions
|
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                         # Shared fixtures: sample_ohlcv (252 bars), short_ohlcv (20 bars), empty_ohlcv, default_backtest_config, sample_signals
|   |-- test_indicators.py                  # Indicator engine and wrapper tests
|   |-- test_signals.py                     # Signal generation and strategy tests
|   |-- test_backtest.py                    # Backtesting engine tests
|   |-- test_analytics.py                   # Metrics and analytics tests
|   |-- test_risk.py                        # Risk management tests
|   |-- test_data_providers.py              # Data provider tests
|
|-- docs/
|   |-- ARCHITECTURE.md                     # Comprehensive architecture reference
```

## Architecture Principles

- **DataFrame-centric**: pandas DataFrame is the universal data container flowing through every pipeline stage. All functions accept and return DataFrames.
- **Signal convention**: `1` = long, `-1` = short, `0` = neutral. This integer convention is used everywhere -- signal generators, backtester, composite signals.
- **Indicator column naming**: follows pandas_ta convention -- `SMA_20`, `RSI_14`, `MACD_12_26_9`, `BBU_20_2.0`, `EMA_12`.
- **ABC pattern for extensibility**: `DataProvider`, `SignalGenerator`, `CustomIndicator`, `PositionSizer`, `CostModel` are all abstract base classes. Extend by subclassing.
- **Decorator-based plugin registry**: `@strategy_registry.register("name")`, `@provider_registry.register("name")`, `@register_indicator()` auto-register implementations.
- **Vectorized backtesting**: the `VectorizedBacktester` processes signal arrays without row-by-row Python loops for the main equity calculation. No event-driven simulation.
- **Dependency pinning**: `pandas >= 2.0, < 3.0` because pandas-ta-classic uses deprecated pandas APIs. Do not upgrade to pandas 3.x.

## Key Conventions

- **Indicator library**: Use `pandas-ta-classic` (not TA-Lib). This avoids requiring C compilation and keeps the project cross-platform.
- **OHLCV DataFrame schema**: `DatetimeIndex` named `"date"`, columns: `open`, `high`, `low`, `close`, `volume` (all lowercase, float64). The `DataProvider.validate()` method enforces this.
- **Adding a new indicator**: (1) write a wrapper function in `ta_framework/indicators/wrappers.py` that takes `(df, **params) -> DataFrame`, (2) add its metadata to `INDICATOR_CATALOG` in `catalog.py`, (3) add it to the `WRAPPER_MAP` dict at the bottom of `wrappers.py`. The `IndicatorEngine` auto-discovers it.
- **Adding a new strategy**: subclass `SignalGenerator`, implement `generate()` and `required_indicators`, decorate with `@strategy_registry.register("name")`. Place in `signals/strategies.py` or a new file.
- **Adding a new data provider**: subclass `DataProvider`, implement `fetch()`, `supported_assets()`, `search_symbols()`, decorate with `@provider_registry.register("name")`.
- **Test location**: all tests in `tests/` with `test_` prefix. Use fixtures from `conftest.py` (`sample_ohlcv`, `short_ohlcv`, `sample_signals`, `default_backtest_config`).
- **All 175 tests must pass before committing**: `pytest tests/ -v`.

## Dependencies and Why

| Package | Version Constraint | Purpose |
|---|---|---|
| `pandas` | >= 2.0, < 3.0 | Core data manipulation library. DataFrames are the universal data container. Pinned below 3.0 for pandas-ta compatibility. |
| `numpy` | >= 1.24 | Numerical array operations underlying pandas. Used directly for vectorized math in backtesting, VaR, Monte Carlo. |
| `pandas-ta-classic` | >= 0.1.0 | 200+ pre-built technical indicator functions (SMA, RSI, MACD, Bollinger, etc.). Drop-in replacement for TA-Lib with no C dependencies. |
| `yfinance` | >= 0.2.0 | Downloads historical OHLCV data from Yahoo Finance. Powers the default `YFinanceProvider`. |
| `plotly` | >= 5.0 | Interactive charting library (candlestick, scatter, heatmaps). All visualizations use Plotly Figure objects. |
| `streamlit` | >= 1.30 | Web framework for the dashboard UI. Turns Python scripts into browser apps with zero HTML/CSS/JS. |
| `scipy` | >= 1.10 | Statistical functions: normal distribution for VaR, Q-Q plots, distribution fitting. |
| `scikit-learn` | >= 1.3 | Machine learning utilities: `KMeans` for regime detection, `StandardScaler` for feature normalization. |

**Optional dependencies** (in pyproject.toml extras):
- `ccxt` >= 4.0 -- cryptocurrency exchange data (`pip install -e ".[crypto]"`)
- `hmmlearn` >= 0.3 -- Hidden Markov Model regime detection (`pip install -e ".[regime]"`)
- `numba` >= 0.57 -- JIT acceleration for numerical loops (`pip install -e ".[accel]"`)
- `pytest` >= 7.0, `pytest-cov` >= 4.0 -- testing (`pip install -e ".[dev]"`)

## Common Patterns

### Computing Indicators

```python
from ta_framework.indicators.engine import IndicatorEngine

engine = IndicatorEngine()

# Single indicator
df = engine.compute(df, "sma", length=20)
df = engine.compute(df, "rsi", length=14)

# Batch computation
df = engine.compute_batch(df, [
    {"name": "ema", "params": {"length": 12}},
    {"name": "ema", "params": {"length": 26}},
    {"name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}},
])
```

### Generating Signals

```python
from ta_framework.core.registry import strategy_registry

# Use a built-in strategy
strategy_cls = strategy_registry.get("ema_cross")
strategy = strategy_cls(fast_period=12, slow_period=26)

# Compute required indicators first
engine = IndicatorEngine()
for spec in strategy.required_indicators:
    name = spec.pop("name")
    df = engine.compute(df, name, **spec)

# Generate signals (adds 'signal' column: 1/-1/0)
df = strategy.generate(df)
```

### Running a Backtest

```python
from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.core.types import BacktestConfig

config = BacktestConfig(
    initial_capital=100_000,
    commission_pct=0.001,
    slippage_pct=0.0005,
    allow_short=True,
)
backtester = VectorizedBacktester(config)
result = backtester.run(df)  # df must have 'signal' and 'close' columns

print(result.summary())
# {'total_return': 0.12, 'max_drawdown': 0.08, 'sharpe_ratio': 1.5, ...}
```

### Creating a Custom Indicator

```python
from ta_framework.indicators.custom import CustomIndicator, register_indicator
from ta_framework.core.types import IndicatorCategory, IndicatorTier

@register_indicator(tier=IndicatorTier.TIER1)
class SpreadIndicator(CustomIndicator):
    name = "spread"
    category = IndicatorCategory.CUSTOM

    def compute(self, df):
        out = df.copy()
        out["SPREAD"] = df["high"] - df["low"]
        return out

    @property
    def output_columns(self):
        return ["SPREAD"]
```
