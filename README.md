# TA Framework

A Python technical analysis toolkit combining 38 indicators, rule-based signal generation, vectorized backtesting, risk management, and interactive Streamlit dashboards into a single importable package.

Built on pandas DataFrames as the universal data container. Uses [pandas-ta-classic](https://github.com/pandas-ta/pandas-ta) for cross-platform indicator computation -- no TA-Lib or C compilation required.

## Features

**Indicators** -- 38 production-quality indicators across trend (SMA, EMA, VWMA, HMA, KAMA, Supertrend, Ichimoku, ADX, Parabolic SAR), momentum (RSI, MACD, Stochastic, CCI, MFI, TSI), volatility (Bollinger Bands, ATR, Keltner Channel, Donchian), volume (OBV, VWAP, CMF, A/D Line), and overlap (Pivot Points, Fibonacci). Custom indicator API for proprietary calculations.

**Strategies** -- 6 pre-built strategies (EMA Crossover, RSI Threshold, MACD Signal Cross, Bollinger Band Mean-Reversion, Supertrend, TTM Squeeze) plus a composite signal combiner supporting voting, weighted, and confirmation modes. Custom strategy API via subclassing.

**Backtesting** -- Vectorized engine with commission/slippage modeling, long and short positions. Grid search parameter optimization. Walk-forward validation. Monte Carlo robustness testing (bootstrap returns, trade shuffling, confidence intervals).

**Risk Management** -- 4 position sizers (fixed fractional, Kelly criterion, volatility-based, risk parity). VaR suite (parametric, historical, Monte Carlo, CVaR). 4 stop-loss types (fixed, ATR, trailing, chandelier). Portfolio-level metrics (volatility, Calmar, Ulcer Index, risk contribution).

**Analytics** -- 25+ metrics (Sharpe, Sortino, Calmar, Omega, CAGR, max drawdown, win rate, profit factor, expectancy, skewness, kurtosis, tail ratio, VaR/CVaR). Tear sheet generation. Benchmark comparison (alpha, beta, information ratio, tracking error, up/down capture). Market regime detection (KMeans, optional HMM).

**Data** -- Yahoo Finance, local CSV/Parquet, cryptocurrency exchanges (via ccxt). Data quality checking (gap detection, outlier detection, OHLC consistency) with auto-cleaning. Multi-timeframe resampling.

**Visualization** -- Interactive Plotly charts: candlestick with indicator overlays, multi-panel layouts, drawdown charts, monthly returns heatmap, correlation heatmap, return distributions with VaR lines, Q-Q plots, trade markers (buy/sell on chart).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run app.py
```

In the browser: enter a symbol (e.g., AAPL), select a timeframe and date range, add indicators, then switch to the Backtester page to run a strategy backtest.

### Python API

```python
from ta_framework.data.yfinance_provider import YFinanceProvider
from ta_framework.indicators.engine import IndicatorEngine
from ta_framework.core.registry import strategy_registry
from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.core.types import BacktestConfig

# Fetch data
provider = YFinanceProvider()
df = provider.fetch("AAPL", timeframe="1d", start="2023-01-01", end="2024-01-01")

# Compute indicators
engine = IndicatorEngine()
df = engine.compute(df, "ema", length=12)
df = engine.compute(df, "ema", length=26)

# Generate signals
strategy_cls = strategy_registry.get("ema_cross")
strategy = strategy_cls(fast_period=12, slow_period=26)
df = strategy.generate(df)

# Backtest
config = BacktestConfig(initial_capital=100_000, commission_pct=0.001, allow_short=True)
backtester = VectorizedBacktester(config)
result = backtester.run(df)

print(result.summary())
# {'total_return': 0.12, 'sharpe_ratio': 1.5, 'max_drawdown': 0.08, ...}
```

## Installation

**Requirements**: Python 3.10+

```bash
# Runtime dependencies only
pip install -r requirements.txt

# Editable install with dev dependencies
pip install -e ".[dev]"

# All optional extras (ccxt, hmmlearn, numba)
pip install -e ".[all]"
```

### Optional Dependencies

| Extra | Package | Purpose |
|-------|---------|---------|
| `crypto` | ccxt >= 4.0 | Cryptocurrency exchange data |
| `regime` | hmmlearn >= 0.3 | HMM-based market regime detection |
| `accel` | numba >= 0.57 | JIT acceleration for numerical loops |
| `dev` | pytest, pytest-cov | Testing |

## Project Structure

```
ta_framework/
  core/           # Types, enums, registry, config, exceptions
  data/           # Data providers (Yahoo Finance, CSV, ccxt), quality checks
  indicators/     # Engine, catalog (38 indicators), wrappers, custom indicator API
  signals/        # Signal generators, rule functions, 6 strategies, composite combiner
  backtest/       # Vectorized engine, results, cost models, optimization, Monte Carlo
  risk/           # Position sizing, VaR, stop-losses, portfolio risk
  analytics/      # 25+ metrics, tear sheets, benchmark comparison, regime detection
  viz/            # Plotly charts (candlestick, drawdown, heatmaps, distributions, trades)
  pages/          # Streamlit UI (Dashboard, Backtester, Compare, Analysis)

app.py            # Streamlit entry point
tests/            # 175 tests
docs/             # Architecture, developer guide, user guide, quickstart
```

## Streamlit Dashboard

Four pages accessible via the sidebar:

- **Dashboard** -- Load market data, overlay indicators, interactive multi-panel charts
- **Backtester** -- Select a strategy, configure parameters, run backtest, view equity curve and trade log
- **Compare** -- Multi-asset performance comparison (normalized curves, correlation heatmap) or multi-strategy comparison (side-by-side metrics)
- **Analysis** -- Tear sheet, VaR/CVaR risk analysis, regime detection, return distributions

## Testing

```bash
# Run all 175 tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_indicators.py -v

# Run a single test
pytest tests/test_indicators.py::TestEngineCompute::test_sma -v
```

## Key Conventions

- **Signal convention**: `1` = long, `-1` = short, `0` = neutral (used everywhere)
- **Column naming**: follows pandas-ta convention (`SMA_20`, `RSI_14`, `MACD_12_26_9`, `BBU_20_2.0`)
- **OHLCV schema**: DatetimeIndex named `"date"`, columns `open`, `high`, `low`, `close`, `volume` (lowercase, float64)
- **pandas version**: pinned `>= 2.0, < 3.0` (pandas-ta-classic compatibility)

## Extending the Framework

**Add an indicator**: Write a wrapper function in `indicators/wrappers.py`, add metadata to `INDICATOR_CATALOG` in `catalog.py`, add to `WRAPPER_MAP`. The engine auto-discovers it.

**Add a strategy**: Subclass `SignalGenerator`, implement `generate()` and `required_indicators`, decorate with `@strategy_registry.register("name")`.

**Add a data provider**: Subclass `DataProvider`, implement `fetch()`, `supported_assets()`, `search_symbols()`, decorate with `@provider_registry.register("name")`.

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) -- Step-by-step first run
- [User Guide](docs/USER_GUIDE.md) -- Dashboard walkthrough and workflows
- [Developer Guide](docs/DEVELOPER_GUIDE.md) -- Extending indicators, strategies, and providers
- [Architecture](docs/ARCHITECTURE.md) -- Design decisions and system overview
- [Financial Professional's Guide](docs/FINANCIAL_PROFESSIONALS_GUIDE.md) -- Capability assessment for PMs, quants, and risk analysts

## License

MIT
