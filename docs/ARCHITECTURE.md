# Architecture Guide

A comprehensive reference for developers who want to understand, extend, or contribute to the Technical Analysis Framework. Written for programmers familiar with C, C++, or Java who may be new to Python and its data science ecosystem.

---

## 1. The Big Picture

The framework implements a linear data pipeline. Raw market data enters at the top, flows through computation stages, and exits as performance reports and interactive charts in a browser.

```
  Data Source (Yahoo Finance / CSV file / Crypto Exchange)
      |
      v
  DataProvider.fetch() --> pd.DataFrame (OHLCV)
      |
      v
  IndicatorEngine.compute() --> DataFrame + indicator columns (SMA_20, RSI_14, ...)
      |
      v
  SignalGenerator.generate() --> DataFrame + signal column (1 / -1 / 0)
      |
      v
  VectorizedBacktester.run() --> BacktestResult
      |                              |
      v                              v
  TearSheet / Analytics         Visualization (Plotly charts)
      |                              |
      v                              v
           Streamlit UI (browser)
```

### What each stage does in plain English

1. **Data Source**: Historical price data lives somewhere -- on Yahoo Finance's servers, in a local CSV file, or on a cryptocurrency exchange. We do not generate this data; we download it.

2. **DataProvider.fetch()**: A provider class connects to the data source, downloads the price history for a given stock symbol and date range, and normalizes it into a standard table format (a pandas DataFrame) with columns `open`, `high`, `low`, `close`, `volume` indexed by date. Think of it as reading a spreadsheet into memory.

3. **IndicatorEngine.compute()**: Technical indicators are mathematical formulas applied to price data. For example, a 20-day Simple Moving Average (SMA) computes the average closing price over the last 20 days for each day. The engine takes the price table, runs the formula, and adds new columns like `SMA_20` or `RSI_14` to the same table.

4. **SignalGenerator.generate()**: A trading strategy reads the indicator columns and decides when to buy or sell. It adds a `signal` column to the table: `1` means "go long (buy)", `-1` means "go short (sell)", `0` means "do nothing". For example, an EMA crossover strategy says "buy when the fast EMA crosses above the slow EMA".

5. **VectorizedBacktester.run()**: The backtester replays the signal column against historical prices to simulate what would have happened if we had actually traded those signals. It tracks equity (account balance), generates a list of trades, and computes performance metrics like total return and Sharpe ratio.

6. **TearSheet / Analytics**: Comprehensive performance analysis -- Sharpe ratio, Sortino ratio, maximum drawdown, win rate, profit factor, VaR, regime detection, and more. Produces structured data for display.

7. **Visualization**: Plotly creates interactive charts -- candlestick charts with indicator overlays, equity curves, drawdown charts, return distribution histograms, heatmaps.

8. **Streamlit UI**: The web dashboard that ties everything together. A user selects a stock, picks a strategy, clicks "Run Backtest", and sees all the results in their browser.

---

## 2. Technology Stack Explained

If you come from a C++ or Java background, many of these tools will be unfamiliar. This section maps each technology to concepts you already know.

### Python

**What it is**: An interpreted, dynamically-typed programming language.

**Why not C++/Java?** Python dominates data science because of its ecosystem. Libraries like pandas, NumPy, and scikit-learn provide functionality that would take thousands of lines of C++ to replicate. Python trades raw execution speed for developer productivity -- a trading strategy that would take a week to prototype in C++ takes an afternoon in Python.

**Key differences from C++/Java**:
- No compilation step. You run the source code directly: `python script.py`.
- No explicit type declarations (though we use type hints as documentation).
- Indentation defines code blocks (no `{}` braces).
- No semicolons at line ends.

### pandas

**What it is**: The core data manipulation library. Think of it as an in-memory SQL database or a 2D spreadsheet in code.

**The C++/Java analogy**: Imagine `std::vector<std::map<string, double>>` but with named columns, a time-series index, automatic alignment, and hundreds of built-in aggregation/transformation functions. Or think of it as `ResultSet` from JDBC but stored in memory with full SQL-like query capabilities.

**Concrete example**:
```
            open    high     low   close     volume
date
2024-01-02  185.3   186.1   184.0  185.8   50000000
2024-01-03  186.0   187.5   185.5  187.2   48000000
2024-01-04  187.0   188.0   186.0  186.5   52000000
```

This is a DataFrame. Each row is a trading day. Each column is a field. The index (left side) is a `DatetimeIndex` -- a special index type that understands dates and enables time-series operations like resampling from daily to weekly data.

**Key operations we use**:
- `df["close"]` -- select a column (returns a Series, like a 1D array with an index)
- `df["close"].rolling(20).mean()` -- compute rolling 20-day average
- `df["close"].pct_change()` -- compute percent change between consecutive rows
- `df.copy()` -- create an independent copy (DataFrames are mutable reference types)
- `df[df["signal"] == 1]` -- filter rows where signal equals 1 (boolean indexing)

### pandas-ta (pandas-ta-classic)

**What it is**: A library of 200+ pre-built technical indicator functions that operate on pandas Series/DataFrames.

**The C++/Java analogy**: Like a specialized math library (think Boost.Math or Apache Commons Math) but for financial formulas. Instead of implementing a 50-line RSI calculation yourself, you call `ta.rsi(close_series, length=14)` and get back a Series of RSI values.

**Why "classic"?** The original `pandas-ta` package has maintenance issues. `pandas-ta-classic` is a community fork that fixes compatibility with modern pandas versions. The API is identical.

**Why not TA-Lib?** TA-Lib is a C library with Python bindings. It requires compiling C code during installation, which fails on many systems (especially Windows without a C compiler). pandas-ta-classic is pure Python -- it works everywhere.

### Streamlit

**What it is**: A web framework that turns Python scripts into browser-based applications. No HTML, CSS, or JavaScript required.

**The Java analogy**: Imagine if you could write a Swing desktop app, but it automatically runs in a web browser, handles all the HTTP routing, and re-renders whenever the user interacts with a widget. That is Streamlit.

**How it works**: You write a Python script that calls functions like `st.text_input("Symbol")`, `st.button("Run")`, `st.line_chart(data)`. Streamlit runs this script, converts each call into an HTML widget, and serves the result as a web page. When the user clicks a button or types in an input, Streamlit **re-runs the entire script from top to bottom** with the new input values.

This is fundamentally different from Spring MVC, Express.js, or any request/response web framework. There are no routes, no controllers, no templates. Just a linear Python script.

### Plotly

**What it is**: An interactive charting library. Charts support zoom, pan, hover tooltips, and click events.

**The Java analogy**: Like JFreeChart but running in the browser with JavaScript-powered interactivity.

**How we use it**: Every chart function in `ta_framework/viz/` returns a `plotly.graph_objects.Figure` object. Streamlit renders it with `st.plotly_chart(fig)`. The user sees an interactive chart they can zoom and hover over.

### pytest

**What it is**: Python's most popular test framework.

**The Java analogy**: JUnit, but simpler. In Java you need a test class, annotations like `@Test`, and assertion methods like `assertEquals()`. In pytest, you just write a function whose name starts with `test_` and use Python's built-in `assert` statement:

```python
# Java (JUnit)                        # Python (pytest)
@Test                                  def test_addition():
public void testAddition() {               assert 1 + 1 == 2
    assertEquals(2, 1 + 1);
}
```

No test class required (though we do use them for organization). No annotations. No `assertEquals` -- just `assert`.

### pip + pyproject.toml

**What they are**: Package manager + build configuration.

**The Java analogy**: `pip` is like Maven or Gradle (downloads and installs packages). `pyproject.toml` is like `pom.xml` or `build.gradle` (declares project metadata, dependencies, and build settings).

**Key commands**:
- `pip install -r requirements.txt` -- install dependencies listed in a file (like `mvn install`)
- `pip install -e ".[dev]"` -- install this project in "editable" mode so changes take effect immediately (like running from source)

### NumPy / SciPy

**What they are**: Numerical computing libraries.

**The C++ analogy**: NumPy is like BLAS/LAPACK for Python -- it provides fast N-dimensional array operations implemented in C under the hood. `numpy.ndarray` is a contiguous block of typed memory, similar to `std::vector<double>`. SciPy builds on NumPy with higher-level scientific functions (statistics, optimization, signal processing).

**Where we use them**: VaR calculations (normal distribution), Monte Carlo simulation (random number generation), portfolio risk (matrix multiplication), Q-Q plots (quantile functions).

### scikit-learn

**What it is**: Machine learning library.

**Where we use it**: `KMeans` clustering for market regime detection (in `analytics/regime.py`). We use KMeans to classify market conditions into "bull", "bear", and "neutral" regimes based on rolling return and volatility features. `StandardScaler` normalizes features before clustering.

---

## 3. Design Patterns (Mapped to Java/C++ Equivalents)

### Abstract Base Class (ABC) = Java Interface

Python's ABC (Abstract Base Class) is the equivalent of a Java interface or a C++ pure virtual class. It defines a contract that subclasses must implement.

**Python (our code)**:
```python
# ta_framework/data/base.py
from abc import ABC, abstractmethod

class DataProvider(ABC):
    @abstractmethod
    def fetch(self, symbol, timeframe, start, end):
        """Must be implemented by subclasses."""

    @abstractmethod
    def supported_assets(self):
        """Must be implemented by subclasses."""

    @abstractmethod
    def search_symbols(self, query):
        """Must be implemented by subclasses."""
```

**Java equivalent**:
```java
public interface DataProvider {
    DataFrame fetch(String symbol, Timeframe tf, Date start, Date end);
    List<AssetClass> supportedAssets();
    List<Map<String, String>> searchSymbols(String query);
}
```

**C++ equivalent**:
```cpp
class DataProvider {
public:
    virtual DataFrame fetch(string symbol, Timeframe tf, Date start, Date end) = 0;
    virtual vector<AssetClass> supported_assets() = 0;
    virtual vector<map<string, string>> search_symbols(string query) = 0;
};
```

**Where we use ABCs in this project**:
- `DataProvider` -- implemented by `YFinanceProvider`, `CSVProvider`, `CCXTProvider`
- `SignalGenerator` -- implemented by `EMACrossStrategy`, `RSIStrategy`, `MACDStrategy`, etc.
- `CustomIndicator` -- base for user-defined indicators
- `PositionSizer` -- implemented by `FixedFractional`, `KellyCriterion`, `VolatilityBased`, `RiskParity`
- `CostModel` -- implemented by `FixedCommission`, `PercentageCommission`, `TieredCommission`, `SlippageModel`

### Registry Pattern = ServiceLocator / Plugin System

The Registry pattern lets us discover and retrieve implementations by name at runtime, without hard-coding imports.

**Python (our code)**:
```python
# ta_framework/core/registry.py
class Registry:
    def __init__(self, name):
        self._store = {}

    def register(self, key, **meta):
        def decorator(obj):
            self._store[key] = {"obj": obj, "meta": meta}
            return obj
        return decorator

    def get(self, key):
        return self._store[key]["obj"]

# Global instances
strategy_registry = Registry("strategies")
provider_registry = Registry("providers")
indicator_registry = Registry("indicators")
```

**Usage**:
```python
@strategy_registry.register("ema_cross")
class EMACrossStrategy(SignalGenerator):
    ...

# Later, look up by name:
cls = strategy_registry.get("ema_cross")
strategy = cls(fast_period=12, slow_period=26)
```

**Java equivalent**: This is like the ServiceLoader pattern or a simple HashMap-based service locator:
```java
Map<String, Class<? extends Strategy>> registry = new HashMap<>();
registry.put("ema_cross", EMACrossStrategy.class);
// Later:
Strategy s = registry.get("ema_cross").newInstance();
```

**Why we use it**: The Streamlit UI needs to present a dropdown of all available strategies. The registry provides `strategy_registry.keys()` without the UI needing to import every strategy class.

### Decorator Pattern = Java Annotations

Python decorators are functions that wrap other functions or classes. They are similar in purpose (but different in mechanism) to Java annotations.

**Python (our code)**:
```python
@strategy_registry.register("ema_cross")
class EMACrossStrategy(SignalGenerator):
    ...
```

The `@strategy_registry.register("ema_cross")` line calls `register("ema_cross")`, which returns a decorator function. That decorator receives `EMACrossStrategy` as its argument, stores it in the registry under the key `"ema_cross"`, and returns the class unchanged.

**Java equivalent**:
```java
@RegisterStrategy("ema_cross")
public class EMACrossStrategy implements SignalGenerator { ... }
// (Plus an annotation processor to discover and register at startup)
```

The key difference: Java annotations are metadata that require separate processing (reflection, annotation processors). Python decorators execute immediately when the class/function is defined.

### Dataclass = Java Record / POJO / C++ Struct

Python dataclasses auto-generate `__init__`, `__repr__`, `__eq__`, and other boilerplate methods.

**Python (our code)**:
```python
# ta_framework/core/types.py
from dataclasses import dataclass, field

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    position_size_pct: float = 1.0
    max_positions: int = 1
    risk_free_rate: float = 0.04
    benchmark_symbol: str | None = None
    allow_short: bool = True
    metadata: dict = field(default_factory=dict)
```

**Java equivalent** (Java 16+ record):
```java
public record BacktestConfig(
    double initialCapital,     // 100_000.0
    double commissionPct,      // 0.001
    double slippagePct,        // 0.0005
    double positionSizePct,    // 1.0
    int maxPositions,          // 1
    double riskFreeRate,       // 0.04
    String benchmarkSymbol,    // null
    boolean allowShort,        // true
    Map<String, Object> metadata
) {}
```

**C++ equivalent**:
```cpp
struct BacktestConfig {
    double initial_capital = 100000.0;
    double commission_pct = 0.001;
    double slippage_pct = 0.0005;
    // ...
};
```

### DataFrame Pipeline = Builder Pattern / Method Chaining

Data flows through the system by repeatedly transforming a DataFrame, similar to the Builder pattern or method chaining.

```python
# Each step takes a DataFrame, adds columns, returns the modified DataFrame
df = provider.fetch("AAPL", Timeframe.D1, "2020-01-01")  # OHLCV
df = engine.compute(df, "ema", length=12)                  # + EMA_12
df = engine.compute(df, "ema", length=26)                  # + EMA_26
df = strategy.generate(df)                                  # + signal
result = backtester.run(df)                                 # -> BacktestResult
```

**Java equivalent** (conceptual):
```java
BacktestResult result = Pipeline.create()
    .fetch("AAPL", Timeframe.D1, "2020-01-01")
    .compute("ema", Map.of("length", 12))
    .compute("ema", Map.of("length", 26))
    .generateSignals(emaCrossStrategy)
    .backtest(config);
```

---

## 4. Module Deep Dive

### 4.1 core

**Purpose**: Foundation types, configuration, plugin registry, and exception hierarchy shared across all other modules.

**Key Files**:
| File | Description |
|---|---|
| `types.py` | Enums (`AssetClass`, `Timeframe`, `SignalDirection`, `IndicatorTier`, `IndicatorCategory`) and dataclasses (`Signal`, `Trade`, `BacktestConfig`, `IndicatorConfig`) |
| `registry.py` | Generic `Registry` class + global instances: `indicator_registry`, `strategy_registry`, `provider_registry` |
| `config.py` | `FrameworkConfig` singleton -- default timeframe, cache settings, OHLCV column list |
| `exceptions.py` | Exception hierarchy rooted at `TAFrameworkError` with specific subclasses per domain |

**Key Classes/Functions**:
- `Registry` -- generic key-value store for plugin registration. Methods: `register(key)` (returns decorator), `get(key)`, `keys()`, `items()`.
- `FrameworkConfig` -- singleton accessed via `FrameworkConfig.get()`. Holds defaults like `default_timeframe=D1`, `cache_ttl_seconds=300`.
- `BacktestConfig` -- passed to `VectorizedBacktester` to configure capital, commissions, slippage, position sizing.
- `Trade` -- represents a completed trade with entry/exit prices, P&L, commissions. Has computed properties `net_pnl` and `is_winner`.

**Dependencies**: None (this is the foundation layer).

**Extension Points**: Add new enums to `types.py` for new asset classes or timeframes. Add new exception classes to `exceptions.py` for new error domains.

**Design Decisions**: The `Registry` class was chosen over Python's `importlib` entry points for simplicity -- no setup.py configuration required, registration happens at import time via decorators.

### 4.2 data

**Purpose**: Fetch, validate, and normalize OHLCV market data from multiple sources.

**Key Files**:
| File | Description |
|---|---|
| `base.py` | `DataProvider` ABC defining the interface + `validate()` method that enforces the OHLCV schema |
| `yfinance_provider.py` | `YFinanceProvider` -- downloads from Yahoo Finance via the `yfinance` library |
| `csv_provider.py` | `CSVProvider` -- reads local CSV or Parquet files |
| `ccxt_provider.py` | `CCXTProvider` -- cryptocurrency data via the `ccxt` library (optional dependency) |
| `quality.py` | `DataQualityChecker` -- detects NaN gaps, price outliers, OHLC inconsistencies; provides a `clean()` method |
| `timeframe.py` | `resample()` for converting between timeframes (e.g., 1h to 4h); `align_timeframes()` for multi-TF analysis |

**Key Classes/Functions**:
- `DataProvider.fetch(symbol, timeframe, start, end)` -- abstract method; returns OHLCV DataFrame.
- `DataProvider.validate(df)` -- concrete method on the base class; lowercases columns, promotes date column to index, checks required columns, casts to float64, sorts chronologically, removes duplicates.
- `YFinanceProvider` -- handles yfinance quirks (Title Case columns, 4h resampling since yfinance does not natively support 4h candles).
- `CSVProvider` -- auto-detects file format from extension (`.csv` vs `.parquet`), auto-discovers the date column from common names.
- `DataQualityChecker.full_check(df)` -- runs gap detection, outlier detection, and OHLC consistency checks. Returns a structured report dict.

**Dependencies**: `core` (types, exceptions, registry, config).

**Extension Points**: Subclass `DataProvider` and decorate with `@provider_registry.register("name")` to add a new data source (e.g., database, REST API, websocket).

**Design Decisions**: The `validate()` method lives on the base class (not each subclass) to guarantee a uniform output format regardless of the data source. All providers call `self.validate(df)` as the last step of `fetch()`.

### 4.3 indicators

**Purpose**: Compute technical indicators on OHLCV DataFrames. Provides 38 Tier 1 indicators out of the box, plus a framework for custom indicators.

**Key Files**:
| File | Description |
|---|---|
| `engine.py` | `IndicatorEngine` -- central compute API (`compute`, `compute_batch`, `register`, `available`) |
| `catalog.py` | `INDICATOR_CATALOG` -- dict mapping indicator names to `IndicatorConfig` metadata (category, tier, default params, description) |
| `wrappers.py` | One wrapper function per indicator (e.g., `sma()`, `rsi()`, `macd()`). Each wraps a pandas-ta call and appends result columns to the DataFrame. `WRAPPER_MAP` dict at the bottom maps names to functions. |
| `custom.py` | `CustomIndicator` ABC + `@register_indicator()` decorator for user-defined indicators |
| `composite.py` | `CompositeIndicator` -- chains multiple indicators sequentially (e.g., compute SMA, then compute RSI of the SMA) |

**Key Classes/Functions**:
- `IndicatorEngine.compute(df, "sma", length=20)` -- looks up the wrapper function, calls it, returns the augmented DataFrame.
- `IndicatorEngine.compute_batch(df, [{"name": "sma", "params": {"length": 20}}, ...])` -- computes multiple indicators in sequence.
- `IndicatorEngine.register(name, func)` -- registers a custom indicator at runtime.
- `WRAPPER_MAP` -- dict of `{name: function}` for all 38 built-in indicators. The engine resolves indicator names by looking here first, then falling back to the registry.

**Dependencies**: `core` (registry, types, exceptions), `pandas-ta-classic`.

**Extension Points**:
1. **Quick**: Add a function to `wrappers.py`, add metadata to `catalog.py`, add to `WRAPPER_MAP`.
2. **Class-based**: Subclass `CustomIndicator`, implement `compute()`, decorate with `@register_indicator()`.
3. **Chaining**: Use `CompositeIndicator` to chain existing indicators without writing new code.

**Design Decisions**: Wrapper functions (not classes) were chosen for built-in indicators because they map naturally to the functional nature of pandas-ta. Each wrapper follows the same signature: `func(df, **params) -> DataFrame`.

### 4.4 signals

**Purpose**: Generate trading signals (buy/sell/neutral) from indicator columns.

**Key Files**:
| File | Description |
|---|---|
| `base.py` | `SignalGenerator` ABC -- defines the interface: `generate(df)` and `required_indicators` |
| `rules.py` | Reusable atomic signal rule functions: `crossover()`, `threshold()`, `divergence()`, `breakout()` |
| `strategies.py` | 6 pre-built strategies: `EMACrossStrategy`, `RSIStrategy`, `MACDStrategy`, `BollingerBandStrategy`, `SupertrendStrategy`, `TTMSqueezeStrategy` |
| `composite.py` | `CompositeSignal` -- combines multiple generators via voting, weighted average, or confirmation (all-agree) |

**Key Classes/Functions**:
- `SignalGenerator.generate(df)` -- abstract. Takes a DataFrame with indicator columns already present, adds a `signal` column (1/-1/0), optionally adds `signal_strength`.
- `SignalGenerator.required_indicators` -- abstract property. Returns list of dicts like `[{"name": "ema", "length": 12}]` describing which indicators must be pre-computed.
- `SignalGenerator.validate_columns(df, columns)` -- concrete helper that raises `SignalError` if required columns are missing.
- `crossover(fast, slow)` -- returns Series of 1/-1/0 when fast line crosses above/below slow line.
- `threshold(series, upper, lower)` -- returns 1 when series drops below lower (buy), -1 when above upper (sell).
- `CompositeSignal` -- wraps multiple generators. Modes: `VOTING` (majority), `WEIGHTED` (weighted sum), `CONFIRMATION` (all must agree).

**Dependencies**: `core` (registry, exceptions), `numpy`.

**Extension Points**: Subclass `SignalGenerator`, implement `generate()` and `required_indicators`, decorate with `@strategy_registry.register("name")`.

**Design Decisions**: Signals are decoupled from indicators. A strategy's `generate()` method assumes the required indicator columns are already in the DataFrame -- it does not compute them. This separation means the same strategy can work on pre-computed DataFrames from any source.

### 4.5 backtest

**Purpose**: Simulate trading strategies against historical data to measure performance.

**Key Files**:
| File | Description |
|---|---|
| `engine.py` | `VectorizedBacktester` -- the core backtesting loop. Processes signal array, tracks positions, computes equity curve. |
| `results.py` | `BacktestResult` dataclass -- holds equity curve, trade list, signals DataFrame, config. Computed properties: `total_return`, `max_drawdown`, `sharpe_ratio`, `win_rate`, `profit_factor`. |
| `costs.py` | Cost model hierarchy: `CostModel` ABC, `FixedCommission`, `PercentageCommission`, `TieredCommission`, `SlippageModel` |
| `optimization.py` | `GridSearchOptimizer` (exhaustive parameter search) and `WalkForwardOptimizer` (rolling in-sample/out-of-sample) |
| `monte_carlo.py` | `MonteCarloSimulator` -- bootstrap equity paths and shuffle trade order for robustness testing |

**Key Classes/Functions**:
- `VectorizedBacktester.run(df, signal_col="signal")` -- main entry point. Reads the signal column, forward-fills positions (hold until next signal), detects trade entries/exits, computes equity bar-by-bar, builds list of `Trade` objects. Returns `BacktestResult`.
- `BacktestResult.summary()` -- returns dict with total_return, max_drawdown, num_trades, win_rate, profit_factor, sharpe_ratio, initial_capital, final_equity.
- `GridSearchOptimizer.run(df)` -- tries every combination of parameters, backtests each, ranks by target metric.
- `WalkForwardOptimizer.run(df)` -- splits data into rolling windows, optimizes in-sample, evaluates out-of-sample.

**Dependencies**: `core` (types, exceptions), `signals` (for optimization), `numpy`.

**Extension Points**: The `CostModel` ABC allows custom commission/slippage models. The optimizer classes accept any `SignalGenerator` subclass.

**Design Decisions**: The backtester uses a forward-fill position model: once a signal fires (1 or -1), the position is held until the next non-zero signal. This avoids the need for explicit "hold" signals. The implementation uses a for-loop (not pure vectorized operations) because position sizing depends on the current capital, which changes after each trade -- this is inherently sequential.

### 4.6 risk

**Purpose**: Position sizing, stop-loss strategies, Value at Risk, and portfolio-level risk metrics.

**Key Files**:
| File | Description |
|---|---|
| `position_sizing.py` | `PositionSizer` ABC + implementations: `FixedFractional`, `KellyCriterion`, `VolatilityBased`, `RiskParity` |
| `var.py` | Value at Risk calculations: `parametric_var`, `historical_var`, `monte_carlo_var`, `cvar` (Conditional VaR / Expected Shortfall) |
| `stops.py` | Stop-loss functions: `fixed_stop`, `atr_stop`, `trailing_stop`, `chandelier_stop` |
| `portfolio.py` | Portfolio risk: `portfolio_volatility`, `max_drawdown_duration`, `calmar_ratio`, `ulcer_index`, `risk_contribution` |

**Key Classes/Functions**:
- `FixedFractional.calculate(capital, price, risk_per_trade)` -- risk a fixed percentage of capital per trade.
- `KellyCriterion.calculate(...)` -- uses Kelly formula: `f = win_rate - (1 - win_rate) / payoff_ratio`. Default half-Kelly for safety.
- `parametric_var(returns, confidence=0.95)` -- Gaussian VaR using mean and standard deviation.
- `historical_var(returns, confidence=0.95)` -- VaR from the actual return distribution (empirical percentile).
- `trailing_stop(close, trail_pct)` -- trailing stop that follows price upward, never moves down.

**Dependencies**: `core` (exceptions), `numpy`, `scipy` (for normal distribution in parametric VaR).

**Extension Points**: Subclass `PositionSizer` for custom sizing logic. Add new stop-loss strategies as functions in `stops.py`.

**Design Decisions**: Position sizers return number of shares (float), not dollar amounts, because the caller knows the price and can multiply. VaR functions return positive numbers representing potential loss (not negative returns).

### 4.7 analytics

**Purpose**: Comprehensive performance measurement, benchmark comparison, and market regime detection.

**Key Files**:
| File | Description |
|---|---|
| `metrics.py` | 25+ individual metric functions: `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`, `max_drawdown`, `win_rate`, `profit_factor`, `expectancy`, `var_95`, `cvar_95`, `skewness`, `kurtosis`, etc. Plus `calculate_all()` aggregate. |
| `tearsheet.py` | `TearSheet` class -- generates a full performance report organized by category (returns, risk, trades, distribution, benchmark). Also: `monthly_returns()` pivot table, `drawdown_analysis()`. |
| `benchmark.py` | Benchmark comparison: `alpha_beta` (CAPM regression), `information_ratio`, `tracking_error`, `up_capture`, `down_capture`, `active_return` |
| `regime.py` | `RegimeDetector` -- `detect_kmeans(returns, n_regimes=3)` and `detect_hmm(returns, n_regimes=3)`. Also `regime_statistics()` for per-regime mean return, volatility, Sharpe, duration. |

**Key Classes/Functions**:
- `TearSheet.generate()` -- returns nested dict: `{"returns": {...}, "risk": {...}, "trades": {...}, "distribution": {...}}`.
- `TearSheet.monthly_returns()` -- pivot table of monthly returns (years as rows, months as columns).
- `RegimeDetector.detect_kmeans(returns, n_regimes=3)` -- uses KMeans clustering on rolling return + rolling volatility features. Labels are sorted by mean return (0=bear, 1=neutral, 2=bull).
- `calculate_all(returns, equity_curve, trades)` -- computes every available metric in one call.

**Dependencies**: `core` (types), `numpy`, `scipy` (statistics), `scikit-learn` (KMeans, StandardScaler), optionally `hmmlearn` (Gaussian HMM).

**Extension Points**: Add new metric functions to `metrics.py`. The `TearSheet.generate()` method can be extended to include new metric categories.

**Design Decisions**: Metrics are implemented as standalone functions (not methods on a class) so they can be used independently or composed. The `TearSheet` class is a convenience wrapper that calls multiple metric functions and organizes results.

### 4.8 viz

**Purpose**: Create interactive Plotly charts for all visualization needs.

**Key Files**:
| File | Description |
|---|---|
| `charts.py` | `candlestick_chart` (OHLC + overlays + volume), `indicator_panel` (oscillators in subplots), `multi_panel_chart` (combined: candles + overlays + volume + oscillators) |
| `drawdown.py` | `drawdown_chart` (underwater chart), `drawdown_periods` (equity curve with top N drawdown periods highlighted) |
| `heatmaps.py` | `correlation_heatmap` (asset correlation matrix), `monthly_returns_heatmap` (year x month return table) |
| `distribution.py` | `returns_histogram` (histogram + normal overlay + VaR lines), `qq_plot` (Q-Q plot vs normal distribution) |
| `trade_plots.py` | `trade_markers` (buy/sell triangles on candlestick chart), `pnl_chart` (cumulative P&L + per-trade bar chart) |

**Key Classes/Functions**: All functions return `plotly.graph_objects.Figure` objects that can be displayed with `st.plotly_chart(fig)` in Streamlit or saved as HTML.

**Dependencies**: `plotly`, `numpy`, `scipy` (for normal PDF in distribution charts), `core` (types for `SignalDirection`).

**Extension Points**: Add new chart types as functions in existing files or new files. Follow the pattern: accept DataFrame/Series, return `go.Figure`.

**Design Decisions**: Every chart function is a pure function -- no side effects, no state. This makes them easy to test, compose, and reuse outside Streamlit.

### 4.9 pages

**Purpose**: Streamlit UI pages that compose all framework components into an interactive web application.

**Key Files**:
| File | Description |
|---|---|
| `components.py` | Shared Streamlit widgets: `symbol_input`, `timeframe_select`, `date_range_input`, `indicator_multiselect`, `strategy_select`, `strategy_params`, `backtest_config_panel`, `fetch_data` (cached) |
| `dashboard.py` | Dashboard page: symbol/indicator selection, multi-panel chart with auto-classified overlays/oscillators |
| `backtester.py` | Backtester page: strategy configuration, backtest execution, results display (metrics, equity curve, drawdown, trades) |
| `compare.py` | Compare page: multi-asset normalized performance + correlation, multi-strategy equity comparison |
| `analysis.py` | Analysis page: tear sheet, VaR risk analysis, KMeans regime detection overlay, return distribution stats |

**Key Classes/Functions**:
- `components.fetch_data(symbol, timeframe, start, end)` -- cached data fetch (5-minute TTL via `@st.cache_data`). All pages use this to avoid redundant API calls.
- `dashboard.render()` -- each page has a `render()` function called by `app.py` based on sidebar navigation.
- `components.strategy_params(strategy_name, key_prefix)` -- dynamically renders parameter inputs based on the selected strategy name.

**Dependencies**: All other modules (data, indicators, signals, backtest, analytics, risk, viz), `streamlit`.

**Extension Points**: Add a new page by creating a new file with a `render()` function, then adding it to the navigation in `app.py`.

**Design Decisions**: Shared widgets live in `components.py` to avoid duplication across pages. Each widget takes a `key` parameter for Streamlit's widget state management (Streamlit requires unique keys when the same widget type appears multiple times).

---

## 5. Complete Data Flow Walkthrough

This section traces exactly what happens when a user clicks "Run Backtest" in the Streamlit UI, from the button click to the results appearing on screen.

### Step 1: User interaction

The user has navigated to the "Backtester" page. In the sidebar, they have:
- Entered `AAPL` as the symbol
- Selected `1d` (daily) timeframe
- Chosen start date 2020-01-01 and end date 2024-12-31
- Selected "EMA Crossover" strategy with fast=12, slow=26
- Configured backtest settings (capital $100,000, commission 0.1%, slippage 0.05%)

They click the "Run Backtest" button.

### Step 2: Streamlit re-runs the script

Streamlit detects the button click and re-runs `app.py` from top to bottom.

**File**: `app.py:15-30`
```python
page = st.sidebar.radio("Navigate", ["Dashboard", "Backtester", "Compare", "Analysis"])
if page == "Backtester":
    backtester.render()
```

Since "Backtester" is selected, `backtester.render()` is called.

### Step 3: Backtester page reads sidebar inputs

**File**: `ta_framework/pages/backtester.py:29-37`

The sidebar widgets have already been rendered with their current values. `symbol_input()` returns `"AAPL"`, `timeframe_select()` returns `Timeframe.D1`, `strategy_select()` returns `"ema_cross"`, `strategy_params("ema_cross")` returns `{"fast_period": 12, "slow_period": 26}`, `backtest_config_panel()` returns `{"initial_capital": 100000, ...}`.

### Step 4: Fetch market data

**File**: `ta_framework/pages/backtester.py:49-54`
```python
df = fetch_data(symbol, timeframe.value, start, end)
```

This calls `components.fetch_data()` which is decorated with `@st.cache_data(ttl=300)`. If AAPL daily data for this date range was fetched in the last 5 minutes, the cached result is returned immediately.

Otherwise, it creates a `YFinanceProvider` and calls `fetch()`:

**File**: `ta_framework/data/yfinance_provider.py:33-79`

1. `yfinance.Ticker("AAPL").history(start="2020-01-01", end="2024-12-31", interval="1d")` downloads the data from Yahoo Finance servers.
2. The returned DataFrame has Title Case columns (`Open`, `High`, `Low`, `Close`, `Volume`). These are lowercased.
3. `self.validate(df)` is called (inherited from `DataProvider`).

**File**: `ta_framework/data/base.py:63-109`

`validate()` ensures:
- Columns are lowercase
- Index is a `DatetimeIndex` named `"date"`
- All required columns (`open`, `high`, `low`, `close`, `volume`) are present
- Values are cast to `float64`
- Rows are sorted chronologically
- Duplicate dates are removed

Result: a clean OHLCV DataFrame with ~1260 rows (5 years of daily data).

### Step 5: Look up the strategy class and compute indicators

**File**: `ta_framework/pages/backtester.py:62-72`
```python
strategy_cls = strategy_registry.get("ema_cross")  # returns EMACrossStrategy class
strategy = strategy_cls(fast_period=12, slow_period=26)
```

**File**: `ta_framework/signals/strategies.py:34-38`

`strategy.required_indicators` returns:
```python
[{"name": "ema", "length": 12}, {"name": "ema", "length": 26}]
```

The page loops over these and computes each indicator:

**File**: `ta_framework/pages/backtester.py:66-72`
```python
engine = IndicatorEngine()
for ind_spec in strategy.required_indicators:
    name = ind_spec.pop("name")
    df = engine.compute(df, name, **ind_spec)
```

**File**: `ta_framework/indicators/engine.py:42-84`

`engine.compute(df, "ema", length=12)`:
1. `_resolve("ema")` looks up `WRAPPER_MAP["ema"]` and finds the `ema()` wrapper function.
2. Calls `ema(df, length=12)`.

**File**: `ta_framework/indicators/wrappers.py:56-60`
```python
def ema(df, length=20, column="close"):
    col_name = f"EMA_{length}"
    result = ta.ema(df[column], length=length)  # pandas-ta computes 12-day EMA
    result.name = col_name
    return _append(df, result, fallback_name=col_name)
```

`_append()` copies the DataFrame and adds the `EMA_12` column. Same process for `EMA_26`.

After this step, the DataFrame has columns: `open, high, low, close, volume, EMA_12, EMA_26`.

### Step 6: Generate signals

**File**: `ta_framework/pages/backtester.py:75-80`
```python
df = strategy.generate(df)
```

**File**: `ta_framework/signals/strategies.py:40-50`
```python
def generate(self, df):
    fast_col = f"EMA_{self.fast_period}"  # "EMA_12"
    slow_col = f"EMA_{self.slow_period}"  # "EMA_26"
    self.validate_columns(df, [fast_col, slow_col])

    df["signal"] = crossover(df[fast_col], df[slow_col])
    # ... compute signal_strength ...
    return df
```

**File**: `ta_framework/signals/rules.py:13-30`
```python
def crossover(fast, slow):
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    cross_up = (fast > slow) & (prev_fast <= prev_slow)    # EMA_12 crosses above EMA_26
    cross_down = (fast < slow) & (prev_fast >= prev_slow)  # EMA_12 crosses below EMA_26
    signal = pd.Series(0, index=fast.index, dtype=np.int8)
    signal[cross_up] = 1    # buy signal
    signal[cross_down] = -1  # sell signal
    return signal
```

After this step, the DataFrame has a `signal` column with values 1, -1, and 0.

### Step 7: Run the backtest

**File**: `ta_framework/pages/backtester.py:83-91`
```python
config = BacktestConfig(**bt_cfg)
backtester = VectorizedBacktester(config)
result = backtester.run(df)
```

**File**: `ta_framework/backtest/engine.py:23-144`

`VectorizedBacktester.run(df)`:
1. Reads the `signal` column and `close` prices.
2. Forward-fills signals: if signal is 0, hold the previous position. This converts sparse signals (only fires on crossover days) into a continuous position series.
3. Iterates bar-by-bar:
   - When position changes (e.g., from 0 to 1): closes the old position, records a `Trade` object, opens a new position using `capital * position_size_pct / price` shares.
   - Deducts commissions and slippage on each trade.
   - Computes mark-to-market equity each bar.
4. Returns a `BacktestResult` containing the equity curve (Series), trade list, signals DataFrame, and config.

### Step 8: Display results

**File**: `ta_framework/pages/backtester.py:93-136`

The page calls `result.summary()` and displays key metrics as `st.metric()` widgets:

**File**: `ta_framework/backtest/results.py:78-89`
```python
def summary(self):
    return {
        "total_return": self.total_return,
        "max_drawdown": self.max_drawdown,
        "num_trades": self.num_trades,
        "win_rate": self.win_rate,
        ...
    }
```

Then it renders:
- Equity curve as a line chart (`st.line_chart(result.equity_curve)`)
- Drawdown chart via `drawdown_chart(result.equity_curve)` from `viz/drawdown.py`
- Trade markers on a candlestick chart via `trade_markers(df, result.trades)` from `viz/trade_plots.py`
- P&L analysis via `pnl_chart(result.trades)` from `viz/trade_plots.py`
- Trade log table via `result.to_dataframe()` displayed with `st.dataframe()`

The user sees all of this in their browser within a few seconds.

---

## 6. Testing Architecture

### How pytest discovers tests

pytest uses a naming convention to find tests automatically:
- It searches directories listed in `testpaths` (configured in `pyproject.toml` as `["tests"]`)
- It looks for files matching `test_*.py` or `*_test.py`
- Within those files, it finds functions named `test_*` or classes named `Test*` with methods named `test_*`
- No registration or configuration is needed -- just name your test correctly.

**Comparison to JUnit**: In JUnit, you annotate methods with `@Test`. In pytest, you just name them `test_something`. In JUnit, test methods must be in a class. In pytest, standalone functions work fine (but we use classes for grouping).

### What fixtures are

Fixtures are pytest's equivalent of JUnit's `@Before` / `@BeforeEach` setup methods. They provide shared test data or configuration to test functions.

**JUnit**:
```java
private DataFrame sampleData;

@Before
public void setUp() {
    sampleData = generateSampleOHLCV();
}

@Test
public void testSMA() {
    // uses sampleData
}
```

**pytest**:
```python
@pytest.fixture
def sample_ohlcv():
    # generate and return 252 days of OHLCV data
    return df

def test_sma(sample_ohlcv):  # pytest sees the parameter name matches a fixture
    engine = IndicatorEngine()
    result = engine.compute(sample_ohlcv, "sma", length=20)
    assert "SMA_20" in result.columns
```

pytest automatically matches function parameter names to fixture names. If a test function takes a parameter called `sample_ohlcv`, pytest calls the `sample_ohlcv` fixture and passes the result.

### conftest.py: project-wide fixtures

**File**: `tests/conftest.py`

This file is automatically loaded by pytest. Any fixture defined here is available to all test files without importing.

Our fixtures:
- `sample_ohlcv` -- 252 days (~1 year) of synthetic OHLCV data starting at price 100 with realistic random walk returns. Uses `np.random.seed(42)` for reproducibility.
- `short_ohlcv` -- 20-bar dataset for quick unit tests.
- `empty_ohlcv` -- empty DataFrame with correct column names (for edge case testing).
- `default_backtest_config` -- `BacktestConfig()` with all defaults.
- `sample_signals` -- `sample_ohlcv` with an added `signal` column (alternating buy/sell every 40 bars).

### How to run tests at different granularities

```bash
# All tests
pytest tests/ -v

# One file
pytest tests/test_indicators.py -v

# One class
pytest tests/test_indicators.py::TestEngineCompute -v

# One test
pytest tests/test_indicators.py::TestEngineCompute::test_sma -v

# Tests matching a keyword
pytest tests/ -v -k "rsi"

# With coverage report
pytest tests/ -v --cov=ta_framework --cov-report=term-missing
```

### How to write a new test

Template:

```python
# tests/test_my_feature.py
import pandas as pd
import pytest

from ta_framework.my_module import MyClass


class TestMyClass:
    """Tests for MyClass."""

    def test_basic_behavior(self, sample_ohlcv):
        """Test that the basic operation works."""
        obj = MyClass()
        result = obj.do_something(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert "expected_column" in result.columns

    def test_edge_case_empty(self, empty_ohlcv):
        """Test behavior with empty input."""
        obj = MyClass()
        with pytest.raises(SomeError):
            obj.do_something(empty_ohlcv)

    def test_parameter_validation(self, sample_ohlcv):
        """Test that invalid parameters are rejected."""
        obj = MyClass()
        with pytest.raises(InvalidParameterError):
            obj.do_something(sample_ohlcv, invalid_param=-1)
```

---

## 7. The Streamlit Model (for Web Framework Newcomers)

### How traditional web apps work

In a traditional web framework (Spring Boot, Express.js, Django):

1. The server starts and listens for HTTP requests.
2. A request comes in: `GET /dashboard?symbol=AAPL`.
3. A controller/handler function receives the request.
4. It processes data, renders a template (HTML with placeholders), and sends the response.
5. The browser displays the HTML. JavaScript handles interactivity.

This requires understanding HTTP, routing, templates, HTML, CSS, JavaScript, and client-server architecture.

### How Streamlit works

Streamlit throws all of that away. Here is the entire mental model:

1. You write a Python script (`app.py`).
2. You run `streamlit run app.py`.
3. Streamlit executes your script from top to bottom.
4. Every `st.something()` call becomes an HTML widget in the browser.
5. When the user interacts with any widget (clicks a button, changes an input), **Streamlit re-runs the entire script from top to bottom** with the new widget values.

That is it. No routes. No controllers. No templates. No HTML. No CSS. No JavaScript.

### Key Streamlit concepts

**Widgets capture input**:
```python
symbol = st.text_input("Symbol", value="AAPL")  # renders a text box, returns the current value
timeframe = st.selectbox("Timeframe", ["1d", "1h", "1wk"])  # dropdown
run = st.button("Run Backtest")  # True on the run where it was clicked, False otherwise
```

**Display functions show output**:
```python
st.line_chart(data)           # line chart
st.dataframe(df)              # interactive table
st.plotly_chart(fig)          # Plotly figure
st.metric("Sharpe", "1.52")  # metric card with big number
```

**`st.sidebar`**: Creates a collapsible sidebar panel for inputs/navigation.
```python
with st.sidebar:
    symbol = st.text_input("Symbol")
```

**`@st.cache_data`**: Memoization decorator. The function is called once; subsequent calls with the same arguments return the cached result. Critical for avoiding redundant API calls:
```python
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_data(symbol, timeframe, start, end):
    provider = YFinanceProvider()
    return provider.fetch(symbol, ...)
```

Without caching, every widget interaction would re-fetch data from Yahoo Finance.

**Re-run model**: This is the most important concept. Every time the user clicks anything, the entire script runs again. That means:
- Variables do not persist between runs (unless cached or stored in `st.session_state`).
- Expensive operations (API calls, heavy computation) must be cached.
- The script must be fast enough to re-run in under a second for responsive UI.

**This is NOT like Spring MVC or Express.js**. There are no request handlers, no routing table, no view templates. It is much simpler -- but the re-run-everything model can be surprising if you come from a traditional web framework.

---

## 8. Common Pitfalls for C++/Java Developers

### No compilation

Python is interpreted. There is no compile step. Errors that C++ or Java would catch at compile time (wrong type, missing method, typo in variable name) only appear when that line of code actually executes at runtime.

**What this means for you**: You will not see type errors until the code runs. Use `pytest` frequently to catch issues early. Type hints (like `def foo(x: int) -> str:`) are optional documentation -- Python does not enforce them at runtime.

### Duck typing

Python does not require objects to declare what interfaces they implement. If an object has a `.compute()` method, you can call it -- regardless of what class it is.

```python
# C++: must declare inheritance
class MyIndicator : public CustomIndicator { ... };

# Python: as long as it has the right methods, it works
class MyIndicator:
    def compute(self, df):
        return df  # no inheritance required (but we use ABCs for clarity)
```

This is called "duck typing" -- if it walks like a duck and quacks like a duck, it is a duck. Our code uses ABCs to document the expected interface, but Python does not strictly require them.

### Mutable default arguments

This is one of the most notorious Python traps:

```python
# WRONG -- the default list is shared across all calls
def add_item(item, items=[]):
    items.append(item)
    return items

add_item("a")  # returns ["a"]
add_item("b")  # returns ["a", "b"] -- NOT ["b"]!
```

The default `[]` is created once when the function is defined, not on each call. Every call shares the same list.

**The fix**:
```python
# CORRECT -- use None and create a new list each call
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

Our codebase uses `field(default_factory=dict)` and `field(default_factory=list)` in dataclasses for the same reason.

### Import system

Python's import system can be confusing:

```python
# Import a module (access members as module.thing)
import pandas as pd
df = pd.DataFrame()

# Import specific names from a module
from ta_framework.core.types import BacktestConfig
config = BacktestConfig()

# Import everything (discouraged -- pollutes namespace)
from ta_framework.core.types import *
```

**Circular imports**: If module A imports module B and module B imports module A, Python will raise an `ImportError`. This is resolved by:
1. Restructuring to avoid the cycle
2. Moving the import inside a function (lazy import)
3. Using `from __future__ import annotations` (defers type hint evaluation)

Our codebase uses `from __future__ import annotations` at the top of most files to avoid circular import issues with type hints.

### Virtual environments

Python does not have a per-project classpath like Java. By default, all packages install globally. Virtual environments (`venv`) solve this by creating an isolated copy of the Python interpreter with its own package directory.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows Git Bash)
source venv/Scripts/activate

# Now pip installs go to venv/, not global Python
pip install -r requirements.txt
```

**The Java analogy**: A virtual environment is like having a separate Maven `.m2` repository per project. Each project gets its own dependency tree without conflicts.

### Indentation matters

Python uses indentation (whitespace) to define code blocks. C++ and Java use `{}` braces.

```python
# Python                              // C++
if x > 0:                             if (x > 0) {
    print("positive")                     cout << "positive";
    y = x * 2                             y = x * 2;
else:                                  } else {
    print("non-positive")                 cout << "non-positive";
                                       }
```

If indentation is wrong, the code either fails with `IndentationError` or (worse) runs with different control flow than intended. Most editors handle this automatically, but be careful when copy-pasting code.

### No semicolons

Lines end without semicolons. Adding a semicolon is not a syntax error (Python ignores it), but it is non-idiomatic.

### Everything is an object

In Python, literally everything is an object -- including functions, classes, modules, and even types themselves. This is why we can store functions in dictionaries (`WRAPPER_MAP`), pass classes as arguments (`strategy_cls`), and use decorators (functions that take functions as arguments and return functions).

```python
# Functions are objects -- you can store them in a dict
WRAPPER_MAP = {"sma": sma, "ema": ema, "rsi": rsi}
func = WRAPPER_MAP["sma"]  # func is now the sma function
result = func(df, length=20)  # call it

# Classes are objects -- you can store them in variables
cls = strategy_registry.get("ema_cross")  # cls is EMACrossStrategy class
instance = cls(fast_period=12, slow_period=26)  # create an instance
```

This is fundamentally different from C++ and Java where functions are not first-class objects (Java has method references and lambdas since Java 8, but they are more limited than Python's approach).
