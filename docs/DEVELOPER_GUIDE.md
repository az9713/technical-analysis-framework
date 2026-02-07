# TA Framework Developer Guide

A comprehensive, step-by-step guide for developers who are new to Python, pandas, web applications, or financial software. If you have experience with C, C++, or Java, this guide maps every concept to what you already know.

---

## Table of Contents

- [Chapter 1: Setting Up Your Development Environment](#chapter-1-setting-up-your-development-environment)
- [Chapter 2: Understanding the Project Structure](#chapter-2-understanding-the-project-structure)
- [Chapter 3: Core Foundation](#chapter-3-core-foundation)
- [Chapter 4: Data Layer](#chapter-4-data-layer)
- [Chapter 5: Indicators](#chapter-5-indicators)
- [Chapter 6: Signals](#chapter-6-signals)
- [Chapter 7: Backtesting](#chapter-7-backtesting)
- [Chapter 8: Risk Management](#chapter-8-risk-management)
- [Chapter 9: Analytics](#chapter-9-analytics)
- [Chapter 10: Visualization](#chapter-10-visualization)
- [Chapter 11: Streamlit UI](#chapter-11-streamlit-ui)
- [Chapter 12: Testing Guide](#chapter-12-testing-guide)
- [Chapter 13: Adding New Features (Tutorials)](#chapter-13-adding-new-features-tutorials)
- [Chapter 14: Troubleshooting](#chapter-14-troubleshooting)
- [Appendix A: Python Quick Reference for C++/Java Developers](#appendix-a-python-quick-reference)
- [Appendix B: pandas Quick Reference](#appendix-b-pandas-quick-reference)
- [Appendix C: Financial Terms Glossary](#appendix-c-financial-terms-glossary)

---

## Chapter 1: Setting Up Your Development Environment

### 1.1 Prerequisites

You need:
- **Python 3.10+** -- the programming language (like Java's JDK or a C++ compiler)
- **pip** -- Python's package manager (like Maven for Java, NuGet for C#, or apt-get for Linux)
- **Git** -- version control (you likely already have this)

### 1.2 Installing Python on Windows

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click "Download Python 3.1x.x" (latest 3.10+ version)
3. Run the installer
4. **CRITICAL**: Check the box "Add Python to PATH" at the bottom of the installer
5. Click "Install Now"
6. When finished, open Command Prompt (press Windows key, type `cmd`, press Enter)
7. Verify Python:
   ```
   python --version
   ```
   You should see `Python 3.10.x` or higher.
8. Verify pip:
   ```
   pip --version
   ```
   You should see `pip 2x.x.x from ...`

If `python` is not recognized, Python was not added to PATH. Uninstall and reinstall, making sure to check the PATH box.

### 1.3 What is a Virtual Environment?

In Java, each project can have its own classpath with specific library versions. Python achieves the same with **virtual environments** -- isolated directories containing a Python interpreter and installed packages, separate from your system Python.

**Why you need one**: Without a virtual environment, all Python projects on your machine share the same packages. If Project A needs pandas 1.5 and Project B needs pandas 2.0, they conflict. A virtual environment gives each project its own isolated set of packages.

Create and activate a virtual environment:

```bash
# Navigate to the project
cd C:\Users\simon\Downloads\pandas_TA_neuralNine

# Create a virtual environment (creates a "venv" folder)
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You'll see (venv) at the start of your prompt:
# (venv) C:\Users\simon\Downloads\pandas_TA_neuralNine>
```

**The `(venv)` prefix means you're inside the virtual environment.** All `pip install` commands will install packages into this isolated space, not system-wide.

To deactivate later: `deactivate`

### 1.4 Installing the Project

With your virtual environment activated:

```bash
# Install all runtime dependencies
pip install -r requirements.txt

# Install the project itself in "editable" mode (changes to code take effect immediately)
pip install -e ".[dev]"
```

**What does `-e` mean?** Editable install. Normally `pip install` copies your code into the Python packages directory. With `-e`, it creates a link instead, so when you edit source files, the changes are immediately available without reinstalling. This is like a symlink.

**What does `[dev]` mean?** Extra dependencies for development. Our `pyproject.toml` defines optional dependency groups. `[dev]` installs pytest and pytest-cov for testing.

### 1.5 Verifying Your Installation

```bash
# Test that the package imports correctly
python -c "import ta_framework; print(ta_framework.__version__)"
# Expected output: 0.1.0

# Run all tests (should show 175 passed)
pytest tests/ -v

# Start the web application
streamlit run app.py
# This opens http://localhost:8501 in your browser
# Press Ctrl+C in the terminal to stop
```

### 1.6 Recommended IDE Setup

**VS Code** (recommended):
1. Install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Open the project folder: File > Open Folder > select `pandas_TA_neuralNine`
3. Install the "Python" extension by Microsoft
4. Press `Ctrl+Shift+P`, type "Python: Select Interpreter"
5. Choose the interpreter inside your `venv` folder
6. Open a terminal in VS Code: Terminal > New Terminal (it auto-activates your venv)

**PyCharm Community Edition** (alternative):
1. Download from [https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/)
2. Open the project folder
3. Configure the Python interpreter to use your venv

### 1.7 Common Installation Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| `python is not recognized` | Python not in PATH | Reinstall Python, check "Add to PATH" |
| `pip is not recognized` | Same as above | Same as above |
| `No module named 'ta_framework'` | Package not installed | Run `pip install -e .` |
| `No module named 'pandas_ta'` | Wrong package name | Run `pip install pandas-ta-classic` (not `pandas-ta`) |
| `SSL certificate errors` | Corporate proxy/firewall | Run `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt` |
| `Permission denied` | Running without venv as admin | Use a virtual environment |
| `Address already in use` (Streamlit) | Another instance running | Close the other terminal or use `streamlit run app.py --server.port 8502` |

---

## Chapter 2: Understanding the Project Structure

### 2.1 Directory Layout

```
pandas_TA_neuralNine/
├── app.py                          # Streamlit entry point (run with: streamlit run app.py)
├── pyproject.toml                  # Package metadata + dependencies (like pom.xml)
├── requirements.txt                # Pinned dependency versions
├── CLAUDE.md                       # AI assistant project conventions
│
├── ta_framework/                   # The Python package (importable as: import ta_framework)
│   ├── __init__.py                 # Package root: version, public exports
│   │
│   ├── core/                       # Foundation types used by all modules
│   │   ├── __init__.py
│   │   ├── types.py                # Enums: AssetClass, Timeframe, SignalDirection
│   │   │                           # Dataclasses: Signal, Trade, BacktestConfig
│   │   ├── config.py               # FrameworkConfig singleton
│   │   ├── registry.py             # Plugin registry (indicator_registry, strategy_registry, provider_registry)
│   │   └── exceptions.py           # Exception hierarchy (DataError, IndicatorError, etc.)
│   │
│   ├── data/                       # Data fetching, validation, and transformation
│   │   ├── __init__.py
│   │   ├── base.py                 # ABC: DataProvider (fetch, validate, supported_assets)
│   │   ├── yfinance_provider.py    # Yahoo Finance implementation (stocks, ETFs, crypto)
│   │   ├── ccxt_provider.py        # Crypto exchanges via CCXT (optional)
│   │   ├── csv_provider.py         # CSV/Parquet file import
│   │   ├── quality.py              # Data quality checks (gaps, outliers, OHLC consistency)
│   │   └── timeframe.py            # Resampling (daily->weekly) and multi-TF alignment
│   │
│   ├── indicators/                 # Technical indicator computation
│   │   ├── __init__.py
│   │   ├── engine.py               # IndicatorEngine: compute(), compute_batch(), register()
│   │   ├── catalog.py              # INDICATOR_CATALOG: metadata for 38 Tier 1 indicators
│   │   ├── wrappers.py             # Typed wrappers around pandas_ta functions
│   │   ├── custom.py               # CustomIndicator ABC + @register_indicator decorator
│   │   └── composite.py            # CompositeIndicator: chain indicators (RSI of SMA)
│   │
│   ├── signals/                    # Trading signal generation
│   │   ├── __init__.py
│   │   ├── base.py                 # ABC: SignalGenerator (generate, required_indicators)
│   │   ├── rules.py                # Signal rules: crossover, threshold, divergence, breakout
│   │   ├── composite.py            # CompositeSignal: combine generators (vote/weight/confirm)
│   │   └── strategies.py           # 6 pre-built strategies: EMA Cross, RSI, MACD, BB, etc.
│   │
│   ├── backtest/                   # Strategy backtesting
│   │   ├── __init__.py
│   │   ├── engine.py               # VectorizedBacktester: core simulation engine
│   │   ├── costs.py                # Commission + slippage models
│   │   ├── results.py              # BacktestResult: equity curve, trades, metrics
│   │   ├── optimization.py         # GridSearch + WalkForward parameter optimization
│   │   └── monte_carlo.py          # Bootstrap resampling, trade shuffling
│   │
│   ├── risk/                       # Risk management
│   │   ├── __init__.py
│   │   ├── position_sizing.py      # FixedFractional, Kelly, VolatilityBased, RiskParity
│   │   ├── var.py                  # VaR (parametric, historical, Monte Carlo), CVaR
│   │   ├── stops.py                # Stop-loss: fixed, ATR, trailing, chandelier
│   │   └── portfolio.py            # Portfolio-level risk metrics
│   │
│   ├── analytics/                  # Performance analytics
│   │   ├── __init__.py
│   │   ├── metrics.py              # 25+ metrics: Sharpe, Sortino, Calmar, Omega, etc.
│   │   ├── tearsheet.py            # TearSheet: comprehensive performance report
│   │   ├── benchmark.py            # Alpha, beta, information ratio vs benchmark
│   │   └── regime.py               # Market regime detection (KMeans, optional HMM)
│   │
│   ├── viz/                        # Visualization (Plotly charts)
│   │   ├── __init__.py
│   │   ├── charts.py               # Candlestick, indicator panels, multi-panel charts
│   │   ├── drawdown.py             # Drawdown/underwater charts
│   │   ├── heatmaps.py             # Correlation + monthly returns heatmaps
│   │   ├── distribution.py         # Return histograms, QQ plots
│   │   └── trade_plots.py          # Trade markers on charts, P&L visualization
│   │
│   └── pages/                      # Streamlit UI pages
│       ├── __init__.py
│       ├── components.py           # Shared widgets: symbol input, strategy select, etc.
│       ├── dashboard.py            # Data + indicators + interactive chart
│       ├── backtester.py           # Strategy testing + results display
│       ├── compare.py              # Multi-asset and multi-strategy comparison
│       └── analysis.py             # Tear sheet, regime detection, risk analysis
│
├── tests/                          # Test suite (175 tests)
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures: sample_ohlcv, short_ohlcv, sample_signals
│   ├── test_data_providers.py      # Tests for data module (29 tests)
│   ├── test_indicators.py          # Tests for indicators (28 tests)
│   ├── test_signals.py             # Tests for signals (22 tests)
│   ├── test_backtest.py            # Tests for backtesting (28 tests)
│   ├── test_risk.py                # Tests for risk management (31 tests)
│   └── test_analytics.py          # Tests for analytics (37 tests)
│
└── docs/                           # Documentation
    ├── ARCHITECTURE.md             # Architecture deep dive for C++/Java developers
    ├── DEVELOPER_GUIDE.md          # This file
    ├── USER_GUIDE.md               # End-user documentation
    └── QUICKSTART.md               # 10 use cases to get started
```

### 2.2 How Python Packages Work

**In Java**, you organize code into packages with `package com.example.myapp;` and each directory needs no special file.

**In Python**, a package is a directory containing a file called `__init__.py`. This file is executed when the package is imported and typically defines what's publicly available.

```python
# When you write:
from ta_framework.indicators import IndicatorEngine

# Python does:
# 1. Finds ta_framework/ directory
# 2. Executes ta_framework/__init__.py
# 3. Finds ta_framework/indicators/ directory
# 4. Executes ta_framework/indicators/__init__.py
# 5. Finds IndicatorEngine in that __init__.py's exports
```

Our `ta_framework/indicators/__init__.py` contains:
```python
from ta_framework.indicators.engine import IndicatorEngine
```
This re-exports `IndicatorEngine` so users can import from the package directly instead of the specific module.

**C++ comparison**: `__init__.py` is like a header file that `#include`s other headers to create a convenient single import point.

### 2.3 How Python Imports Work

| Python | Java Equivalent | C++ Equivalent |
|--------|----------------|----------------|
| `import pandas` | `import java.util.*;` | `#include <pandas>` |
| `from pandas import DataFrame` | `import java.util.ArrayList;` | `using std::vector;` |
| `import pandas as pd` | (no direct equivalent) | `namespace pd = pandas;` |
| `from ta_framework.core.types import Signal` | `import com.ta.core.types.Signal;` | `#include "core/types.h"` |

**Common mistake: circular imports.** If module A imports module B and module B imports module A, Python raises an error. We avoid this by having a clear dependency hierarchy:
```
core/ <- data/, indicators/, signals/, backtest/, risk/, analytics/, viz/
data/ <- pages/
indicators/ <- pages/
signals/ <- backtest/, pages/
backtest/ <- pages/
```

### 2.4 The pyproject.toml File

This is like `pom.xml` (Maven) or `build.gradle` (Gradle). It defines:

```toml
[project]
name = "ta-framework"          # Package name (like artifactId in Maven)
version = "0.1.0"              # Semantic version
requires-python = ">=3.10"     # Minimum Python version

dependencies = [               # Runtime dependencies (like <dependencies> in pom.xml)
    "pandas>=2.0,<3.0",       # Version ranges supported
    "numpy>=1.24",
    ...
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]          # Like Maven profiles or Gradle configurations
crypto = ["ccxt>=4.0"]

[tool.pytest.ini_options]      # Tool-specific configuration
testpaths = ["tests"]
```

---

## Chapter 3: Core Foundation

The `core/` module defines the types and infrastructure that ALL other modules depend on. Nothing in `core/` imports from other modules.

### 3.1 types.py -- The Type System

#### Enums

**AssetClass** -- What kind of financial instrument:
```python
class AssetClass(str, Enum):
    EQUITY = "equity"      # Stocks: AAPL, MSFT, GOOGL
    CRYPTO = "crypto"      # Cryptocurrency: BTC, ETH
    FOREX = "forex"        # Foreign exchange: EUR/USD
    FUTURES = "futures"    # Derivative contracts
    ETF = "etf"            # Exchange-Traded Funds: SPY, QQQ
    INDEX = "index"        # Market indices: S&P 500
```

**Timeframe** -- How data is grouped by time:
```python
class Timeframe(str, Enum):
    M1 = "1m"    # 1-minute bars (intraday)
    M5 = "5m"    # 5-minute bars
    M15 = "15m"  # 15-minute bars
    M30 = "30m"  # 30-minute bars
    H1 = "1h"    # 1-hour bars
    H4 = "4h"    # 4-hour bars
    D1 = "1d"    # Daily bars (most common)
    W1 = "1wk"   # Weekly bars
    MN1 = "1mo"  # Monthly bars
```

Each timeframe has helper methods:
- `to_pandas_freq()`: converts to pandas resampling frequency string
- `to_minutes()`: returns the period in minutes (useful for comparisons)

**SignalDirection** -- Trading actions:
- `LONG`: buy (expect price to go UP)
- `SHORT`: sell (expect price to go DOWN)
- `NEUTRAL`: do nothing

#### Dataclasses

Python `@dataclass` is like a Java `record` or C++ `struct` with auto-generated `__init__`, `__repr__`, and `__eq__`.

**Signal** -- A single trading signal:
```python
@dataclass
class Signal:
    timestamp: datetime          # When the signal was generated
    direction: SignalDirection   # LONG, SHORT, or NEUTRAL
    strength: float = 0.0       # 0.0 to 1.0 (how strong the signal is)
    confidence: float = 0.0     # 0.0 to 1.0 (how confident we are)
    source: str = ""            # Which strategy generated it
    metadata: dict = field(default_factory=dict)  # Extra data
```

**Trade** -- A completed buy+sell pair:
```python
@dataclass
class Trade:
    entry_time: datetime         # When we bought/sold
    exit_time: datetime | None   # When we closed the position
    entry_price: float           # Price at entry
    exit_price: float            # Price at exit
    quantity: float              # How many shares/units
    direction: SignalDirection   # LONG or SHORT
    pnl: float                  # Profit/loss before costs
    commission: float            # Trading fees paid
    slippage: float              # Execution cost
    holding_period: int          # Number of bars held

    @property
    def net_pnl(self) -> float:       # P&L after costs
        return self.pnl - self.commission - self.slippage

    @property
    def is_winner(self) -> bool:       # Did this trade make money?
        return self.net_pnl > 0
```

**BacktestConfig** -- Settings for a backtest:
```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0    # Starting money ($100,000)
    commission_pct: float = 0.001         # 0.1% per trade
    slippage_pct: float = 0.0005          # 0.05% execution cost
    position_size_pct: float = 1.0        # Use 100% of capital per trade
    max_positions: int = 1                # One position at a time
    risk_free_rate: float = 0.04          # 4% annual (for Sharpe ratio)
    allow_short: bool = True              # Allow short selling
```

### 3.2 registry.py -- The Plugin System

The registry pattern allows new indicators, strategies, and data providers to be added without modifying existing code.

**Java equivalent**: Think of it like Spring's `@Component` + `ApplicationContext`, or Java's `ServiceLoader`.

```python
# How the Registry works:

# 1. Create a registry
indicator_registry = Registry("indicators")

# 2. Register something using a decorator (like @Component in Spring)
@indicator_registry.register("sma")
def compute_sma(df, length=20):
    ...

# 3. Retrieve it later
func = indicator_registry.get("sma")  # Returns the compute_sma function
result = func(df, length=50)

# 4. List all registered items
indicator_registry.keys()  # ["sma", "ema", "rsi", ...]
```

**Three global registries**:
- `indicator_registry` -- technical indicators (SMA, RSI, MACD, etc.)
- `strategy_registry` -- trading strategies (EMA Cross, RSI Threshold, etc.)
- `provider_registry` -- data providers (yfinance, csv, ccxt)

### 3.3 exceptions.py -- Error Hierarchy

```
TAFrameworkError (base)
├── DataError
│   ├── InvalidSymbolError       # Ticker not found
│   ├── DataQualityError         # Data fails quality checks
│   └── InsufficientDataError    # Not enough bars for computation
├── IndicatorError
│   └── InvalidParameterError    # Bad indicator parameters
├── SignalError                  # Signal generation failure
├── BacktestError                # Backtesting failure
├── ConfigError                  # Configuration problems
└── RegistryError                # Registry lookup failure
```

**Catching errors**:
```python
from ta_framework.core.exceptions import InvalidSymbolError, IndicatorError

try:
    df = provider.fetch("INVALID_TICKER", Timeframe.D1, "2020-01-01")
except InvalidSymbolError as e:
    print(f"Symbol not found: {e}")
except IndicatorError as e:
    print(f"Indicator computation failed: {e}")
```

---

## Chapter 4: Data Layer

### 4.1 What is OHLCV Data?

Financial markets record price data as **OHLCV** for each time period:

| Field | Meaning | Example |
|-------|---------|---------|
| **Open** | Price at the start of the period | $150.00 |
| **High** | Highest price during the period | $155.50 |
| **Low** | Lowest price during the period | $149.25 |
| **Close** | Price at the end of the period | $153.00 |
| **Volume** | Number of shares/units traded | 45,000,000 |

For daily data, each row represents one trading day. For hourly data, each row is one hour, and so on.

### 4.2 What is a pandas DataFrame?

For C++/Java developers, a `DataFrame` is:
- An **in-memory table** (like a SQL table, but in RAM)
- A **2D array with named columns** and an index (like a primary key)
- The universal data container in this project

```python
import pandas as pd

# A DataFrame looks like this when printed:
#              open    high     low   close     volume
# date
# 2024-01-02  150.0   155.5  149.3   153.0  45000000
# 2024-01-03  153.0   157.2  152.1   156.5  38000000
# 2024-01-04  156.5   158.0  154.8   155.0  42000000

# Access a column (returns a Series - like a 1D array with an index):
prices = df["close"]          # Series: [153.0, 156.5, 155.0]

# Access multiple columns:
subset = df[["open", "close"]]

# Access by position:
first_row = df.iloc[0]        # First row
last_row = df.iloc[-1]        # Last row

# Access by index label:
row = df.loc["2024-01-03"]    # Row for that date

# Common operations:
df["close"].mean()             # Average close price
df["close"].pct_change()       # Daily return percentages
df["close"].rolling(20).mean() # 20-day moving average
```

### 4.3 The DataProvider ABC

All data providers implement this interface:

```python
class DataProvider(ABC):
    @abstractmethod
    def fetch(self, symbol, timeframe, start, end) -> pd.DataFrame:
        """Fetch OHLCV data. Returns DataFrame with DatetimeIndex."""

    @abstractmethod
    def supported_assets(self) -> list[AssetClass]:
        """What asset types this provider supports."""

    @abstractmethod
    def search_symbols(self, query) -> list[dict]:
        """Search for symbols matching a query."""

    def validate(self, df) -> pd.DataFrame:
        """Normalize column names, sort by date, check for OHLCV columns."""
```

**The contract**: Every provider returns a DataFrame with:
- `DatetimeIndex` named `"date"`
- Columns: `open`, `high`, `low`, `close`, `volume` (all lowercase)

### 4.4 YFinanceProvider

The primary data provider. Uses Yahoo Finance (free, no API key needed).

```python
from ta_framework.data.yfinance_provider import YFinanceProvider
from ta_framework.core.types import Timeframe

provider = YFinanceProvider()

# Fetch Apple daily data for the last 5 years
df = provider.fetch("AAPL", Timeframe.D1, "2020-01-01", "2025-01-01")

# Available symbols:
# Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA
# Crypto: BTC-USD, ETH-USD, SOL-USD
# ETFs: SPY, QQQ, IWM, GLD
# Forex: EURUSD=X, GBPUSD=X, JPYUSD=X
# Indices: ^GSPC (S&P 500), ^IXIC (NASDAQ)
```

### 4.5 CSVProvider

Load your own data from CSV or Parquet files:

```python
from ta_framework.data.csv_provider import CSVProvider

provider = CSVProvider("path/to/data.csv")
df = provider.fetch("MY_DATA", Timeframe.D1, "2020-01-01")
```

**Required CSV format:**
```csv
date,open,high,low,close,volume
2024-01-02,150.0,155.5,149.3,153.0,45000000
2024-01-03,153.0,157.2,152.1,156.5,38000000
```

Column names are case-insensitive (Open, OPEN, open all work -- they get normalized to lowercase).

### 4.6 DataQualityChecker

Validates and cleans data before analysis:

```python
from ta_framework.data.quality import DataQualityChecker

checker = DataQualityChecker()

# Check for issues
report = checker.full_check(df)
# Returns: {"gaps": {...}, "outliers": {...}, "ohlc": {...}}

# Clean the data
clean_df = checker.clean(df, fill_method="ffill", remove_outliers=True)
```

### 4.7 Tutorial: Creating Your Own Data Provider

Here's how to create a provider for Alpha Vantage (or any API):

```python
# File: ta_framework/data/alphavantage_provider.py

from ta_framework.data.base import DataProvider
from ta_framework.core.types import AssetClass, Timeframe
from ta_framework.core.registry import provider_registry
import pandas as pd

@provider_registry.register("alphavantage")
class AlphaVantageProvider(DataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, symbol, timeframe, start, end=None):
        # Your API call logic here
        # Must return a DataFrame with DatetimeIndex and OHLCV columns
        ...
        return self.validate(df)  # Always validate before returning

    def supported_assets(self):
        return [AssetClass.EQUITY, AssetClass.FOREX]

    def search_symbols(self, query):
        return [{"symbol": query, "name": "Unknown"}]
```

---

## Chapter 5: Indicators

### 5.1 What Are Technical Indicators?

Technical indicators are mathematical calculations applied to price/volume data that help identify patterns:

- **Moving Averages (SMA, EMA)**: Smooth out price noise to reveal trends. An SMA_20 is the average of the last 20 closing prices.
- **RSI (Relative Strength Index)**: Measures momentum on a 0-100 scale. Above 70 = overbought (might fall). Below 30 = oversold (might rise).
- **MACD**: Uses the difference between two moving averages to detect trend changes and momentum.
- **Bollinger Bands**: Price envelope that shows the "normal" range of price movement (mean +/- 2 standard deviations).
- **ATR (Average True Range)**: Measures how much price typically moves per period -- a measure of volatility.

### 5.2 The IndicatorEngine

Central hub for computing indicators:

```python
from ta_framework.indicators.engine import IndicatorEngine

engine = IndicatorEngine()

# Compute a single indicator
df = engine.compute(df, "sma", length=20)      # Adds column "SMA_20"
df = engine.compute(df, "rsi", length=14)      # Adds column "RSI_14"

# Compute multiple at once
df = engine.compute_batch(df, [
    {"name": "sma", "params": {"length": 20}},
    {"name": "rsi", "params": {"length": 14}},
    {"name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}},
])

# List all available indicators
print(engine.available())  # ['adx', 'aroon', 'atr', 'bbands', ...]

# Register a custom function at runtime
def my_indicator(df, length=10):
    df[f"MY_{length}"] = df["close"].rolling(length).mean() * 2
    return df

engine.register("my_custom", my_indicator)
```

### 5.3 The Indicator Catalog (38 Tier 1 Indicators)

| Name | Category | What It Measures | Output Columns |
|------|----------|-----------------|----------------|
| sma | Trend | Simple Moving Average | SMA_{length} |
| ema | Trend | Exponential Moving Average | EMA_{length} |
| wma | Trend | Weighted Moving Average | WMA_{length} |
| dema | Trend | Double EMA | DEMA_{length} |
| tema | Trend | Triple EMA | TEMA_{length} |
| hma | Trend | Hull Moving Average | HMA_{length} |
| kama | Trend | Kaufman Adaptive MA | KAMA_{length} |
| t3 | Trend | Tillson T3 | T3_{length} |
| supertrend | Trend | Trend direction | SUPERT_{length}_{mult}, SUPERTd_{length}_{mult} |
| ichimoku | Trend | Cloud indicator | ISA_9, ISB_26, ITS_9, IKS_26 |
| adx | Trend | Trend strength (0-100) | ADX_{length} |
| aroon | Trend | Trend direction | AROONU_{length}, AROOND_{length} |
| psar | Trend | Parabolic Stop and Reverse | PSARl, PSARs |
| vwma | Trend | Volume-Weighted MA | VWMA_{length} |
| rsi | Momentum | Relative Strength Index | RSI_{length} |
| macd | Momentum | Moving Average Convergence Divergence | MACD_{f}_{s}_{sig}, MACDs_{f}_{s}_{sig}, MACDh_{f}_{s}_{sig} |
| stoch | Momentum | Stochastic Oscillator | STOCHk_{k}_{d}_{smooth}, STOCHd_{k}_{d}_{smooth} |
| cci | Momentum | Commodity Channel Index | CCI_{length} |
| willr | Momentum | Williams %R | WILLR_{length} |
| roc | Momentum | Rate of Change | ROC_{length} |
| mfi | Momentum | Money Flow Index | MFI_{length} |
| tsi | Momentum | True Strength Index | TSI_{fast}_{slow} |
| uo | Momentum | Ultimate Oscillator | UO |
| ao | Momentum | Awesome Oscillator | AO |
| bbands | Volatility | Bollinger Bands | BBL, BBM, BBU, BBB, BBP |
| atr | Volatility | Average True Range | ATR_{length} |
| kc | Volatility | Keltner Channel | KCLe, KCBe, KCUe |
| donchian | Volatility | Donchian Channel | DCL, DCM, DCU |
| true_range | Volatility | True Range | TRUERANGE |
| stdev | Volatility | Standard Deviation | STDEV_{length} |
| obv | Volume | On-Balance Volume | OBV |
| vwap | Volume | Volume-Weighted Avg Price | VWAP |
| ad | Volume | Accumulation/Distribution | AD |
| cmf | Volume | Chaikin Money Flow | CMF_{length} |
| fi | Volume | Force Index | FI_{length} |
| sma_volume | Volume | Volume SMA | SMA_VOLUME_{length} |
| pivots | Overlap | Pivot Points | PP, S1, S2, R1, R2 |
| fib | Overlap | Fibonacci Levels | FIB_236, FIB_382, FIB_500, FIB_618 |

### 5.4 Tutorial: Creating a Custom Indicator

Let's create a "Momentum Ratio" indicator that compares short-term momentum to long-term momentum:

```python
# In ta_framework/indicators/custom.py (or a new file)

from ta_framework.indicators.custom import CustomIndicator, register_indicator
from ta_framework.core.types import IndicatorCategory
import pandas as pd

@register_indicator
class MomentumRatio(CustomIndicator):
    name = "momentum_ratio"
    category = IndicatorCategory.MOMENTUM

    def compute(self, df: pd.DataFrame, short=5, long=20) -> pd.DataFrame:
        short_mom = df["close"].pct_change(short)
        long_mom = df["close"].pct_change(long)
        df[f"MOMRATIO_{short}_{long}"] = short_mom / long_mom.replace(0, float("nan"))
        return df

    @property
    def output_columns(self) -> list[str]:
        return ["MOMRATIO_5_20"]
```

Now use it:
```python
engine = IndicatorEngine()
df = engine.compute(df, "momentum_ratio", short=5, long=20)
```

Write a test:
```python
# In tests/test_indicators.py
def test_momentum_ratio(sample_ohlcv):
    engine = IndicatorEngine()
    df = engine.compute(sample_ohlcv, "momentum_ratio", short=5, long=20)
    assert "MOMRATIO_5_20" in df.columns
    assert not df["MOMRATIO_5_20"].isna().all()
```

---

## Chapter 6: Signals

### 6.1 What is a Trading Signal?

A signal tells the system WHEN to act:
- `1` = **Buy/Go Long** (expect price to rise)
- `-1` = **Sell/Go Short** (expect price to fall)
- `0` = **Do nothing** (no clear direction)

### 6.2 The SignalGenerator ABC

Every strategy implements this interface:

```python
class SignalGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'signal' column (1/-1/0) to df.
        Assumes indicator columns already exist."""

    @property
    @abstractmethod
    def required_indicators(self) -> list[dict]:
        """What indicators must be computed before generate() is called.
        Example: [{'name': 'ema', 'length': 20}]"""
```

**Key design decision**: Strategies do NOT compute indicators themselves. They assume indicator columns already exist in the DataFrame. This separation allows:
- Reusing indicators across multiple strategies
- Computing indicators once, using them many times
- Mixing and matching indicators freely

### 6.3 Signal Rules

Reusable building blocks for strategies:

```python
from ta_framework.signals.rules import crossover, threshold, breakout

# Crossover: signal when fast line crosses slow line
signals = crossover(df["EMA_12"], df["EMA_26"])
# Returns: 1 where EMA_12 crosses ABOVE EMA_26
#         -1 where EMA_12 crosses BELOW EMA_26
#          0 otherwise

# Threshold: signal at extreme values
signals = threshold(df["RSI_14"], upper=70, lower=30)
# Returns: 1 when RSI drops below 30 (oversold -> buy)
#         -1 when RSI rises above 70 (overbought -> sell)
#          0 in between

# Breakout: signal at band boundaries
signals = breakout(df["close"], df["BBU_20_2.0"], df["BBL_20_2.0"])
# Returns: 1 when close breaks below lower band (mean reversion buy)
#         -1 when close breaks above upper band (mean reversion sell)
```

### 6.4 Pre-Built Strategies

| Strategy | Logic | Best For |
|----------|-------|----------|
| **EMACrossStrategy** | Buy when fast EMA > slow EMA, sell when fast < slow | Trending markets |
| **RSIStrategy** | Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought) | Range-bound markets |
| **MACDStrategy** | Buy when MACD crosses above signal line, sell when below | Trend + momentum |
| **BollingerBandStrategy** | Buy at lower band, sell at upper band | Mean reversion |
| **SupertrendStrategy** | Follow supertrend direction changes | Strong trends |
| **TTMSqueezeStrategy** | Trade when volatility squeeze releases | Breakouts |

### 6.5 Tutorial: Creating Your Own Strategy

Let's create a "Golden Cross" strategy (50-day SMA crosses above 200-day SMA):

```python
# In ta_framework/signals/strategies.py

from ta_framework.core.registry import strategy_registry
from ta_framework.signals.base import SignalGenerator
from ta_framework.signals.rules import crossover

@strategy_registry.register("golden_cross")
class GoldenCrossStrategy(SignalGenerator):
    name = "golden_cross"

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def required_indicators(self) -> list[dict]:
        return [
            {"name": "sma", "length": self.fast_period},
            {"name": "sma", "length": self.slow_period},
        ]

    def generate(self, df):
        fast_col = f"SMA_{self.fast_period}"
        slow_col = f"SMA_{self.slow_period}"
        self.validate_columns(df, [fast_col, slow_col])

        df["signal"] = crossover(df[fast_col], df[slow_col])
        return df
```

Don't forget to add it to the strategy list in `pages/components.py` if you want it in the UI.

---

## Chapter 7: Backtesting

### 7.1 What is Backtesting?

Backtesting answers: "If I had used this strategy over the past N years, how much money would I have made or lost?"

You take historical data, apply your strategy's buy/sell rules, simulate the trades, and measure performance. It's not a guarantee of future results, but it helps filter out strategies that would have failed historically.

### 7.2 Vectorized vs Event-Driven

| Approach | How It Works | Speed | Realism |
|----------|-------------|-------|---------|
| **Event-Driven** | Process each bar one at a time, maintaining full state | Slow | High |
| **Vectorized** | Apply operations to entire arrays at once | Fast (100-1000x) | Medium |

We use vectorized backtesting because this is a research tool, not a live trading system. The speed advantage lets you test thousands of parameter combinations quickly.

**C++ analogy**: Event-driven is like a `for` loop processing one element at a time. Vectorized is like SIMD operations or NumPy array operations that process all elements simultaneously.

### 7.3 The VectorizedBacktester

```python
from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.core.types import BacktestConfig

config = BacktestConfig(
    initial_capital=100_000,     # Start with $100K
    commission_pct=0.001,        # 0.1% commission per trade
    slippage_pct=0.0005,         # 0.05% slippage
    position_size_pct=1.0,       # Use 100% of capital per trade
    allow_short=True,            # Allow short selling
)

backtester = VectorizedBacktester(config)
result = backtester.run(df)  # df must have 'close' and 'signal' columns

# Access results
print(result.total_return)    # e.g., 0.25 = 25% return
print(result.max_drawdown)    # e.g., 0.15 = 15% worst decline
print(result.num_trades)      # Number of completed trades
print(result.win_rate)        # Fraction of profitable trades
print(result.sharpe_ratio)    # Risk-adjusted return
```

### 7.4 BacktestResult

The `result` object contains everything:
- `result.equity_curve` -- a pandas Series of portfolio value at each bar
- `result.trades` -- a list of `Trade` objects (every buy/sell pair)
- `result.summary()` -- a dictionary of all key metrics
- `result.to_dataframe()` -- trades as a DataFrame for easy viewing

### 7.5 Optimization

**Grid Search** -- Try every parameter combination:

```python
from ta_framework.backtest.optimization import GridSearchOptimizer
from ta_framework.signals.strategies import EMACrossStrategy

# Pre-compute ALL indicator columns you might need
import pandas_ta as ta
for period in [5, 10, 15, 20, 25, 30]:
    df[f"EMA_{period}"] = ta.ema(df["close"], length=period)

optimizer = GridSearchOptimizer(
    strategy_cls=EMACrossStrategy,
    param_grid={
        "fast_period": [5, 10, 15],
        "slow_period": [20, 25, 30],
    },
    metric="sharpe_ratio",  # Optimize for this metric
)

report = optimizer.run(df)
print(report.best.params)   # Best parameter combination
print(report.best.metrics)  # Its performance metrics
print(report.to_dataframe()) # All results as a table
```

**Walk-Forward Optimization** -- More realistic testing that avoids overfitting:

```python
from ta_framework.backtest.optimization import WalkForwardOptimizer

wf = WalkForwardOptimizer(
    strategy_cls=EMACrossStrategy,
    param_grid={"fast_period": [5, 10, 15], "slow_period": [20, 25, 30]},
    in_sample_pct=0.7,   # Use 70% for finding best params
    n_windows=4,          # Roll through 4 windows
)
report = wf.run(df)
```

### 7.6 Monte Carlo Simulation

Test whether your results are due to skill or luck:

```python
from ta_framework.backtest.monte_carlo import MonteCarloSimulator

mc = MonteCarloSimulator(seed=42)

# Resample returns - would the strategy work with different return sequences?
paths = mc.bootstrap_returns(result.equity_curve, n_simulations=1000)

# Shuffle trades - does the order of trades matter?
final_equities = mc.shuffle_trades(result.trades, n_simulations=1000)

# Get confidence intervals
ci = mc.confidence_intervals(final_equities)
# {0.05: 85000, 0.25: 95000, 0.5: 102000, 0.75: 110000, 0.95: 125000}
```

---

## Chapter 8: Risk Management

### 8.1 Position Sizing

How much capital to allocate per trade:

```python
from ta_framework.risk.position_sizing import FixedFractional, KellyCriterion, VolatilityBased

# Risk 2% of capital per trade
sizer = FixedFractional()
shares = sizer.calculate(capital=100_000, price=150.0, risk_per_trade=0.02)
# Result: number of shares to buy

# Kelly Criterion (mathematically optimal, often too aggressive)
kelly = KellyCriterion()
shares = kelly.calculate(
    capital=100_000, price=150.0, risk_per_trade=0.02,
    win_rate=0.55, payoff_ratio=1.5
)

# Volatility-based (smaller positions when market is volatile)
vol = VolatilityBased()
shares = vol.calculate(
    capital=100_000, price=150.0, risk_per_trade=0.02,
    atr=3.5  # Average True Range
)
```

### 8.2 Value at Risk (VaR)

"What's the most I could lose in a day with 95% confidence?"

```python
from ta_framework.risk.var import parametric_var, historical_var, monte_carlo_var, cvar

returns = df["close"].pct_change().dropna()

# Different VaR methods (all return positive numbers)
p_var = parametric_var(returns, confidence=0.95)    # Assumes normal distribution
h_var = historical_var(returns, confidence=0.95)    # Uses actual historical data
mc_var = monte_carlo_var(returns, confidence=0.95)  # Simulates scenarios

# CVaR: average loss in the worst 5% of days
expected_shortfall = cvar(returns, confidence=0.95)
# CVaR is always >= VaR (it's the average of the tail, not just the boundary)
```

### 8.3 Stop-Loss Strategies

Automatically exit losing positions:

```python
from ta_framework.risk.stops import fixed_stop, atr_stop, trailing_stop, chandelier_stop

# Fixed: exit if price drops 5% from entry
stop_price = fixed_stop(entry_price=150.0, stop_pct=0.05)  # $142.50

# ATR-based: adaptive to market volatility
stop_series = atr_stop(df, multiplier=2.0, period=14)  # Series of stop prices

# Trailing: follows price up, never moves down
stop_series = trailing_stop(df["close"], trail_pct=0.05)

# Chandelier: highest high minus ATR*multiplier
stop_series = chandelier_stop(df, period=22, multiplier=3.0)
```

---

## Chapter 9: Analytics

### 9.1 Performance Metrics (All 25+)

| Metric | What It Measures | Good Value | Formula |
|--------|-----------------|------------|---------|
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 | (return - risk_free) / volatility |
| **Sortino Ratio** | Downside risk-adjusted return | > 1.5 | (return - risk_free) / downside_dev |
| **Calmar Ratio** | Return / max drawdown | > 1.0 | annualized_return / max_drawdown |
| **Omega Ratio** | Gain-weighted probability | > 1.0 | sum(gains) / sum(losses) |
| **Max Drawdown** | Worst peak-to-trough decline | < 20% | (peak - trough) / peak |
| **CAGR** | Compound annual growth rate | > 10% | (end/start)^(1/years) - 1 |
| **Volatility** | Annual standard deviation | < 20% | daily_std * sqrt(252) |
| **Win Rate** | % of profitable trades | > 50% | winners / total_trades |
| **Profit Factor** | Gross profit / gross loss | > 1.5 | sum(wins) / sum(losses) |
| **Expectancy** | Average profit per trade | > 0 | mean(all_trade_pnls) |
| **VaR 95%** | Daily loss limit (95% confidence) | < 3% | 5th percentile of returns |
| **CVaR 95%** | Average worst-case loss | < 5% | mean(worst 5% returns) |
| **Tail Ratio** | Right tail / left tail | > 1.0 | 95th percentile / |5th percentile| |
| **Skewness** | Return distribution asymmetry | > 0 | scipy.stats.skew |
| **Kurtosis** | Fat tails (extreme events) | near 0 | scipy.stats.kurtosis |
| **Recovery Factor** | Net profit / max drawdown | > 2.0 | net_profit / max_dd_amount |

### 9.2 TearSheet

```python
from ta_framework.analytics.tearsheet import TearSheet

ts = TearSheet(
    equity_curve=result.equity_curve,
    trades=result.trades,
    benchmark=spy_equity_curve,  # optional
    risk_free_rate=0.04,
)

report = ts.generate()        # Dict organized by category
table = ts.summary_table()    # DataFrame of all metrics
monthly = ts.monthly_returns() # Year x Month pivot table
dd = ts.drawdown_analysis()   # Top 5 drawdowns with dates
```

### 9.3 Regime Detection

```python
from ta_framework.analytics.regime import RegimeDetector

detector = RegimeDetector()
returns = df["close"].pct_change().dropna()

# Detect 3 regimes: bear (0), neutral (1), bull (2)
regimes = detector.detect_kmeans(returns, n_regimes=3)

# Get statistics per regime
stats = detector.regime_statistics(returns, regimes)
# Returns DataFrame: mean_return, volatility, sharpe, avg_duration per regime
```

---

## Chapter 10: Visualization

All chart functions return `plotly.graph_objects.Figure` objects. These are interactive charts you can zoom, pan, and hover over.

```python
from ta_framework.viz.charts import candlestick_chart, multi_panel_chart
from ta_framework.viz.drawdown import drawdown_chart
from ta_framework.viz.heatmaps import correlation_heatmap, monthly_returns_heatmap
from ta_framework.viz.distribution import returns_histogram
from ta_framework.viz.trade_plots import trade_markers, pnl_chart

# Candlestick chart with overlays
fig = candlestick_chart(df, overlays=["SMA_20", "EMA_50"])
fig.show()  # Opens in browser

# Multi-panel chart: candlestick + volume + oscillators
fig = multi_panel_chart(df, overlays=["EMA_12"], oscillators=["RSI_14"])

# Drawdown underwater chart
fig = drawdown_chart(result.equity_curve)

# Correlation heatmap (needs dict of return series)
fig = correlation_heatmap({"AAPL": aapl_returns, "MSFT": msft_returns})

# Monthly returns heatmap
fig = monthly_returns_heatmap(result.equity_curve)

# Return distribution histogram
fig = returns_histogram(returns)

# Trade markers on price chart
fig = trade_markers(df, result.trades)

# P&L chart
fig = pnl_chart(result.trades)
```

---

## Chapter 11: Streamlit UI

### 11.1 How Streamlit Works

**Traditional web app** (Spring MVC, Express.js):
```
Client (HTML/JS) <-> Server (routes/controllers) <-> Database
```

**Streamlit**:
```
Python script IS the entire app.
Script re-runs top-to-bottom on every user interaction.
No HTML, no CSS, no JavaScript, no routes.
```

```python
# This is a complete Streamlit app:
import streamlit as st

st.title("My App")
name = st.text_input("Your name")  # Renders a text box
if st.button("Greet"):             # Renders a button
    st.write(f"Hello, {name}!")    # Displays text
```

Key concepts:
- `st.sidebar` -- left panel for controls
- `st.columns(N)` -- create N columns for layout
- `st.tabs(["A", "B"])` -- create tabbed views
- `st.expander("Title")` -- collapsible sections
- `@st.cache_data` -- memoize expensive function calls (like `@Cacheable` in Spring)

### 11.2 Adding a New Page

1. Create `ta_framework/pages/mypage.py`:
```python
import streamlit as st

def render():
    st.header("My New Page")
    st.write("Hello from my new page!")
```

2. Import and add to navigation in `app.py`:
```python
from ta_framework.pages import dashboard, backtester, compare, analysis, mypage

page = st.sidebar.radio("Navigate", ["Dashboard", "Backtester", "Compare", "Analysis", "My Page"])
...
elif page == "My Page":
    mypage.render()
```

---

## Chapter 12: Testing Guide

### 12.1 pytest Basics

**Java equivalent**: pytest is like JUnit, but simpler.

```python
# JUnit (Java):
# @Test
# public void testAdd() {
#     assertEquals(4, Calculator.add(2, 2));
# }

# pytest (Python):
def test_add():
    assert Calculator.add(2, 2) == 4
```

No test classes required (but we use them for organization). No `@Test` annotation needed -- just start the function name with `test_`.

### 12.2 Fixtures

Fixtures provide shared test data. Like JUnit's `@Before` setup methods, but more flexible.

```python
# In conftest.py (available to ALL test files):
@pytest.fixture
def sample_ohlcv():
    """252 days of realistic OHLCV data."""
    np.random.seed(42)
    # ... generates DataFrame ...
    return df

# In test files, just use the fixture name as a parameter:
def test_sma(sample_ohlcv):           # sample_ohlcv is auto-injected
    engine = IndicatorEngine()
    df = engine.compute(sample_ohlcv, "sma", length=20)
    assert "SMA_20" in df.columns
```

Our fixtures:
- `sample_ohlcv` -- 252 bars (~1 year) of realistic daily data
- `short_ohlcv` -- 20 bars for quick tests
- `empty_ohlcv` -- empty DataFrame with correct columns
- `sample_signals` -- OHLCV data with pre-computed signal column
- `default_backtest_config` -- BacktestConfig with defaults

### 12.3 Running Tests

```bash
# Run ALL tests (175)
pytest tests/ -v

# Run one file
pytest tests/test_indicators.py -v

# Run one test class
pytest tests/test_indicators.py::TestEngineCompute -v

# Run one specific test
pytest tests/test_indicators.py::TestEngineCompute::test_sma -v

# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# With coverage report
pytest tests/ --cov=ta_framework
```

### 12.4 Writing a New Test

Template:

```python
# tests/test_myfeature.py
import pytest
from ta_framework.my_module import MyClass

class TestMyClass:
    def test_basic_usage(self, sample_ohlcv):
        obj = MyClass()
        result = obj.do_something(sample_ohlcv)
        assert result is not None
        assert "expected_column" in result.columns

    def test_edge_case_empty(self, empty_ohlcv):
        obj = MyClass()
        with pytest.raises(SomeError):
            obj.do_something(empty_ohlcv)

    def test_specific_value(self):
        # Test with known inputs and expected outputs
        result = MyClass.calculate(input_value=100)
        assert result == pytest.approx(42.0, rel=0.01)  # Within 1%
```

### 12.5 Current Test Coverage

| Test File | Tests | What's Covered |
|-----------|-------|---------------|
| test_data_providers.py | 29 | Validation, CSV/Parquet I/O, quality checks, resampling, alignment |
| test_indicators.py | 28 | SMA, EMA, RSI, MACD, BBands, ATR, ADX, Stoch, OBV, batch, composite, custom |
| test_signals.py | 22 | Crossover, threshold, breakout, divergence, EMA/RSI/MACD strategies, composite |
| test_backtest.py | 28 | Engine, costs, results, grid search, Monte Carlo, edge cases |
| test_risk.py | 31 | Position sizing, VaR, stops (fixed/ATR/trailing/chandelier), portfolio metrics |
| test_analytics.py | 37 | All metrics, tear sheet, benchmark, regime detection, calculate_all |

---

## Chapter 13: Adding New Features (Tutorials)

### Tutorial 1: Adding a New Indicator (VWAP)

VWAP (Volume-Weighted Average Price) is already in our catalog, but here's how you'd add a completely new one.

**Step 1**: Add the wrapper function in `indicators/wrappers.py`:
```python
def vwap_custom(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Custom VWAP calculation."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df[f"VWAPC_{length}"] = (
        (typical_price * df["volume"]).rolling(length).sum()
        / df["volume"].rolling(length).sum()
    )
    return df
```

**Step 2**: Add to `WRAPPER_MAP` in the same file:
```python
WRAPPER_MAP["vwap_custom"] = vwap_custom
```

**Step 3**: Add metadata in `indicators/catalog.py`:
```python
"vwap_custom": IndicatorConfig(
    name="vwap_custom",
    category=IndicatorCategory.VOLUME,
    tier=IndicatorTier.TIER1,
    params={"length": 20},
    description="Custom Volume-Weighted Average Price",
),
```

**Step 4**: Write a test in `tests/test_indicators.py`:
```python
def test_vwap_custom(self, sample_ohlcv):
    engine = IndicatorEngine()
    df = engine.compute(sample_ohlcv, "vwap_custom", length=20)
    assert "VWAPC_20" in df.columns
    assert df["VWAPC_20"].notna().sum() > 0
```

**Step 5**: Run tests: `pytest tests/test_indicators.py -v`

### Tutorial 2: Adding a New Strategy (Golden Cross)

**Step 1**: Add to `signals/strategies.py` (see Chapter 6.5 above for full code).

**Step 2**: Update `pages/components.py` strategy list:
```python
strategies = ["ema_cross", "rsi", "macd", "bbands", "supertrend", "ttm_squeeze", "golden_cross"]
labels = {
    ...
    "golden_cross": "Golden Cross (50/200 SMA)",
}
```

**Step 3**: Add parameters in `strategy_params()`:
```python
elif strategy_name == "golden_cross":
    c1, c2 = st.columns(2)
    with c1:
        params["fast_period"] = st.number_input("Fast SMA", 10, 100, 50, key=f"{key_prefix}_fast")
    with c2:
        params["slow_period"] = st.number_input("Slow SMA", 50, 300, 200, key=f"{key_prefix}_slow")
```

**Step 4**: Write tests and verify.

### Tutorial 3: Adding a New Streamlit Page

**Step 1**: Create `ta_framework/pages/optimization.py`:
```python
import streamlit as st

def render():
    st.header("Strategy Optimization")
    st.write("Find the best parameters for your strategy.")
    # Add your UI logic here
```

**Step 2**: Import in `app.py`:
```python
from ta_framework.pages import dashboard, backtester, compare, analysis, optimization

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Backtester", "Compare", "Analysis", "Optimization"],
)
...
elif page == "Optimization":
    optimization.render()
```

**Step 3**: Run `streamlit run app.py` and verify your new page appears.

---

## Chapter 14: Troubleshooting

### Common Errors

**`ModuleNotFoundError: No module named 'ta_framework'`**
- **Cause**: Package not installed in your virtual environment
- **Fix**: `pip install -e .` from the project root
- **Also check**: Is your virtual environment activated? (look for `(venv)` in prompt)

**`ModuleNotFoundError: No module named 'pandas_ta'`**
- **Cause**: Wrong package name
- **Fix**: `pip install pandas-ta-classic` (NOT `pip install pandas-ta`)

**`No data returned for symbol X`**
- **Cause**: Invalid ticker symbol or yfinance rate limit
- **Fix**: Check the symbol on Yahoo Finance website first. Wait a few seconds between requests.

**`SignalError: missing required columns ['EMA_12']`**
- **Cause**: Indicators weren't computed before calling strategy.generate()
- **Fix**: Use IndicatorEngine to compute all required indicators first

**`BacktestError: Signal column 'signal' not found`**
- **Cause**: Strategy's generate() method wasn't called
- **Fix**: Call `df = strategy.generate(df)` before `backtester.run(df)`

**`Address already in use` (Streamlit)**
- **Cause**: Another Streamlit instance is running
- **Fix**: Close the other terminal, or use `streamlit run app.py --server.port 8502`

**pandas `FutureWarning` messages**
- **Cause**: pandas-ta uses some deprecated pandas APIs
- **Fix**: These are warnings, not errors. They don't affect functionality. We pin pandas < 3.0 to avoid breaking changes.

**Tests fail after making changes**
- **Fix**: Run `pytest tests/ -v --tb=long` to see full error details
- Check that you haven't broken the expected column naming convention
- Check that all required fixtures are available

---

## Appendix A: Python Quick Reference

| Concept | C++/Java | Python |
|---------|----------|--------|
| Variable declaration | `int x = 5;` | `x = 5` (no type needed) |
| Type hint (optional) | `int x = 5;` | `x: int = 5` |
| Function | `int add(int a, int b) { return a+b; }` | `def add(a, b): return a + b` |
| Function with types | same | `def add(a: int, b: int) -> int:` |
| Class | `class Dog { ... }` | `class Dog:` |
| Constructor | `Dog(String name) { this.name = name; }` | `def __init__(self, name): self.name = name` |
| Inheritance | `class Puppy extends Dog` | `class Puppy(Dog):` |
| Interface/ABC | `interface Runnable` | `class Runnable(ABC):` |
| this/self | `this.name` | `self.name` |
| null | `null` / `nullptr` | `None` |
| boolean | `true` / `false` | `True` / `False` |
| String format | `String.format("Hi %s", name)` | `f"Hi {name}"` |
| List | `ArrayList<Integer>` | `[1, 2, 3]` |
| Dictionary/Map | `HashMap<String, Integer>` | `{"key": value}` |
| For loop | `for (int i : items)` | `for i in items:` |
| List comprehension | `items.stream().map(x -> x*2).collect(...)` | `[x*2 for x in items]` |
| Try/catch | `try { ... } catch (Exception e)` | `try: ... except Exception as e:` |
| With/resources | `try (var f = new File(...))` | `with open(...) as f:` |
| Decorator | `@Override` / `@Autowired` | `@property` / `@register` |
| No semicolons | `int x = 5;` | `x = 5` |
| Indentation = blocks | `{ ... }` | Indent with 4 spaces |

---

## Appendix B: pandas Quick Reference

```python
import pandas as pd
import numpy as np

# Create a DataFrame
df = pd.DataFrame({
    "close": [100, 102, 101, 105, 103],
    "volume": [1000, 1500, 1200, 1800, 1100],
}, index=pd.date_range("2024-01-01", periods=5))

# Select column (returns Series)
prices = df["close"]

# Select multiple columns
subset = df[["close", "volume"]]

# Filter rows
high_volume = df[df["volume"] > 1200]

# Basic statistics
df["close"].mean()       # Average: 102.2
df["close"].std()        # Standard deviation
df["close"].min()        # Minimum
df["close"].max()        # Maximum

# Time series operations
df["close"].pct_change()           # Daily % change
df["close"].rolling(3).mean()      # 3-period moving average
df["close"].shift(1)               # Previous day's value
df["close"].cummax()               # Running maximum
df["close"].diff()                 # Difference from previous

# Add a new column
df["sma_3"] = df["close"].rolling(3).mean()

# Resampling (daily to weekly)
weekly = df.resample("W").agg({"close": "last", "volume": "sum"})

# Access by position
df.iloc[0]      # First row
df.iloc[-1]     # Last row
df.iloc[1:3]    # Rows 1 and 2

# Access by index label
df.loc["2024-01-01"]

# Shape
df.shape        # (5, 2) = 5 rows, 2 columns
len(df)         # 5 rows
df.columns      # Index(['close', 'volume'])
```

---

## Appendix C: Financial Terms Glossary

| Term | Definition |
|------|-----------|
| **Alpha** | Return above what the market provided (excess return vs benchmark) |
| **ATR** | Average True Range -- average daily price movement, measures volatility |
| **Backtest** | Testing a strategy on historical data to evaluate performance |
| **Bear Market** | Declining prices (pessimistic market) |
| **Benchmark** | A reference index (e.g., S&P 500) to compare strategy performance against |
| **Beta** | How much an asset moves relative to the market (beta=1 means same as market) |
| **Bollinger Bands** | Price envelope: middle SMA +/- 2 standard deviations |
| **Bull Market** | Rising prices (optimistic market) |
| **CAGR** | Compound Annual Growth Rate -- smoothed annual return |
| **Calmar Ratio** | Annualized return divided by max drawdown |
| **Candlestick** | Chart showing OHLC as a "candle" with body and wicks |
| **Commission** | Fee charged by broker per trade |
| **CVaR** | Conditional Value at Risk -- average loss in the worst X% of scenarios |
| **Drawdown** | Decline from a peak to a trough in portfolio value |
| **EMA** | Exponential Moving Average -- weighted more toward recent prices |
| **Equity Curve** | Chart of portfolio value over time |
| **Futures** | Contracts to buy/sell an asset at a future date/price |
| **Long** | Buying an asset (profiting when price rises) |
| **MACD** | Moving Average Convergence Divergence -- momentum/trend indicator |
| **Max Drawdown** | Largest peak-to-trough decline (measures worst-case loss) |
| **Momentum** | Rate of price change (how fast price is moving) |
| **Moving Average** | Average of the last N prices, smooths out noise |
| **OHLCV** | Open, High, Low, Close, Volume -- standard price data format |
| **Omega Ratio** | Probability-weighted ratio of gains to losses |
| **Overbought** | Asset may be overpriced (RSI > 70 typically) |
| **Oversold** | Asset may be underpriced (RSI < 30 typically) |
| **P&L** | Profit and Loss |
| **Position Sizing** | How much capital to allocate per trade |
| **Profit Factor** | Gross profits divided by gross losses |
| **Regime** | Market phase: bull (rising), bear (falling), or neutral (sideways) |
| **Resampling** | Converting data from one timeframe to another (e.g., daily to weekly) |
| **Risk-Free Rate** | Return from a "safe" investment (e.g., US Treasury bonds, ~4%) |
| **ROC** | Rate of Change -- percentage price change over N periods |
| **RSI** | Relative Strength Index -- momentum oscillator (0-100 scale) |
| **Sharpe Ratio** | Risk-adjusted return: (return - risk_free) / volatility. Above 1.0 is good. |
| **Short** | Selling an asset you don't own (profiting when price falls) |
| **Signal** | A buy (1), sell (-1), or neutral (0) trading instruction |
| **Slippage** | Difference between expected trade price and actual execution price |
| **SMA** | Simple Moving Average -- unweighted average of last N prices |
| **Sortino Ratio** | Like Sharpe but only penalizes downside volatility |
| **Stop-Loss** | Automatic exit when price moves against you by a set amount |
| **Supertrend** | Trend-following indicator based on ATR |
| **Tear Sheet** | Comprehensive performance report for a strategy |
| **Trailing Stop** | Stop-loss that follows price upward but never downward |
| **VaR** | Value at Risk -- maximum expected loss at a given confidence level |
| **Volatility** | How much price varies over time (standard deviation of returns) |
| **Volume** | Number of shares/units traded in a period |
| **VWAP** | Volume-Weighted Average Price |
| **Win Rate** | Percentage of trades that are profitable |
