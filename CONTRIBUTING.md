# Contributing to TA Framework

Thank you for your interest in contributing. This guide covers the development workflow, coding standards, and how to add new features.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/az9713/technical-analysis-framework.git
cd technical-analysis-framework

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify everything works
pytest tests/ -v
```

## Development Workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests first** -- add tests to the appropriate file in `tests/`.
3. **Implement** your changes.
4. **Run the full test suite** -- all 175+ tests must pass before submitting.
5. **Open a pull request** against `main` with a clear description of the change.

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_indicators.py -v

# Run with coverage
pytest tests/ --cov=ta_framework --cov-report=term-missing
```

## Project Conventions

### Signal Convention

Signals are integers throughout the entire framework:

- `1` = long
- `-1` = short
- `0` = neutral

### Column Naming

Indicator columns follow the pandas-ta convention: `SMA_20`, `RSI_14`, `MACD_12_26_9`, `BBU_20_2.0`, `EMA_12`.

### OHLCV Schema

All DataFrames use a `DatetimeIndex` named `"date"` with lowercase columns: `open`, `high`, `low`, `close`, `volume` (all `float64`).

### pandas Version

pandas is pinned to `>= 2.0, < 3.0` because pandas-ta-classic uses deprecated pandas APIs. Do not upgrade to pandas 3.x.

## Adding New Features

### New Indicator

1. Write a wrapper function in `ta_framework/indicators/wrappers.py`:

```python
def compute_my_indicator(df: pd.DataFrame, **params) -> pd.DataFrame:
    """Compute My Indicator."""
    length = params.get("length", 20)
    result = df.ta.my_indicator(length=length)
    return pd.concat([df, result], axis=1)
```

2. Add metadata to `INDICATOR_CATALOG` in `ta_framework/indicators/catalog.py`:

```python
"my_indicator": IndicatorConfig(
    name="my_indicator",
    category=IndicatorCategory.MOMENTUM,
    tier=IndicatorTier.TIER1,
    params={"length": 20},
    description="My Custom Indicator",
),
```

3. Add to `WRAPPER_MAP` at the bottom of `wrappers.py`:

```python
"my_indicator": compute_my_indicator,
```

4. Add tests in `tests/test_indicators.py`.

### New Strategy

Subclass `SignalGenerator`, implement `generate()` and `required_indicators`, and register it:

```python
from ta_framework.signals.base import SignalGenerator
from ta_framework.core.registry import strategy_registry

@strategy_registry.register("my_strategy")
class MyStrategy(SignalGenerator):
    name = "my_strategy"

    def __init__(self, period: int = 14) -> None:
        self.period = period

    @property
    def required_indicators(self) -> list[dict]:
        return [{"name": "rsi", "length": self.period}]

    def generate(self, df):
        self.validate_columns(df, [f"RSI_{self.period}"])
        # Your signal logic here -- set df["signal"] to 1, -1, or 0
        df["signal"] = 0
        return df
```

Add tests in `tests/test_signals.py`.

### New Data Provider

Subclass `DataProvider` and register it:

```python
from ta_framework.data.base import DataProvider
from ta_framework.core.registry import provider_registry

@provider_registry.register("my_provider")
class MyProvider(DataProvider):
    def fetch(self, symbol, timeframe="1d", start=None, end=None):
        # Return a validated OHLCV DataFrame
        ...

    def supported_assets(self):
        return ["equity"]

    def search_symbols(self, query):
        return []
```

Add tests in `tests/test_data_providers.py`.

## Code Style

- **Python 3.10+** -- use modern type hints (`list[dict]`, `str | None`, etc.).
- **No unnecessary abstractions** -- three similar lines are better than a premature helper function.
- **Docstrings** -- required for public classes and functions. Use NumPy-style docstrings.
- **Imports** -- use `from __future__ import annotations` at the top of every module.

## What We're Looking For

Contributions in these areas are particularly welcome:

- **New indicators** -- especially Tier 2/3 indicators not yet in the catalog
- **New strategies** -- novel signal generation approaches
- **New data providers** -- additional data sources beyond Yahoo Finance, CSV, and ccxt
- **Bug fixes** -- with a test that reproduces the bug
- **Performance improvements** -- especially in the backtesting engine
- **Documentation** -- tutorials, examples, workflow guides

## What to Avoid

- Do not upgrade pandas beyond 2.x (breaks pandas-ta-classic compatibility)
- Do not add TA-Lib as a dependency (the project deliberately avoids C dependencies)
- Do not add event-driven backtesting (the project uses vectorized backtesting by design)
- Do not add live trading / broker connectivity (this is a research tool)

## Reporting Issues

Open an issue on GitHub with:

- A clear title describing the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
