# Financial Professional's Guide to TA Framework

A capability assessment and workflow guide for portfolio managers, quantitative analysts, research analysts, and risk professionals evaluating this tool for their work.

---

## 1. Executive Summary

### What This Tool Is

TA Framework is an open-source, research-grade technical analysis and backtesting toolkit built in Python. It provides 38 production-quality technical indicators, 6 pre-built trading strategies, a vectorized backtesting engine, risk management tools (VaR, position sizing, stop-losses), and a 25+ metric analytics suite -- all wrapped in an interactive Streamlit dashboard you can run locally.

The framework is designed for **strategy research and evaluation**, not live execution. It processes daily/weekly/monthly OHLCV bar data from Yahoo Finance, CSV files, or cryptocurrency exchanges (via ccxt), and produces the quantitative outputs that inform trading decisions: equity curves, Sharpe ratios, drawdown analysis, regime detection, and Monte Carlo confidence intervals.

### What This Tool Is NOT

- **Not a trading platform.** There is no order management, no broker connectivity, no live execution.
- **Not a real-time system.** It operates on historical bars, not streaming tick data.
- **Not a portfolio management system.** Single-asset backtest focus; no multi-asset portfolio rebalancing.
- **Not regulatory-compliant software.** No audit trail, no regulatory capital calculations, no compliance checks.

### Who It's For

| Role | Primary Use |
|------|-------------|
| **Quantitative Analyst** | Strategy hypothesis testing, parameter optimization, walk-forward validation |
| **Research Analyst** | Screening setups, indicator overlays, regime analysis |
| **Portfolio Manager** | Multi-asset comparison, benchmark analysis, risk assessment |
| **Risk Professional** | VaR/CVaR calculations, drawdown analysis, stress testing via Monte Carlo |
| **Trader** | Signal prototyping, visual confirmation of setups, strategy comparison |

---

## 2. Feature Matrix

| Category | Supported | Not Supported |
|----------|-----------|---------------|
| **Indicators** | 38 Tier 1 indicators across trend, momentum, volatility, volume, and overlap categories. Custom indicator framework for proprietary calculations. | No proprietary vendor indicators (Bloomberg BVAL, Refinitiv, etc.). No options Greeks. No fixed-income analytics. |
| **Strategies** | 6 pre-built: EMA Crossover, RSI Threshold, MACD Signal Cross, Bollinger Band Mean-Reversion, Supertrend, TTM Squeeze. Composite signal combiner (voting, weighted, confirmation). Custom strategy API. | No ML/AI-based strategies. No multi-leg option strategies. No pairs trading. No market-making. |
| **Backtesting** | Vectorized engine with commission and slippage modeling. Long and short positions. Grid search parameter optimization. Walk-forward validation. Monte Carlo robustness testing. | No event-driven simulation. No order book modeling. No partial fills. No intraday execution simulation. No multi-asset portfolio backtest. |
| **Risk Management** | 4 position sizers (fixed fractional, Kelly criterion, volatility-based, risk parity). Parametric/historical/Monte Carlo VaR + CVaR. 4 stop-loss types. Portfolio volatility and risk contribution. | No real-time risk limits. No margin calculations. No counterparty risk. No regulatory capital (Basel III/IV). |
| **Analytics** | 25+ metrics: Sharpe, Sortino, Calmar, Omega, CAGR, max drawdown, VaR/CVaR, win rate, profit factor, expectancy, skewness, kurtosis, tail ratio, recovery factor. Tear sheet generation. Benchmark comparison (alpha, beta, information ratio, tracking error, up/down capture). | No factor analysis (Fama-French). No transaction cost analysis (TCA). No attribution analysis. |
| **Data Sources** | Yahoo Finance (equities, ETFs, indices, forex). Local CSV/Parquet files. Cryptocurrency exchanges via ccxt (optional). Data quality checking (gaps, outliers, OHLC consistency). | No Bloomberg. No Refinitiv/LSEG. No proprietary data feeds. No tick data. No fundamental data. |
| **Visualization** | Interactive Plotly charts: candlestick with indicator overlays, multi-panel charts, drawdown (underwater) charts, monthly returns heatmap, correlation heatmap, returns distribution with VaR lines, Q-Q plots, trade markers (buy/sell on chart). | No real-time streaming charts. No order flow visualization. No depth-of-market. |
| **Timeframes** | 1-minute through monthly bars. Multi-timeframe resampling and alignment. | No tick data. No custom session times. |
| **Asset Classes** | Equities, ETFs, indices, forex, crypto. | No options. No futures (except via CSV). No fixed income. No commodities (except via CSV/crypto). |

---

## 3. Indicator Coverage

### Trend-Following Indicators (14)

| Indicator | Typical Institutional Use | Key Parameters |
|-----------|--------------------------|----------------|
| SMA (Simple MA) | Trend identification, support/resistance levels | length (default: 20) |
| EMA (Exponential MA) | Faster trend detection, crossover signals | length (default: 20) |
| WMA (Weighted MA) | Price-weighted trend smoothing | length (default: 20) |
| DEMA (Double EMA) | Reduced lag trend detection | length (default: 20) |
| TEMA (Triple EMA) | Minimal lag trend detection | length (default: 20) |
| VWMA (Volume-Weighted MA) | Institutional flow confirmation | length (default: 20) |
| HMA (Hull MA) | Ultra-smooth trend with minimal lag | length (default: 20) |
| KAMA (Kaufman Adaptive MA) | Adaptive smoothing for noisy markets | length, fast, slow |
| T3 (Tillson T3) | Smooth trend following with adjustable responsiveness | length, a (volume factor) |
| Supertrend | Trend direction with built-in stop levels | length (7), multiplier (3.0) |
| Ichimoku Cloud | Multi-component trend system (support, resistance, momentum) | tenkan (9), kijun (26), senkou (52) |
| ADX (Avg Directional Index) | Trend strength measurement (not direction) | length (default: 14) |
| Aroon | Trend identification via time since highs/lows | length (default: 25) |
| Parabolic SAR | Trailing stop and trend reversal detection | af0, af (0.02), max_af (0.2) |

### Momentum / Mean-Reversion Indicators (10)

| Indicator | Typical Institutional Use | Key Parameters |
|-----------|--------------------------|----------------|
| RSI | Overbought/oversold screening, divergence detection | length (default: 14) |
| MACD | Momentum shifts, signal-line crossovers | fast (12), slow (26), signal (9) |
| Stochastic | Short-term overbought/oversold, mean-reversion timing | k (14), d (3), smooth_k (3) |
| CCI (Commodity Channel) | Cyclical extreme detection | length (default: 20) |
| Williams %R | Fast overbought/oversold oscillator | length (default: 14) |
| ROC (Rate of Change) | Momentum screening, sector rotation | length (default: 10) |
| MFI (Money Flow Index) | Volume-weighted RSI equivalent | length (default: 14) |
| TSI (True Strength) | Smoothed momentum with signal line | fast (13), slow (25) |
| Ultimate Oscillator | Multi-timeframe momentum | fast (7), medium (14), slow (28) |
| Awesome Oscillator | Momentum via MA differential | fast (5), slow (34) |

### Volatility Indicators (6)

| Indicator | Typical Institutional Use | Key Parameters |
|-----------|--------------------------|----------------|
| Bollinger Bands | Volatility regime, mean-reversion bands, squeeze detection | length (20), std (2.0) |
| ATR (Avg True Range) | Position sizing, stop-loss calibration, volatility normalization | length (default: 14) |
| Keltner Channel | Trend channel, squeeze detection (with Bollinger) | length (20), scalar (1.5) |
| Donchian Channel | Breakout system (Turtle Trading), range identification | lower/upper_length (20) |
| Standard Deviation | Raw volatility measurement | length (default: 20) |
| True Range | Single-bar volatility for ATR calculation | (no params) |

### Volume Indicators (6)

| Indicator | Typical Institutional Use | Key Parameters |
|-----------|--------------------------|----------------|
| OBV (On Balance Volume) | Accumulation/distribution confirmation | (no params) |
| VWAP | Institutional execution benchmark, intraday fair value | (no params) |
| A/D Line | Buying/selling pressure measurement | (no params) |
| CMF (Chaikin Money Flow) | Money flow into/out of a security | length (default: 20) |
| Force Index | Volume-weighted price change | length (default: 13) |
| Volume SMA | Relative volume analysis | length (default: 20) |

### Overlap / Support-Resistance (2)

| Indicator | Typical Institutional Use | Key Parameters |
|-----------|--------------------------|----------------|
| Pivot Points | Intraday support/resistance levels | method (traditional) |
| Fibonacci Retracements | Retracement targets from recent swing | length (default: 20) |

### Extending the Catalog

The framework supports custom indicators via subclassing. If your desk uses a proprietary indicator not in the catalog, you can implement it once and it becomes available throughout the UI and backtesting engine. See Workflow 6 below.

---

## 4. Risk Management Capabilities

### Position Sizing

| Model | Method | When to Use |
|-------|--------|-------------|
| **Fixed Fractional** | Risk a fixed % of capital per trade | Standard risk management, straightforward allocation |
| **Kelly Criterion** | Optimal bet sizing from win rate and payoff ratio | Strategy with known, stable edge; use half-Kelly (default) for safety |
| **Volatility-Based** | Size inversely proportional to ATR | Equalizing dollar risk across instruments of different volatility |
| **Risk Parity** | Equalize risk contribution via inverse-volatility weighting | Multi-asset allocation with equal risk budget per asset |

### Value at Risk Suite

| Method | Description | Use Case |
|--------|-------------|----------|
| **Parametric VaR** | Gaussian assumption, mean + z-score * sigma | Quick estimates, normally distributed returns |
| **Historical VaR** | Empirical percentile of actual return distribution | Non-normal returns, fat tails |
| **Monte Carlo VaR** | Simulated returns from fitted distribution (10,000 paths default) | Forward-looking risk scenarios, customizable horizon |
| **CVaR (Expected Shortfall)** | Average loss beyond the VaR threshold | Tail risk assessment, regulatory preference over VaR |

All VaR methods accept configurable confidence levels (default 95%). Parametric and Monte Carlo VaR also support multi-period time horizons.

### Stop-Loss Strategies

| Type | Description | Parameters |
|------|-------------|------------|
| **Fixed Stop** | Percentage below entry price | stop_pct (e.g., 0.02 for 2%) |
| **ATR Stop** | Dynamic stop at close - (ATR * multiplier) | multiplier (default: 2.0), period (14) |
| **Trailing Stop** | Follows price upward, locks in gains | trail_pct (e.g., 0.05 for 5%) |
| **Chandelier Exit** | Highest high minus ATR * multiplier | period (22), multiplier (3.0) |

### Portfolio-Level Risk Metrics

- **Portfolio Volatility**: Annualized standard deviation using weights and covariance matrix
- **Maximum Drawdown Duration**: Longest period (in bars) from peak to recovery
- **Calmar Ratio**: Annualized return / max drawdown
- **Ulcer Index**: Root mean square of percentage drawdowns (measures pain of drawdowns)
- **Risk Contribution**: Marginal risk contribution per asset in a portfolio context

### What's NOT Here

- No real-time position monitoring or risk limit enforcement
- No margin requirement calculations
- No regulatory capital models (Basel III/IV, Solvency II)
- No counterparty or credit risk modeling
- No scenario analysis with user-defined shocks (Monte Carlo uses historical distribution only)

---

## 5. Common Workflows

### Workflow 1: Morning Market Screening

**Scenario**: A research analyst wants to scan for technical setups before the open -- RSI extremes, moving average crosses, squeeze releases.

**Step-by-step**:

1. **Launch the dashboard**: `streamlit run app.py` opens the Dashboard page in your browser.
2. **Enter a symbol**: Type a ticker (e.g., AAPL) in the symbol input. Select timeframe (1d recommended for daily screening) and date range (past 6-12 months).
3. **Fetch data**: The app downloads OHLCV data from Yahoo Finance and caches it for the session.
4. **Add indicator overlays**: Use the indicator multi-select to add your screening indicators:
   - RSI (14) -- look for sub-30 (oversold) or above-70 (overbought) readings
   - EMA 12 and EMA 26 -- check for recent crossovers
   - Bollinger Bands (20, 2.0) -- identify squeeze conditions
   - VWAP -- compare current price to institutional fair value
5. **Visual scan**: The multi-panel chart displays candlesticks with overlays in the main panel and oscillators (RSI, MACD) in sub-panels. Identify setups visually.

**Limitation**: This is a per-symbol workflow. There is no automated multi-symbol screener that scans a watchlist and returns a ranked list of setups. For universe screening, you would need to iterate symbols manually or script against the Python API.

---

### Workflow 2: Strategy Research & Backtesting

**Scenario**: A quant wants to test whether an EMA crossover strategy works on a specific equity over the past 3 years.

**Step-by-step**:

1. **Navigate to the Backtester page** in the Streamlit app.
2. **Configure the backtest**:
   - Enter the symbol and date range (3 years of daily data gives ~756 bars)
   - Select the "EMA Cross" strategy
   - Set parameters: fast period = 12, slow period = 26
   - Configure capital ($100,000), commission (0.1%), slippage (0.05%)
   - Enable or disable short selling as appropriate
3. **Run the backtest**: The engine computes indicators, generates signals, and runs the vectorized backtest.
4. **Review results**: The output includes:
   - **Equity curve chart** showing the growth of $100K
   - **Summary metrics**: total return, Sharpe ratio, max drawdown, win rate, profit factor, number of trades
   - **Trade list**: Entry/exit dates, prices, P&L per trade, holding period
   - **Buy/sell markers** overlaid on the price chart

**Parameter optimization**: Use the Python API for exhaustive parameter search:

```python
from ta_framework.backtest.optimization import GridSearchOptimizer
from ta_framework.signals.strategies import EMACrossStrategy

optimizer = GridSearchOptimizer(
    strategy_cls=EMACrossStrategy,
    param_grid={
        "fast_period": [5, 8, 12, 15, 20],
        "slow_period": [20, 26, 30, 40, 50],
    },
    metric="sharpe_ratio",
)
report = optimizer.run(df)  # df with pre-computed indicators
print(report.best.params)   # {'fast_period': 8, 'slow_period': 30}
print(report.best.metrics)  # {'sharpe_ratio': 1.23, ...}
```

**Walk-forward validation** (to guard against overfitting):

```python
from ta_framework.backtest.optimization import WalkForwardOptimizer

wf = WalkForwardOptimizer(
    strategy_cls=EMACrossStrategy,
    param_grid={"fast_period": [8, 12, 15], "slow_period": [26, 30, 40]},
    in_sample_pct=0.7,    # 70% train, 30% test per window
    n_windows=4,           # 4 rolling windows
    metric="sharpe_ratio",
)
report = wf.run(df)
# Each window uses in-sample to find best params, then evaluates out-of-sample
```

---

### Workflow 3: Multi-Asset Comparison

**Scenario**: A portfolio manager wants to compare AAPL, MSFT, and GOOGL performance over the same period -- normalized returns, correlation, and summary statistics.

**Step-by-step**:

1. **Navigate to the Compare page** in the Streamlit app.
2. **Select the Multi-Asset tab**.
3. **Enter symbols**: AAPL, MSFT, GOOGL.
4. **Set date range and timeframe** (e.g., 1 year of daily data).
5. **Run comparison**: The page displays:
   - **Normalized equity curves**: All starting at the same base for direct comparison
   - **Correlation heatmap**: Pairwise return correlations
   - **Summary statistics table**: Total return, volatility, Sharpe ratio, max drawdown per asset

---

### Workflow 4: Strategy Comparison & Selection

**Scenario**: A quant wants to evaluate EMA Crossover vs RSI vs MACD on the same asset to determine which strategy to deploy.

**Step-by-step**:

1. **Navigate to the Compare page**, select the **Multi-Strategy tab**.
2. **Enter symbol and date range**.
3. **Select strategies** to compare: EMA Cross, RSI, MACD. Configure parameters for each.
4. **Run comparison**: The page displays side-by-side:
   - Equity curves for each strategy on the same chart
   - Comparison table: Sharpe ratio, win rate, profit factor, max drawdown, number of trades per strategy

This workflow helps answer: "Which signal generation approach has historically performed best on this particular instrument?"

---

### Workflow 5: Risk Assessment & Tear Sheet Generation

**Scenario**: A risk analyst needs a comprehensive risk report for a strategy -- VaR, drawdown analysis, regime detection, return distribution characteristics.

**Step-by-step**:

1. **Navigate to the Analysis page**.
2. **Enter symbol, run a strategy backtest** (or load results from a previous backtest).
3. **Review the tabs**:

   **Tear Sheet tab**: Full performance report with 25+ metrics organized into:
   - Returns: CAGR, Sharpe, Sortino, Calmar, Omega, annualized volatility
   - Risk: max drawdown, max drawdown duration, recovery factor, 95% VaR, 95% CVaR, downside deviation
   - Trades: win rate, profit factor, expectancy, avg win/loss, largest win/loss, consecutive wins/losses
   - Distribution: skewness, kurtosis, tail ratio, common sense ratio
   - Benchmark (if configured): CAPM alpha, beta, information ratio, tracking error

   **Risk tab**: VaR/CVaR analysis at configurable confidence levels with multiple methodologies (parametric, historical, Monte Carlo).

   **Regime tab**: Market regime detection using KMeans clustering (3 regimes: bear, neutral, bull). Per-regime statistics: mean return, volatility, Sharpe ratio, average duration. Optional HMM-based detection if `hmmlearn` is installed.

   **Distribution tab**: Return histogram with normal distribution overlay and VaR confidence lines. Q-Q plot for assessing normality of returns.

**Programmatic tear sheet generation**:

```python
from ta_framework.analytics.tearsheet import TearSheet

ts = TearSheet(
    equity_curve=result.equity_curve,
    trades=result.trades,
    benchmark=benchmark_equity,  # optional
    risk_free_rate=0.04,
)
report = ts.generate()
# report['returns']['sharpe_ratio'], report['risk']['max_drawdown'], etc.

monthly = ts.monthly_returns()       # Year x Month pivot table
drawdowns = ts.drawdown_analysis()   # Top 5 drawdown periods with depth, duration, recovery
```

---

### Workflow 6: Custom Strategy Development

**Scenario**: An experienced quant wants to implement a proprietary signal -- for example, a momentum strategy that goes long when RSI is oversold AND price is above the 200-day SMA (a filtered mean-reversion approach).

**Using the Composite Signal combiner** (no code required):

```python
from ta_framework.signals.strategies import RSIStrategy, EMACrossStrategy
from ta_framework.signals.composite import CompositeSignal, CombineMode

# Require both signals to agree
composite = CompositeSignal(mode=CombineMode.CONFIRMATION)
composite.add_generator(RSIStrategy(period=14, overbought=70, oversold=30))
composite.add_generator(EMACrossStrategy(fast_period=50, slow_period=200))

# Compute indicators, then generate
df = composite.generate(df)  # signal fires only when BOTH agree
```

**Writing a fully custom strategy** (for logic the built-in rules can't express):

```python
from ta_framework.signals.base import SignalGenerator
from ta_framework.core.registry import strategy_registry

@strategy_registry.register("rsi_above_sma200")
class RSIAboveSMA200Strategy(SignalGenerator):
    name = "rsi_above_sma200"

    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def required_indicators(self) -> list[dict]:
        return [
            {"name": "rsi", "length": self.rsi_period},
            {"name": "sma", "length": 200},
        ]

    def generate(self, df):
        rsi_col = f"RSI_{self.rsi_period}"
        self.validate_columns(df, [rsi_col, "SMA_200"])

        import pandas as pd
        signal = pd.Series(0, index=df.index, dtype=int)

        # Long: RSI oversold AND price above SMA 200 (trend filter)
        long_mask = (df[rsi_col] < self.oversold) & (df["close"] > df["SMA_200"])
        # Short: RSI overbought AND price below SMA 200
        short_mask = (df[rsi_col] > self.overbought) & (df["close"] < df["SMA_200"])

        signal[long_mask] = 1
        signal[short_mask] = -1
        df["signal"] = signal
        return df
```

Once registered, this strategy becomes available throughout the framework -- including in the Streamlit UI, optimizer, and compare page.

---

### Workflow 7: Data Import & Quality Assurance

**Scenario**: An analyst has proprietary OHLCV data exported from an internal system as a CSV file and wants to verify data quality before running analysis.

**Step-by-step**:

1. **Prepare the CSV**: Ensure columns include `date`, `open`, `high`, `low`, `close`, `volume` (case-insensitive; the provider normalizes them).

2. **Load via CSVProvider**:
```python
from ta_framework.data.csv_provider import CSVProvider

provider = CSVProvider(base_path="/path/to/data/")
df = provider.fetch("my_data.csv", timeframe="1d")
```

3. **Run quality checks**:
```python
from ta_framework.data.quality import DataQualityChecker

checker = DataQualityChecker()
report = checker.full_check(df)

# report['gaps'] -> {nan_rows, nan_pct, passed, gap_locations}
# report['outliers'] -> {outlier_count, outlier_pct, passed, outlier_locations}
# report['ohlc_consistency'] -> {high_violations, low_violations, volume_violations, passed}
```

4. **Clean if needed**:
```python
df_clean = checker.clean(df, fill_method="ffill", remove_outliers=True, z_threshold=5.0)
```

The cleaner forward-fills NaN gaps, optionally removes statistical outliers (log-returns beyond 5 sigma), clips negative volume to zero, and enforces OHLC consistency (high >= max of OHLC, low <= min of OHLC).

---

## 6. Capability & Limitation Matrix

### "Can I Do X?" Reference

| Question | Answer | Details |
|----------|--------|---------|
| **Can I execute live trades?** | No | This is a research tool. No broker API, no OMS, no execution. |
| **Can I trade options or multi-leg strategies?** | No | Equity/ETF/crypto single-instrument only. No options pricing, no Greeks. |
| **Can I do high-frequency or tick-level backtesting?** | No | Minimum granularity is 1-minute bars. No tick data, no order book simulation. |
| **Can I backtest a portfolio of multiple assets simultaneously?** | No | Single-asset backtesting only. You can compare assets side-by-side but not simulate rebalancing across them. |
| **Can I use machine learning for signal generation?** | Not built-in | The framework uses KMeans for regime detection but has no ML signal generators. However, you can write a custom SignalGenerator that calls scikit-learn or any ML library internally. |
| **Can I run factor analysis (Fama-French, Carhart)?** | No | Benchmark comparison includes alpha/beta, but no multi-factor decomposition. |
| **Can I calculate regulatory capital requirements?** | No | No Basel III/IV, no Solvency II, no regulatory reporting. |
| **Can I generate an audit trail of signals and trades?** | Partial | BacktestResult stores all trades with timestamps, prices, and P&L. Signals are stored in the DataFrame. But there is no persistent audit logging or compliance-grade record-keeping. |
| **Can I use Bloomberg or Refinitiv data?** | Indirectly | Export from Bloomberg/Refinitiv to CSV, then load via CSVProvider. No direct API integration. |
| **Can I test stop-loss and take-profit in backtests?** | Partial | Stop-loss functions exist as utilities (fixed, ATR, trailing, chandelier) but are not integrated into the backtest engine's execution loop. The backtester reacts to signal changes only. |
| **Can I combine multiple strategies?** | Yes | CompositeSignal supports voting, weighted combination, and unanimous confirmation modes. |
| **Can I optimize parameters and guard against overfitting?** | Yes | GridSearchOptimizer for exhaustive search + WalkForwardOptimizer for rolling in-sample/out-of-sample validation. |
| **Can I stress-test a strategy?** | Partial | MonteCarloSimulator bootstraps returns or shuffles trade order to produce confidence intervals. No user-defined stress scenarios. |
| **Can I detect market regimes?** | Yes | KMeans clustering (built-in) and HMM (with optional hmmlearn). Per-regime statistics including mean return, volatility, Sharpe, and duration. |
| **Can I import my own data from internal systems?** | Yes | CSVProvider loads CSV/Parquet files. DataQualityChecker validates and cleans the data. Custom DataProvider subclass for other formats. |
| **Can I use this from Jupyter notebooks?** | Yes | The entire Python API is importable. All classes and functions work in notebooks without the Streamlit UI. |
| **Can I extend the indicator set?** | Yes | Subclass CustomIndicator, implement compute(), and it auto-registers into the engine. |
| **Can I add a new data source?** | Yes | Subclass DataProvider, implement fetch(), supported_assets(), search_symbols(). |

---

## 7. Integration with Existing Workflows

### Alongside Bloomberg Terminal / Refinitiv

TA Framework complements, rather than replaces, institutional data platforms. The recommended workflow:

1. **Export** OHLCV data from your terminal to CSV (Bloomberg: `BDH` function -> Excel -> CSV export; Refinitiv: Eikon Data API -> CSV).
2. **Load** into the framework via `CSVProvider`.
3. **Run analysis** (backtesting, VaR, regime detection) that your terminal may not support or may charge extra for.
4. **Export results** back to your existing tools. BacktestResult provides `to_dataframe()` for easy CSV export of trade logs.

### From Jupyter Notebooks

Every component is available as a Python import:

```python
from ta_framework.data.yfinance_provider import YFinanceProvider
from ta_framework.indicators.engine import IndicatorEngine
from ta_framework.signals.strategies import EMACrossStrategy
from ta_framework.backtest.engine import VectorizedBacktester
from ta_framework.analytics.tearsheet import TearSheet
from ta_framework.analytics.regime import RegimeDetector
from ta_framework.risk.var import parametric_var, historical_var, cvar
```

Results are pandas DataFrames and standard Python dicts -- compatible with any downstream tool (matplotlib, seaborn, Excel export, database storage).

### With Excel

- Export any DataFrame to Excel: `df.to_excel("output.xlsx")`
- Export backtest trades: `result.to_dataframe().to_excel("trades.xlsx")`
- Export tear sheet metrics: `pd.DataFrame([ts.generate()]).to_excel("tearsheet.xlsx")`

---

## 8. Getting Started for Financial Professionals

### 5-Minute Quick Start

```bash
# 1. Install (requires Python 3.10+)
pip install -r requirements.txt

# 2. Launch the interactive dashboard
streamlit run app.py

# 3. In the browser:
#    - Enter "AAPL" as the symbol
#    - Select "1d" timeframe, last 1 year
#    - Click to load data
#    - Add RSI (14) and Bollinger Bands (20, 2.0) from the indicator selector
#    - Switch to the Backtester page and run an EMA Cross backtest
```

### Recommended First Steps by Role

**If you're a Research Analyst**:
1. Start with the Dashboard page. Load a symbol you follow daily.
2. Add your usual indicator overlay (e.g., 50/200 SMA, RSI, VWAP).
3. Compare the charting to what you see on your Bloomberg or trading platform -- this validates the data and indicator calculations match your expectations.

**If you're a Quantitative Analyst**:
1. Start with the Backtester page. Pick a hypothesis you've been meaning to test.
2. Run it with the default parameters, then try the grid search optimizer via the Python API.
3. Run walk-forward validation to see if the edge persists out-of-sample.
4. Generate a tear sheet and Monte Carlo confidence intervals.

**If you're a Portfolio Manager**:
1. Start with the Compare page. Enter 3-5 holdings from your portfolio.
2. Review the normalized equity curves and correlation heatmap.
3. Switch to the Analysis page for VaR/CVaR calculations on individual positions.

**If you're a Risk Professional**:
1. Start with the Analysis page. Load a symbol and run a backtest.
2. Review the Risk tab for VaR/CVaR at your standard confidence levels.
3. Check the Regime tab to understand how strategy performance varies across market conditions.
4. Use the Python API for Monte Carlo stress testing with `MonteCarloSimulator`.

---

## Appendix: Technical Specifications

| Specification | Value |
|---------------|-------|
| Python version | 3.10+ |
| pandas version | >= 2.0, < 3.0 |
| Indicator library | pandas-ta-classic (no C/TA-Lib dependency) |
| Backtest engine | Vectorized (signal-driven, not event-driven) |
| Signal convention | 1 = long, -1 = short, 0 = neutral |
| Default commission | 0.1% per trade |
| Default slippage | 0.05% per trade |
| Default initial capital | $100,000 |
| Default risk-free rate | 4.0% annual |
| VaR default confidence | 95% |
| Monte Carlo default simulations | 1,000 (bootstrap) / 10,000 (VaR) |
| Regime detection | KMeans (3 clusters) or HMM (optional hmmlearn) |
| Cost models | Fixed, percentage, tiered commission; base + volume-impact slippage |
| Data validation | Gap detection, outlier detection (z-score), OHLC consistency, auto-cleaning |
