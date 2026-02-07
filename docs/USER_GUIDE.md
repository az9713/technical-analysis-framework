# TA Framework User Guide

A complete guide to using the Technical Analysis Framework application. This guide is written for users with basic computer skills -- no experience with Python, finance, or web applications is required.

---

## Table of Contents

- [Part 1: Getting Started](#part-1-getting-started)
  - [What is the TA Framework?](#what-is-the-ta-framework)
  - [What is Technical Analysis?](#what-is-technical-analysis)
  - [Installing the Application](#installing-the-application)
  - [Starting the Application](#starting-the-application)
  - [Stopping and Restarting](#stopping-and-restarting)
- [Part 2: The Dashboard Page](#part-2-the-dashboard-page)
  - [What You Will See](#what-you-will-see)
  - [Entering a Symbol](#entering-a-symbol)
  - [Choosing a Timeframe](#choosing-a-timeframe)
  - [Setting the Date Range](#setting-the-date-range)
  - [Adding Indicators](#adding-indicators)
  - [Reading the Chart](#reading-the-chart)
  - [Data Table and Summary Statistics](#data-table-and-summary-statistics)
- [Part 3: The Backtester Page](#part-3-the-backtester-page)
  - [What is Backtesting?](#what-is-backtesting)
  - [Choosing a Strategy](#choosing-a-strategy)
  - [Setting Strategy Parameters](#setting-strategy-parameters)
  - [Configuring the Backtest](#configuring-the-backtest)
  - [Running the Backtest](#running-the-backtest)
  - [Understanding the Results](#understanding-the-results)
  - [Reading the Charts](#reading-the-charts)
  - [The Trade Log](#the-trade-log)
- [Part 4: The Compare Page](#part-4-the-compare-page)
  - [Multi-Asset Comparison](#multi-asset-comparison)
  - [Multi-Strategy Comparison](#multi-strategy-comparison)
- [Part 5: The Analysis Page](#part-5-the-analysis-page)
  - [Running an Analysis](#running-an-analysis)
  - [Tear Sheet Tab](#tear-sheet-tab)
  - [Risk Analysis Tab](#risk-analysis-tab)
  - [Regime Detection Tab](#regime-detection-tab)
  - [Distributions Tab](#distributions-tab)
- [Part 6: Tips and Best Practices](#part-6-tips-and-best-practices)
- [Part 7: Frequently Asked Questions](#part-7-frequently-asked-questions)
- [Part 8: Troubleshooting](#part-8-troubleshooting)

---

## Part 1: Getting Started

### What is the TA Framework?

The TA Framework is a tool that helps you analyze financial markets using historical price data. It runs in your web browser and provides an interactive set of charts, indicators, and analysis tools.

With the TA Framework you can:

- **View price charts** for stocks, cryptocurrencies, ETFs, and forex pairs, with professional candlestick charts and dozens of technical indicators overlaid.
- **Test trading strategies** on historical data to answer the question: "If I had used this strategy over the past 5 years, would I have made or lost money?"
- **Compare investments** side by side to see how different assets performed relative to each other, and how correlated their price movements are.
- **Compare strategies** to find out which trading approach works best for a given asset.
- **Get detailed performance reports** including risk metrics, return breakdowns, drawdown analysis, market regime detection, and statistical distributions.

This application is designed for research and education. It does not connect to any brokerage or execute real trades.

### What is Technical Analysis?

Technical analysis is a method of evaluating financial assets by studying their past price and trading volume data. Instead of looking at a company's earnings reports, balance sheets, or business model (which is called *fundamental analysis*), technical analysis focuses entirely on what has already happened in the market -- the price movements and the patterns within them.

The core idea is that price movements are not entirely random. Trends tend to continue, patterns tend to repeat, and the collective behavior of buyers and sellers leaves footprints in the data. Technical analysts use mathematical formulas called *indicators* to measure things like trend direction, momentum (how fast a price is moving), volatility (how much the price swings), and trading volume (how many shares or coins changed hands).

For example, a *moving average* smooths out daily price noise to reveal the underlying trend. The *Relative Strength Index (RSI)* measures whether an asset has been bought or sold too aggressively in recent days. *Bollinger Bands* create a dynamic envelope around the price to show when prices are unusually high or low compared to their recent history.

Technical analysis is widely used by day traders, swing traders, institutional investors, and quantitative researchers. It is not a crystal ball -- no indicator can predict the future with certainty -- but it provides a structured, data-driven framework for making informed decisions.

### Installing the Application

Follow these steps carefully. You only need to do this once.

#### Step 1: Install Python

Python is the programming language that powers this application. You need to install it on your computer.

1. Open your web browser and go to **https://www.python.org/downloads/**
2. Click the large yellow button that says **"Download Python 3.x.x"** (the exact version number may vary; anything 3.10 or higher is fine).
3. Once the download finishes, open the installer file.
4. **CRITICAL**: On the very first screen of the installer, you will see a checkbox at the bottom that says **"Add Python to PATH"**. **You MUST check this box.** If you skip this step, the commands in the next steps will not work.
5. Click **"Install Now"** and wait for the installation to complete.
6. To verify the installation worked:
   - Press the **Windows key** on your keyboard.
   - Type **cmd** and press Enter. This opens the Command Prompt (a black window where you can type commands).
   - Type the following and press Enter:
     ```
     python --version
     ```
   - You should see something like `Python 3.12.1`. If you see an error instead, Python was not installed correctly -- go back and make sure you checked "Add Python to PATH".

#### Step 2: Open Command Prompt

If you closed the Command Prompt from the previous step, open it again:

1. Press the **Windows key** on your keyboard.
2. Type **cmd**.
3. Click **"Command Prompt"** in the search results.

You will see a black window with a blinking cursor. This is where you will type the remaining commands.

#### Step 3: Navigate to the Project Folder

Type the following command and press Enter:

```
cd C:\Users\simon\Downloads\pandas_TA_neuralNine
```

This tells the Command Prompt to "change directory" into the project folder. You should see the path change in the prompt line.

#### Step 4: Install Required Software

Type the following command and press Enter:

```
pip install -r requirements.txt
```

This command reads the file `requirements.txt` (which lists all the software libraries the application needs) and downloads and installs them automatically. You will see a lot of text scrolling by as it downloads. **This may take 2 to 5 minutes** depending on your internet speed. Wait until you see a message that says the installations are complete and you get your cursor back.

If you see an error about `pip` not being found, try this alternative command instead:

```
python -m pip install -r requirements.txt
```

### Starting the Application

#### Step 5: Launch the Application

Make sure you are still in the project folder (from Step 3), then type:

```
streamlit run app.py
```

After a few seconds, two things will happen:

1. The Command Prompt will display a message that says **"You can now view your Streamlit app in your browser"** along with a local URL (usually `http://localhost:8501`).
2. Your default web browser will open automatically and load the application.

If the browser does not open automatically, copy the URL from the Command Prompt and paste it into your browser's address bar.

**Note:** The Command Prompt must remain open while you use the application. It is running the server that powers the web interface. Do not close it.

### Stopping and Restarting

#### Stopping the Application

1. Go back to the Command Prompt window (you can click it in the taskbar at the bottom of the screen).
2. Press **Ctrl+C** (hold the Ctrl key and press C). You may need to press it twice.
3. The server will stop, and the browser page will no longer update.
4. You can now close the browser tab and the Command Prompt window.

#### Restarting the Application (After First Install)

After the initial installation, you only need two commands to start the application again. Open the Command Prompt and type:

```
cd C:\Users\simon\Downloads\pandas_TA_neuralNine
streamlit run app.py
```

You do not need to reinstall the requirements unless the application has been updated.

---

## Part 2: The Dashboard Page

The Dashboard is the first page you see when you open the application. It lets you view price charts for any supported financial instrument and overlay technical indicators on top.

### What You Will See

The screen is divided into two areas:

- **Left sidebar**: This is where all your controls are. You will find the symbol input box, the timeframe selector, date range pickers, and the indicator selection menus.
- **Main area**: This is the large area on the right where the interactive chart, data table, and summary statistics are displayed.

### Entering a Symbol

At the top of the sidebar, you will see a text box labeled **"Symbol"**. This is where you type the ticker symbol for the asset you want to analyze. The default value is `AAPL` (Apple Inc.).

Here are examples of valid symbols you can enter:

| Asset Type | Example Symbols | Description |
|---|---|---|
| **US Stocks** | `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, `META`, `NVDA` | Major American companies |
| **Cryptocurrency** | `BTC-USD`, `ETH-USD`, `SOL-USD`, `DOGE-USD` | Crypto prices in US Dollars |
| **ETFs** | `SPY`, `QQQ`, `IWM`, `DIA`, `VTI`, `GLD` | Exchange-Traded Funds |
| **Forex** | `EURUSD=X`, `GBPUSD=X`, `USDJPY=X` | Currency pairs |
| **Indices** | `^GSPC`, `^DJI`, `^IXIC` | S&P 500, Dow Jones, NASDAQ |

Type the symbol exactly as shown (capitalization does not matter -- the app will convert it to uppercase). Then press Enter or click elsewhere on the page.

**Tip**: If you are not sure what a company's ticker symbol is, search "AAPL stock" (replace with the company name) in any search engine. The ticker is the short abbreviation shown in results (e.g., AAPL for Apple, MSFT for Microsoft).

### Choosing a Timeframe

Below the symbol input, you will see a dropdown labeled **"Timeframe"**. This controls the time period that each bar (or candlestick) on the chart represents. The default is **1d** (daily).

| Timeframe | What Each Bar Represents | Best For |
|---|---|---|
| **1m** | 1 minute | Very short-term intraday analysis |
| **5m** | 5 minutes | Short-term intraday analysis |
| **15m** | 15 minutes | Intraday trading |
| **30m** | 30 minutes | Intraday trading |
| **1h** | 1 hour | Short-term swing analysis |
| **4h** | 4 hours | Medium-term swing analysis |
| **1d** | 1 day | Most common; general analysis |
| **1wk** | 1 week | Longer-term trend analysis |
| **1mo** | 1 month | Very long-term, big-picture view |

**For beginners**: Start with **1d** (daily). This is the most commonly used timeframe and gives you a clear view of price trends over weeks and months.

**Important note about intraday data**: Yahoo Finance only provides limited historical data for short timeframes. For 1-minute bars, you may only get the last 7 days of data. For hourly bars, you may get up to 2 years. Daily and weekly bars go back many years.

### Setting the Date Range

Below the timeframe selector, you will see two date pickers side by side:

- **Start Date**: The beginning of the date range. Default is 5 years ago.
- **End Date**: The end of the date range. Default is today.

Click on either date picker to open a calendar. Select the dates you want, then click elsewhere to confirm. The chart will update automatically.

### Adding Indicators

Below the date range, separated by a divider line, you will see several multiselect menus grouped by indicator category. Each menu lets you select one or more indicators to overlay on the chart.

#### Trend Indicators

Trend indicators help you identify the direction of the market -- is the price going up, down, or sideways?

| Indicator | Full Name | What It Shows |
|---|---|---|
| `sma` | Simple Moving Average | Average of the last N closing prices. Smooths out noise to reveal the trend. Appears as a line on the price chart. |
| `ema` | Exponential Moving Average | Like SMA but gives more weight to recent prices. Reacts faster to price changes. |
| `wma` | Weighted Moving Average | Linearly weighted average; most weight on recent prices. |
| `dema` | Double Exponential MA | Faster-reacting smoothed average using double exponential calculation. |
| `tema` | Triple Exponential MA | Even faster-reacting average using triple exponential calculation. |
| `vwma` | Volume Weighted MA | Moving average that also accounts for trading volume. |
| `hma` | Hull Moving Average | Designed to reduce lag while remaining smooth. |
| `kama` | Kaufman Adaptive MA | Adapts its speed based on market conditions -- fast in trends, slow in choppy markets. |
| `t3` | Tillson T3 | Ultra-smooth moving average with minimal lag. |
| `supertrend` | Supertrend | Shows the current trend direction. Line flips between support (below price = uptrend) and resistance (above price = downtrend). |
| `ichimoku` | Ichimoku Cloud | A comprehensive system showing support, resistance, trend direction, and momentum all at once. Appears as a shaded "cloud" on the chart. |
| `adx` | Average Directional Index | Measures trend strength (not direction). Values above 25 suggest a strong trend. Appears in a separate panel below the chart. |
| `aroon` | Aroon Indicator | Measures how long since the highest high and lowest low. Helps identify trend changes. |
| `psar` | Parabolic SAR | Dots that appear above or below the price. Below = uptrend, above = downtrend. When dots flip sides, it signals a potential trend change. |

#### Momentum Indicators

Momentum indicators measure how fast the price is moving and whether the current move is gaining or losing strength.

| Indicator | Full Name | What It Shows |
|---|---|---|
| `rsi` | Relative Strength Index | Oscillates between 0 and 100. Above 70 = overbought (may be due for a drop). Below 30 = oversold (may be due for a bounce). |
| `macd` | Moving Average Convergence Divergence | Shows the relationship between two moving averages. When the MACD line crosses above the signal line, it suggests upward momentum. |
| `stoch` | Stochastic Oscillator | Compares the closing price to the price range over a period. Above 80 = overbought, below 20 = oversold. |
| `cci` | Commodity Channel Index | Measures the deviation of price from its average. Extreme readings suggest overbought or oversold conditions. |
| `willr` | Williams %R | Similar to Stochastic but inverted. Ranges from -100 to 0. Above -20 = overbought, below -80 = oversold. |
| `roc` | Rate of Change | The percentage change in price over N periods. Positive = price rising, negative = price falling. |
| `mfi` | Money Flow Index | Like RSI but also factors in volume. Above 80 = overbought, below 20 = oversold. |
| `tsi` | True Strength Index | A momentum oscillator that smooths price changes to identify trend direction and overbought/oversold levels. |
| `uo` | Ultimate Oscillator | Combines short, medium, and long-term momentum into one indicator. |
| `ao` | Awesome Oscillator | Measures market momentum using the difference between 5-period and 34-period simple moving averages of the midpoint price. |

#### Volatility Indicators

Volatility indicators measure how much the price is swinging. High volatility means big price swings; low volatility means calm, steady movement.

| Indicator | Full Name | What It Shows |
|---|---|---|
| `bbands` | Bollinger Bands | Three lines: a middle line (20-period SMA) with upper and lower bands 2 standard deviations away. When bands are wide, volatility is high. When they squeeze together, a big move may be coming. |
| `atr` | Average True Range | The average size of price bars over N periods. Higher ATR = more volatile market. |
| `kc` | Keltner Channel | Similar to Bollinger Bands but uses ATR instead of standard deviation. |
| `donchian` | Donchian Channel | Shows the highest high and lowest low over N periods. Creates a channel around the price. |
| `stdev` | Standard Deviation | Raw measure of how spread out prices are from their average. |
| `true_range` | True Range | The range of each individual bar, accounting for gaps between bars. |

#### Volume Indicators

Volume indicators analyze trading volume (how many shares or coins are being traded) to confirm price movements or spot divergences.

| Indicator | Full Name | What It Shows |
|---|---|---|
| `obv` | On Balance Volume | Cumulative volume that adds volume on up days and subtracts on down days. Rising OBV confirms an uptrend. |
| `vwap` | Volume Weighted Average Price | The average price weighted by volume. Often used as a benchmark -- price above VWAP suggests bullish sentiment. |
| `ad` | Accumulation/Distribution | Shows whether a stock is being accumulated (bought) or distributed (sold) based on price and volume. |
| `cmf` | Chaikin Money Flow | Measures money flow volume over a period. Positive = buying pressure, negative = selling pressure. |
| `fi` | Force Index | Combines price change and volume to measure the strength of a move. |
| `volume_sma` | Volume Moving Average | Simple moving average of volume. Helps identify when volume is above or below normal. |

#### Overlap Indicators

| Indicator | Full Name | What It Shows |
|---|---|---|
| `pivot` | Pivot Points | Key support and resistance levels calculated from previous period's high, low, and close. |
| `fib` | Fibonacci Retracement | Horizontal lines at key Fibonacci ratios (23.6%, 38.2%, 50%, 61.8%) indicating potential support/resistance. |

**How to select indicators**: Click on a multiselect box for any category. A dropdown will appear showing the available indicators. Click on one or more to add them. To remove an indicator, click the "X" next to its name in the selection box.

### Reading the Chart

The main chart is an interactive **candlestick chart** powered by Plotly. Here is how to read it:

#### Candlesticks

Each candlestick represents one time period (one day if you selected the 1d timeframe). A candlestick has four components:

- **Open**: The price at the beginning of the period (the top or bottom edge of the "body").
- **Close**: The price at the end of the period (the opposite edge of the body).
- **High**: The highest price during the period (the top of the thin line, called the "wick" or "shadow").
- **Low**: The lowest price during the period (the bottom of the thin line).

**Green candlestick** = the price went UP during this period (close is higher than open). The bottom of the green body is the open price, and the top is the close price.

**Red candlestick** = the price went DOWN during this period (close is lower than open). The top of the red body is the open price, and the bottom is the close price.

#### Interacting with the Chart

- **Hover**: Move your mouse over any candlestick to see the exact Open, High, Low, Close values and date.
- **Zoom in**: Click and drag on the chart to select a region to zoom into.
- **Zoom out**: Double-click anywhere on the chart to reset the zoom.
- **Pan**: After zooming in, click and drag while holding the Shift key to move around.
- **Toolbar**: In the top-right corner of the chart, you will find icons for zoom, pan, home (reset), and download (save as PNG image).

#### Volume Bars

Below the candlestick chart, you will see colored bars representing trading volume:

- **Green bars**: Volume on days when the price closed higher than it opened.
- **Red bars**: Volume on days when the price closed lower than it opened.

Tall bars mean a lot of trading activity. Short bars mean quiet trading. Unusually high volume often accompanies significant price moves.

#### Indicator Panels

When you add oscillator-type indicators (like RSI, MACD, Stochastic), they appear in separate panels below the volume chart. Each panel has its own scale and axis. Overlay-type indicators (like moving averages, Bollinger Bands) appear directly on the candlestick chart because they share the same price scale.

### Data Table and Summary Statistics

Below the chart, you will find two expandable sections. Click the arrow or the section title to expand them.

#### Raw Data

Click **"Raw Data"** to see the last 50 rows of price data in table form. Each row represents one time period and shows:

| Column | Meaning |
|---|---|
| **open** | The price at the start of the period |
| **high** | The highest price during the period |
| **low** | The lowest price during the period |
| **close** | The price at the end of the period |
| **volume** | The number of shares/coins traded |
| *(indicator columns)* | Any computed indicator values (e.g., SMA_20, RSI_14) |

#### Summary Statistics

Click **"Summary Statistics"** to see four key metrics:

- **Close**: The most recent closing price.
- **High**: The highest price in the entire date range.
- **Low**: The lowest price in the entire date range.
- **Avg Daily Return**: The average daily percentage change (a positive number means the asset tended to go up on most days).

---

## Part 3: The Backtester Page

### What is Backtesting?

Backtesting answers the question: **"If I had followed this trading strategy over the past several years, how much money would I have made or lost?"**

Here is how it works:

1. You choose a trading strategy (a set of rules for when to buy and when to sell).
2. The application goes through historical price data day by day.
3. On each day, it checks whether the strategy's rules say to buy, sell, or do nothing.
4. It tracks a simulated portfolio, including trading costs, and calculates performance metrics at the end.

Backtesting does not guarantee future results -- past performance is not a prediction of the future. But it helps you understand how a strategy behaves in different market conditions and lets you compare strategies objectively.

### Choosing a Strategy

To navigate to the Backtester, click **"Backtester"** in the left sidebar navigation. The sidebar will update to show strategy controls.

The **"Strategy"** dropdown offers six pre-built strategies. Here is what each one does in plain English:

#### 1. EMA Crossover

**How it works**: This strategy uses two Exponential Moving Averages -- a fast one (short-term average, default 12 periods) and a slow one (long-term average, default 26 periods).

- **Buy signal**: When the fast average crosses *above* the slow average. This suggests the short-term trend is turning upward.
- **Sell signal**: When the fast average crosses *below* the slow average. This suggests the short-term trend is turning downward.

**Strengths**: Simple, easy to understand, good at catching major trends.
**Weaknesses**: Generates false signals in choppy, sideways markets.

#### 2. RSI Threshold

**How it works**: The RSI (Relative Strength Index) measures whether an asset is "overbought" or "oversold" on a scale of 0 to 100.

- **Buy signal**: When the RSI drops below the oversold threshold (default 30). This means the price has fallen sharply and may bounce back.
- **Sell signal**: When the RSI rises above the overbought threshold (default 70). This means the price has risen sharply and may pull back.

**Strengths**: Good at catching reversals, works well in range-bound markets.
**Weaknesses**: Can signal too early in strong trends ("catching a falling knife").

#### 3. MACD Signal Line

**How it works**: MACD (Moving Average Convergence Divergence) calculates the difference between a fast and slow moving average, then compares it to a "signal line" (a smoothed version of that difference).

- **Buy signal**: When the MACD line crosses *above* the signal line. This indicates rising momentum.
- **Sell signal**: When the MACD line crosses *below* the signal line. This indicates falling momentum.

**Strengths**: Combines trend-following and momentum, widely used by professional traders.
**Weaknesses**: Lags behind the actual price, can be late to signal entries/exits.

#### 4. Bollinger Bands

**How it works**: Bollinger Bands create an upper and lower band around a moving average based on price volatility.

- **Buy signal**: When the price touches or drops below the lower band. This suggests the price is unusually low compared to its recent range.
- **Sell signal**: When the price touches or rises above the upper band. This suggests the price is unusually high.

**Strengths**: Adapts to volatility, good for mean-reversion trading.
**Weaknesses**: In strong trends, price can "ride the band" for extended periods, causing premature exits.

#### 5. Supertrend

**How it works**: Supertrend draws a line that flips between acting as support (below the price during uptrends) and resistance (above the price during downtrends).

- **Buy signal**: When the Supertrend flips from above the price to below the price (uptrend begins).
- **Sell signal**: When the Supertrend flips from below the price to above the price (downtrend begins).

**Strengths**: Clear trend-following signals, good at staying in trends.
**Weaknesses**: Whipsaws (rapid back-and-forth signals) in sideways markets.

#### 6. TTM Squeeze

**How it works**: The TTM Squeeze detects when Bollinger Bands move inside Keltner Channels, indicating that volatility is compressed ("squeezing"). When the squeeze releases, it fires a signal based on momentum direction.

- **Buy signal**: When the squeeze releases and momentum is positive.
- **Sell signal**: When the squeeze releases and momentum is negative.

**Strengths**: Specifically designed to catch the start of big moves after periods of low volatility.
**Weaknesses**: Not all squeezes lead to significant moves; some fizzle out.

### Setting Strategy Parameters

After selecting a strategy, you will see parameter inputs specific to that strategy. Here is what each parameter does and suggested starting values:

#### EMA Crossover Parameters

| Parameter | Default | What It Does | How Changes Affect Results |
|---|---|---|---|
| **Fast Period** | 12 | The number of periods for the short-term average | Lower = more signals, more whipsaws. Higher = fewer signals, smoother. |
| **Slow Period** | 26 | The number of periods for the long-term average | Lower = faster reaction to trends. Higher = catches only major trends. |

**Suggested starting values**: Fast=12, Slow=26 (these are the classic MACD periods and work well across many assets).

#### RSI Parameters

| Parameter | Default | What It Does | How Changes Affect Results |
|---|---|---|---|
| **Period** | 14 | How many bars to use in the RSI calculation | Lower = more sensitive, more signals. Higher = smoother, fewer signals. |
| **Overbought** | 70 | RSI level that triggers a sell signal | Lower = sells sooner. Higher = waits for more extreme overbought conditions. |
| **Oversold** | 30 | RSI level that triggers a buy signal | Lower = buys only at extreme oversold. Higher = buys sooner. |

**Suggested starting values**: Period=14, Overbought=70, Oversold=30 (the standard settings used by most traders).

#### MACD Parameters

| Parameter | Default | What It Does | How Changes Affect Results |
|---|---|---|---|
| **Fast** | 12 | Short-term EMA period | Lower = more responsive, more signals. |
| **Slow** | 26 | Long-term EMA period | Higher = smoother, fewer but stronger signals. |
| **Signal** | 9 | Smoothing period for the signal line | Lower = signal line reacts faster. Higher = more confirmation needed. |

**Suggested starting values**: Fast=12, Slow=26, Signal=9 (the original Gerald Appel settings).

#### Bollinger Bands Parameters

| Parameter | Default | What It Does | How Changes Affect Results |
|---|---|---|---|
| **Period** | 20 | Number of bars for the moving average and standard deviation | Lower = tighter bands, more signals. Higher = wider perspective. |
| **Std Dev** | 2.0 | Number of standard deviations for the bands | Lower = narrower bands, more signals. Higher = wider bands, fewer signals. |

**Suggested starting values**: Period=20, Std Dev=2.0 (John Bollinger's original recommendation).

#### Supertrend Parameters

| Parameter | Default | What It Does | How Changes Affect Results |
|---|---|---|---|
| **Period** | 10 | ATR lookback period | Lower = more sensitive to short-term changes. Higher = smoother. |
| **Multiplier** | 3.0 | ATR multiplier for the Supertrend bands | Lower = closer to price, more signals. Higher = farther from price, fewer signals. |

**Suggested starting values**: Period=10, Multiplier=3.0.

#### TTM Squeeze Parameters

| Parameter | Default | What It Does | How Changes Affect Results |
|---|---|---|---|
| **BB Period** | 20 | Bollinger Band lookback period | Affects how the squeeze zone is calculated. |
| **KC Period** | 20 | Keltner Channel lookback period | Should usually match BB Period. |
| **KC Mult** | 1.5 | Keltner Channel multiplier | Lower = more frequent squeezes. Higher = rarer, potentially more significant squeezes. |

**Suggested starting values**: BB Period=20, KC Period=20, KC Mult=1.5.

### Configuring the Backtest

Below the strategy parameters, you will see the **"Backtest Settings"** section with the following controls:

| Setting | Default | What It Means |
|---|---|---|
| **Initial Capital ($)** | $100,000 | The amount of money your simulated portfolio starts with. |
| **Commission (%)** | 0.1% | The fee charged per trade. This is deducted from your portfolio each time you buy or sell. Most online brokers charge between 0% and 0.5%. |
| **Slippage (%)** | 0.05% | The difference between the expected price and the actual execution price. In real trading, you rarely get the exact price you see on the screen. |
| **Position Size (%)** | 100% | What percentage of your available capital to put into each trade. 100% means "go all in" on each trade. 50% means only use half your capital per trade. |
| **Allow Short** | Checked | Whether the strategy can "short sell" (bet that the price will go down). When checked, sell signals will open short positions. When unchecked, sell signals only close existing long positions. |

**For beginners**: Keep the defaults to start. Once you are comfortable, try changing the commission and slippage to see how trading costs affect your results.

### Running the Backtest

Once you have configured the symbol, timeframe, date range, strategy, parameters, and backtest settings:

1. Click the **"Run Backtest"** button in the main area (the large blue button).
2. You will see spinners that say "Fetching data...", "Computing indicators...", "Generating signals...", and "Running backtest..." as each step completes.
3. After a few seconds, the results will appear below.

### Understanding the Results

The results section starts with **eight metric cards** arranged in two rows. Here is what each one means:

#### Row 1

| Metric | What It Means | Good Values |
|---|---|---|
| **Total Return** | The percentage gain or loss of your portfolio from start to finish. +25% means your $100,000 became $125,000. -10% means it dropped to $90,000. | Positive is good. Compare to the asset's buy-and-hold return. |
| **Max Drawdown** | The largest peak-to-trough decline during the entire period. If your portfolio grew to $120,000 then fell to $96,000 before recovering, the max drawdown is 20%. | Smaller is better. Below -30% is concerning. |
| **Sharpe Ratio** | A measure of risk-adjusted return. It answers: "How much return did I get for each unit of risk I took?" | Above 1.0 is decent. Above 2.0 is very good. Below 0 means you lost money. |
| **Win Rate** | The percentage of trades that were profitable. If 60 out of 100 trades made money, the win rate is 60%. | Above 50% is typical for most strategies. Some profitable strategies have low win rates but large average wins. |

#### Row 2

| Metric | What It Means | Good Values |
|---|---|---|
| **Trades** | The total number of completed buy-sell pairs (round trips). | Depends on the strategy. Too few means not enough data to be statistically meaningful. Too many may indicate over-trading. |
| **Profit Factor** | Gross profits divided by gross losses. A profit factor of 2.0 means you made $2 for every $1 you lost. | Above 1.0 means profitable overall. Above 1.5 is good. Above 2.0 is excellent. |
| **Final Equity** | The dollar value of your portfolio at the end of the backtest period. | Higher than Initial Capital means the strategy was profitable. |
| **Initial Capital** | The starting dollar value (what you entered in settings). | Reference value. |

### Reading the Charts

Below the metrics, you will see several charts:

#### Equity Curve

This is a line chart showing your portfolio value over time. The x-axis is time (dates), and the y-axis is your portfolio value in dollars.

- An **upward-sloping line** means the strategy was making money during that period.
- A **downward-sloping line** means the strategy was losing money.
- **Flat sections** mean the strategy was not in a trade (sitting in cash).

#### Drawdown Chart

This chart shows the "underwater" plot -- how far below the all-time high your portfolio was at each point in time.

- The y-axis shows the drawdown as a negative percentage.
- Deeper dips mean larger losses from the peak.
- The deepest point on this chart corresponds to the Max Drawdown metric.

This chart helps you understand **how painful** the strategy would have been to follow. Even a profitable strategy can have periods of significant loss.

#### Trade Signals

This chart overlays buy and sell markers on the candlestick price chart.

- **Buy markers** (typically green triangles pointing up) show where the strategy entered long positions.
- **Sell markers** (typically red triangles pointing down) show where the strategy exited or went short.

This helps you visually verify that the strategy is behaving as expected and trading at reasonable points.

#### P&L Analysis

This chart shows the profit or loss of each individual trade as a bar chart.

- **Green bars** = winning trades (positive P&L).
- **Red bars** = losing trades (negative P&L).
- Tall bars = large gains or losses.

Look for patterns: are the wins larger than the losses? Are there clusters of losing trades?

### The Trade Log

At the bottom of the results, click **"Trade Log"** to expand the full list of every trade the strategy executed.

| Column | Meaning |
|---|---|
| **entry_time** | The date/time the trade was opened |
| **exit_time** | The date/time the trade was closed |
| **entry_price** | The price at which the trade was opened |
| **exit_price** | The price at which the trade was closed |
| **quantity** | The number of shares/units traded |
| **direction** | "long" (bought first, sold later) or "short" (sold first, bought back later) |
| **pnl** | Raw profit or loss (before costs) |
| **pnl_pct** | Profit or loss as a percentage of the trade value |
| **commission** | Trading fees for this trade |
| **slippage** | Slippage cost for this trade |
| **net_pnl** | Final profit or loss after deducting commission and slippage |
| **holding_period** | How many bars (days, if using daily timeframe) the position was held |

**Tip**: Look at the losing trades first. Understanding *why* trades lost money is more educational than celebrating the winners.

---

## Part 4: The Compare Page

The Compare page lets you compare multiple assets or multiple strategies side by side. Click **"Compare"** in the sidebar navigation.

You will see two tabs at the top: **"Multi-Asset"** and **"Multi-Strategy"**.

### Multi-Asset Comparison

This tab lets you compare the performance of several assets over the same time period.

#### How to Use It

1. In the sidebar, select a **Timeframe** and set the **Date Range**.
2. In the main area, enter your symbols in the **"Symbols (comma-separated)"** text box. The default is `AAPL, MSFT, GOOGL, BTC-USD`.
3. Click **"Compare Assets"**.

#### What You Will See

**Normalized Performance Chart**: A line chart showing each asset's cumulative return, starting at 1.0. This makes it easy to compare assets with very different price levels. For example, if AAPL starts at 1.0 and ends at 1.5, it returned 50% over the period.

**Return Summary Table**: A table with one row per asset showing:

| Column | Meaning |
|---|---|
| **Symbol** | The ticker symbol |
| **Total Return** | Total percentage return over the period |
| **Ann. Volatility** | Annualized volatility (a measure of risk -- higher means more price swings) |
| **Sharpe** | Sharpe Ratio (risk-adjusted return) |
| **Max DD** | Maximum drawdown (worst peak-to-trough decline) |

**Correlation Matrix**: A heatmap showing how closely each pair of assets moves together.

- A value close to **+1.0** means the two assets tend to move in the same direction at the same time (highly correlated).
- A value close to **-1.0** means they tend to move in opposite directions (negatively correlated).
- A value close to **0** means their movements are mostly independent.

This is useful for **diversification**: if you own two assets that are highly correlated, they will both go down at the same time. Ideally, you want a mix of assets with low or negative correlation.

### Multi-Strategy Comparison

This tab lets you test multiple trading strategies on the same asset and compare their results.

#### How to Use It

1. Enter a **Symbol** (e.g., `AAPL`).
2. Select a **Timeframe** and set the **Date Range**.
3. In the **"Strategies to compare"** multiselect, choose two or more strategies. The defaults are `ema_cross` and `rsi`.
4. Configure the **Backtest Settings** (capital, commission, slippage, etc.).
5. Click **"Compare Strategies"**.

Note: All strategies will use their default parameters for this comparison.

#### What You Will See

**Equity Curves**: A line chart with one line per strategy, showing how each strategy's portfolio value changed over time. The strategies that end higher performed better.

**Strategy Comparison Table**: A table with one row per strategy showing:

| Column | Meaning |
|---|---|
| **Strategy** | Strategy name |
| **Total Return** | Total percentage return |
| **Max Drawdown** | Worst peak-to-trough decline |
| **Sharpe** | Risk-adjusted return |
| **Win Rate** | Percentage of winning trades |
| **Trades** | Number of completed trades |
| **Profit Factor** | Gross profit / gross loss |

**Tip**: Do not assume the strategy with the highest total return is the "best." Look at the Sharpe Ratio and Max Drawdown too. A strategy that returns 30% but has a 50% drawdown may be worse than one that returns 20% with only a 10% drawdown.

---

## Part 5: The Analysis Page

The Analysis page provides the deepest level of analysis available in the application. It combines a backtest with comprehensive performance analytics, risk measurement, market regime detection, and return distribution analysis.

Click **"Analysis"** in the sidebar navigation.

### Running an Analysis

The sidebar controls are the same as the Backtester:

1. Enter a **Symbol**, select a **Timeframe**, and set the **Date Range**.
2. Choose a **Strategy** and adjust its **Parameters**.
3. Configure the **Backtest Settings**.
4. Click the **"Run Analysis"** button in the main area.

The analysis runs a full backtest behind the scenes and then generates four tabs of detailed results.

### Tear Sheet Tab

A "tear sheet" is a comprehensive one-page summary of a strategy's performance. Professional fund managers use tear sheets to evaluate trading systems. This tab organizes metrics into several categories.

#### Returns Metrics

| Metric | What It Means |
|---|---|
| **CAGR** | Compound Annual Growth Rate -- the annualized return. If your strategy returned 50% over 5 years, the CAGR is the steady annual rate that would produce that result (about 8.4% per year). |
| **Sharpe Ratio** | Risk-adjusted return. How much excess return you earned per unit of risk. Above 1.0 is acceptable, above 2.0 is strong. |
| **Sortino Ratio** | Like Sharpe but only penalizes downside volatility (bad volatility). Ignores upside swings, which are desirable. Higher is better. |
| **Calmar Ratio** | Annualized return divided by maximum drawdown. Tells you how much return you got per unit of worst-case risk. Higher is better. |
| **Omega Ratio** | The probability-weighted ratio of gains over losses. Above 1.0 means gains outweigh losses. |
| **Volatility** | Annualized standard deviation of returns. Higher means more unpredictable returns. |

#### Risk Metrics

| Metric | What It Means |
|---|---|
| **Max Drawdown** | The largest peak-to-trough decline in portfolio value. |
| **Max Drawdown Duration** | The longest period (in bars/days) the portfolio stayed below its all-time high. |
| **Recovery Factor** | Net profit divided by maximum drawdown (in dollar terms). Higher means the strategy recovered from losses more effectively. |
| **VaR 95%** | Value at Risk at the 95% confidence level. "On 95% of days, the daily loss will not exceed this percentage." |
| **CVaR 95%** | Conditional Value at Risk (also called Expected Shortfall). The average loss on the worst 5% of days. This is a more conservative risk measure than VaR. |
| **Downside Deviation** | Standard deviation of only the negative returns. A more relevant risk measure than total volatility because it only counts the "bad" volatility. |

#### Trade Metrics

| Metric | What It Means |
|---|---|
| **Total Trades** | Number of completed round-trip trades. |
| **Win Rate** | Percentage of trades that made money. |
| **Profit Factor** | Total profit from winners / total loss from losers. |
| **Expectancy** | Average dollar profit or loss per trade. Positive means the strategy is profitable on average. |
| **Avg Win** | Average profit of winning trades. |
| **Avg Loss** | Average loss of losing trades (shown as a negative number). |
| **Largest Win** | The single most profitable trade. |
| **Largest Loss** | The single worst trade. |
| **Consecutive Wins** | The longest winning streak. |
| **Consecutive Losses** | The longest losing streak. |

#### Distribution Metrics

| Metric | What It Means |
|---|---|
| **Skewness** | Whether returns are lopsided. Positive skew means more frequent small losses but occasional large wins. Negative skew means the opposite. |
| **Kurtosis** | Whether returns have "fat tails" (extreme events more common than a normal bell curve would predict). Higher kurtosis means more surprises. |
| **Tail Ratio** | Ratio of the right tail (gains) to the left tail (losses). Above 1.0 means the upside potential is larger than the downside risk. |
| **Common Sense Ratio** | Tail ratio multiplied by profit factor. A combined measure of edge quality. |

#### Monthly Returns Heatmap

Below the metrics, you will see a color-coded table showing returns for each month of each year:

- **Green cells** = positive return that month (the darker the green, the higher the return).
- **Red cells** = negative return that month (the darker the red, the larger the loss).

This helps you spot seasonal patterns and understand when the strategy performs best or worst.

#### Drawdown Analysis

Below the heatmap, you will see:

- The **Max Drawdown** value.
- A table listing the **top 5 worst drawdown periods**, showing when each started, when it reached its deepest point, when it recovered, how deep it went, and how long recovery took.
- A chart highlighting these drawdown periods visually.

### Risk Analysis Tab

This tab focuses specifically on risk measurement.

#### Value at Risk Section

Three metrics are displayed:

| Metric | What It Means |
|---|---|
| **Parametric VaR (95%)** | Calculated assuming returns follow a normal (bell curve) distribution. "With 95% confidence, your daily loss will not exceed X%." |
| **Historical VaR (95%)** | Calculated using the actual historical return data (no distribution assumption). Generally more accurate than parametric VaR. |
| **CVaR (95%)** | The average loss on the worst 5% of days. More conservative than VaR because VaR only tells you the threshold, while CVaR tells you how bad it gets beyond that threshold. |

**Example**: If Historical VaR is 2.5% and your portfolio is $100,000, then on 95% of days you would expect to lose no more than $2,500. However, on the worst 5% of days, losses could exceed that amount. CVaR tells you how much you lose on average during those worst days.

#### Drawdown Chart

An "underwater" plot showing how far below the portfolio's all-time high it was at each point in time. This is the same chart type as in the Backtester but is included here for the risk-focused context.

### Regime Detection Tab

This tab uses a machine learning algorithm (KMeans clustering) to automatically detect different **market regimes** -- distinct phases where the market behaves differently.

#### What Are Market Regimes?

Financial markets are not static. They cycle through different "moods":

- **Bull market**: Prices are generally rising. Optimism is high.
- **Bear market**: Prices are generally falling. Pessimism is dominant.
- **Neutral/sideways market**: Prices are moving in a range without a clear direction.

The regime detector analyzes historical returns and groups them into three clusters based on their characteristics.

#### What You Will See

**Regime Statistics Table**: Shows the mean return, standard deviation, and number of observations for each detected regime. You can see how different the regimes are from each other.

**Price Chart with Regime Overlay**: The price chart with colored dots overlaid:

- **Red dots** = Bear regime (negative average returns).
- **Gray dots** = Neutral regime (near-zero average returns).
- **Green dots** = Bull regime (positive average returns).

This visualization helps you understand that your strategy's performance is partly determined by which regime the market was in. A trend-following strategy will excel in bull and bear regimes but struggle in neutral regimes.

### Distributions Tab

This tab shows the statistical distribution of your strategy's returns.

#### Return Histogram

A histogram (bar chart) showing how frequently each return level occurred. The x-axis is the daily return percentage, and the y-axis is how many days had that return.

A perfectly "normal" distribution would look like a symmetric bell curve centered on zero. Real financial returns often have:

- **Fat tails**: Extreme events happen more often than expected.
- **Negative skew**: Large losses are more frequent than large gains.
- **Leptokurtosis**: The peak is taller and tails are fatter than a normal curve.

#### Key Statistics

Four summary statistics are displayed:

| Statistic | What It Means |
|---|---|
| **Mean** | The average daily return. A small positive number is typical for profitable strategies. |
| **Std Dev** | Standard deviation of daily returns. Measures how spread out the returns are. |
| **Skewness** | Asymmetry of the distribution. Positive = right-skewed (long right tail). Negative = left-skewed (long left tail). Zero = symmetric. |
| **Kurtosis** | "Peakedness" of the distribution relative to a normal curve. Above 0 = fat tails (more extreme events). Below 0 = thin tails (fewer extreme events). |

---

## Part 6: Tips and Best Practices

### For Beginners

1. **Start simple.** Use the Dashboard with well-known stocks (AAPL, MSFT, SPY) on the daily timeframe before diving into backtesting.

2. **Learn one indicator at a time.** Add a single indicator to the Dashboard, study how it moves relative to the price, and understand it before adding more.

3. **Use default parameters first.** The default values for each strategy and indicator are industry-standard settings that have been used for decades. Change them only after you understand what they do.

4. **Compare with costs vs. without.** Run a backtest with 0% commission and slippage, then run it again with realistic costs (0.1% commission, 0.05% slippage). The difference is often eye-opening.

5. **Use enough historical data.** Test strategies over at least 3-5 years to capture different market conditions (bull markets, bear markets, sideways periods). Testing over just 6 months may give misleading results.

6. **No strategy works all the time.** Every strategy has periods where it loses money. The key is whether it makes more money than it loses over the long run.

7. **The Sharpe Ratio matters more than Total Return.** A strategy with 50% return and a Sharpe of 0.5 is probably worse than a strategy with 20% return and a Sharpe of 2.0. The second strategy achieved its returns with much less risk.

### Common Mistakes

1. **Overfitting**: Tweaking strategy parameters until they work perfectly on historical data. This is like studying the answers to last year's exam -- it will not help you on the next one. If you try 100 parameter combinations and pick the best one, you have probably found noise, not a real pattern.

2. **Ignoring transaction costs**: A strategy that trades 500 times per year with 0.1% commission per trade spends 50% of its capital on commissions alone. Always include realistic costs.

3. **Survivorship bias**: Only testing on stocks that exist today. Companies that went bankrupt are not in the data. This makes strategies appear more profitable than they actually are.

4. **Testing on too short a time period**: A strategy that works over 6 months may just be lucky. Use at least 3-5 years of data.

5. **Using too many indicators at once**: Adding 10 indicators to a chart does not make you a better analyst. Most of them will give conflicting signals. Focus on 2-3 indicators that provide different types of information (e.g., one trend indicator and one momentum indicator).

6. **Confusing correlation with causation**: Just because an indicator signal preceded a price move does not mean the indicator caused or predicted the move.

### Keyboard Shortcuts and Tips

- **Ctrl+C** in the Command Prompt: Stops the application server.
- **F5** or **Ctrl+R** in the browser: Refreshes the page and resets all inputs to defaults.
- **Ctrl+Click** on chart toolbar icons to activate different chart interaction modes.
- If the application becomes slow or unresponsive, stop it with Ctrl+C in the Command Prompt and restart with `streamlit run app.py`.

---

## Part 7: Frequently Asked Questions

**Q: Can I use this for real trading?**
A: This application is designed for research and education only. It does not connect to any brokerage account and cannot execute real trades. Backtesting results are hypothetical and do not guarantee future performance. If you decide to trade with real money based on insights from this tool, you do so at your own risk.

**Q: Is the data free?**
A: Yes. The application uses Yahoo Finance data, which is free for personal use. No API key or subscription is required.

**Q: How far back does the historical data go?**
A: It depends on the asset and timeframe. For major US stocks on a daily timeframe, you can typically get 20+ years of data. For cryptocurrency, data usually starts from when the coin was first listed (e.g., Bitcoin from around 2014 on Yahoo Finance). For intraday data (1-minute or 5-minute bars), Yahoo Finance only provides the last 7-30 days.

**Q: Why are some indicator values blank or missing at the start of the chart?**
A: Most indicators need a "warmup" period. For example, a 20-day SMA needs at least 20 days of data before it can calculate its first value. The first 19 days will be blank. This is normal.

**Q: Can I save my results?**
A: Currently, results are displayed in the browser and are not automatically saved. However, you can:
- Take screenshots of the charts.
- Use the download button in the top-right corner of Plotly charts to save them as PNG images.
- Copy data from tables by selecting and using Ctrl+C.

**Q: What if I get an error saying "No data returned"?**
A: This usually means the symbol you entered is not recognized by Yahoo Finance, or you selected a date range where no data exists. Double-check the ticker symbol. For crypto, remember to add `-USD` (e.g., `BTC-USD`, not just `BTC`). For forex, use the `=X` suffix (e.g., `EURUSD=X`).

**Q: Why does the backtest take a long time?**
A: The first time you fetch data for a symbol, it is downloaded from the internet. Subsequent requests for the same symbol within 5 minutes will use cached data and be much faster. Large date ranges with short timeframes (e.g., 5 years of hourly data) involve a lot of data and may take longer to process.

**Q: Can I add my own custom strategy?**
A: Yes, but this requires Python programming knowledge. See the Developer Guide for instructions on creating custom strategies using the framework's strategy registry.

**Q: What does "Allow Short" mean?**
A: "Shorting" or "short selling" is a trading technique where you sell an asset you do not own, with the intent of buying it back later at a lower price. You profit when the price goes down. When "Allow Short" is checked, the backtester will open short positions on sell signals. When unchecked, sell signals only close existing long positions.

**Q: Why do different strategies show different numbers of trades?**
A: Each strategy has its own logic for deciding when to buy and sell. Some strategies (like RSI) may trade frequently, while others (like Supertrend) may hold positions for long periods. The number of trades also depends on the asset's price behavior during the test period.

**Q: What is a good Sharpe Ratio?**
A: As a rough guide:
- Below 0: The strategy lost money.
- 0 to 0.5: Poor risk-adjusted return.
- 0.5 to 1.0: Below average.
- 1.0 to 2.0: Good.
- 2.0 to 3.0: Very good.
- Above 3.0: Exceptional (and possibly too good -- double-check for overfitting).

---

## Part 8: Troubleshooting

### Problem: "python is not recognized as an internal or external command"

**What you see**: When you type `python --version` in the Command Prompt, you get an error message.

**Why it happens**: Python was not added to your system's PATH during installation.

**How to fix it**:
1. Uninstall Python: Go to Windows Settings > Apps > Installed Apps, find Python, and click Uninstall.
2. Re-download the Python installer from https://www.python.org/downloads/.
3. Run the installer and this time **make sure to check the "Add Python to PATH" checkbox** on the first screen.
4. Click "Install Now."
5. Open a new Command Prompt and try `python --version` again.

---

### Problem: "pip is not recognized as an internal or external command"

**What you see**: When you type `pip install -r requirements.txt`, you get an error.

**Why it happens**: The `pip` command is not on your PATH, even though Python is installed.

**How to fix it**: Use the alternative command:
```
python -m pip install -r requirements.txt
```

---

### Problem: "ModuleNotFoundError: No module named 'streamlit'"

**What you see**: When you run `streamlit run app.py`, you get an import error.

**Why it happens**: The required packages were not installed, or they were installed for a different Python version.

**How to fix it**:
1. Make sure you are in the project directory:
   ```
   cd C:\Users\simon\Downloads\pandas_TA_neuralNine
   ```
2. Reinstall the requirements:
   ```
   pip install -r requirements.txt
   ```
3. If you have multiple Python versions installed, try:
   ```
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

---

### Problem: "Failed to fetch data for [symbol]" or "No data returned"

**What you see**: An error message appears in the application after entering a symbol.

**Why it happens**: The symbol is not recognized by Yahoo Finance, or there is no data for the date range you selected.

**How to fix it**:
1. Double-check the ticker symbol. Common mistakes:
   - Crypto must include `-USD`: use `BTC-USD`, not `BTC`.
   - Forex must include `=X`: use `EURUSD=X`, not `EURUSD`.
   - Indices must include `^`: use `^GSPC`, not `GSPC`.
2. Try a shorter date range or a different timeframe. Yahoo Finance has limited intraday data history.
3. Make sure you have an internet connection.
4. Some symbols may not be available on Yahoo Finance. Try searching for the symbol on https://finance.yahoo.com/ first.

---

### Problem: "Could not compute [indicator name]"

**What you see**: A warning appears saying an indicator could not be computed.

**Why it happens**: The indicator requires more data than is available, or the indicator is not compatible with the selected asset type (e.g., VWAP requires intraday data with volume).

**How to fix it**:
1. Extend your date range to include more historical data.
2. Switch to a timeframe that provides more bars (e.g., use daily instead of weekly).
3. Try a different indicator.

---

### Problem: "Backtest failed" or "Signal generation failed"

**What you see**: An error appears after clicking "Run Backtest."

**Why it happens**: There may not be enough data for the strategy's indicators to calculate values, or the data may have unexpected gaps.

**How to fix it**:
1. Extend your date range (use at least 1-2 years of daily data).
2. Check that you selected a valid symbol with available data.
3. If using short timeframes (1m, 5m), note that limited history may not provide enough bars for indicators with longer lookback periods.

---

### Problem: The application is slow or the browser is not responding

**What you see**: The page takes a long time to load, or the browser shows "waiting" for an extended period.

**Why it happens**: Large amounts of data (e.g., 5 years of hourly data) or many indicators can slow down processing.

**How to fix it**:
1. Reduce the date range.
2. Use a longer timeframe (daily instead of hourly).
3. Remove some indicators from the Dashboard.
4. Close other browser tabs and applications to free up memory.
5. If the app is completely frozen, go to the Command Prompt and press Ctrl+C to stop it, then restart with `streamlit run app.py`.

---

### Problem: Charts are blank or not displaying

**What you see**: The chart area is empty even though data appears to have loaded.

**Why it happens**: The browser may not be rendering Plotly charts correctly.

**How to fix it**:
1. Try refreshing the page (F5 or Ctrl+R).
2. Try a different browser (Chrome, Firefox, or Edge all work well with Plotly).
3. Make sure JavaScript is enabled in your browser settings.
4. Clear your browser cache and reload.

---

### Problem: "Address already in use" when starting the application

**What you see**: An error about port 8501 being in use.

**Why it happens**: A previous instance of the application is still running.

**How to fix it**:
1. Check if another Command Prompt window is running the app and close it (Ctrl+C).
2. If you cannot find the old instance, you can specify a different port:
   ```
   streamlit run app.py --server.port 8502
   ```
   Then open `http://localhost:8502` in your browser.

---

*This guide covers version 0.1.0 of the TA Framework. For developer documentation, see the Developer Guide. For a quick hands-on tutorial, see the Quick Start Guide.*
