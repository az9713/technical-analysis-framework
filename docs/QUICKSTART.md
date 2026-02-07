# Quick Start Guide: 10 Things to Try

Welcome to the TA Framework. This guide walks you through 10 hands-on exercises, each taking 2-5 minutes. By the end, you will know how to use every page of the application and you will have learned core concepts of technical analysis along the way.

**No financial knowledge required.** Everything is explained as we go.

---

## How to Use This Guide

- Do the exercises in order -- each one builds on what you learned in the previous one.
- Each exercise has a **Goal** (what you will accomplish), **What You Will Learn** (the concept), step-by-step instructions, and a **Takeaway** at the end.
- If you have not installed the application yet, follow the installation steps in the [User Guide](USER_GUIDE.md#installing-the-application) first.

### Before You Start

Open the Command Prompt and start the application:

```
cd C:\Users\simon\Downloads\pandas_TA_neuralNine
streamlit run app.py
```

Your browser will open to the application. You should see the **Dashboard** page with a chart already loaded for AAPL (Apple Inc.).

---

## Use Case 1: View Your First Stock Chart

**Goal**: See Apple's stock price history displayed as a professional candlestick chart.

**What you will learn**: How to read candlestick charts and what OHLCV (Open, High, Low, Close, Volume) data looks like.

### Steps

1. You should already be on the **Dashboard** page. If not, click **"Dashboard"** in the left sidebar.
2. The Symbol field is pre-filled with **AAPL** (Apple Inc.). Leave it as-is.
3. The Timeframe is set to **1d** (daily). Leave it as-is.
4. The date range covers the last 5 years. Leave it as-is.
5. Look at the chart in the main area. You are seeing Apple's stock price history displayed as candlesticks.
6. **Hover your mouse** over any candlestick. A tooltip appears showing the exact Open, High, Low, and Close prices for that day, along with the date.
7. Notice the colors:
   - **Green candlesticks** = the price went UP that day (the close was higher than the open).
   - **Red candlesticks** = the price went DOWN that day (the close was lower than the open).
8. Look at the **volume bars** at the bottom of the chart. These colored bars show how many shares of Apple were traded each day. Tall bars mean heavy trading activity; short bars mean quiet days.
9. **Zoom in** by clicking and dragging across a section of the chart. This lets you focus on a specific time period.
10. **Double-click** anywhere on the chart to zoom back out to the full view.
11. Now try changing the symbol: clear the text box in the sidebar and type **MSFT** (Microsoft). Press Enter. The chart updates to show Microsoft's price history.
12. Try **GOOGL** (Google), **TSLA** (Tesla), or **BTC-USD** (Bitcoin) to see how different assets look.

### Takeaway

Stock prices are recorded as four values for each period: Open (the price at the start of the day), High (the highest price during the day), Low (the lowest), and Close (the price at the end of the day). Volume tells you how actively the asset was traded. This "OHLCV" data is the foundation of all technical analysis.

---

## Use Case 2: Add a Moving Average

**Goal**: Overlay a Simple Moving Average (SMA) on the chart to see the underlying trend.

**What you will learn**: What moving averages are and how they smooth out daily price noise to reveal trends.

### Steps

1. Make sure you are on the **Dashboard** page with **AAPL** loaded.
2. In the left sidebar, scroll down past the date range pickers. You will see multiselect boxes grouped by category.
3. Find the box labeled **"Trend Indicators"** and click on it.
4. Select **sma** from the dropdown. The chart updates immediately.
5. You now see a smooth line weaving through the candlesticks. This is the **20-day Simple Moving Average** -- it calculates the average closing price of the last 20 days, plotted for each day.
6. Notice how the SMA line is smoother than the choppy day-to-day candlestick movement. This is the whole point: it filters out the noise.
7. Look at periods where the price is consistently **above the SMA line**. These are uptrends. The price is higher than its recent average, meaning it has been going up.
8. Now look at periods where the price is consistently **below the SMA line**. These are downtrends. The price has been falling relative to its recent average.
9. Now add **ema** from the same Trend Indicators menu. A second line appears. This is the **Exponential Moving Average**, which gives more weight to recent prices and therefore reacts faster to price changes.
10. Compare the two lines: the EMA (Exponential) hugs the price more closely and turns sooner. The SMA (Simple) is smoother and slower to react.

### Takeaway

Moving averages smooth out daily price fluctuations to reveal the underlying trend. When the price is above the moving average, the trend is generally up. When below, the trend is generally down. Shorter-period averages react faster; longer-period averages are smoother. This simple concept is the basis of many trading strategies.

---

## Use Case 3: Spot Overbought and Oversold Conditions with RSI

**Goal**: Add the RSI indicator and learn to identify when a stock might be overbought or oversold.

**What you will learn**: How the Relative Strength Index (RSI) works and what "overbought" and "oversold" mean.

### Steps

1. Still on the **Dashboard** with AAPL loaded, look at the sidebar.
2. Find the **"Momentum Indicators"** multiselect box and click on it.
3. Select **rsi**. A new panel appears below the main price chart.
4. This panel shows the RSI on a scale from 0 to 100.
5. Notice two important levels:
   - **70 line** (overbought zone): When RSI rises above 70, the stock has been bought aggressively and may be due for a pullback or pause. Think of it like a rubber band stretched too far in one direction.
   - **30 line** (oversold zone): When RSI drops below 30, the stock has been sold aggressively and may be due for a bounce. The rubber band has been stretched too far in the other direction.
6. Scroll through the chart and find a moment where the RSI dipped below 30. Now look at what the price did shortly after. In many cases, the price bounced upward.
7. Find a moment where the RSI climbed above 70. The price often paused or pulled back afterward.
8. **Important caveat**: This does not work every time. In strong uptrends, the RSI can stay above 70 for weeks while the price keeps rising. RSI is one tool among many, not a standalone trading system.

### Takeaway

RSI measures momentum -- how fast and how far the price has moved in one direction. Readings above 70 suggest overbought conditions (a possible pullback), and readings below 30 suggest oversold conditions (a possible bounce). It is a useful gauge, but should always be considered alongside other factors.

---

## Use Case 4: Your First Backtest

**Goal**: Test the EMA Crossover strategy on Apple stock and see whether it would have been profitable.

**What you will learn**: How backtesting works, what the key results mean, and how to read an equity curve.

### Steps

1. Click **"Backtester"** in the left sidebar navigation. The page changes to the Strategy Backtester.
2. In the sidebar:
   - **Symbol**: AAPL (should already be set).
   - **Timeframe**: 1d (daily).
   - **Date Range**: Leave the defaults (last 5 years).
3. **Strategy**: The dropdown should show **"EMA Crossover"**. This strategy buys when a fast-moving average crosses above a slow-moving average, and sells when it crosses below.
4. **Parameters**: Leave the defaults: Fast Period = 12, Slow Period = 26.
5. **Backtest Settings**: Leave the defaults: $100,000 initial capital, 0.1% commission, 0.05% slippage, 100% position size, Allow Short checked.
6. Click the **"Run Backtest"** button (the large blue button in the main area).
7. Wait a few seconds. The results appear below.
8. Look at the four key metrics in the first row:
   - **Total Return**: Did the strategy make or lose money overall? A positive percentage means profit.
   - **Max Drawdown**: The worst decline from peak to trough. This tells you how painful the worst losing streak was.
   - **Sharpe Ratio**: A measure of risk-adjusted return. Above 1.0 is considered decent.
   - **Win Rate**: What percentage of trades were winners?
9. Scroll down to the **Equity Curve**. This line chart shows your simulated portfolio value over time. An upward slope means the strategy was making money; a downward slope means it was losing.
10. Below that, look at the **Drawdown** chart. This "underwater" plot shows how far below the all-time high your portfolio was at each point. Deep dips represent painful losing periods.
11. Scroll down further to **Trade Signals** to see buy and sell markers on the price chart.
12. Finally, click **"Trade Log"** to expand the full list of every trade: when it entered, when it exited, the entry price, exit price, and profit or loss.

### Takeaway

Backtesting lets you test a "what if" scenario on historical data. It answers: "If I had followed this strategy over the past N years, what would have happened?" The results give you objective metrics to evaluate the strategy. Remember that past performance does not guarantee future results, but it helps you understand how a strategy behaves.

---

## Use Case 5: See How Transaction Costs Affect Returns

**Goal**: Run the same backtest with higher trading costs and observe the impact.

**What you will learn**: Transaction costs can dramatically reduce (or even eliminate) a strategy's profitability.

### Steps

1. You should still be on the **Backtester** page with the EMA Crossover results from Use Case 4.
2. **Write down** (or remember) the current **Total Return** and **Final Equity** values.
3. In the sidebar, scroll down to **Backtest Settings**.
4. Change **Commission** from 0.1% to **0.5%** (five times higher).
5. Change **Slippage** from 0.05% to **0.2%** (four times higher).
6. Click **"Run Backtest"** again.
7. Compare the new results to what you wrote down:
   - How much did the Total Return change?
   - Is the Final Equity much lower?
   - Look at the Equity Curve -- is it flatter or even declining now?
8. The difference between the two runs shows you exactly how much of the strategy's profit was consumed by trading costs.
9. Now try the extreme: set Commission to **1.0%** and Slippage to **0.5%**. Run the backtest again. You may find that the strategy is now a net loser.
10. **Reset** the values back to the defaults (Commission = 0.1%, Slippage = 0.05%) when you are done.

### Takeaway

Always account for transaction costs when evaluating a strategy. A strategy that looks profitable in theory may become unprofitable after factoring in commissions and slippage. This is especially true for strategies that trade frequently. The more trades a strategy makes, the more costs eat into profits. In real-world trading, costs can be the difference between a winning and losing strategy.

---

## Use Case 6: Compare Apple vs Bitcoin

**Goal**: See how a traditional stock (Apple) compares to a cryptocurrency (Bitcoin) in terms of return and risk.

**What you will learn**: Different asset classes have very different risk/return profiles, and diversification matters.

### Steps

1. Click **"Compare"** in the left sidebar navigation.
2. You will see two tabs at the top: **"Multi-Asset"** and **"Multi-Strategy"**. Make sure **"Multi-Asset"** is selected.
3. In the sidebar, keep the Timeframe as **1d** and the date range as the last 5 years.
4. In the main area, you will see a text box labeled **"Symbols (comma-separated)"**. Clear it and type:
   ```
   AAPL, BTC-USD
   ```
5. Click **"Compare Assets"**.
6. Look at the **Normalized Performance** chart. Both assets start at 1.0 on the left side. The line that ends higher had a better total return. Notice how much more volatile (wiggly) the Bitcoin line is compared to Apple.
7. Look at the **Return Summary** table:
   - Compare the **Total Return**: Which asset made more money?
   - Compare the **Ann. Volatility**: Which asset had more price swings? (Higher = more volatile.)
   - Compare the **Max DD** (Max Drawdown): Which asset had a worse peak-to-trough decline?
8. Look at the **Correlation Matrix** heatmap. The value in the off-diagonal cell shows how closely Apple and Bitcoin move together. A low number (close to 0) means they are relatively independent.
9. Now try a broader comparison. Change the symbols to:
   ```
   AAPL, MSFT, BTC-USD, ETH-USD, GLD
   ```
   (Apple, Microsoft, Bitcoin, Ethereum, and Gold.)
10. Click **"Compare Assets"** again. The normalized chart now shows five lines. Compare how they performed and how correlated they are.

### Takeaway

Cryptocurrency is typically much more volatile than traditional stocks -- it can deliver higher returns, but with much larger drawdowns. The correlation matrix shows you that assets which move independently of each other provide better diversification. Owning assets with low correlation means when one goes down, the other may not follow, reducing overall portfolio risk.

---

## Use Case 7: RSI Strategy vs MACD Strategy

**Goal**: Compare two different trading strategies on the same stock to see which performs better.

**What you will learn**: There is no single "best" strategy. Different strategies have different strengths and weaknesses.

### Steps

1. On the **Compare** page, click the **"Multi-Strategy"** tab.
2. In the **Symbol** field, type **AAPL**.
3. Select **Timeframe**: 1d and set the date range to the last 5 years.
4. In the **"Strategies to compare"** multiselect, select: **ema_cross** and **rsi**. (These may already be selected as defaults.)
5. Keep the **Backtest Settings** at their defaults ($100,000 capital, 0.1% commission, etc.).
6. Click **"Compare Strategies"**.
7. Look at the **Equity Curves** chart. Two lines appear, one for each strategy. The line that ends higher had a better total return. But also notice which one was smoother (less volatile) vs. which one had sharper ups and downs.
8. Look at the **Strategy Comparison** table:
   - Which strategy had a higher **Total Return**?
   - Which had a lower **Max Drawdown** (less risk)?
   - Which had a higher **Sharpe Ratio** (better risk-adjusted return)?
   - Which had a higher **Win Rate**?
   - Which had more **Trades**?
9. Now add more strategies. Go back to the multiselect and also add **macd** and **bbands**. Click **"Compare Strategies"** again.
10. With four strategies on the chart, you can see how they diverge over time. Some strategies that start well may falter later, while others improve as market conditions change.

### Takeaway

No single strategy wins in all market conditions. A trend-following strategy (like EMA Crossover) excels when the market has clear trends but struggles in sideways markets. A mean-reversion strategy (like RSI or Bollinger Bands) works well in ranges but gets crushed in strong trends. The "best" strategy depends on the asset and the time period. Smart analysis means understanding these trade-offs rather than searching for a perfect strategy.

---

## Use Case 8: Deep Dive with the Tear Sheet

**Goal**: Generate a comprehensive performance report (tear sheet) for a strategy applied to the S&P 500.

**What you will learn**: How professional traders and fund managers evaluate trading strategies with detailed metrics.

### Steps

1. Click **"Analysis"** in the left sidebar navigation.
2. In the sidebar:
   - **Symbol**: Type **SPY** (this is the ETF that tracks the S&P 500 index, representing the overall US stock market).
   - **Timeframe**: 1d.
   - **Date Range**: Last 5 years (the default).
   - **Strategy**: Select **"MACD Signal Line"**.
   - Leave all parameters and backtest settings at their defaults.
3. Click the **"Run Analysis"** button.
4. The **Tear Sheet** tab is shown by default. You will see metrics organized into categories:
5. **Returns metrics**: Look at the **CAGR** (Compound Annual Growth Rate). This is the annualized return. Also note the **Sharpe Ratio** and **Sortino Ratio** (Sortino only penalizes downside risk, so it is often a more relevant measure).
6. **Risk metrics**: Find the **Max Drawdown** -- how bad was the worst decline? The **Max Drawdown Duration** tells you how many trading days the portfolio stayed below its peak.
7. **Trade metrics**: Check the **Win Rate**, **Profit Factor**, and **Expectancy** (average profit per trade). A positive expectancy means the strategy makes money on average per trade.
8. Scroll down to the **Monthly Returns** heatmap. This is a color-coded grid:
   - Rows = years, Columns = months (Jan through Dec).
   - **Green cells** = positive returns that month.
   - **Red cells** = negative returns that month.
   - Look for seasonal patterns. Are certain months consistently green or red?
9. Continue scrolling to the **Drawdown Analysis** section. You will see a table listing the top 5 worst drawdown periods: when each started, how deep it went, and how long recovery took.
10. Below the table, a chart highlights these drawdown periods visually.

### Takeaway

A tear sheet gives you the complete picture of strategy performance in a standardized format. Professional fund managers review tear sheets before allocating capital to any strategy. The key insight is that you should never judge a strategy by return alone. Risk metrics (drawdown, VaR) and consistency metrics (Sharpe, Sortino) are equally important. A strategy that returns 15% per year with a 10% max drawdown is often preferable to one that returns 25% per year with a 50% max drawdown.

---

## Use Case 9: Detect Market Regimes

**Goal**: See how the market automatically shifts between bull, bear, and neutral phases, and understand how regimes affect strategy performance.

**What you will learn**: Markets have different "moods" (regimes), and strategies perform differently in each one.

### Steps

1. You should still be on the **Analysis** page with SPY results loaded. If not, repeat the setup from Use Case 8 and click "Run Analysis."
2. Click the **"Regime Detection"** tab (the third tab).
3. The heading says **"Market Regime Detection (KMeans)"**. The application has used a machine learning algorithm to automatically identify three different market regimes from the historical price data.
4. Look at the **regime statistics table** first. It shows three groups (numbered 0, 1, 2) with their mean return and standard deviation. The regime with the most negative mean return is the **Bear** regime. The one with the most positive mean return is the **Bull** regime. The middle one is **Neutral**.
5. Now look at the **price chart with regime overlay**. Each data point is colored:
   - **Red dots** = Bear regime (prices tend to fall during these periods).
   - **Gray dots** = Neutral regime (prices move sideways with no clear direction).
   - **Green dots** = Bull regime (prices tend to rise).
6. Trace through the chart from left to right. Notice how the market shifts from one regime to another. Major economic events (recessions, rallies, crises) often correspond to regime transitions.
7. Think about strategy performance in the context of regimes:
   - Trend-following strategies (EMA Crossover, Supertrend) tend to do well in Bull and Bear regimes because there is a clear direction to follow.
   - Mean-reversion strategies (RSI, Bollinger Bands) tend to do well in Neutral regimes because the price oscillates within a range.
   - No strategy does well in all regimes. This is a fundamental challenge of trading.

### Takeaway

Markets cycle through different regimes: bull (rising), bear (falling), and neutral (sideways). Understanding regimes helps explain why even a good strategy has losing periods. It also suggests that the best approach might be to use different strategies for different market conditions, or to accept that losing periods are inevitable and focus on long-term performance.

---

## Use Case 10: Analyze Risk

**Goal**: Understand how much money you could lose with a strategy, and learn to read a return distribution.

**What you will learn**: Risk measurement is as important as return measurement. You need to know how much you could lose before you can decide how much to invest.

### Steps

1. You should still be on the **Analysis** page with SPY results loaded.
2. Click the **"Risk Analysis"** tab (the second tab).
3. You will see three key risk metrics:
   - **Parametric VaR (95%)**: This assumes returns follow a normal (bell curve) distribution and calculates: "With 95% confidence, the daily loss will not exceed X%."
   - **Historical VaR (95%)**: Same concept, but calculated from the actual historical data without assuming any distribution. This is usually more accurate.
   - **CVaR (95%)** (Conditional Value at Risk): "On the worst 5% of days, the *average* loss will be X%." This tells you how bad it gets when things go wrong.
4. **Interpret the numbers**: If your portfolio is $100,000 and the Historical VaR is 2.0%, then on 95% of days you would expect to lose no more than $2,000. But on the remaining 5% of days, losses could be larger -- and CVaR tells you how much larger on average.
5. Below the VaR metrics, look at the **Drawdown** chart. This underwater plot shows every period where the portfolio was below its all-time high. The deeper the dip, the larger the loss. The wider the dip, the longer it took to recover. Imagine living through the deepest dip -- would you have the patience to hold on?
6. Now click the **"Distributions"** tab (the fourth and final tab).
7. You will see a **Return Histogram** -- a bar chart showing how frequently each return level occurred. The x-axis is the daily return percentage, and the y-axis is frequency.
8. Notice the shape:
   - Most days cluster around zero (small positive or negative returns).
   - A few days are far to the left (large losses) or far to the right (large gains).
   - The distribution may not be perfectly symmetric. If the left tail is longer or fatter, the strategy is prone to large unexpected losses.
9. Below the histogram, look at the four **Key Statistics**:
   - **Mean**: The average daily return. A small positive number for profitable strategies.
   - **Std Dev**: How spread out the returns are. Higher = more unpredictable.
   - **Skewness**: If negative, large losses are more common than large gains. If positive, large gains are more common.
   - **Kurtosis**: If positive (which it usually is for financial data), extreme events happen more often than a normal bell curve would predict. This is the famous "fat tails" phenomenon.
10. Put it all together: the risk analysis tells you not just whether a strategy makes money, but how *unpredictable* and *painful* the journey is. Two strategies with the same return can have very different risk profiles.

### Takeaway

Understanding risk helps you size your positions and set realistic expectations. A strategy with a 40% max drawdown means your $100,000 portfolio could temporarily drop to $60,000 before recovering. Could you handle that emotionally? If not, you might want a less aggressive strategy with a smaller drawdown, even if it means lower returns. Risk management is not about avoiding risk entirely -- it is about taking only the risks you can afford and are comfortable with.

---

## What's Next?

Congratulations -- you have completed all 10 exercises. Here is a summary of what you have learned:

| Exercise | Key Concept |
|---|---|
| 1. First Chart | OHLCV data and candlestick reading |
| 2. Moving Average | Trend identification and noise filtering |
| 3. RSI | Momentum and overbought/oversold conditions |
| 4. First Backtest | Strategy testing on historical data |
| 5. Transaction Costs | The impact of real-world trading costs |
| 6. Asset Comparison | Risk/return profiles and diversification |
| 7. Strategy Comparison | No single best strategy; trade-offs everywhere |
| 8. Tear Sheet | Professional-grade performance evaluation |
| 9. Regime Detection | Market phases and their impact on strategies |
| 10. Risk Analysis | Risk measurement and return distributions |

### Ideas for Further Exploration

Now that you know how to use the application, here are some things to try on your own:

- **Experiment with different symbols**: Try analyzing tech stocks (NVDA, AMD), commodities (GLD for gold, USO for oil), or international ETFs (EWJ for Japan, EWZ for Brazil).
- **Test strategies on crypto**: Run the backtester on BTC-USD or ETH-USD. Crypto markets behave very differently from stock markets.
- **Adjust strategy parameters**: Go back to the Backtester and change the EMA Crossover periods from 12/26 to 50/200 (the classic "golden cross"). How do the results change?
- **Combine indicators on the Dashboard**: Add both a trend indicator (SMA or EMA) and a momentum indicator (RSI or MACD) to the Dashboard. Observe how they give different types of information about the same price data.
- **Compare time periods**: Run the same strategy on two different date ranges (e.g., 2019-2021 vs. 2022-2024). How much does performance vary?
- **Explore the full indicator catalog**: The application has 38 indicators across 5 categories. Try adding some you have not used yet, like Bollinger Bands, Ichimoku Cloud, or On Balance Volume.
- **Learn Python and extend the framework**: If you are interested in creating your own strategies or indicators, see the Developer Guide for instructions on using the ta_framework package programmatically.

You now know more about technical analysis and strategy evaluation than most people. Use this knowledge wisely -- remember that all backtesting is based on historical data, and past performance never guarantees future results. Happy analyzing!
