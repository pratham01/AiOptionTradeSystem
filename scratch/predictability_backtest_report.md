# 📊 Quantitative Return Predictability Backtest Report
    
**Report Generated**: 2026-05-31 16:34:44  
**Data Universe**: Selected Index and Major F&O stocks from the SQLite database.  

This backtest evaluates whether the close-to-close direction of the next trading day is statistically predictable based on different technical indicator signals today. We run a two-sample independent t-test to compare the mean return on signal days versus baseline days.

## 🔬 Statistical Edge Results

| Symbol    | Indicator            |   Threshold |   Total Days |   Signals Triggered | Win Rate (Signal)   | Win Rate (Baseline)   | Avg Return (Signal)   | Avg Return (Baseline)   |   t-stat |   p-value | Predictable Edge   |
|:----------|:---------------------|------------:|-------------:|--------------------:|:--------------------|:----------------------|:----------------------|:------------------------|---------:|----------:|:-------------------|
| NIFTY50   | Momentum Surge       |         1.5 |         3723 |                 106 | 10.4%               | 32.4%                 | +0.002%               | +0.027%                 |   -0.51  |    0.6111 | ❌ NO              |
| NIFTY50   | Volume Ratio         |         2   |         3723 |                  10 | 10.0%               | 31.8%                 | -0.212%               | +0.027%                 |   -1.914 |    0.0872 | ❌ NO              |
| NIFTY50   | RSI Oversold         |        35   |         3723 |                 770 | 28.3%               | 32.6%                 | -0.005%               | +0.034%                 |   -0.93  |    0.3524 | ❌ NO              |
| NIFTY50   | RSI Overbought       |        65   |         3723 |                1241 | 32.9%               | 31.2%                 | +0.039%               | +0.020%                 |    0.811 |    0.4172 | ❌ NO              |
| HDFCBANK  | Momentum Surge       |         1.5 |          253 |                  17 | 35.3%               | 44.1%                 | -0.452%               | -0.059%                 |   -1.168 |    0.2581 | ❌ NO              |
| HDFCBANK  | Volume Ratio         |         2   |          253 |                  24 | 45.8%               | 43.2%                 | -0.260%               | -0.067%                 |   -0.715 |    0.4808 | ❌ NO              |
| HDFCBANK  | RSI Oversold         |        35   |          253 |                  61 | 44.3%               | 43.2%                 | -0.158%               | -0.062%                 |   -0.429 |    0.6689 | ❌ NO              |
| HDFCBANK  | RSI Overbought       |        65   |          253 |                  24 | 50.0%               | 42.8%                 | -0.186%               | -0.075%                 |   -0.555 |    0.5824 | ❌ NO              |
| ICICIBANK | Momentum Surge       |         1.5 |          253 |                  19 | 36.8%               | 43.6%                 | -0.183%               | -0.029%                 |   -0.582 |    0.5665 | ❌ NO              |
| ICICIBANK | Volume Ratio         |         2   |          253 |                  25 | 36.0%               | 43.9%                 | -0.261%               | -0.016%                 |   -1.336 |    0.1901 | ❌ NO              |
| ICICIBANK | RSI Oversold         |        35   |          253 |                  66 | 48.5%               | 41.2%                 | -0.062%               | -0.033%                 |   -0.17  |    0.8651 | ❌ NO              |
| ICICIBANK | RSI Overbought       |        65   |          253 |                  32 | 34.4%               | 44.3%                 | -0.314%               | -0.001%                 |   -1.413 |    0.1654 | ❌ NO              |
| AXISBANK  | Momentum Surge       |         1.5 |          253 |                  34 | 50.0%               | 48.4%                 | -0.030%               | +0.059%                 |   -0.319 |    0.7513 | ❌ NO              |
| AXISBANK  | Volume Ratio         |         2   |          253 |                  29 | 51.7%               | 48.2%                 | +0.146%               | +0.034%                 |    0.414 |    0.6811 | ❌ NO              |
| AXISBANK  | RSI Oversold         |        35   |          253 |                  61 | 54.1%               | 46.9%                 | +0.015%               | +0.057%                 |   -0.197 |    0.844  | ❌ NO              |
| AXISBANK  | RSI Overbought       |        65   |          253 |                  62 | 41.9%               | 50.8%                 | -0.094%               | +0.093%                 |   -0.996 |    0.3211 | ❌ NO              |
| AXISBANK  | High-Volume Breakout |         2   |          253 |                   6 | 66.7%               | 48.2%                 | +0.135%               | +0.045%                 |    0.144 |    0.8905 | ❌ NO              |
| HCLTECH   | Momentum Surge       |         1.5 |          253 |                  28 | 57.1%               | 47.1%                 | +0.099%               | -0.154%                 |    0.915 |    0.366  | ❌ NO              |
| HCLTECH   | Volume Ratio         |         2   |          253 |                  32 | 46.9%               | 48.4%                 | +0.045%               | -0.151%                 |    0.839 |    0.4053 | ❌ NO              |
| HCLTECH   | RSI Oversold         |        35   |          253 |                  72 | 44.4%               | 49.7%                 | -0.374%               | -0.028%                 |   -1.598 |    0.1125 | ❌ NO              |
| HCLTECH   | RSI Overbought       |        65   |          253 |                  51 | 47.1%               | 48.5%                 | -0.205%               | -0.107%                 |   -0.334 |    0.7394 | ❌ NO              |
| HCLTECH   | High-Volume Breakout |         2   |          253 |                   5 | 20.0%               | 48.8%                 | -0.805%               | -0.113%                 |   -1.533 |    0.1934 | ❌ NO              |

---
## 🧠 Key Takeaways
1. **t-statistic & p-value**: A t-statistic with absolute value >= 2.0 (or p-value < 0.05) indicates that the difference in mean returns is statistically significant and not due to random walk.
2. **RSI Oversold / Overbought**: These reversion conditions often show the most significant predictability on indices (like NIFTY50) due to mean-reverting index dynamics.
3. **Volume Breakouts**: High volume ratios usually represent smart money entry. Comparing next-day returns after volume surges helps confirm if they represent follow-through or exhaustion.
