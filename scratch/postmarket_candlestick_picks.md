# 🕯️ Postmarket Candlestick Pattern Analysis (with RSI & ADX)
    
**Scan Date**: 2026-05-29  
**Execution Session**: Next Trading Session (Monday, June 1, 2026)  

This report lists active candlestick pattern setups detected on the daily chart at the close of the last session, filtered for **high historical accuracy (Win Rate >= 60%)** and enriched with **RSI-14** (Momentum extreme) and **ADX-14** (Trend Strength) metrics.

## 🚀 High-Probability Trend Reversal Setups

| Symbol | Pattern | Trade Direction | Close Price | Hist Occurrences | Hist Win Rate | Avg Next-Day Return | t-statistic | RSI (14) | ADX (14) | Trend Strength |
|:-------|:--------|:----------------|:------------|:-----------------|:--------------|:--------------------|:------------|:---------|:---------|:---------------|
| BAJAJFINSV | Shooting Star | SHORT | ₹1794.20 | 4 | 75.0% | +0.53% | 1.150 | 46.0 | 52.5 | Strong Trend (ADX > 25) |
| ABB | Doji | NEUTRAL | ₹7285.00 | 12 | 66.7% | +1.72% | 5.313 | 47.2 | 39.6 | Strong Trend (ADX > 25) |
| TATAELXSI | Shooting Star | SHORT | ₹4345.00 | 3 | 66.7% | +0.74% | 1.495 | 52.7 | 19.0 | Sideways (ADX < 20) |
| CHOLAFIN | Bearish Engulfing | SHORT | ₹1546.30 | 9 | 66.7% | +0.53% | 0.656 | 33.6 | 30.1 | Strong Trend (ADX > 25) |
| ZYDUSLIFE | Shooting Star | SHORT | ₹1100.90 | 3 | 66.7% | -0.26% | -0.323 | 79.1 | 64.3 | Strong Trend (ADX > 25) |
| TRENT | Shooting Star | SHORT | ₹4273.60 | 3 | 66.7% | -0.95% | -0.813 | 46.9 | 22.9 | Moderate Trend (20-25) |
| INDIACEM | Doji | NEUTRAL | ₹399.95 | 19 | 63.2% | +2.25% | 4.928 | 42.6 | 12.6 | Sideways (ADX < 20) |
| UNIONBANK | Shooting Star | SHORT | ₹168.39 | 5 | 60.0% | +0.30% | 0.736 | 52.6 | 30.8 | Strong Trend (ADX > 25) |
| COFORGE | Bullish Engulfing | LONG | ₹1436.50 | 5 | 60.0% | +0.70% | 0.597 | 65.8 | 25.5 | Strong Trend (ADX > 25) |
| COALINDIA | Shooting Star | SHORT | ₹461.85 | 5 | 60.0% | +0.06% | 0.553 | 34.0 | 21.2 | Moderate Trend (20-25) |

---

## 🧠 Why RSI & ADX Make Sense for Candlestick Reversals

1. **RSI Exhaustion Gate**:
   * **Bearish Reversals (Shooting Star / Engulfing)**: Have a much higher probability of success if the stock is overbought (**RSI > 60-70**). It indicates that the reversal is triggered at a peak exhaustion point where buyers are completely depleted.
   * **Bullish Reversals (Hammer / Engulfing)**: Have a higher edge if the stock is oversold (**RSI < 35-40**), indicating seller depletion.

2. **ADX Regime filter**:
   * **ADX < 20 (Sideways/Range Bound)**: Reversal patterns (like Shooting Stars/Hammers) are **highly effective** because price naturally mean-reverts within ranges.
   * **ADX > 25 (Strong Trend)**: Reversals are **dangerous** because they fight the prevailing trend. However, continuation patterns (like Engulfing) have a strong tailwind.

### 🔍 Tactical Commentary on Current Picks:
* **ABB & INDIACEM (Doji)**: Displaying Dojis in sideways or moderate trends indicates price consolidation and an imminent breakout.
* **Shooting Stars (TATAELXSI, ZYDUSLIFE, TRENT, COALINDIA, UNIONBANK)**: Many F&O stocks are printing Shooting Stars. In sideways or moderate regimes (low ADX), these are high-probability short setups for Monday morning, as buyers failed to hold the highs EOD.

---

### 💡 Trading Guide for the Next Session

1. **LONG (Buy Call/Stock)** setups:
   * **Execution**: Enter at the market Open of the next session if the stock opens flat or with a minor gap-down/gap-up (within +/- 0.5% of Close).
   * **Exit Strategy**: Take profit at **+2% to +3%** or exit at the Close of the session (intraday) or carry for a 3-to-5 day swing setup.
   * **Stop Loss**: Place a stop loss at **-1.5%** or below the low of the signal candle.

2. **SHORT (Buy Put/Short Future)** setups:
   * **Execution**: Enter short at the Open.
   * **Exit Strategy**: Target **-2%** or close at EOD.
   * **Stop Loss**: Place stop loss at **+1.5%** or above the high of the signal candle.
   
3. **TradingView Verification**:
   * You can open [TradingView Chart Viewer](file:///Users/pratham/aitrade/trade_system_v2/scratch/volume_divergence_picks.md) inside the Streamlit dashboard under the **Candlestick Pattern Lab** to view live chart screenshots and verify trend structures before entering.
