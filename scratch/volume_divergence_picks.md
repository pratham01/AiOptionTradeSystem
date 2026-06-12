# 📊 Daily Volume Divergence Reversal Picks
        
Scan executed on **2026-05-29** using optimized parameters (`pivot_window=5`, `lookback=50`). 

These signals represent high-probability trend reversals confirmed by On-Balance Volume (OBV) divergence, enriched with **RSI-14** (Momentum extreme) and **ADX-14** (Trend Strength) metrics.

| Symbol     | Reversal Type    | Close Price   |   RSI (14) |   ADX (14) | Trend Strength    | Explanation                                                                                                                                         |
|:-----------|:-----------------|:--------------|-----------:|-----------:|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| CHOLAFIN   | 🚀 BULLISH (Buy) | ₹1546.30      |       30   |       31   | Strong (ADX > 25) | Bullish Divergence: Price low on 2026-05-21 (₹1504.00) is LOWER than 2026-04-28 (₹1536.40), but OBV (-3e+07) is HIGHER than on 2026-04-28 (-3e+07). |
| JINDALSTEL | 🚀 BULLISH (Buy) | ₹1219.80      |       49.2 |       28.6 | Strong (ADX > 25) | Bullish Divergence: Price low on 2026-05-21 (₹1197.50) is LOWER than 2026-05-12 (₹1215.00), but OBV (2e+07) is HIGHER than on 2026-05-12 (2e+07).   |

---
### 🧠 Why RSI & ADX Make Sense for Volume Divergences

1. **RSI Exhaustion Gate**:
   * **Bullish Reversals**: Have a higher probability of success if the stock is oversold or in a lower range (**RSI < 40**), indicating seller depletion.
   * **Bearish Reversals**: Have a higher probability of success if the stock is overbought (**RSI > 60**), indicating buyer depletion.

2. **ADX Regime Filter**:
   * **ADX < 20 (Sideways/Range Bound)**: Reversals are highly effective because price naturally mean-reverts within ranges.
   * **ADX > 25 (Strong Trend)**: Reversals are dangerous because you are fighting a strong trend. Check if the ADX line has started sloping downward or rolling over.

---
### 💡 Strategic Execution Guide
1. **Holding Period**: Our backtests show that holding these trend reversal setups for **5 trading sessions** yields the highest historical win rate (53.1%) and average return (+0.54% overall, +0.63% on Shorts).
2. **Stop Loss**: A tight stop loss of **3%** from entry price helps manage risk, while seeking a **5% target** (Profit Factor: 1.55 on Shorts).
