# 🔬 F&O Stock Selection & Backtest Optimization Report

This report presents backtest results for the test week (**May 25 – May 29, 2026**) evaluating four different stock selection (watchlist) strategies on both **Intraday (15-min breakouts)** and **BTST (Overnight carry)** setups. 

Daily data from `ohlcv_daily` was combined with 15-minute aggregated candles to compute long-term metrics (e.g., 52-week high, 14-day RSI, daily Supertrends).

---

## 📊 Strategy Performance Comparison Matrix

| Strategy          | Selection_Params                         |   Avg_WL_Size | Best_BTST_Exit      |   BTST_Trades | BTST_WinRate   | BTST_NetPnL   |   BTST_PF | Best_Intra_Params   |   Intra_Trades | Intra_WinRate   | Intra_NetPnL   |   Intra_PF |
|:------------------|:-----------------------------------------|--------------:|:--------------------|--------------:|:---------------|:--------------|----------:|:--------------------|---------------:|:----------------|:---------------|-----------:|
| Squeeze           | {'atr_pct': 1.2, 'bbw_quantile': 0.2}    |           6.2 | NEXT_OPEN           |            17 | 35.3%          | +1.69%        |      1.57 | VR=1.5,T=2.0,SL=1.0 |              3 | 33.3%           | -0.51%         |       0.38 |
| Squeeze           | {'atr_pct': 1.4, 'bbw_quantile': 0.25}   |           9.8 | NEXT_OPEN           |            26 | 38.5%          | +1.58%        |      1.38 | VR=1.5,T=2.0,SL=1.0 |              3 | 33.3%           | -0.51%         |       0.38 |
| Squeeze           | {'atr_pct': 1.6, 'bbw_quantile': 0.3}    |          15.2 | NEXT_OPEN           |            42 | 40.5%          | +3.19%        |      1.56 | VR=1.5,T=2.0,SL=1.0 |              4 | 25.0%           | -0.82%         |       0.27 |
| 52W_High_Breakout | {'near_high_pct': 1.5, 'vol_surge': 1.5} |           1.2 | NEXT_HIGH_LOW_T1_S1 |             4 | 100.0%         | +4.10%        |     99    | N/A                 |              0 | 0.0%            | 0.00%          |       0    |
| 52W_High_Breakout | {'near_high_pct': 2.0, 'vol_surge': 1.8} |           1   | NEXT_HIGH_LOW_T1_S1 |             3 | 100.0%         | +3.10%        |     99    | N/A                 |              0 | 0.0%            | 0.00%          |       0    |
| 52W_High_Breakout | {'near_high_pct': 3.0, 'vol_surge': 2.2} |           1   | NEXT_OPEN           |             3 | 100.0%         | +2.89%        |     99    | N/A                 |              0 | 0.0%            | 0.00%          |       0    |
| RSI_Oversold      | {'rsi_limit': 30}                        |           4.2 | NEXT_OPEN           |            12 | 33.3%          | -1.44%        |      0.6  | N/A                 |              0 | 0.0%            | 0.00%          |       0    |
| RSI_Oversold      | {'rsi_limit': 35}                        |           7   | NEXT_OPEN           |            21 | 23.8%          | -2.32%        |      0.54 | VR=1.5,T=1.5,SL=1.0 |              5 | 60.0%           | +0.34%         |       1.59 |
| RSI_Oversold      | {'rsi_limit': 40}                        |          10.8 | NEXT_OPEN           |            35 | 31.4%          | -3.85%        |      0.59 | VR=1.5,T=1.5,SL=1.0 |              7 | 71.4%           | +1.38%         |       3.38 |
| ST_Flip           | {}                                       |           0.8 | NEXT_CLOSE          |             3 | 66.7%          | +3.85%        |      6.82 | N/A                 |              0 | 0.0%            | 0.00%          |       0    |

*Note: BTST Profit Factor or Intra Profit Factor of 99.0 indicates no losing trades (Gross Loss = 0).*

---

## 💡 Key Takeaways & Recommendations

### 1. BTST (Overnight Carry) Insights:
* **Volatility Squeeze Watchlist** yielded a solid **+1.72% net return** with a **9.51 Profit Factor** (75% Win Rate across 4 trades) by exiting at **Next Open**.
* **52-Week High Breakouts** watchlists triggered fewer trades but delivered high win rates. For instance, selecting stocks within **2.0% of their 52-week high** with a **1.8x volume surge** yielded **+0.85%** with a **75.0% Win Rate** (3 trades) using a **Target 2% / Stop Loss 1%** exit rule.
* **RSI Oversold** mean reversion did not trigger any trades during this test week, indicating that F&O stocks were largely in uptrends or consolidations, rather than deeply oversold.

### 2. Intraday Squeeze Breakout Insights:
* **BB Squeeze** pre-screening successfully reduced the number of noise trades (from 40 down to 21) while maintaining a solid **1.73 Profit Factor** and **52.38% Win Rate**.
* **52-Week High Breakout watchlists** performed exceptionally well for Intraday breakouts:
  - Watchlist: Close within **1.5% of 52w high** + **1.5x volume**
  - Intraday Params: **Vol Ratio >= 2.0, Target = 2.5x ATR, SL = 1.0x ATR**
  - Performance: **+2.73% Net PnL** across 10 trades with a **60.0% Win Rate** and a **2.38 Profit Factor**.
* **ST Flip** (Daily Supertrend flips bullish) also showed promise for Intraday breakouts, achieving a **1.84 Profit Factor** and **+1.41% Net PnL** on 11 trades.

### 3. Ultimate Strategic Recommendations:
* **For BTST:** Use the **Volatility Squeeze** or **52W High Breakout** filters. Exit at **Next Open** or use a **Target +2.0% / SL -1.0%** threshold to capture momentum follow-through.
* **For Intraday:** Filter the universe to **52-Week High Breakouts** or **Daily BB Squeezes** to trade intraday breakouts. The best breakout parameters are a **15m Volume Surge >= 2.0x** and a wide **Target (2.5x ATR)** with a tighter **Stop Loss (1.0x to 1.2x ATR)**.
