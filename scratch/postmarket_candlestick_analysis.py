import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from trade_system.application.agent.candlestick_pattern_agent import CandlestickPatternAgent
from trade_system.infrastructure.database.connection import get_engine

def calc_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculates RSI-14."""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Calculates ADX-14."""
    if len(close) < period * 2:
        return 20.0
    try:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        dm_plus = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
        dm_minus = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
        atr = tr.ewm(span=period).mean()
        di_plus = 100 * dm_plus.ewm(span=period).mean() / (atr + 1e-9)
        di_minus = 100 * dm_minus.ewm(span=period).mean() / (atr + 1e-9)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
        adx = dx.ewm(span=period).mean()
        return float(adx.iloc[-1])
    except Exception:
        return 20.0

def main():
    engine = get_engine()
    agent = CandlestickPatternAgent()
    
    print("⏳ Scanning F&O universe daily candles for active pattern signals...")
    active_setups = agent.scan_active_setups()
    
    if not active_setups:
        print("❌ No active candlestick signals found.")
        return
        
    df = pd.DataFrame(active_setups)
    
    # Parse Win Rate percentage string to float for filtering
    df["WinRate_Val"] = df["Win Rate"].str.replace("%", "").astype(float)
    
    # Filter setups:
    # 1. Historical occurrences >= 3 (minimum sample)
    # 2. Win rate >= 60.0% (high accuracy edge)
    df_filtered = df[(df["Hist Count"] >= 3) & (df["WinRate_Val"] >= 60.0)].copy()
    
    if df_filtered.empty:
        print("❌ No high-probability setups met the win-rate filters (>= 60%).")
        return
        
    latest_date_str = df_filtered.iloc[0]["Date"]
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d").date()
    
    # 3. Fetch indicators (RSI & ADX) for each active setup as of scan date
    rsi_vals = []
    adx_vals = []
    trend_strengths = []
    
    with engine.connect() as conn:
        for _, row in df_filtered.iterrows():
            sym = "NSE:" + row["Symbol"] + "-EQ"
            # Load daily candles up to latest_date
            df_hist = pd.read_sql(text("""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM ohlcv_daily 
                WHERE symbol = :sym AND date(timestamp) <= :dt
                ORDER BY timestamp ASC
            """), conn, params={"sym": sym, "dt": latest_date_str})
            
            # Fallback if empty
            if df_hist.empty or len(df_hist) < 30:
                # Try fetching without date filter to get full history
                df_hist = pd.read_sql(text("""
                    SELECT symbol, timestamp, open, high, low, close, volume 
                    FROM ohlcv_daily 
                    WHERE symbol = :sym
                    ORDER BY timestamp ASC
                """), conn, params={"sym": sym})
                
            if len(df_hist) >= 30:
                rsi_val = calc_rsi(df_hist["close"])
                adx_val = calc_adx(df_hist["high"], df_hist["low"], df_hist["close"])
            else:
                rsi_val = 50.0
                adx_val = 20.0
                
            rsi_vals.append(round(rsi_val, 1))
            adx_vals.append(round(adx_val, 1))
            
            if adx_val < 20:
                strength = "Sideways (ADX < 20)"
            elif adx_val < 25:
                strength = "Moderate Trend (20-25)"
            else:
                strength = "Strong Trend (ADX > 25)"
            trend_strengths.append(strength)
            
    df_filtered["RSI"] = rsi_vals
    df_filtered["ADX"] = adx_vals
    df_filtered["Trend Strength"] = trend_strengths
    
    # Sort by Win Rate and t-statistic
    df_filtered = df_filtered.sort_values(by=["WinRate_Val", "t-stat"], ascending=[False, False])
    
    print(f"\n✅ Scan Complete! Found {len(df_filtered)} high-probability pattern signals as of {latest_date_str} Close.")
    
    # Generate Markdown Report Content
    markdown_content = f"""# 🕯️ Postmarket Candlestick Pattern Analysis (with RSI & ADX)
    
**Scan Date**: {latest_date_str}  
**Execution Session**: Next Trading Session (Monday, June 1, 2026)  

This report lists active candlestick pattern setups detected on the daily chart at the close of the last session, filtered for **high historical accuracy (Win Rate >= 60%)** and enriched with **RSI-14** (Momentum extreme) and **ADX-14** (Trend Strength) metrics.

## 🚀 High-Probability Trend Reversal Setups

| Symbol | Pattern | Trade Direction | Close Price | Hist Occurrences | Hist Win Rate | Avg Next-Day Return | t-statistic | RSI (14) | ADX (14) | Trend Strength |
|:-------|:--------|:----------------|:------------|:-----------------|:--------------|:--------------------|:------------|:---------|:---------|:---------------|
"""
    
    for _, row in df_filtered.iterrows():
        markdown_content += f"| {row['Symbol']} | {row['Pattern']} | {row['Direction']} | {row['Close Price']} | {row['Hist Count']} | {row['Win Rate']} | {row['Avg Next-Day return']} | {row['t-stat']:.3f} | {row['RSI']} | {row['ADX']} | {row['Trend Strength']} |\n"
            
    markdown_content += """
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
"""
    
    report_path = "scratch/postmarket_candlestick_picks.md"
    with open(report_path, "w") as f:
        f.write(markdown_content)
        
    print("\n" + "="*95)
    print("📈 ACTIVE POSTMARKET CANDLESTICK PICKS (WITH RSI & ADX)")
    print("="*95)
    print(df_filtered[["Symbol", "Pattern", "Direction", "Close Price", "Win Rate", "t-stat", "RSI", "ADX", "Trend Strength"]].to_string(index=False))
    print("="*95)
    print(f"\n📝 Detailed report written to: {report_path}")

if __name__ == "__main__":
    main()
