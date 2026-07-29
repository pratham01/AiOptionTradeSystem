from __future__ import annotations

import pandas as pd
from trade_system.domains.strategy.application.indicators.base import BaseIndicator


class RSIIndicator(BaseIndicator):
    """
    Relative Strength Index (RSI) Indicator.
    Used to identify overbought (>70) and oversold (<30) conditions.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty or len(data) < self.period:
            return data

        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # --- Divergence Detection ---
        df['bull_div'] = False
        df['bear_div'] = False
        
        lookback = 30
        p = 2 # Pivot window
        
        for i in range(lookback, len(df)):
            # 1. Bullish Divergence (Price lower low, RSI higher low)
            # Find local lows in price
            if df['close'].iloc[i] == df['close'].iloc[i-p:i+1].min():
                curr_price_low = df['close'].iloc[i]
                curr_rsi_low = df['rsi'].iloc[i]
                
                # Search for a previous low in the lookback window
                for j in range(i - lookback, i - p):
                    if df['close'].iloc[j] == df['close'].iloc[j-p:j+p+1].min():
                        prev_price_low = df['close'].iloc[j]
                        prev_rsi_low = df['rsi'].iloc[j]
                        
                        if curr_price_low < prev_price_low and curr_rsi_low > prev_rsi_low:
                            df.at[df.index[i], 'bull_div'] = True
                            break

            # 2. Bearish Divergence (Price higher high, RSI lower high)
            if df['high'].iloc[i] == df['high'].iloc[i-p:i+1].max():
                curr_price_high = df['high'].iloc[i]
                curr_rsi_high = df['rsi'].iloc[i]
                
                for j in range(i - lookback, i - p):
                    if df['high'].iloc[j] == df['high'].iloc[j-p:j+p+1].max():
                        prev_price_high = df['high'].iloc[j]
                        prev_rsi_high = df['rsi'].iloc[j]
                        
                        if curr_price_high > prev_price_high and curr_rsi_high < prev_rsi_high:
                            df.at[df.index[i], 'bear_div'] = True
                            break
                            
        return df
