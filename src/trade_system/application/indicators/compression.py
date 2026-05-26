from __future__ import annotations

import pandas as pd
import numpy as np
from trade_system.application.indicators.base import BaseIndicator


class CompressionIndicator(BaseIndicator):
    """
    Identifies price and volatility compression (coiling).
    A state of low volatility that often precedes an explosive move.
    
    Metrics:
    - Tightness: Average Range (HL) relative to ATR.
    - Squeeze: Bollinger Bands width getting narrower than Keltner Channels.
    - Coiling: 3-4 inside bars or narrow range candles (NR7).
    """

    def __init__(self, atr_period: int = 14, lookback: int = 4):
        self.atr_period = atr_period
        self.lookback = lookback

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty or len(data) < self.atr_period:
            return data

        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # 1. ATR for base volatility
        high_low = df['high'] - df['low']
        high_cp = (df['high'] - df['close'].shift(1)).abs()
        low_cp = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()

        # 2. Local Range Tightness
        # Is the current 4-bar average range < 80% of ATR?
        df['current_range'] = df['high'] - df['low']
        df['avg_range_lookback'] = df['current_range'].rolling(window=self.lookback).mean()
        
        # Compression Score (0-100)
        # 1. Price Range Compression (Higher = More Compressed)
        df['range_compression'] = (1 - (df['avg_range_lookback'] / df['atr'])).clip(0, 1) * 100
        
        # 2. Narrow Range 7 (NR7) detection
        # Is today's range the smallest in the last 7 bars?
        df['nr7'] = df['current_range'] == df['current_range'].rolling(7).min()
        
        # 3. Inside Bar detection
        df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        
        # Combined Compression Signal
        df['is_compressed'] = (df['range_compression'] > 20) | (df['nr7']) | (df['inside_bar'])
        
        return df
