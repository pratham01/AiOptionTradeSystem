import pandas as pd
import numpy as np
import logging
from typing import Tuple

LOGGER = logging.getLogger(__name__)

class RsiDivergence:
    """
    Python implementation of RSI Divergence (Fast RSI - Slow RSI).
    """
    def __init__(self, len_fast: int = 5, len_slow: int = 14):
        self.len_fast = len_fast
        self.len_slow = len_slow

    def _rma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Pine Script's rma (Relative Moving Average).
        It's an EMA with alpha = 1/length.
        """
        return series.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

    def _calculate_rsi(self, series: pd.Series, length: int) -> pd.Series:
        """Helper to calculate RSI using RMA."""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        avg_up = self._rma(up, length)
        avg_down = self._rma(down, length)
        
        rs = avg_up / avg_down
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero
        rsi.loc[avg_down == 0] = 100
        rsi.loc[(avg_down == 0) & (avg_up == 0)] = 50 # Neutral if no movement
        
        return rsi

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates fast RSI, slow RSI, and their divergence.
        """
        df = df.copy()
        df['rsi_fast'] = self._calculate_rsi(df['close'], self.len_fast)
        df['rsi_slow'] = self._calculate_rsi(df['close'], self.len_slow)
        df['divergence'] = df['rsi_fast'] - df['rsi_slow']
        
        # Signals: 1 (Bullish) when divergence crosses above 0, -1 (Bearish) when below 0
        df['rsi_signal'] = 0
        df.loc[(df['divergence'] > 0) & (df['divergence'].shift(1) <= 0), 'rsi_signal'] = 1
        df.loc[(df['divergence'] < 0) & (df['divergence'].shift(1) >= 0), 'rsi_signal'] = -1
        
        return df
