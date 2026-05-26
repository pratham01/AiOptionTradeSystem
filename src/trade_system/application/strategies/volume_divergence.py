from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from trade_system.core.models.domain import Signal
from trade_system.application.strategies.base import BaseStrategy
from trade_system.application.indicators.obv import OBVIndicator

logger = logging.getLogger(__name__)

class VolumeDivergenceStrategy(BaseStrategy):
    """
    Strategy: Volume Divergence with Price Action (using OBV).
    """
    name = "volume_divergence"

    def __init__(self, pivot_window: int = 5, lookback: int = 50):
        if pivot_window < 2:
            raise ValueError("pivot_window must be >= 2")
        if lookback <= pivot_window * 2:
            raise ValueError("lookback must be greater than pivot_window * 2")
        self.pivot_window = pivot_window
        self.lookback = lookback
        self.obv_indicator = OBVIndicator()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data

        df = data.copy().sort_values("timestamp").reset_index(drop=True)
        df.columns = [col.lower() for col in df.columns]
        
        # 1. Calculate OBV
        df = self.obv_indicator.calculate(df)
        
        # 2. Pre-calculate signals
        df['final_signal'] = 0
        
        for i in range(self.lookback, len(df)):
            window_df = df.iloc[i-self.lookback:i+1]
            p = self.pivot_window

            # Find local lows
            price_lows = []
            for j in range(p, len(window_df) - p):
                if window_df['close'].iloc[j] == window_df['close'].iloc[j-p:j+p+1].min():
                    price_lows.append((j, window_df['close'].iloc[j], window_df['obv'].iloc[j]))

            if len(price_lows) >= 2:
                last_low = price_lows[-1]
                prev_low = price_lows[-2]
                if last_low[1] < prev_low[1] and last_low[2] > prev_low[2]:
                    # If the latest low was recent
                    if (len(window_df) - 1 - last_low[0]) <= p:
                        df.loc[df.index[i], 'final_signal'] = 1

            # Find local highs
            price_highs = []
            for j in range(p, len(window_df) - p):
                if window_df['close'].iloc[j] == window_df['close'].iloc[j-p:j+p+1].max():
                    price_highs.append((j, window_df['close'].iloc[j], window_df['obv'].iloc[j]))

            if len(price_highs) >= 2:
                last_high = price_highs[-1]
                prev_high = price_highs[-2]
                if last_high[1] > prev_high[1] and last_high[2] < prev_high[2]:
                    if (len(window_df) - 1 - last_high[0]) <= p:
                        df.loc[df.index[i], 'final_signal'] = -1

        return df

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        reason = "Bullish Volume Divergence" if row['final_signal'] == 1 else "Bearish Volume Divergence"
        return f"Divergence | {symbol} | {reason} | Price: {row['close']:.2f}"
