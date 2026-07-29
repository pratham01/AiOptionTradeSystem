import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from .base import BaseStrategy
from ..indicators.supertrend import SupertrendIndicator

logger = logging.getLogger(__name__)

class RvolTrendStrategy(BaseStrategy):
    """
    Strategy: High Relative Volume (RVOL) + Supertrend Confirmation.
    """
    name = "rvol_trend"

    def __init__(self, rvol_threshold: float = 2.0, st_period: int = 7, st_multiplier: int = 3):
        self.rvol_threshold = rvol_threshold
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.supertrend = SupertrendIndicator(period=st_period, multiplier=st_multiplier)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates buy/sell signals based on RVOL and Supertrend.
        """
        if data is None or data.empty or len(data) < 21:
            return data

        df = data.copy()
        
        # 1. Calculate Average Volume (20 period)
        df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
        
        # 2. Calculate RVOL
        df['rvol'] = df['volume'] / df['avg_volume_20']
        
        # 3. Calculate Supertrend
        df = self.supertrend.calculate(df)
        
        # 4. Refine Signal with RVOL filter
        # Only keep Supertrend BUY signals if RVOL is above threshold
        df['final_signal'] = 0
        
        # We look for where supertrend_signal was 1 (Buy flip) 
        # AND rvol > threshold on that same candle
        df.loc[(df['supertrend_signal'] == 1) & (df['rvol'] >= self.rvol_threshold), 'final_signal'] = 1
        
        # Sell signal is just the Supertrend sell flip (usually no volume filter for exits)
        df.loc[df['supertrend_signal'] == -1, 'final_signal'] = -1
        
        return df

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        """Custom message for this strategy."""
        action = "🚀 HIGH VOLUME BUY" if row['final_signal'] == 1 else "🛑 EXIT"
        color = "🟢" if row['final_signal'] == 1 else "🔴"
        
        message = (
            f"{color} <b>{action} Signal!</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Price:</b> ₹{row['close']:.2f}\n"
            f"<b>RVOL:</b> {row['rvol']:.2f}x\n"
            f"<b>Avg Vol:</b> {row['avg_volume_20']:.0f}\n"
            f"<b>Time:</b> {row['timestamp']}\n"
        )
        return message
