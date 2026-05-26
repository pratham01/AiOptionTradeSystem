import pandas as pd
import numpy as np
from .base import BaseIndicator
from .supertrend import calculate_supertrend

class EMASupertrendCloudIndicator(BaseIndicator):
    """
    Implementation of EMA 5-13-50 + Supertrend + Cloud strategy.
    
    Components:
    - EMA 5, 13, 50
    - Supertrend (10, 3)
    - Cloud (Area between EMA 5 and 13)
    """
    def __init__(self, ema_fast: int = 5, ema_medium: int = 13, ema_slow: int = 50, 
                 st_period: int = 10, st_multiplier: float = 3.0):
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        self.st_period = st_period
        self.st_multiplier = st_multiplier

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data
            
        df = data.copy()
        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate EMAs
        df[f'ema_{self.ema_fast}'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df[f'ema_{self.ema_medium}'] = df['close'].ewm(span=self.ema_medium, adjust=False).mean()
        df[f'ema_{self.ema_slow}'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Calculate Supertrend
        # The project's calculate_supertrend already adds 'supertrend' and 'supertrend_direction'
        st_df = calculate_supertrend(df, period=self.st_period, multiplier=self.st_multiplier)
        df['supertrend'] = st_df['supertrend']
        df['supertrend_direction'] = st_df['supertrend_direction']
        
        # Strategy Logic
        # Cloud: EMA 5 > EMA 13 is Bullish Cloud, EMA 5 < EMA 13 is Bearish Cloud
        df['cloud_direction'] = np.where(df[f'ema_{self.ema_fast}'] > df[f'ema_{self.ema_medium}'], 1, -1)
        
        # Trend Filter (EMA 50)
        df['trend_filter'] = np.where(df['close'] > df[f'ema_{self.ema_slow}'], 1, -1)
        
        # Final Signals
        # Long: Price > EMA 50 AND Cloud is Bullish AND Supertrend is Buy (1)
        df['long_signal'] = (df['trend_filter'] == 1) & (df['cloud_direction'] == 1) & (df['supertrend_direction'] == 1)
        
        # Short: Price < EMA 50 AND Cloud is Bearish AND Supertrend is Sell (-1)
        df['short_signal'] = (df['trend_filter'] == -1) & (df['cloud_direction'] == -1) & (df['supertrend_direction'] == -1)
        
        # Convert signals to 1, -1, 0 for easier analysis
        df['strategy_signal'] = 0
        df.loc[df['long_signal'], 'strategy_signal'] = 1
        df.loc[df['short_signal'], 'strategy_signal'] = -1
        
        return df

def calculate_ema_supertrend_cloud(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return EMASupertrendCloudIndicator(**kwargs).calculate(data)
