import pandas as pd
import numpy as np
import logging
from .base import BaseIndicator
from .supertrend import calculate_supertrend

logger = logging.getLogger(__name__)

class AdvancedSupertrendIndicator(BaseIndicator):
    """
    Advanced Supertrend Indicator incorporating sideways market detection
    using Choppiness Index (CHOP) and Bollinger Band Squeeze.
    """
    def __init__(self, st_period: int = 7, st_multiplier: float = 3.0, 
                 chop_period: int = 14, bb_period: int = 20, 
                 bb_std: float = 2.0, kc_mult: float = 1.5):
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.chop_period = chop_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_mult = kc_mult

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data
            
        df = data.copy()
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.columns = [col.lower() for col in df.columns]
        
        # 1. Base Supertrend
        st_df = calculate_supertrend(df, period=self.st_period, multiplier=self.st_multiplier)
        df['supertrend'] = st_df['supertrend']
        df['supertrend_direction'] = st_df['supertrend_direction']
        
        # 2. True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr_st'] = df['tr'].rolling(window=self.st_period).mean() # ATR for stops
        
        # 3. Choppiness Index (CHOP)
        # CHOP = 100 * LOG10( SUM(ATR(1), Period) / ( MaxHigh(Period) - MinLow(Period) ) ) / LOG10(Period)
        atr1_sum = df['tr'].rolling(window=self.chop_period).sum()
        highest_high = df['high'].rolling(window=self.chop_period).max()
        lowest_low = df['low'].rolling(window=self.chop_period).min()
        
        # Prevent division by zero
        range_hl = (highest_high - lowest_low).replace(0, np.nan)
        df['chop'] = 100 * np.log10(atr1_sum / range_hl) / np.log10(self.chop_period)
        
        # 4. Bollinger Bands & Keltner Channels (Squeeze Indicator)
        df['bb_mid'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * self.bb_std)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * self.bb_std)
        
        # For Keltner Channels, KC Mid is BB Mid, KC range is ATR(20)
        df['atr20'] = df['tr'].rolling(window=self.bb_period).mean()
        df['kc_upper'] = df['bb_mid'] + (df['atr20'] * self.kc_mult)
        df['kc_lower'] = df['bb_mid'] - (df['atr20'] * self.kc_mult)
        
        # Squeeze condition: Bollinger Bands sit entirely inside Keltner Channels
        # When Bollinger Bands contract, volatility is low -> sideways market
        df['is_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        
        # 5. EMA Filters (Trend Filter)
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Clean intermediate columns to keep memory usage low
        cols_to_drop = ['tr0', 'tr1', 'tr2', 'tr', 'atr20']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        return df

def calculate_advanced_supertrend(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return AdvancedSupertrendIndicator(**kwargs).calculate(data)
