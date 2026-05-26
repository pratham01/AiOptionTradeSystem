import pandas as pd
import numpy as np
from .base import BaseIndicator

class OBVIndicator(BaseIndicator):
    """
    On-Balance Volume (OBV) Indicator.
    OBV is a technical trading momentum indicator that uses volume flow to predict changes in stock price.
    """
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data
            
        df = data.copy()
        # Ensure lowercase column names for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate OBV
        # If current close > previous close, OBV = prev_OBV + current_volume
        # If current close < previous close, OBV = prev_OBV - current_volume
        # If current close == previous close, OBV = prev_OBV
        
        df['price_change'] = df['close'].diff()
        df['vol_direction'] = np.where(df['price_change'] > 0, 1, 
                                     np.where(df['price_change'] < 0, -1, 0))
        
        df['obv'] = (df['vol_direction'] * df['volume']).cumsum()
        
        # Cleanup
        df.drop(columns=['price_change', 'vol_direction'], inplace=True)
        
        return df
