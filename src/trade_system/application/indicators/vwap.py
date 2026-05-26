from __future__ import annotations

import pandas as pd
from trade_system.application.indicators.base import BaseIndicator


class VWAPIndicator(BaseIndicator):
    """
    Volume Weighted Average Price (VWAP) Indicator.
    Resets at the start of each day.
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        df = data.copy()
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol'] = df['tp'] * df['volume']
        
        # Group by date and calculate cumulative sum
        groups = df.groupby('date')
        df['cum_tp_vol'] = groups['tp_vol'].cumsum()
        df['cum_vol'] = groups['volume'].cumsum()
        
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # Clean up temporary columns
        df.drop(columns=['tp', 'tp_vol', 'cum_tp_vol', 'cum_vol'], inplace=True)
        
        return df
