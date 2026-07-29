from __future__ import annotations

import pandas as pd
import numpy as np
from trade_system.domains.strategy.application.indicators.base import BaseIndicator


class VolumeDeltaIndicator(BaseIndicator):
    """
    Calculates estimated Volume Delta and Cumulative Volume Delta (CVD) from OHLCV data.
    Uses the Money Flow Multiplier to estimate buying vs selling pressure.
    
    Delta = Volume * ((Close - Low) - (High - Close)) / (High - Low)
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # Calculate Money Flow Multiplier
        # Handle division by zero for flat candles
        price_range = df["high"] - df["low"]
        multiplier = np.where(
            price_range > 0,
            ((df["close"] - df["low"]) - (df["high"] - df["close"])) / price_range,
            0
        )

        df["delta"] = multiplier * df["volume"]
        
        # Cumulative Volume Delta (CVD)
        # Reset CVD at the start of each day if timestamp is available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
            df['date'] = df['timestamp'].dt.date
            df['cvd'] = df.groupby('date')['delta'].cumsum()
        else:
            df['cvd'] = df['delta'].cumsum()

        return df
