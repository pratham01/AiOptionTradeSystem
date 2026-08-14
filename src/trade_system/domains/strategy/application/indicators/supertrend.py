import pandas as pd
import numpy as np
import logging
from .base import BaseIndicator

logger = logging.getLogger(__name__)

class SupertrendIndicator(BaseIndicator):
    def __init__(self, period: int = 7, multiplier: int = 3):
        self.period = period
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates Supertrend on the provided OHLC data."""
        if data is None or data.empty or len(data) < self.period:
            return data

        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # --- DATA CONTINUITY SAFEGUARD ---
        # Detect if there are massive gaps (e.g., > 45 mins) in the intraday time-series.
        # This prevents GIGO (Garbage-In Garbage-Out) calculation corruption.
        time_series = None
        if isinstance(df.index, pd.DatetimeIndex):
            time_series = df.index.to_series()
        elif 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            time_series = df['timestamp']
        
        if time_series is not None and len(time_series) > 1:
            gaps = time_series.groupby(time_series.dt.date).diff()
            max_gap = gaps.max()
            if not pd.isna(max_gap) and max_gap > pd.Timedelta(minutes=45):
                logger.error(f"CRITICAL: Data gap of {max_gap} detected! Suppressing Supertrend calculation to prevent corrupted signals.")
                return pd.DataFrame() # Return empty to prevent false signals
        # ---------------------------------

        # Calculate True Range (TR)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

        # Calculate Average True Range (ATR)
        df['atr'] = df['tr'].ewm(alpha=1/self.period, adjust=False).mean()

        # Calculate basic upper and lower bands
        df['hl2'] = (df['high'] + df['low']) / 2
        df['basic_upperband'] = df['hl2'] + self.multiplier * df['atr']
        df['basic_lowerband'] = df['hl2'] - self.multiplier * df['atr']

        # Calculate final upper and lower bands
        df['final_upperband'] = 0.0
        df['final_lowerband'] = 0.0

        for i in range(1, len(df)):
            prev_f_up = df.loc[df.index[i-1], 'final_upperband']
            prev_f_low = df.loc[df.index[i-1], 'final_lowerband']
            prev_close = df.loc[df.index[i-1], 'close']
            curr_b_up = df.loc[df.index[i], 'basic_upperband']
            curr_b_low = df.loc[df.index[i], 'basic_lowerband']

            # Update final upper band
            if curr_b_up < prev_f_up or prev_close > prev_f_up:
                df.loc[df.index[i], 'final_upperband'] = curr_b_up
            else:
                df.loc[df.index[i], 'final_upperband'] = prev_f_up

            # Update final lower band
            if curr_b_low > prev_f_low or prev_close < prev_f_low:
                df.loc[df.index[i], 'final_lowerband'] = curr_b_low
            else:
                df.loc[df.index[i], 'final_lowerband'] = prev_f_low

        # Calculate Supertrend
        df['supertrend'] = 0.0
        for i in range(1, len(df)):
            prev_st = df.loc[df.index[i-1], 'supertrend']
            prev_f_up = df.loc[df.index[i-1], 'final_upperband']
            prev_f_low = df.loc[df.index[i-1], 'final_lowerband']
            curr_close = df.loc[df.index[i], 'close']

            if prev_st == prev_f_up:
                if curr_close > df.loc[df.index[i], 'final_upperband']:
                    df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lowerband']
                else:
                    df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_upperband']
            else:
                if curr_close < df.loc[df.index[i], 'final_lowerband']:
                    df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_upperband']
                else:
                    df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lowerband']

        # Calculate Supertrend direction
        df['supertrend_direction'] = np.where(df['close'] > df['supertrend'], 1, -1)

        # Calculate Supertrend signal
        df['supertrend_signal'] = 0
        df.loc[df['supertrend_direction'] > df['supertrend_direction'].shift(1), 'supertrend_signal'] = 1
        df.loc[df['supertrend_direction'] < df['supertrend_direction'].shift(1), 'supertrend_signal'] = -1

        # Drop intermediate columns
        cols_to_drop = ['tr0', 'tr1', 'tr2', 'tr', 'atr', 'hl2', 'basic_upperband', 'basic_lowerband', 'final_upperband', 'final_lowerband']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        return df


def calculate_supertrend(data: pd.DataFrame, period: int = 7, multiplier: int = 3) -> pd.DataFrame:
    return SupertrendIndicator(period=period, multiplier=multiplier).calculate(data)
