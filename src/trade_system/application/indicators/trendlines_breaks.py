import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TrendlinesWithBreaks:
    """
    Python implementation of LuxAlgo - Trendlines with Breaks.
    """
    def __init__(
        self, 
        length: int = 14, 
        mult: float = 1.0, 
        calc_method: str = 'Atr'
    ):
        self.length = length
        self.mult = mult
        self.calc_method = calc_method

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates trendlines and breakout signals.
        """
        if len(df) < self.length * 2:
            return df

        df = df.copy()
        n = len(df)
        
        # 1. Pivot High / Low Detection (Vectorized)
        highs = df['high'].values
        lows = df['low'].values
        ph = np.full(n, np.nan)
        pl = np.full(n, np.nan)

        for i in range(self.length, n - self.length):
            # Pivot High: current high is >= all highs in [i-length, i+length]
            window_h = highs[i - self.length : i + self.length + 1]
            if highs[i] == np.max(window_h):
                ph[i] = highs[i]
                
            # Pivot Low: current low is <= all lows in [i-length, i+length]
            window_l = lows[i - self.length : i + self.length + 1]
            if lows[i] == np.min(window_l):
                pl[i] = lows[i]

        # Shift pivots forward by 'length' bars to avoid lookahead bias
        # In Pine, ta.pivothigh(14, 14) returns a value at bar i only when bar i+14 is reached.
        df['ph'] = pd.Series(ph).shift(self.length)
        df['pl'] = pd.Series(pl).shift(self.length)

        # 2. ATR Calculation for Slope
        high_low = df['high'] - df['low']
        high_cp = np.abs(df['high'] - df['close'].shift())
        low_cp = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.length).mean()

        # 3. Slope Calculation
        if self.calc_method == 'Atr':
            df['slope_base'] = df['atr'] / self.length * self.mult
        else:
            # Fallback to ATR if method not implemented
            df['slope_base'] = df['atr'] / self.length * self.mult

        # 4. Iterative Trendline Calculation (Stateful)
        upper = np.zeros(n)
        lower = np.zeros(n)
        slope_ph = np.zeros(n)
        slope_pl = np.zeros(n)
        upos = np.zeros(n)
        dnos = np.zeros(n)

        # Loop through data to maintain state
        for i in range(1, n):
            # Slopes
            if not np.isnan(df['ph'].iloc[i]):
                slope_ph[i] = df['slope_base'].iloc[i]
            else:
                slope_ph[i] = slope_ph[i-1]

            if not np.isnan(df['pl'].iloc[i]):
                slope_pl[i] = df['slope_base'].iloc[i]
            else:
                slope_pl[i] = slope_pl[i-1]

            # Trendline Values
            if not np.isnan(df['ph'].iloc[i]):
                upper[i] = df['ph'].iloc[i]
            else:
                upper[i] = upper[i-1] - slope_ph[i]

            if not np.isnan(df['pl'].iloc[i]):
                lower[i] = df['pl'].iloc[i]
            else:
                lower[i] = lower[i-1] + slope_pl[i]

            # Breakouts (Internal Oscillator state in Pine)
            # upos := ph ? 0 : close > upper - slope_ph * length ? 1 : upos
            # Note: Pine's 'upper' in that context is the value at bar i.
            if not np.isnan(df['ph'].iloc[i]):
                upos[i] = 0
            elif df['close'].iloc[i] > (upper[i] - slope_ph[i] * self.length):
                upos[i] = 1
            else:
                upos[i] = upos[i-1]

            if not np.isnan(df['pl'].iloc[i]):
                dnos[i] = 0
            elif df['close'].iloc[i] < (lower[i] + slope_pl[i] * self.length):
                dnos[i] = 1
            else:
                dnos[i] = dnos[i-1]

        df['upper_line'] = upper
        df['lower_line'] = lower
        df['upos'] = upos
        df['dnos'] = dnos
        
        # Signals: 1 for Buy, -1 for Sell
        df['trendline_signal'] = 0
        # Bullish Break: upos transitions from 0 to 1
        df.loc[(df['upos'] == 1) & (df['upos'].shift(1) == 0), 'trendline_signal'] = 1
        # Bearish Break: dnos transitions from 0 to 1
        df.loc[(df['dnos'] == 1) & (df['dnos'].shift(1) == 0), 'trendline_signal'] = -1

        return df
