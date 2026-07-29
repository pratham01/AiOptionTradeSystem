from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from trade_system.domains.strategy.application.indicators.base import BaseIndicator

LOGGER = logging.getLogger(__name__)

@dataclass
class FVG:
    type: str  # "BFVG" (Bullish) or "BKFVG" (Bearish)
    top: float
    bottom: float
    timestamp: pd.Timestamp
    filled: bool = False

@dataclass
class RetracementZone:
    type: str  # "FIB", "FVG", "BREAKOUT"
    level: float
    top: float
    bottom: float
    meta: Dict[str, Any]

class RetracementIndicator(BaseIndicator):
    """
    Identifies high-probability retracement zones:
    1. Fibonacci Levels (0.5, 0.618)
    2. Fair Value Gaps (FVG)
    3. Breakout Zone Retests (S/R Flip)
    """

    def __init__(self, pivot_length: int = 5, fvg_min_size_pct: float = 0.05):
        self.pivot_length = pivot_length
        self.fvg_min_size_pct = fvg_min_size_pct

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < self.pivot_length * 2:
            return df

        df = df.copy()
        df['retracement_signal'] = 0
        df['retracement_zone'] = ""

        # 1. Fibonacci & Breakout Zones (Pivot based)
        df = self._calculate_pivots_and_fib(df)
        
        # 2. FVG Detection
        df = self._calculate_fvg(df)

        return df

    def _calculate_pivots_and_fib(self, df: pd.DataFrame) -> pd.DataFrame:
        highs = df['high'].values
        lows = df['low'].values
        
        df['last_ph'] = np.nan
        df['last_pl'] = np.nan
        df['fib_50'] = np.nan
        df['fib_618'] = np.nan
        df['breakout_zone'] = np.nan

        last_ph = None
        last_pl = None
        
        for i in range(self.pivot_length, len(df)):
            # Detect Pivot High
            window_h = highs[i - self.pivot_length : i + 1]
            if highs[i - self.pivot_length] == np.max(highs[i - 2*self.pivot_length : i + 1] if i >= 2*self.pivot_length else window_h):
                # This is a simplified pivot check to avoid lookahead in real-time
                pass
            
            # Use actual rolling max/min for simpler implementation
            rolling_max = df['high'].shift(1).rolling(window=self.pivot_length*2).max()
            rolling_min = df['low'].shift(1).rolling(window=self.pivot_length*2).min()
            
            df['last_ph'] = rolling_max
            df['last_pl'] = rolling_min
            
            # Fib Levels of the move from PL to PH (or vice versa)
            df['fib_50'] = df['last_pl'] + (df['last_ph'] - df['last_pl']) * 0.5
            df['fib_618'] = df['last_pl'] + (df['last_ph'] - df['last_pl']) * 0.382 # Retracement to 61.8 from top is 38.2 from bottom
            # Correcting: 0.618 retracement from high means price is at PL + (PH-PL)*0.382
            # Let's define them as distance from bottom
            # 0.5 = 50%
            # 0.618 = 61.8% retracement (price is at 38.2% of the move)
            
        return df

    def _calculate_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan
        
        for i in range(2, len(df)):
            c1 = df.iloc[i-2]
            c3 = df.iloc[i]
            
            avg_price = (c1['close'] + c3['close']) / 2
            min_gap = avg_price * (self.fvg_min_size_pct / 100)

            # Bullish FVG
            if c3['low'] > c1['high'] + min_gap:
                df.loc[df.index[i], 'fvg_top'] = c3['low']
                df.loc[df.index[i], 'fvg_bottom'] = c1['high']
            
            # Bearish FVG
            elif c3['high'] < c1['low'] - min_gap:
                df.loc[df.index[i], 'fvg_top'] = c1['low']
                df.loc[df.index[i], 'fvg_bottom'] = c3['high']
                
        return df

    def get_active_zones(self, df: pd.DataFrame) -> List[RetracementZone]:
        """Returns zones the current price is currently 'visiting'."""
        if df.empty: return []
        
        latest = df.iloc[-1]
        price = latest['close']
        zones = []
        
        # 1. Check Fib 50-61.8 Zone
        fib_high = latest['fib_50']
        fib_low = latest['fib_618']
        if not np.isnan(fib_high) and not np.isnan(fib_low):
            # Sort them
            f_min, f_max = min(fib_high, fib_low), max(fib_high, fib_low)
            if f_min <= price <= f_max:
                zones.append(RetracementZone("FIB", 0.618, f_max, f_min, {}))

        # 2. Check FVG (Unfilled)
        # Scan recent bars for FVG
        for i in range(len(df)-1, max(-1, len(df)-50), -1):
            f_top = df.iloc[i]['fvg_top']
            f_bot = df.iloc[i]['fvg_bottom']
            if not np.isnan(f_top):
                if f_bot <= price <= f_top:
                    zones.append(RetracementZone("FVG", 0.0, f_top, f_bot, {"timestamp": df.iloc[i]['timestamp']}))
                    break # Only latest one

        # 3. Breakout Retest
        last_ph = latest['last_ph']
        if not np.isnan(last_ph):
            # If price was above PH and now retesting it within 0.2%
            if abs(price - last_ph) / last_ph <= 0.002:
                zones.append(RetracementZone("BREAKOUT", 0.0, last_ph * 1.002, last_ph * 0.998, {"level": "PH"}))

        return zones
