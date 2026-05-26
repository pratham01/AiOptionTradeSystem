import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

LOGGER = logging.getLogger(__name__)

@dataclass
class Pivot:
    index: int
    value: float
    timestamp: pd.Timestamp

class SandRYata:
    """
    Python implementation of S&R • Yata.
    Calculates Support & Resistance channels and Fibonacci levels.
    """
    def __init__(
        self,
        length: int = 21,
        max_pivots: int = 5,
        fib_levels: Optional[List[float]] = None,
        ext_levels: Optional[List[float]] = None
    ):
        self.length = length
        self.max_pivots = max_pivots
        self.fib_levels = fib_levels or [0.236, 0.382, 0.5, 0.618, 0.786]
        self.ext_levels = ext_levels or [1.618, -0.618]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates pivot-based S&R channels and Fibonacci levels.
        """
        if len(df) < self.length * 2:
            return df

        df = df.copy()
        n = len(df)
        
        # 1. Pivot Detection
        highs = df['high'].values
        lows = df['low'].values
        
        ph_indices = []
        ph_values = []
        pl_indices = []
        pl_values = []
        
        # Series for results
        df['res_line'] = np.nan
        df['sup_line'] = np.nan
        
        # Fibonacci level columns
        for i, level in enumerate(self.fib_levels, 1):
            df[f'fib_r{i}'] = np.nan
        for i, level in enumerate(self.ext_levels, 1):
            df[f'fib_e{i}'] = np.nan

        # Iterate through bars to simulate Pine Script's bar-by-bar processing
        # and avoid lookahead bias.
        for i in range(self.length, n):
            # Check for pivot high at index i - length
            # A pivot is confirmed only after 'length' bars pass
            target_idx = i - self.length
            if target_idx >= self.length:
                window_h = highs[target_idx - self.length : target_idx + self.length + 1]
                if highs[target_idx] == np.max(window_h):
                    ph_indices.append(target_idx)
                    ph_values.append(highs[target_idx])
                    if len(ph_indices) > self.max_pivots:
                        ph_indices.pop(0)
                        ph_values.pop(0)

                window_l = lows[target_idx - self.length : target_idx + self.length + 1]
                if lows[target_idx] == np.min(window_l):
                    pl_indices.append(target_idx)
                    pl_values.append(lows[target_idx])
                    if len(pl_indices) > self.max_pivots:
                        pl_indices.pop(0)
                        pl_values.pop(0)

            # 2. Channel Calculation (If we have enough pivots)
            # Resistance Line (Channel High)
            best_res_val = np.nan
            res_line_info = None
            
            if len(ph_indices) >= 2:
                min_diff = float('inf')
                for idx_a in range(len(ph_indices)):
                    for idx_b in range(idx_a + 1, len(ph_indices)):
                        x1, y1 = ph_indices[idx_a], ph_values[idx_a]
                        x2, y2 = ph_indices[idx_b], ph_values[idx_b]
                        
                        # Line equation: y = mx + c
                        m = (y2 - y1) / (x2 - x1)
                        c = y1 - m * x1
                        
                        # Current price of this line at bar i
                        current_lp = m * i + c
                        
                        if current_lp > highs[i]:
                            diff = abs(current_lp - df['close'].iloc[i])
                            if diff < min_diff:
                                min_diff = diff
                                best_res_val = current_lp
                                res_line_info = (m, c)
            
            df.loc[df.index[i], 'res_line'] = best_res_val

            # Support Line (Channel Low)
            best_sup_val = np.nan
            sup_line_info = None
            
            if len(pl_indices) >= 2:
                min_diff = float('inf')
                for idx_a in range(len(pl_indices)):
                    for idx_b in range(idx_a + 1, len(pl_indices)):
                        x1, y1 = pl_indices[idx_a], pl_values[idx_a]
                        x2, y2 = pl_indices[idx_b], pl_values[idx_b]
                        
                        m = (y2 - y1) / (x2 - x1)
                        c = y1 - m * x1
                        current_lp = m * i + c
                        
                        if current_lp < lows[i]:
                            diff = abs(current_lp - df['close'].iloc[i])
                            if diff < min_diff:
                                min_diff = diff
                                best_sup_val = current_lp
                                sup_line_info = (m, c)
            
            df.loc[df.index[i], 'sup_line'] = best_sup_val

            # 3. Fibonacci Levels
            if not np.isnan(best_res_val) and not np.isnan(best_sup_val):
                diff = best_res_val - best_sup_val
                
                for j, level in enumerate(self.fib_levels, 1):
                    df.loc[df.index[i], f'fib_r{j}'] = best_sup_val + diff * level
                
                for j, level in enumerate(self.ext_levels, 1):
                    df.loc[df.index[i], f'fib_e{j}'] = best_sup_val + diff * level

        return df
