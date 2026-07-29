from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from .base import BaseIndicator

@dataclass
class ChartPrimeVPResult:
    poc: float
    volume_delta: float
    pivots: list[dict] # {price, index, is_high, volume_percent}
    bins: np.ndarray
    bin_edges: np.ndarray

class ChartPrimeVPIndicator(BaseIndicator):
    """
    Python implementation of "Volume Profile + Pivot Levels [ChartPrime]"
    Combines Volume Profile bucketing with significant Pivot detection filtered by volume intensity.
    """
    
    def __init__(
        self, 
        period: int = 200, 
        num_bins: int = 50, 
        pivot_length: int = 10, 
        pivot_filter: float = 20.0
    ):
        self.period = period
        self.num_bins = num_bins
        self.pivot_length = pivot_length
        self.pivot_filter = pivot_filter

    def calculate(self, data: pd.DataFrame) -> ChartPrimeVPResult | None:
        if data.empty or len(data) < self.period:
            return None

        # Work on the lookback window
        df = data.tail(self.period).copy()
        df.columns = [col.lower() for col in df.columns]
        
        # 1. Identify Pivots (ta.pivothigh / ta.pivotlow)
        # A pivot is high if it's the max in its window [i-len, i+len]
        df['is_high'] = df['high'] == df['high'].rolling(window=2*self.pivot_length+1, center=True).max()
        df['is_low'] = df['low'] == df['low'].rolling(window=2*self.pivot_length+1, center=True).min()
        
        # 2. Volume Profile & Delta Calculation
        h_max = df['high'].max()
        l_min = df['low'].min()
        
        if h_max == l_min:
            return None
            
        bin_size = (h_max - l_min) / self.num_bins
        bin_edges = np.linspace(l_min, h_max, self.num_bins + 1)
        
        bins_vol = np.zeros(self.num_bins)
        bins_delta = np.zeros(self.num_bins)
        
        # Bucketing logic following Pine Script:
        # close[j] >= bin_low-bin_size and close[j] < bin_high+bin_size
        for i in range(self.num_bins):
            b_low = bin_edges[i]
            b_high = bin_edges[i+1]
            
            # Fuzzy match as per original script
            mask = (df['close'] >= b_low - bin_size) & (df['close'] < b_high + bin_size)
            bins_vol[i] = df.loc[mask, 'volume'].sum()
            
            # Delta: close > open ? vol : -vol
            signed_vol = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
            bins_delta[i] = signed_vol[mask].sum()

        max_vol = bins_vol.max()
        if max_vol == 0:
            return None
            
        # POC: Midpoint of bin with max volume
        poc_idx = np.argmax(bins_vol)
        poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        
        # 3. Filter Pivots by Volume Percentage
        detected_pivots = []
        
        # Extract pivot points
        high_pivots = df[df['is_high']]
        low_pivots = df[df['is_low']]
        
        # Process Highs
        for idx, row in high_pivots.iterrows():
            p_val = row['high']
            # Find which bin this pivot belongs to
            b_idx = np.digitize(p_val, bin_edges) - 1
            b_idx = min(max(0, b_idx), self.num_bins - 1)
            
            vol_pct = (bins_vol[b_idx] / max_vol) * 100
            if vol_pct >= self.pivot_filter:
                detected_pivots.append({
                    "price": float(p_val),
                    "index": idx,
                    "is_high": True,
                    "vol_pct": float(vol_pct)
                })

        # Process Lows
        for idx, row in low_pivots.iterrows():
            p_val = row['low']
            b_idx = np.digitize(p_val, bin_edges) - 1
            b_idx = min(max(0, b_idx), self.num_bins - 1)
            
            vol_pct = (bins_vol[b_idx] / max_vol) * 100
            if vol_pct >= self.pivot_filter:
                detected_pivots.append({
                    "price": float(p_val),
                    "index": idx,
                    "is_high": False,
                    "vol_pct": float(vol_pct)
                })

        return ChartPrimeVPResult(
            poc=float(poc_price),
            volume_delta=float(bins_delta.sum()),
            pivots=detected_pivots,
            bins=bins_vol,
            bin_edges=bin_edges
        )
