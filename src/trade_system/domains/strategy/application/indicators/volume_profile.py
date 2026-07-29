from __future__ import annotations

import numpy as np
import pandas as pd
from trade_system.domains.strategy.application.indicators.base import BaseIndicator
from trade_system.shared import VolumeProfileSummary


class VolumeProfileIndicator(BaseIndicator):
    """
    Calculates Volume Profile (POC, VAH, VAL) for a given dataset.
    Uses a high-accuracy distribution model where volume is distributed 
    across the price range of each candle.
    """

    def __init__(self, price_step: float = 2.0, value_area_pct: float = 0.70):
        self.price_step = price_step
        self.value_area_pct = value_area_pct

    def calculate(self, data: pd.DataFrame) -> VolumeProfileSummary | None:
        """
        Calculates Volume Profile levels from OHLCV data.
        Returns a VolumeProfileSummary object.
        """
        if data.empty or "close" not in data.columns or "volume" not in data.columns:
            return None

        df = data.copy()
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure numeric types
        for col in ["high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = df[df["volume"] > 0]
        if df.empty:
            return None

        min_price = df["low"].min()
        max_price = df["high"].max()
        
        if min_price == max_price:
            return VolumeProfileSummary(
                point_of_control=min_price,
                value_area_low=min_price,
                value_area_high=min_price,
                total_volume=df["volume"].sum()
            )

        # Create price bins
        bins = np.arange(
            np.floor(min_price / self.price_step) * self.price_step,
            np.ceil(max_price / self.price_step) * self.price_step + self.price_step,
            self.price_step
        )
        
        bin_counts = len(bins) - 1
        volume_distribution = np.zeros(bin_counts)

        # Distribute volume across bins for each candle
        # We use a simple weighted model: 50% volume at typical price, 
        # and remaining 50% distributed across the range.
        for _, row in df.iterrows():
            high = row["high"]
            low = row["low"]
            close = row["close"]
            vol = row["volume"]
            typical = (high + low + close) / 3
            
            # 1. 50% at Typical Price
            typical_bin = np.digitize(typical, bins) - 1
            if 0 <= typical_bin < bin_counts:
                volume_distribution[typical_bin] += vol * 0.5
            
            # 2. 50% distributed across High-Low Range
            if high > low:
                start_bin = np.digitize(low, bins) - 1
                end_bin = np.digitize(high, bins) - 1
                start_bin = max(0, start_bin)
                end_bin = min(bin_counts - 1, end_bin)
                
                num_bins = end_bin - start_bin + 1
                vol_per_bin = (vol * 0.5) / num_bins
                for i in range(start_bin, end_bin + 1):
                    volume_distribution[i] += vol_per_bin
            else:
                if 0 <= typical_bin < bin_counts:
                    volume_distribution[typical_bin] += vol * 0.5

        total_volume = volume_distribution.sum()
        if total_volume == 0:
            return None

        # Find POC
        poc_idx = np.argmax(volume_distribution)
        poc_price = bins[poc_idx] + (self.price_step / 2)

        # Find Value Area (VAH, VAL)
        target_vol = total_volume * self.value_area_pct
        current_vol = volume_distribution[poc_idx]
        
        low_idx = poc_idx
        high_idx = poc_idx
        
        while current_vol < target_vol:
            # Check expansion potential
            can_go_lower = low_idx > 0
            can_go_higher = high_idx < bin_counts - 1
            
            if not can_go_lower and not can_go_higher:
                break
                
            vol_low = volume_distribution[low_idx - 1] if can_go_lower else -1
            vol_high = volume_distribution[high_idx + 1] if can_go_higher else -1
            
            if vol_high >= vol_low:
                high_idx += 1
                current_vol += volume_distribution[high_idx]
            else:
                low_idx -= 1
                current_vol += volume_distribution[low_idx]

        val = bins[low_idx]
        vah = bins[high_idx] + self.price_step

        return VolumeProfileSummary(
            point_of_control=float(poc_price),
            value_area_low=float(val),
            value_area_high=float(vah),
            total_volume=float(total_volume)
        )
