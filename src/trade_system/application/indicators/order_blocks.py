"""
Volumetric Order Blocks & Demand/Supply Zone Detector
=====================================================
Identifies high-probability institutional Order Blocks (OB) based on:
1. Swing Pivots (Highs/Lows)
2. Break of Structure (BOS) / Change of Character (CHoCH)
3. Originating Volume & Range of base candles (Volumetric Signature)
4. Mitigation status (whether the zone has been tested/mitigated by price action)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

@dataclass
class OrderBlockZone:
    """Represents a Supply or Demand Order Block Zone."""
    is_bullish: bool       # True = Demand Zone (Buy), False = Supply Zone (Sell)
    entry_price: float     # Top of the zone (for Demand) or bottom of the zone (for Supply)
    stop_price: float      # Bottom of the zone (for Demand) or top of the zone (for Supply)
    origin_bar_idx: int    # Bar index where the OB was formed
    breakout_bar_idx: int  # Bar index where the Break of Structure occurred
    volume: float          # Cumulative volume of the base candle(s)
    volume_ratio: float    # Base volume / average volume (liquidity footprint)
    is_mitigated: bool     # True = already tested, False = active/unmitigated zone
    mitigated_bar_idx: int | None = None

class VolumetricOrderBlockDetector:
    """
    Scans historical OHLCV data to detect structural order blocks,
    calculates volume signatures, and filters out mitigated zones.
    """

    def __init__(
        self,
        pivot_left: int = 5,
        pivot_right: int = 5,
        vol_avg_period: int = 20,
    ) -> None:
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.vol_avg_period = vol_avg_period

    def _find_swings(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Detect swing highs and swing lows."""
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        
        swing_highs = np.zeros(n)
        swing_lows = np.zeros(n)

        for i in range(self.pivot_left, n - self.pivot_right):
            # Check swing high
            window_high = highs[i - self.pivot_left : i + self.pivot_right + 1]
            if highs[i] == window_high.max():
                swing_highs[i] = highs[i]

            # Check swing low
            window_low = lows[i - self.pivot_left : i + self.pivot_right + 1]
            if lows[i] == window_low.min():
                swing_lows[i] = lows[i]

        return swing_highs, swing_lows

    def calculate(self, df: pd.DataFrame) -> list[OrderBlockZone]:
        """
        Scans data to build and track Order Blocks.
        Returns a list of active (unmitigated) and historical order block zones.
        """
        if df.empty or len(df) < (self.pivot_left + self.pivot_right + 1):
            return []

        df = df.copy()
        df["vol_sma"] = df["volume"].rolling(window=self.vol_avg_period, min_periods=1).mean()
        
        swing_highs, swing_lows = self._find_swings(df)
        
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values
        vol_smas = df["vol_sma"].values
        n = len(df)

        active_bullish_obs: list[OrderBlockZone] = []
        active_bearish_obs: list[OrderBlockZone] = []
        all_obs: list[OrderBlockZone] = []

        # Keep track of active swings for BOS checks
        last_swing_high_idx = -1
        last_swing_low_idx = -1

        for i in range(n):
            # Update latest confirmed swings
            # A swing at index j is only confirmed after self.pivot_right bars have passed (i.e. at index j + self.pivot_right)
            conf_idx = i - self.pivot_right
            if conf_idx >= 0:
                if swing_highs[conf_idx] > 0:
                    last_swing_high_idx = conf_idx
                if swing_lows[conf_idx] > 0:
                    last_swing_low_idx = conf_idx

            # ── 1. Check Mitigation for Active Zones ────────────────────────
            # Mitigate Bullish Zones
            for ob in active_bullish_obs[:]:
                if lows[i] < ob.stop_price:
                    # Invalidated completely (broke below stop)
                    active_bullish_obs.remove(ob)
                elif lows[i] <= ob.entry_price:
                    # Mitigated (touched)
                    ob.is_mitigated = True
                    ob.mitigated_bar_idx = i
                    active_bullish_obs.remove(ob)

            # Mitigate Bearish Zones
            for ob in active_bearish_obs[:]:
                if highs[i] > ob.stop_price:
                    # Invalidated completely (broke above stop)
                    active_bearish_obs.remove(ob)
                elif highs[i] >= ob.entry_price:
                    # Mitigated (touched)
                    ob.is_mitigated = True
                    ob.mitigated_bar_idx = i
                    active_bearish_obs.remove(ob)

            # ── 2. Check for Break of Structure (BOS) ────────────────────────
            # Bullish BOS (Close breaks above last swing high)
            if last_swing_high_idx != -1 and closes[i] > highs[last_swing_high_idx]:
                # Look back to find the origin base candle (most recent bearish candle before the push)
                origin_idx = -1
                for j in range(i - 1, last_swing_high_idx - 1, -1):
                    if closes[j] < opens[j]:  # Bearish base candle
                        origin_idx = j
                        break
                
                if origin_idx == -1:
                    origin_idx = last_swing_high_idx  # Fallback

                # Define Demand Zone range (from high to low of base candle)
                entry = highs[origin_idx]
                stop = lows[origin_idx]
                vol = volumes[origin_idx]
                avg_vol = vol_smas[origin_idx] if vol_smas[origin_idx] > 0 else 1.0
                ratio = vol / avg_vol

                # Create Bullish OB (Demand Zone)
                ob = OrderBlockZone(
                    is_bullish=True,
                    entry_price=round(entry, 2),
                    stop_price=round(stop, 2),
                    origin_bar_idx=origin_idx,
                    breakout_bar_idx=i,
                    volume=vol,
                    volume_ratio=round(ratio, 2),
                    is_mitigated=False
                )
                
                # Check if it has been mitigated immediately
                mitigated = False
                for k in range(origin_idx + 1, i + 1):
                    if lows[k] <= entry:
                        mitigated = True
                        ob.is_mitigated = True
                        ob.mitigated_bar_idx = k
                        break

                all_obs.append(ob)
                if not mitigated:
                    active_bullish_obs.append(ob)

                # Reset so we don't trigger multiple BOS on the same swing
                last_swing_high_idx = -1

            # Bearish BOS (Close breaks below last swing low)
            elif last_swing_low_idx != -1 and closes[i] < lows[last_swing_low_idx]:
                # Look back to find the origin base candle (most recent bullish candle before the push)
                origin_idx = -1
                for j in range(i - 1, last_swing_low_idx - 1, -1):
                    if closes[j] > opens[j]:  # Bullish base candle
                        origin_idx = j
                        break
                
                if origin_idx == -1:
                    origin_idx = last_swing_low_idx  # Fallback

                # Define Supply Zone range (from low to high of base candle)
                entry = lows[origin_idx]
                stop = highs[origin_idx]
                vol = volumes[origin_idx]
                avg_vol = vol_smas[origin_idx] if vol_smas[origin_idx] > 0 else 1.0
                ratio = vol / avg_vol

                # Create Bearish OB (Supply Zone)
                ob = OrderBlockZone(
                    is_bullish=False,
                    entry_price=round(entry, 2),
                    stop_price=round(stop, 2),
                    origin_bar_idx=origin_idx,
                    breakout_bar_idx=i,
                    volume=vol,
                    volume_ratio=round(ratio, 2),
                    is_mitigated=False
                )
                
                # Check mitigation immediately
                mitigated = False
                for k in range(origin_idx + 1, i + 1):
                    if highs[k] >= entry:
                        mitigated = True
                        ob.is_mitigated = True
                        ob.mitigated_bar_idx = k
                        break

                all_obs.append(ob)
                if not mitigated:
                    active_bearish_obs.append(ob)

                # Reset
                last_swing_low_idx = -1

        return all_obs
