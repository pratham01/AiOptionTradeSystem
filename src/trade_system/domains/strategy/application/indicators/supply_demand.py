"""
Institutional Supply & Demand Engine (JeaFx Masterclass Framework)
==================================================================
Identifies high-probability institutional Supply & Demand zones based on:
1. Imbalance / Fair Value Gap (FVG) Validation:
   - A base candle is ONLY valid if the subsequent impulse candle leaves an open Imbalance
     (gap between Candle 1 wick and Candle 3 wick >= alpha * ATR).
   - If adjacent candles overlap, the zone is deemed "efficient" and discarded.
2. Extreme vs. Decisional Zones:
   - Extreme Zone: Absolute origin base candle at the swing pivot.
     Highest probability of holding because no institutional imbalances remain behind it.
   - Decisional Zone: The base candle / consolidation formed immediately prior to the breakout
     candle that decided to break market structure (BOS / CHoCH).
3. Mitigation & Touch Tracking:
   - Fresh (0 touches): Highest unmitigated institutional order inventory.
   - Mitigated (1 touch): Partially filled.
   - Exhausted (2+ touches): High risk of being broken through.
4. Dealing Range Equilibrium:
   - High-probability Demand zones MUST reside in Discount (< 50% dealing range).
   - High-probability Supply zones MUST reside in Premium (> 50% dealing range).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from trade_system.domains.strategy.application.indicators.smc_structure import (
    SMCStructureState, SMCStructurePoint, PointType, StructureTrend
)

LOGGER = logging.getLogger(__name__)


class SDZoneType(str, Enum):
    EXTREME_DEMAND = "EXTREME_DEMAND"
    DECISIONAL_DEMAND = "DECISIONAL_DEMAND"
    EXTREME_SUPPLY = "EXTREME_SUPPLY"
    DECISIONAL_SUPPLY = "DECISIONAL_SUPPLY"


@dataclass
class SupplyDemandZone:
    """Represents an institutional Supply or Demand Zone."""
    zone_type: SDZoneType
    top: float                    # Upper price bound
    bottom: float                 # Lower price bound
    origin_bar_idx: int           # Index of base candle
    breakout_bar_idx: int         # Index of the candle that confirmed the break
    origin_date: str              # Timestamp of base candle
    volume: float                 # Volume of base candle
    volume_ratio: float           # Volume / 20-bar SMA
    imbalance_size: float         # FVG gap magnitude
    has_valid_imbalance: bool     # True if verified by 3-candle FVG rule
    is_mitigated: bool = False    # True if subsequent price entered the zone
    touch_count: int = 0          # Number of times retested
    mitigated_date: Optional[str] = None
    is_valid: bool = True         # False if price closed through the zone
    is_discount: bool = False     # In lower 50% dealing range
    is_premium: bool = False      # In upper 50% dealing range

    @property
    def mid(self) -> float:
        return round((self.top + self.bottom) / 2.0, 2)

    @property
    def is_fresh(self) -> bool:
        return not self.is_mitigated and self.touch_count == 0 and self.is_valid

    @property
    def is_demand(self) -> bool:
        return self.zone_type in (SDZoneType.EXTREME_DEMAND, SDZoneType.DECISIONAL_DEMAND)

    @property
    def is_supply(self) -> bool:
        return self.zone_type in (SDZoneType.EXTREME_SUPPLY, SDZoneType.DECISIONAL_SUPPLY)


class SupplyDemandEngine:
    """
    Automated Institutional Supply & Demand Zone Detector.
    
    Adheres strictly to the JeaFx methodology:
    - Requires impulse candle leaving an open FVG.
    - Classifies Extreme vs Decisional origins.
    - Monitors mitigation, touch exhaustion, and 50% equilibrium.
    """

    def __init__(
        self,
        swing_length: int = 5,
        fvg_min_atr_mult: float = 0.25,
        vol_avg_period: int = 20,
    ) -> None:
        self.swing_length = swing_length
        self.fvg_min_atr_mult = fvg_min_atr_mult
        self.vol_avg_period = vol_avg_period

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates Average True Range (ATR)."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    def detect_zones(
        self,
        df: pd.DataFrame,
        structure_state: Optional[SMCStructureState] = None,
    ) -> List[SupplyDemandZone]:
        """
        Detects Extreme and Decisional Supply & Demand zones across historical data.
        Returns all valid and active zones.
        """
        n = len(df)
        if n < 15:
            return []

        clean_df = df.copy().sort_values("timestamp").reset_index(drop=True)
        if "vol_sma" not in clean_df.columns:
            clean_df["vol_sma"] = clean_df["volume"].rolling(self.vol_avg_period, min_periods=1).mean()

        atr_series = self.calculate_atr(clean_df, period=14)
        atr_vals = atr_series.values
        opens = clean_df["open"].values
        highs = clean_df["high"].values
        lows = clean_df["low"].values
        closes = clean_df["close"].values
        volumes = clean_df["volume"].values
        vol_smas = clean_df["vol_sma"].values
        dates = clean_df["timestamp"].astype(str).values

        # Compute dealing range equilibrium from recent window
        recent_window = clean_df.tail(min(50, n))
        range_high = float(recent_window["high"].max())
        range_low = float(recent_window["low"].min())
        equilibrium = (range_high + range_low) / 2.0

        zones: List[SupplyDemandZone] = []

        # Scan for swing pivot points or strong momentum breakout bars
        k = self.swing_length
        for i in range(max(1, k), n - 1):
            atr_i = atr_vals[i] if atr_vals[i] > 0 else (closes[i] * 0.01)
            min_gap = atr_i * self.fvg_min_atr_mult

            # ── 1. CANDIDATE DEMAND ZONES (Bullish Impulse) ──
            # Look for strong bullish expansion candle at i: large green candle
            is_bull_impulse = (closes[i] - opens[i]) >= (atr_i * 0.7) and closes[i] > highs[i - 1]
            if is_bull_impulse:
                # Check Imbalance / FVG Validation Rule:
                # Candle i+1 low > Candle i-1 high (3-candle gap)
                fvg_gap = lows[i + 1] - highs[i - 1] if (i + 1 < n) else 0.0
                has_imbalance = fvg_gap >= min_gap

                if has_imbalance:
                    # The base candle is candle i-1 (the indecision / last bearish candle before impulse)
                    base_idx = i - 1
                    base_top = float(highs[base_idx])
                    base_bottom = float(lows[base_idx])

                    # Classify: Extreme vs Decisional
                    # If this base is near the local swing low -> Extreme Demand
                    local_low_slice = lows[max(0, base_idx - k) : base_idx + 1]
                    is_extreme = base_bottom <= float(local_low_slice.min()) + (atr_i * 0.3)
                    z_type = SDZoneType.EXTREME_DEMAND if is_extreme else SDZoneType.DECISIONAL_DEMAND

                    vol = float(volumes[base_idx])
                    avg_v = float(vol_smas[base_idx]) if vol_smas[base_idx] > 0 else 1.0
                    ratio = vol / avg_v

                    zone = SupplyDemandZone(
                        zone_type=z_type,
                        top=round(base_top, 2),
                        bottom=round(base_bottom, 2),
                        origin_bar_idx=base_idx,
                        breakout_bar_idx=i,
                        origin_date=dates[base_idx],
                        volume=vol,
                        volume_ratio=round(ratio, 2),
                        imbalance_size=round(float(fvg_gap), 2),
                        has_valid_imbalance=True,
                        is_mitigated=False,
                        touch_count=0,
                        is_valid=True,
                        is_discount=base_top < equilibrium,
                        is_premium=base_bottom > equilibrium,
                    )

                    # Evaluate mitigation & touches from subsequent bars
                    for m in range(i + 2, n):
                        if lows[m] < zone.bottom:
                            # Price traded completely below demand -> Invalidated
                            zone.is_valid = False
                            break
                        elif lows[m] <= zone.top:
                            zone.is_mitigated = True
                            zone.touch_count += 1
                            if zone.mitigated_date is None:
                                zone.mitigated_date = dates[m]

                    zones.append(zone)

            # ── 2. CANDIDATE SUPPLY ZONES (Bearish Impulse) ──
            # Look for strong bearish expansion candle at i: large red candle
            is_bear_impulse = (opens[i] - closes[i]) >= (atr_i * 0.7) and closes[i] < lows[i - 1]
            if is_bear_impulse:
                # Check Imbalance / FVG Validation Rule:
                # Candle i-1 low > Candle i+1 high
                fvg_gap = lows[i - 1] - highs[i + 1] if (i + 1 < n) else 0.0
                has_imbalance = fvg_gap >= min_gap

                if has_imbalance:
                    base_idx = i - 1
                    base_top = float(highs[base_idx])
                    base_bottom = float(lows[base_idx])

                    # Classify: Extreme vs Decisional
                    local_high_slice = highs[max(0, base_idx - k) : base_idx + 1]
                    is_extreme = base_top >= float(local_high_slice.max()) - (atr_i * 0.3)
                    z_type = SDZoneType.EXTREME_SUPPLY if is_extreme else SDZoneType.DECISIONAL_SUPPLY

                    vol = float(volumes[base_idx])
                    avg_v = float(vol_smas[base_idx]) if vol_smas[base_idx] > 0 else 1.0
                    ratio = vol / avg_v

                    zone = SupplyDemandZone(
                        zone_type=z_type,
                        top=round(base_top, 2),
                        bottom=round(base_bottom, 2),
                        origin_bar_idx=base_idx,
                        breakout_bar_idx=i,
                        origin_date=dates[base_idx],
                        volume=vol,
                        volume_ratio=round(ratio, 2),
                        imbalance_size=round(float(fvg_gap), 2),
                        has_valid_imbalance=True,
                        is_mitigated=False,
                        touch_count=0,
                        is_valid=True,
                        is_discount=base_top < equilibrium,
                        is_premium=base_bottom > equilibrium,
                    )

                    for m in range(i + 2, n):
                        if highs[m] > zone.top:
                            # Price traded completely above supply -> Invalidated
                            zone.is_valid = False
                            break
                        elif highs[m] >= zone.bottom:
                            zone.is_mitigated = True
                            zone.touch_count += 1
                            if zone.mitigated_date is None:
                                zone.mitigated_date = dates[m]

                    zones.append(zone)

        # Filter: retain valid active zones and sort by recency
        valid_zones = [z for z in zones if z.is_valid]
        return valid_zones

    def find_opposing_target_zone(
        self,
        current_price: float,
        is_bullish: bool,
        active_zones: List[SupplyDemandZone],
    ) -> Optional[SupplyDemandZone]:
        """
        Identifies the nearest unmitigated opposing zone for Range-to-Range target management.
        If Bullish (Long): Finds nearest unmitigated Supply Zone above current price.
        If Bearish (Short): Finds nearest unmitigated Demand Zone below current price.
        """
        if is_bullish:
            supply_targets = [
                z for z in active_zones
                if z.is_supply and z.bottom > current_price and z.touch_count <= 1
            ]
            if supply_targets:
                # Return the closest supply zone above current price
                return min(supply_targets, key=lambda z: z.bottom)
        else:
            demand_targets = [
                z for z in active_zones
                if z.is_demand and z.top < current_price and z.touch_count <= 1
            ]
            if demand_targets:
                # Return the closest demand zone below current price
                return max(demand_targets, key=lambda z: z.top)

        return None
