"""
AMDPhaseEngine — Institutional Accumulation, Manipulation & Distribution (Power of 3) Cycle Detector.

Decodes the 3 essential institutional market phases:
1. Accumulation (A): Tight range consolidation where smart money builds orders while volatility compresses.
2. Manipulation (M) - The "Judas Swing" / Spring / UTAD:
   - False break that sweeps liquidity (stop runs) beyond the accumulation boundary.
   - Characterized by quick wick rejection, volume absorption, and reclaim back inside the range.
3. Distribution (D):
   - The true directional expansion trend once trapped retail liquidity is captured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class AMDPhaseInfo:
    """Consolidated status of the Wyckoff / Power of 3 (AMD) phase."""
    phase: str                  # "ACCUMULATION", "MANIPULATION_SPRING", "MANIPULATION_UTAD", "DISTRIBUTION_BULLISH", "DISTRIBUTION_BEARISH", "CONSOLIDATION"
    range_high: float           # Accumulation upper boundary
    range_low: float            # Accumulation lower boundary
    range_mid: float            # 50% equilibrium level
    manipulation_level: float   # Extreme price reached by the sweep wick
    invalidation_stop: float    # Exact structural stop loss price
    target_1: float             # Primary target (opposite boundary of range)
    target_2: float             # Expansion target (Fibonacci 1.618 or liquidity wall)
    confidence: int             # 0 to 100
    description: str            # Institutional narrative
    is_actionable: bool         # True if currently in high-probability sniper trigger window
    recommended_action: str     # "BUY_SWEEP_RECLAIM", "SELL_UPTHRUST_REJECT", "STAND_ASIDE_ACCUMULATION", "RIDE_DISTRIBUTION"


class AMDPhaseEngine:
    """
    Detects Accumulation ranges, identifies Manipulation liquidity sweeps (Springs/UTADs),
    and tracks Distribution expansions for Indices and F&O Stocks.
    """

    def __init__(self, lookback_bars: int = 24, min_compression_bars: int = 8) -> None:
        self.lookback_bars = lookback_bars
        self.min_compression_bars = min_compression_bars

    def analyze(
        self,
        price_df: pd.DataFrame,
        spot_price: float,
        strike_step: float = 50.0,
        atm_vol_delta: int = 0,
        atr: Optional[float] = None,
    ) -> AMDPhaseInfo:
        """
        Execute AMD cycle analysis on candle series.
        """
        if price_df is None or price_df.empty or len(price_df) < self.min_compression_bars:
            return AMDPhaseInfo(
                phase="CONSOLIDATION",
                range_high=spot_price + strike_step,
                range_low=spot_price - strike_step,
                range_mid=spot_price,
                manipulation_level=spot_price,
                invalidation_stop=spot_price - strike_step,
                target_1=spot_price + strike_step,
                target_2=spot_price + (strike_step * 2),
                confidence=30,
                description="Insufficient historical bars to detect structural AMD accumulation range.",
                is_actionable=False,
                recommended_action="STAND_ASIDE_ACCUMULATION",
            )

        df = price_df.tail(self.lookback_bars).copy().reset_index(drop=True)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values

        # Estimate ATR if not provided
        if atr is None or atr <= 0.0:
            ranges = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1])) for i in range(1, len(df))]
            atr = float(np.mean(ranges[-7:])) if len(ranges) >= 7 else strike_step * 0.6
            atr = max(atr, strike_step * 0.4)

        # 1. Identify Accumulation Base (prior consolidation window excluding the latest 2-3 sweep bars)
        eval_len = len(df)
        base_window = df.iloc[:max(self.min_compression_bars, eval_len - 3)]
        range_high = float(base_window["high"].quantile(0.90))
        range_low = float(base_window["low"].quantile(0.10))
        range_height = range_high - range_low
        range_mid = (range_high + range_low) / 2.0

        latest_close = float(closes[-1])
        latest_low = float(lows[-1])
        latest_high = float(highs[-1])

        prev_low = float(lows[-2]) if eval_len >= 2 else latest_low
        prev_high = float(highs[-2]) if eval_len >= 2 else latest_high

        recent_min_low = min(latest_low, prev_low)
        recent_max_high = max(latest_high, prev_high)

        # 2. Check for Phase 2: Manipulation (Judas Swing / Spring / UTAD)
        # Condition for BULLISH SPRING:
        # - Recent low dipped below Accumulation Low (stop run by at least 0.08 * ATR)
        # - Current spot price has reclaimed back ABOVE Accumulation Low
        is_bullish_spring = (recent_min_low < (range_low - (0.08 * atr))) and (latest_close >= range_low)

        # Condition for BEARISH UTAD (Upthrust After Distribution):
        # - Recent high pushed above Accumulation High (liquidity sweep by at least 0.08 * ATR)
        # - Current spot price has closed back BELOW Accumulation High
        is_bearish_utad = (recent_max_high > (range_high + (0.08 * atr))) and (latest_close <= range_high)

        # 3. Check for Phase 3: Distribution (The True Trend Expansion)
        is_bullish_distribution = (latest_close > range_high + (0.3 * atr)) and (eval_len >= 4 and closes[-2] > range_high)
        is_bearish_distribution = (latest_close < range_low - (0.3 * atr)) and (eval_len >= 4 and closes[-2] < range_low)

        # ── Classification Logic ─────────────────────────────────────────────
        if is_bullish_spring:
            sweep_depth = range_low - recent_min_low
            stop_loss = round(recent_min_low - (0.15 * atr), 2)
            t1 = round(range_high, 2)
            t2 = round(range_high + (range_height * 0.618), 2)
            conf = 85 if atm_vol_delta > 0 else 75

            return AMDPhaseInfo(
                phase="MANIPULATION_SPRING",
                range_high=round(range_high, 2),
                range_low=round(range_low, 2),
                range_mid=round(range_mid, 2),
                manipulation_level=round(recent_min_low, 2),
                invalidation_stop=stop_loss,
                target_1=t1,
                target_2=t2,
                confidence=conf,
                description=(
                    f"⚡ BULLISH MANIPULATION (WYCKOFF SPRING)! Smart money swept retail stops below "
                    f"₹{range_low:,.1f} (Sweep Low: ₹{recent_min_low:,.1f}, -{sweep_depth:.1f} pts) "
                    f"and reclaimed the range. Trap sprung on retail shorts — prime long entry window."
                ),
                is_actionable=True,
                recommended_action="BUY_SWEEP_RECLAIM",
            )

        elif is_bearish_utad:
            sweep_height = recent_max_high - range_high
            stop_loss = round(recent_max_high + (0.15 * atr), 2)
            t1 = round(range_low, 2)
            t2 = round(range_low - (range_height * 0.618), 2)
            conf = 85 if atm_vol_delta < 0 else 75

            return AMDPhaseInfo(
                phase="MANIPULATION_UTAD",
                range_high=round(range_high, 2),
                range_low=round(range_low, 2),
                range_mid=round(range_mid, 2),
                manipulation_level=round(recent_max_high, 2),
                invalidation_stop=stop_loss,
                target_1=t1,
                target_2=t2,
                confidence=conf,
                description=(
                    f"⚡ BEARISH MANIPULATION (UPTHRUST / UTAD)! Smart money baited breakout buyers above "
                    f"₹{range_high:,.1f} (Sweep High: ₹{recent_max_high:,.1f}, +{sweep_height:.1f} pts) "
                    f"and rejected back into the range. Trap sprung on retail longs — prime short entry window."
                ),
                is_actionable=True,
                recommended_action="SELL_UPTHRUST_REJECT",
            )

        elif is_bullish_distribution:
            stop_loss = round(range_mid, 2)
            t1 = round(range_high + range_height, 2)
            t2 = round(range_high + (range_height * 1.618), 2)

            return AMDPhaseInfo(
                phase="DISTRIBUTION_BULLISH",
                range_high=round(range_high, 2),
                range_low=round(range_low, 2),
                range_mid=round(range_mid, 2),
                manipulation_level=round(recent_min_low, 2),
                invalidation_stop=stop_loss,
                target_1=t1,
                target_2=t2,
                confidence=80,
                description=(
                    f"🚀 BULLISH DISTRIBUTION EXPANSION! Price has cleared the accumulation ceiling at "
                    f"₹{range_high:,.1f}. Institutional markup phase in progress. Targeting ₹{t1:,.1f} - ₹{t2:,.1f}."
                ),
                is_actionable=False,
                recommended_action="RIDE_DISTRIBUTION",
            )

        elif is_bearish_distribution:
            stop_loss = round(range_mid, 2)
            t1 = round(range_low - range_height, 2)
            t2 = round(range_low - (range_height * 1.618), 2)

            return AMDPhaseInfo(
                phase="DISTRIBUTION_BEARISH",
                range_high=round(range_high, 2),
                range_low=round(range_low, 2),
                range_mid=round(range_mid, 2),
                manipulation_level=round(recent_max_high, 2),
                invalidation_stop=stop_loss,
                target_1=t1,
                target_2=t2,
                confidence=80,
                description=(
                    f"🔻 BEARISH DISTRIBUTION EXPANSION! Price broke down below the accumulation floor at "
                    f"₹{range_low:,.1f}. Institutional markdown phase in progress. Targeting ₹{t1:,.1f} - ₹{t2:,.1f}."
                ),
                is_actionable=False,
                recommended_action="RIDE_DISTRIBUTION",
            )

        else:
            # Phase 1: Accumulation (Compression)
            return AMDPhaseInfo(
                phase="ACCUMULATION",
                range_high=round(range_high, 2),
                range_low=round(range_low, 2),
                range_mid=round(range_mid, 2),
                manipulation_level=round(spot_price, 2),
                invalidation_stop=round(range_low - (0.2 * atr), 2),
                target_1=round(range_high, 2),
                target_2=round(range_high + range_height, 2),
                confidence=70,
                description=(
                    f"📦 PHASE 1: ACCUMULATION (RANGE COMPRESSION). Price is oscillating between "
                    f"₹{range_low:,.1f} and ₹{range_high:,.1f} (Height: {range_height:.1f} pts). "
                    f"Institutions are building inventory. VETO any breakout attempts until manipulation occurs."
                ),
                is_actionable=False,
                recommended_action="STAND_ASIDE_ACCUMULATION",
            )
