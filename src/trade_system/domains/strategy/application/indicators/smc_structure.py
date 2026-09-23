"""
Institutional SMC Market Structure Engine (Photon Trading Framework)
=====================================================================
Direct implementation of mechanical Smart Money Concepts (SMC) market structure:
1. Multi-Resolution Pivots:
   - Swing Structure (Macro institutional trend)
   - Internal Structure (Sub-structure / pullback leg)
2. Strong (Protected) vs Weak (Target) Points:
   - Strong Low: The low that successfully broke a prior swing high (Swing BOS).
     Must be protected; institutional invalidation / stop-loss level.
   - Weak High: High that failed to break a strong low. Targeted for Buy-Side Liquidity (BSL) take-profit.
   - Strong High: The high that successfully broke a prior swing low (Swing BOS down).
     Protected level for short trades.
   - Weak Low: Low that failed to break a strong high. Targeted for Sell-Side Liquidity (SSL) take-profit.
3. Break of Structure (BOS) vs Change of Character (CHoCH):
   - Swing BOS: Trend continuation (signals impulse completion -> pullback expected).
   - Swing CHoCH: Break of a Strong point -> macro trend reversal.
   - Internal CHoCH: First break against the internal pullback leg.
     Signals that the pullback has ended and the swing impulse has resumed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class StructureTrend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class PointType(str, Enum):
    STRONG_LOW = "STRONG_LOW"       # Protected Swing Low
    WEAK_HIGH = "WEAK_HIGH"         # Target Swing High
    STRONG_HIGH = "STRONG_HIGH"     # Protected Swing High
    WEAK_LOW = "WEAK_LOW"           # Target Swing Low
    INTERNAL_HIGH = "INTERNAL_HIGH" # Minor pullback high
    INTERNAL_LOW = "INTERNAL_LOW"   # Minor pullback low


class BreakType(str, Enum):
    SWING_BOS = "SWING_BOS"                 # Macro continuation
    SWING_CHOCH = "SWING_CHOCH"             # Macro trend flip
    INTERNAL_BOS = "INTERNAL_BOS"           # Minor continuation
    INTERNAL_CHOCH_BULL = "INTERNAL_CHOCH_BULL" # Pullback ended, long realignment
    INTERNAL_CHOCH_BEAR = "INTERNAL_CHOCH_BEAR" # Pullback ended, short realignment


@dataclass
class SMCStructurePoint:
    """Represents a validated structural pivot point."""
    bar_idx: int
    timestamp: str
    price: float
    point_type: PointType
    is_strong: bool
    is_swing: bool  # True = Swing structure, False = Internal structure


@dataclass
class SMCStructureBreak:
    """Represents a structural break event (BOS or CHoCH)."""
    bar_idx: int
    timestamp: str
    break_type: BreakType
    price: float
    broken_level: float
    broken_bar_idx: int
    description: str


@dataclass
class SMCStructureState:
    """Complete institutional market structure state at the most recent bar."""
    swing_trend: StructureTrend
    internal_trend: StructureTrend
    strong_protected_level: Optional[float]
    weak_target_level: Optional[float]
    strong_point: Optional[SMCStructurePoint]
    weak_point: Optional[SMCStructurePoint]
    is_pullback: bool
    is_internal_realigned: bool  # True when internal CHoCH fired in direction of swing trend
    realignment_event: Optional[SMCStructureBreak]
    recent_breaks: List[SMCStructureBreak] = field(default_factory=list)
    swing_points: List[SMCStructurePoint] = field(default_factory=list)
    internal_points: List[SMCStructurePoint] = field(default_factory=list)

    @property
    def alignment_description(self) -> str:
        if self.is_internal_realigned:
            return f"REALIGNED_{self.swing_trend.value} (Internal CHoCH Trigger)"
        if self.is_pullback:
            return f"PULLBACK_{self.swing_trend.value} (Counter Internal Structure)"
        return f"ALIGNED_{self.swing_trend.value}"


class SMCStructureEngine:
    """
    Mechanical SMC Market Structure Engine.
    
    Implements Photon Trading's mechanical rules for differentiating
    Swing Structure from Internal Structure, labeling Strong vs Weak
    pivots, and identifying high-probability Internal CHoCH realignments.
    """

    def __init__(
        self,
        swing_length: int = 5,
        internal_length: int = 2,
    ) -> None:
        self.swing_length = swing_length
        self.internal_length = internal_length

    def _find_pivots(
        self, highs: np.ndarray, lows: np.ndarray, length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pivot highs and lows with symmetric lookback/lookforward."""
        n = len(highs)
        pivot_highs = np.zeros(n)
        pivot_lows = np.zeros(n)

        for i in range(length, n - length):
            window_h = highs[i - length : i + length + 1]
            if highs[i] == window_h.max() and highs[i] > highs[i - 1]:
                pivot_highs[i] = highs[i]

            window_l = lows[i - length : i + length + 1]
            if lows[i] == window_l.min() and lows[i] < lows[i - 1]:
                pivot_lows[i] = lows[i]

        return pivot_highs, pivot_lows

    def analyze(self, df: pd.DataFrame) -> SMCStructureState:
        """
        Processes historical OHLCV data to classify structure points,
        detect BOS/CHoCH, and determine the current market structure state.
        """
        n = len(df)
        if n < (self.swing_length * 2 + 5):
            return SMCStructureState(
                swing_trend=StructureTrend.SIDEWAYS,
                internal_trend=StructureTrend.SIDEWAYS,
                strong_protected_level=None,
                weak_target_level=None,
                strong_point=None,
                weak_point=None,
                is_pullback=False,
                is_internal_realigned=False,
                realignment_event=None,
            )

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        dates = df["timestamp"].astype(str).values

        # 1. Detect Swing & Internal Pivots
        swing_high_flags, swing_low_flags = self._find_pivots(highs, lows, self.swing_length)
        internal_high_flags, internal_low_flags = self._find_pivots(highs, lows, self.internal_length)

        # 2. Sequential Structural Evaluation
        swing_trend = StructureTrend.SIDEWAYS
        internal_trend = StructureTrend.SIDEWAYS

        last_swing_high: Optional[Tuple[int, float]] = None
        last_swing_low: Optional[Tuple[int, float]] = None
        strong_low: Optional[SMCStructurePoint] = None
        weak_high: Optional[SMCStructurePoint] = None
        strong_high: Optional[SMCStructurePoint] = None
        weak_low: Optional[SMCStructurePoint] = None

        last_internal_high: Optional[Tuple[int, float]] = None
        last_internal_low: Optional[Tuple[int, float]] = None

        all_breaks: List[SMCStructureBreak] = []
        swing_points: List[SMCStructurePoint] = []
        internal_points: List[SMCStructurePoint] = []

        is_internal_realigned = False
        realignment_event: Optional[SMCStructureBreak] = None

        for i in range(n):
            conf_swing = i - self.swing_length
            if conf_swing >= 0:
                if swing_high_flags[conf_swing] > 0:
                    last_swing_high = (conf_swing, swing_high_flags[conf_swing])
                if swing_low_flags[conf_swing] > 0:
                    last_swing_low = (conf_swing, swing_low_flags[conf_swing])

            conf_internal = i - self.internal_length
            if conf_internal >= 0:
                if internal_high_flags[conf_internal] > 0:
                    last_internal_high = (conf_internal, internal_high_flags[conf_internal])
                    internal_points.append(
                        SMCStructurePoint(
                            bar_idx=conf_internal,
                            timestamp=dates[conf_internal],
                            price=internal_high_flags[conf_internal],
                            point_type=PointType.INTERNAL_HIGH,
                            is_strong=False,
                            is_swing=False,
                        )
                    )
                if internal_low_flags[conf_internal] > 0:
                    last_internal_low = (conf_internal, internal_low_flags[conf_internal])
                    internal_points.append(
                        SMCStructurePoint(
                            bar_idx=conf_internal,
                            timestamp=dates[conf_internal],
                            price=internal_low_flags[conf_internal],
                            point_type=PointType.INTERNAL_LOW,
                            is_strong=False,
                            is_swing=False,
                        )
                    )

            # ── SWING STRUCTURE CHECKS ──
            # Bullish Swing BOS / CHoCH: Close breaks above last swing high
            if last_swing_high is not None and closes[i] > last_swing_high[1]:
                h_idx, h_price = last_swing_high
                if swing_trend == StructureTrend.BEARISH:
                    # Bearish to Bullish flip = Swing CHoCH
                    swing_trend = StructureTrend.BULLISH
                    b_type = BreakType.SWING_CHOCH
                    desc = f"Bullish Swing CHoCH: Broke Strong High ₹{h_price:.2f}"
                else:
                    swing_trend = StructureTrend.BULLISH
                    b_type = BreakType.SWING_BOS
                    desc = f"Bullish Swing BOS: Broke Swing High ₹{h_price:.2f}"

                # The lowest low between the previous high and this break is the STRONG LOW
                search_start = max(0, h_idx - self.swing_length)
                low_slice = lows[search_start : i + 1]
                min_idx = search_start + int(np.argmin(low_slice))
                strong_low = SMCStructurePoint(
                    bar_idx=min_idx,
                    timestamp=dates[min_idx],
                    price=float(lows[min_idx]),
                    point_type=PointType.STRONG_LOW,
                    is_strong=True,
                    is_swing=True,
                )
                swing_points.append(strong_low)

                # Prior high is now a weak high that was harvested
                weak_high = SMCStructurePoint(
                    bar_idx=h_idx,
                    timestamp=dates[h_idx],
                    price=h_price,
                    point_type=PointType.WEAK_HIGH,
                    is_strong=False,
                    is_swing=True,
                )
                swing_points.append(weak_high)

                all_breaks.append(
                    SMCStructureBreak(
                        bar_idx=i,
                        timestamp=dates[i],
                        break_type=b_type,
                        price=float(closes[i]),
                        broken_level=h_price,
                        broken_bar_idx=h_idx,
                        description=desc,
                    )
                )
                last_swing_high = None  # Consumed

            # Bearish Swing BOS / CHoCH: Close breaks below last swing low
            elif last_swing_low is not None and closes[i] < last_swing_low[1]:
                l_idx, l_price = last_swing_low
                if swing_trend == StructureTrend.BULLISH:
                    # Bullish to Bearish flip = Swing CHoCH
                    swing_trend = StructureTrend.BEARISH
                    b_type = BreakType.SWING_CHOCH
                    desc = f"Bearish Swing CHoCH: Broke Strong Low ₹{l_price:.2f}"
                else:
                    swing_trend = StructureTrend.BEARISH
                    b_type = BreakType.SWING_BOS
                    desc = f"Bearish Swing BOS: Broke Swing Low ₹{l_price:.2f}"

                # The highest high between the previous low and this break is the STRONG HIGH
                search_start = max(0, l_idx - self.swing_length)
                high_slice = highs[search_start : i + 1]
                max_idx = search_start + int(np.argmax(high_slice))
                strong_high = SMCStructurePoint(
                    bar_idx=max_idx,
                    timestamp=dates[max_idx],
                    price=float(highs[max_idx]),
                    point_type=PointType.STRONG_HIGH,
                    is_strong=True,
                    is_swing=True,
                )
                swing_points.append(strong_high)

                weak_low = SMCStructurePoint(
                    bar_idx=l_idx,
                    timestamp=dates[l_idx],
                    price=l_price,
                    point_type=PointType.WEAK_LOW,
                    is_strong=False,
                    is_swing=True,
                )
                swing_points.append(weak_low)

                all_breaks.append(
                    SMCStructureBreak(
                        bar_idx=i,
                        timestamp=dates[i],
                        break_type=b_type,
                        price=float(closes[i]),
                        broken_level=l_price,
                        broken_bar_idx=l_idx,
                        description=desc,
                    )
                )
                last_swing_low = None  # Consumed

            # ── INTERNAL STRUCTURE & REALIGNMENT CHECKS ──
            # In a Bullish swing trend, if price is in a pullback, internal structure is bearish.
            # When price breaks above the last internal lower high -> BULLISH INTERNAL CHOCH!
            if swing_trend == StructureTrend.BULLISH:
                if last_internal_high is not None and closes[i] > last_internal_high[1]:
                    ih_idx, ih_price = last_internal_high
                    internal_trend = StructureTrend.BULLISH
                    ev = SMCStructureBreak(
                        bar_idx=i,
                        timestamp=dates[i],
                        break_type=BreakType.INTERNAL_CHOCH_BULL,
                        price=float(closes[i]),
                        broken_level=ih_price,
                        broken_bar_idx=ih_idx,
                        description=f"Bullish Internal CHoCH: Pullback ended, broke minor high ₹{ih_price:.2f}",
                    )
                    all_breaks.append(ev)
                    # If this occurred recently (last 3 bars), flag realignment!
                    if i >= n - 3:
                        is_internal_realigned = True
                        realignment_event = ev
                    last_internal_high = None

                elif last_internal_low is not None and closes[i] < last_internal_low[1]:
                    il_idx, il_price = last_internal_low
                    internal_trend = StructureTrend.BEARISH
                    last_internal_low = None

            # In a Bearish swing trend, if price is in a pullback, internal structure is bullish.
            # When price breaks below the last internal higher low -> BEARISH INTERNAL CHOCH!
            elif swing_trend == StructureTrend.BEARISH:
                if last_internal_low is not None and closes[i] < last_internal_low[1]:
                    il_idx, il_price = last_internal_low
                    internal_trend = StructureTrend.BEARISH
                    ev = SMCStructureBreak(
                        bar_idx=i,
                        timestamp=dates[i],
                        break_type=BreakType.INTERNAL_CHOCH_BEAR,
                        price=float(closes[i]),
                        broken_level=il_price,
                        broken_bar_idx=il_idx,
                        description=f"Bearish Internal CHoCH: Pullback ended, broke minor low ₹{il_price:.2f}",
                    )
                    all_breaks.append(ev)
                    if i >= n - 3:
                        is_internal_realigned = True
                        realignment_event = ev
                    last_internal_low = None

                elif last_internal_high is not None and closes[i] > last_internal_high[1]:
                    ih_idx, ih_price = last_internal_high
                    internal_trend = StructureTrend.BULLISH
                    last_internal_high = None

        # Determine pullback status
        is_pullback = False
        if swing_trend == StructureTrend.BULLISH and internal_trend == StructureTrend.BEARISH:
            is_pullback = True
        elif swing_trend == StructureTrend.BEARISH and internal_trend == StructureTrend.BULLISH:
            is_pullback = True

        # Determine active strong level and weak target level
        strong_lvl = None
        weak_lvl = None
        active_strong_point = None
        active_weak_point = None

        if swing_trend == StructureTrend.BULLISH:
            if strong_low is not None:
                strong_lvl = strong_low.price
                active_strong_point = strong_low
            # Weak target is the highest swing high made in the current move
            current_max = float(highs[max(0, n - 20) : n].max())
            weak_lvl = current_max
            active_weak_point = SMCStructurePoint(
                bar_idx=n - 1,
                timestamp=dates[-1],
                price=weak_lvl,
                point_type=PointType.WEAK_HIGH,
                is_strong=False,
                is_swing=True,
            )
        elif swing_trend == StructureTrend.BEARISH:
            if strong_high is not None:
                strong_lvl = strong_high.price
                active_strong_point = strong_high
            current_min = float(lows[max(0, n - 20) : n].min())
            weak_lvl = current_min
            active_weak_point = SMCStructurePoint(
                bar_idx=n - 1,
                timestamp=dates[-1],
                price=weak_lvl,
                point_type=PointType.WEAK_LOW,
                is_strong=False,
                is_swing=True,
            )

        return SMCStructureState(
            swing_trend=swing_trend,
            internal_trend=internal_trend,
            strong_protected_level=strong_lvl,
            weak_target_level=weak_lvl,
            strong_point=active_strong_point,
            weak_point=active_weak_point,
            is_pullback=is_pullback,
            is_internal_realigned=is_internal_realigned,
            realignment_event=realignment_event,
            recent_breaks=all_breaks[-10:],
            swing_points=swing_points[-10:],
            internal_points=internal_points[-10:],
        )
