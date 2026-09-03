"""
Multi-Touch Support/Resistance Breakout & Breakdown Proximity Screener.

Identifies F&O stocks on the daily timeframe that have tested a critical horizontal
level multiple times (>= 3 touches) and are currently compressing right at or near
the verge of a breakdown or breakout.

Pattern Anatomy:
1. Multi-Touch Horizontal Floor/Ceiling: Price repeatedly bounces from/rejects at
   a tight price band (+- 1.0% to 1.2%), absorbing liquidity.
2. Compression / Price Squeeze: Range contracts (5d ATR < 20d ATR, BBW squeeze),
   forming descending highs into flat support (breakdown) or ascending lows into
   flat resistance (breakout).
3. Proximity: Current market price is within 0.0% to 2.5% of the trigger level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trade_system.domains.market_data.infrastructure.data.fo_universe import (
    get_fo_universe,
    get_sector_mapping,
)
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass
class ProximitySetup:
    """Represents a stock detected near a multi-touch breakout or breakdown level."""
    symbol: str
    clean_symbol: str
    sector: str
    setup_type: str            # "NEAR_BREAKDOWN", "AT_BREAKDOWN", "JUST_BROKEN_DOWN", "NEAR_BREAKOUT", "AT_BREAKOUT", "JUST_BROKEN_OUT"
    direction: str             # "BEARISH" or "BULLISH"
    current_price: float
    key_level: float           # Support floor or Resistance ceiling
    distance_pct: float        # % distance from current price to the level
    distance_pts: float        # Absolute point distance
    touch_count: int           # Number of tests/touches of this level
    lookback_days: int
    compression_score: float   # 0 to 100
    atr_ratio: float           # 5-day ATR / 20-day ATR (<1.0 = volatility compression)
    bollinger_bandwidth_pct: float
    swing_dates: List[str] = field(default_factory=list)
    recent_range_high: float = 0.0
    recent_range_low: float = 0.0
    analysis_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "clean_symbol": self.clean_symbol,
            "sector": self.sector,
            "setup_type": self.setup_type,
            "direction": self.direction,
            "current_price": round(self.current_price, 2),
            "key_level": round(self.key_level, 2),
            "distance_pct": round(self.distance_pct, 2),
            "distance_pts": round(self.distance_pts, 2),
            "touch_count": self.touch_count,
            "lookback_days": self.lookback_days,
            "compression_score": round(self.compression_score, 1),
            "atr_ratio": round(self.atr_ratio, 2),
            "bollinger_bandwidth_pct": round(self.bollinger_bandwidth_pct, 2),
            "swing_dates": self.swing_dates,
            "analysis_summary": self.analysis_summary,
        }


class BreakoutBreakdownProximityScreener:
    """
    Multi-Touch Support/Resistance Proximity Screener on Daily Timeframe.
    """

    def __init__(
        self,
        lookback_days: int = 30,
        cluster_tolerance_pct: float = 0.012,
        min_touches: int = 3,
        max_proximity_pct: float = 2.5,
    ) -> None:
        self.lookback_days = lookback_days
        self.cluster_tolerance_pct = cluster_tolerance_pct
        self.min_touches = min_touches
        self.max_proximity_pct = max_proximity_pct
        self.sector_map = get_sector_mapping()

    def analyze_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
    ) -> List[ProximitySetup]:
        """
        Analyze a daily OHLCV DataFrame for multi-touch support/resistance proximity.
        """
        if df.empty or len(df) < 20:
            return []

        lookback = lookback or self.lookback_days
        data = df.copy()

        if "timestamp" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed", errors="coerce")
            data = data.dropna(subset=["timestamp"])
            data = data.set_index("timestamp")

        data = data.sort_index()
        sub = data.iloc[-lookback:].copy()
        if len(sub) < 15:
            return []

        current_close = float(sub["close"].iloc[-1])
        current_low = float(sub["low"].iloc[-1])
        current_high = float(sub["high"].iloc[-1])
        clean_sym = symbol.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-INDEX", "")
        sector = self.sector_map.get(symbol, "GENERAL")

        # Technical Indicators: ATR & Bollinger Bands
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs()
        ], axis=1).max(axis=1)

        atr_5 = float(tr.rolling(5, min_periods=3).mean().iloc[-1])
        atr_20 = float(tr.rolling(20, min_periods=10).mean().iloc[-1])
        atr_ratio = round(atr_5 / atr_20, 2) if atr_20 > 0 else 1.0

        sma_20 = data["close"].rolling(20, min_periods=10).mean().iloc[-1]
        std_20 = data["close"].rolling(20, min_periods=10).std().iloc[-1]
        bb_width_pct = round(((2 * 2.0 * std_20) / sma_20) * 100.0, 2) if sma_20 > 0 else 0.0

        lows = sub["low"].values
        highs = sub["high"].values
        dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in sub.index]

        setups: List[ProximitySetup] = []

        # -------------------------------------------------------------
        # 1. Multi-Touch Support Floor (Breakdown Candidates)
        # -------------------------------------------------------------
        support_clusters = self._cluster_levels(
            levels=lows,
            dates=dates,
            tolerance_pct=self.cluster_tolerance_pct,
            min_touches=self.min_touches,
            is_support=True,
        )

        for cluster in support_clusters:
            floor_level = cluster["floor_level"]
            avg_level = cluster["avg_level"]
            touches = cluster["touches"]
            touch_dates = cluster["dates"]

            # Distance from current close to floor level
            dist_pct = ((current_close - floor_level) / floor_level) * 100.0
            dist_pts = current_close - floor_level

            # Check if stock is near or at breakdown level
            if -3.0 <= dist_pct <= self.max_proximity_pct:
                # Classify status
                if dist_pct < -0.5:
                    setup_type = "JUST_BROKEN_DOWN"
                elif -0.5 <= dist_pct <= 0.5:
                    setup_type = "AT_BREAKDOWN_POINT"
                else:
                    setup_type = "NEAR_BREAKDOWN"

                # Check for Descending Triangle (Lower Highs compressing into support)
                recent_highs = highs[-10:] if len(highs) >= 10 else highs
                is_lower_highs = bool(recent_highs[-1] < np.max(recent_highs[:5])) if len(recent_highs) >= 6 else False

                # Compression Score calculation (0 to 100)
                comp_score = 50.0
                comp_score += min(25.0, (touches - self.min_touches + 1) * 6.0) # More touches = higher score
                if atr_ratio < 0.85:
                    comp_score += 15.0 # ATR compression
                elif atr_ratio < 1.0:
                    comp_score += 8.0
                if bb_width_pct < 6.0:
                    comp_score += 10.0 # Bollinger squeeze
                if is_lower_highs:
                    comp_score += 10.0 # Descending structure
                comp_score = min(100.0, comp_score)

                summary = (
                    f"Tested support floor ₹{floor_level:.2f} {touches} times in {len(sub)} sessions; "
                    f"currently {dist_pct:+.2f}% ({dist_pts:+.2f} pts) from breakdown level."
                )

                setups.append(
                    ProximitySetup(
                        symbol=symbol,
                        clean_symbol=clean_sym,
                        sector=sector,
                        setup_type=setup_type,
                        direction="BEARISH",
                        current_price=current_close,
                        key_level=floor_level,
                        distance_pct=dist_pct,
                        distance_pts=dist_pts,
                        touch_count=touches,
                        lookback_days=len(sub),
                        compression_score=comp_score,
                        atr_ratio=atr_ratio,
                        bollinger_bandwidth_pct=bb_width_pct,
                        swing_dates=touch_dates,
                        recent_range_high=float(np.max(highs)),
                        recent_range_low=float(np.min(lows)),
                        analysis_summary=summary,
                    )
                )

        # -------------------------------------------------------------
        # 2. Multi-Touch Resistance Ceiling (Breakout Candidates)
        # -------------------------------------------------------------
        resistance_clusters = self._cluster_levels(
            levels=highs,
            dates=dates,
            tolerance_pct=self.cluster_tolerance_pct,
            min_touches=self.min_touches,
            is_support=False,
        )

        for cluster in resistance_clusters:
            ceiling_level = cluster["ceiling_level"]
            avg_level = cluster["avg_level"]
            touches = cluster["touches"]
            touch_dates = cluster["dates"]

            # Distance from current close to ceiling level (negative means below ceiling)
            dist_pct = ((current_close - ceiling_level) / ceiling_level) * 100.0
            dist_pts = ceiling_level - current_close

            # Check if stock is near or at breakout level
            if -self.max_proximity_pct <= dist_pct <= 3.0:
                if dist_pct > 0.5:
                    setup_type = "JUST_BROKEN_OUT"
                elif -0.5 <= dist_pct <= 0.5:
                    setup_type = "AT_BREAKOUT_POINT"
                else:
                    setup_type = "NEAR_BREAKOUT"

                # Check for Ascending Triangle (Higher Lows compressing into resistance)
                recent_lows = lows[-10:] if len(lows) >= 10 else lows
                is_higher_lows = bool(recent_lows[-1] > np.min(recent_lows[:5])) if len(recent_lows) >= 6 else False

                comp_score = 50.0
                comp_score += min(25.0, (touches - self.min_touches + 1) * 6.0)
                if atr_ratio < 0.85:
                    comp_score += 15.0
                elif atr_ratio < 1.0:
                    comp_score += 8.0
                if bb_width_pct < 6.0:
                    comp_score += 10.0
                if is_higher_lows:
                    comp_score += 10.0
                comp_score = min(100.0, comp_score)

                summary = (
                    f"Tested resistance ceiling ₹{ceiling_level:.2f} {touches} times in {len(sub)} sessions; "
                    f"currently {dist_pct:+.2f}% ({dist_pts:.2f} pts) from breakout level."
                )

                setups.append(
                    ProximitySetup(
                        symbol=symbol,
                        clean_symbol=clean_sym,
                        sector=sector,
                        setup_type=setup_type,
                        direction="BULLISH",
                        current_price=current_close,
                        key_level=ceiling_level,
                        distance_pct=dist_pct,
                        distance_pts=dist_pts,
                        touch_count=touches,
                        lookback_days=len(sub),
                        compression_score=comp_score,
                        atr_ratio=atr_ratio,
                        bollinger_bandwidth_pct=bb_width_pct,
                        swing_dates=touch_dates,
                        recent_range_high=float(np.max(highs)),
                        recent_range_low=float(np.min(lows)),
                        analysis_summary=summary,
                    )
                )

        return setups

    def _cluster_levels(
        self,
        levels: np.ndarray,
        dates: List[str],
        tolerance_pct: float,
        min_touches: int,
        is_support: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Group price points into distinct horizontal support or resistance clusters.
        """
        if len(levels) == 0:
            return []

        clusters: List[Dict[str, Any]] = []
        used_indices = set()

        # Sort indices by extreme (lowest for support, highest for resistance)
        sorted_indices = np.argsort(levels) if is_support else np.argsort(levels)[::-1]

        for pivot_idx in sorted_indices:
            if pivot_idx in used_indices:
                continue

            ref_val = levels[pivot_idx]
            match_indices = []

            for i, val in enumerate(levels):
                if abs(val - ref_val) / ref_val <= tolerance_pct:
                    match_indices.append(i)

            if len(match_indices) >= min_touches:
                matched_levels = [levels[i] for i in match_indices]
                matched_dates = [dates[i] for i in match_indices]

                # Key boundary level
                boundary_level = float(np.min(matched_levels)) if is_support else float(np.max(matched_levels))
                avg_level = float(np.mean(matched_levels))

                clusters.append({
                    "floor_level" if is_support else "ceiling_level": boundary_level,
                    "avg_level": avg_level,
                    "touches": len(match_indices),
                    "dates": matched_dates,
                })

                for idx in match_indices:
                    used_indices.add(idx)

        # Sort clusters by touch count descending
        clusters.sort(key=lambda x: x["touches"], reverse=True)
        return clusters

    def scan_universe(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        max_distance_pct: Optional[float] = None,
        min_touches: Optional[int] = None,
    ) -> Dict[str, List[ProximitySetup]]:
        """
        Scan a dictionary of symbol -> daily DataFrame across the universe.
        Returns:
            {
                "breakdown_setups": [...],
                "breakout_setups": [...],
            }
        """
        if max_distance_pct is not None:
            self.max_proximity_pct = max_distance_pct
        if min_touches is not None:
            self.min_touches = min_touches

        breakdowns: List[ProximitySetup] = []
        breakouts: List[ProximitySetup] = []

        for symbol, df in symbols_data.items():
            try:
                setups = self.analyze_symbol(symbol, df)
                for s in setups:
                    if s.direction == "BEARISH":
                        breakdowns.append(s)
                    else:
                        breakouts.append(s)
            except Exception as exc:
                LOGGER.error("Error analyzing %s for proximity: %s", symbol, exc)

        # Sort breakdowns by closest distance (ascending absolute distance) and compression score
        breakdowns.sort(key=lambda x: (abs(x.distance_pct), -x.compression_score))
        breakouts.sort(key=lambda x: (abs(x.distance_pct), -x.compression_score))

        return {
            "breakdown_setups": breakdowns,
            "breakout_setups": breakouts,
        }
