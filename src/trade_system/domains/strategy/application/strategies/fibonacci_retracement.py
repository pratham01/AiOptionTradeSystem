"""
Fibonacci Retracement & Golden Pocket Strategy Engine
=====================================================
Calculates standard institutional Fibonacci retracement and extension levels
for momentum stocks identified in Sector Scope and intraday/daily scans:

Key Levels:
- 0.0%: Swing High (Bullish impulse origin / Initial Target)
- 23.6%: Shallow pullback
- 38.2%: Momentum pullback
- 50.0%: Halfway Equilibrium
- 61.8%: The Golden Pocket (Primary Institutional Entry Zone: 50.0% - 61.8%)
- 78.6%: Deep Institutional Discount / Invalidation SL threshold
- 100.0%: Swing Low (Impulse start)
- -27.2% / 127.2%: Fibonacci Trend Extension Target 1
- -61.8% / 161.8%: Golden Ratio Extension Target 2

Provides actionable trade entry, invalidation stop-loss, and multi-tier profit targets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class FibonacciGrid:
    """Represents a calculated Fibonacci Retracement & Extension grid for a swing leg."""
    direction: str                # "CALL" (Long) or "PUT" (Short)
    swing_high: float             # Highest price in impulse leg
    swing_low: float              # Lowest price in impulse leg
    swing_high_idx: int
    swing_low_idx: int
    impulse_range: float          # swing_high - swing_low
    
    # Retracement Levels
    level_0: float                # 0.0% (Swing High for CALL, Swing Low for PUT)
    level_236: float              # 23.6%
    level_382: float              # 38.2%
    level_500: float              # 50.0% (Equilibrium)
    level_618: float              # 61.8% (Golden Ratio)
    level_786: float              # 78.6% (Deep Discount / SL Invalidation)
    level_1000: float             # 100.0% (Swing Low for CALL, Swing High for PUT)
    
    # Extension Targets
    ext_1272: float               # 127.2% Extension (-27.2% Target)
    ext_1618: float               # 161.8% Golden Extension (-61.8% Target)
    ext_2000: float               # 200.0% Extended Target

    # Zone Status
    current_zone: str             # "GOLDEN_POCKET", "PULLBACK_382", "DEEP_DISCOUNT", "EXPANSION", "INVALIDATED"
    is_in_golden_pocket: bool = False
    is_approaching: bool = False


@dataclass
class FibonacciSetup:
    """Actionable trade setup generated from Fibonacci Retracement analysis."""
    symbol: str
    action: str                   # "BUY CALL" or "BUY PUT"
    direction: str                # "CALL" or "PUT"
    setup_name: str               # "FIB_GOLDEN_POCKET_RETEST", "FIB_50_EQUILIBRIUM_BOUNCE", "FIB_DEEP_DISCOUNT_TAP"
    status: str                   # "TRIGGERED", "APPROACHING", "WAITING", "INVALIDATED"
    entry_price: float            # Recommended Entry price
    stop_loss: float              # Invalidation Stop-Loss level
    target_1: float               # Target 1 (0.0% Swing level)
    target_2: float               # Target 2 (127.2% Extension)
    target_3: float               # Target 3 (161.8% Golden Extension)
    risk_reward_ratio: float      # Calculated RRR for Target 1
    extended_rrr: float           # Calculated RRR for Target 2
    grid: FibonacciGrid
    reasons: List[str] = field(default_factory=list)
    spot_price: float = 0.0

    def format_summary(self) -> str:
        icon = "🟢" if self.direction == "CALL" else "🔴"
        sym_clean = self.symbol.replace("NSE:", "").replace("-EQ", "")
        return (
            f"{icon} <b>{sym_clean}</b> | <b>{self.action}</b> [{self.setup_name}] — {self.status}\n"
            f"   Golden Pocket: ₹{self.grid.level_618:.2f} - ₹{self.grid.level_500:.2f} | 78.6% SL: ₹{self.grid.level_786:.2f}\n"
            f"   Entry: ₹{self.entry_price:.2f} | SL: ₹{self.stop_loss:.2f} | T1: ₹{self.target_1:.2f} (1:{self.risk_reward_ratio:.1f} RRR) | T2: ₹{self.target_2:.2f} (1:{self.extended_rrr:.1f} RRR)\n"
            f"   Rationale: {', '.join(self.reasons)}"
        )


class FibonacciRetracementStrategy:
    """
    Identifies impulse swings and calculates Golden Pocket Fibonacci Retracements.
    """

    def __init__(self, swing_lookback: int = 30, atr_period: int = 14) -> None:
        self.swing_lookback = swing_lookback
        self.atr_period = atr_period

    def calculate_grid(self, df: pd.DataFrame, direction: str = "CALL") -> Optional[FibonacciGrid]:
        """
        Detects the relevant swing high and swing low from candle history and computes the Fib Grid.
        """
        if df.empty or len(df) < 5:
            return None

        recent_df = df.iloc[-self.swing_lookback:].copy().reset_index(drop=True)
        highs = recent_df["high"].values
        lows = recent_df["low"].values
        n = len(recent_df)

        if direction.upper() == "CALL":
            # For Bullish: find the lowest low followed by highest high
            low_idx = int(np.argmin(lows))
            if low_idx >= n - 2:  # Low is at the very end (downtrending)
                # Look back further
                low_idx = int(np.argmin(lows[:max(1, n - 3)]))

            sub_highs = highs[low_idx:]
            if len(sub_highs) == 0:
                high_idx = int(np.argmax(highs))
            else:
                high_idx = low_idx + int(np.argmax(sub_highs))

            swing_low = float(lows[low_idx])
            swing_high = float(highs[high_idx])

            if swing_high <= swing_low:
                return None

            diff = swing_high - swing_low
            
            # Retracement levels from High (0.0%) down to Low (100.0%)
            lvl_0 = swing_high
            lvl_236 = swing_high - (0.236 * diff)
            lvl_382 = swing_high - (0.382 * diff)
            lvl_500 = swing_high - (0.500 * diff)
            lvl_618 = swing_high - (0.618 * diff)
            lvl_786 = swing_high - (0.786 * diff)
            lvl_1000 = swing_low

            # Extensions above swing high
            ext_1272 = swing_high + (0.272 * diff)
            ext_1618 = swing_high + (0.618 * diff)
            ext_2000 = swing_high + (1.000 * diff)

            current_price = float(df["close"].iloc[-1])

            # Determine Zone
            if lvl_618 <= current_price <= lvl_500:
                current_zone = "GOLDEN_POCKET"
                in_pocket = True
                approaching = False
            elif lvl_500 < current_price <= lvl_382:
                current_zone = "PULLBACK_382"
                in_pocket = False
                approaching = True
            elif lvl_786 <= current_price < lvl_618:
                current_zone = "DEEP_DISCOUNT"
                in_pocket = False
                approaching = False
            elif current_price > lvl_382:
                current_zone = "EXPANSION"
                in_pocket = False
                approaching = False
            else:
                current_zone = "INVALIDATED"
                in_pocket = False
                approaching = False

            return FibonacciGrid(
                direction="CALL",
                swing_high=round(swing_high, 2),
                swing_low=round(swing_low, 2),
                swing_high_idx=high_idx,
                swing_low_idx=low_idx,
                impulse_range=round(diff, 2),
                level_0=round(lvl_0, 2),
                level_236=round(lvl_236, 2),
                level_382=round(lvl_382, 2),
                level_500=round(lvl_500, 2),
                level_618=round(lvl_618, 2),
                level_786=round(lvl_786, 2),
                level_1000=round(lvl_1000, 2),
                ext_1272=round(ext_1272, 2),
                ext_1618=round(ext_1618, 2),
                ext_2000=round(ext_2000, 2),
                current_zone=current_zone,
                is_in_golden_pocket=in_pocket,
                is_approaching=approaching
            )

        else:  # PUT (Bearish)
            high_idx = int(np.argmax(highs))
            if high_idx >= n - 2:
                high_idx = int(np.argmax(highs[:max(1, n - 3)]))

            sub_lows = lows[high_idx:]
            if len(sub_lows) == 0:
                low_idx = int(np.argmin(lows))
            else:
                low_idx = high_idx + int(np.argmin(sub_lows))

            swing_high = float(highs[high_idx])
            swing_low = float(lows[low_idx])

            if swing_high <= swing_low:
                return None

            diff = swing_high - swing_low

            # Retracement levels from Low (0.0%) up to High (100.0%)
            lvl_0 = swing_low
            lvl_236 = swing_low + (0.236 * diff)
            lvl_382 = swing_low + (0.382 * diff)
            lvl_500 = swing_low + (0.500 * diff)
            lvl_618 = swing_low + (0.618 * diff)
            lvl_786 = swing_low + (0.786 * diff)
            lvl_1000 = swing_high

            # Extensions below swing low
            ext_1272 = swing_low - (0.272 * diff)
            ext_1618 = swing_low - (0.618 * diff)
            ext_2000 = swing_low - (1.000 * diff)

            current_price = float(df["close"].iloc[-1])

            if lvl_500 <= current_price <= lvl_618:
                current_zone = "GOLDEN_POCKET"
                in_pocket = True
                approaching = False
            elif lvl_382 <= current_price < lvl_500:
                current_zone = "PULLBACK_382"
                in_pocket = False
                approaching = True
            elif lvl_618 < current_price <= lvl_786:
                current_zone = "DEEP_DISCOUNT"
                in_pocket = False
                approaching = False
            elif current_price < lvl_382:
                current_zone = "EXPANSION"
                in_pocket = False
                approaching = False
            else:
                current_zone = "INVALIDATED"
                in_pocket = False
                approaching = False

            return FibonacciGrid(
                direction="PUT",
                swing_high=round(swing_high, 2),
                swing_low=round(swing_low, 2),
                swing_high_idx=high_idx,
                swing_low_idx=low_idx,
                impulse_range=round(diff, 2),
                level_0=round(lvl_0, 2),
                level_236=round(lvl_236, 2),
                level_382=round(lvl_382, 2),
                level_500=round(lvl_500, 2),
                level_618=round(lvl_618, 2),
                level_786=round(lvl_786, 2),
                level_1000=round(lvl_1000, 2),
                ext_1272=round(ext_1272, 2),
                ext_1618=round(ext_1618, 2),
                ext_2000=round(ext_2000, 2),
                current_zone=current_zone,
                is_in_golden_pocket=in_pocket,
                is_approaching=approaching
            )

    def evaluate_setup(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        direction: str = "CALL",
        atr: float = 0.0,
    ) -> Optional[FibonacciSetup]:
        """
        Evaluates current price vs Fibonacci Grid and returns high-conviction Entry, SL, and Target levels.
        """
        grid = self.calculate_grid(df, direction=direction)
        if not grid:
            return None

        current_close = float(df["close"].iloc[-1])
        current_atr = atr if atr > 0 else (grid.impulse_range * 0.1)

        reasons = []

        if direction.upper() == "CALL":
            # Stop loss placed below 78.6% level with ATR buffer
            sl = grid.level_786 - (current_atr * 0.3)
            entry = current_close
            t1 = grid.level_0           # 0.0% Swing High
            t2 = grid.ext_1272          # 127.2% Extension
            t3 = grid.ext_1618          # 161.8% Golden Extension

            risk = max(entry - sl, current_atr * 0.5)
            rrr_t1 = (t1 - entry) / risk if risk > 0 else 1.5
            rrr_t2 = (t2 - entry) / risk if risk > 0 else 2.5

            if grid.is_in_golden_pocket:
                status = "TRIGGERED"
                setup_name = "FIB_GOLDEN_POCKET_RETEST"
                reasons.append(f"Price currently in Golden Pocket (50.0% - 61.8% Fib: ₹{grid.level_618:.2f} - ₹{grid.level_500:.2f})")
                reasons.append(f"Optimal risk-defined pullback from Swing High ₹{grid.swing_high:.2f}")
            elif grid.is_approaching:
                status = "APPROACHING"
                setup_name = "FIB_PULLBACK_APPROACHING"
                reasons.append(f"Pulling back towards 50.0% Equilibrium (₹{grid.level_500:.2f}) & 61.8% Golden Pocket")
            elif grid.current_zone == "DEEP_DISCOUNT":
                status = "TRIGGERED"
                setup_name = "FIB_DEEP_DISCOUNT_TAP"
                reasons.append(f"Testing Deep Discount 78.6% level at ₹{grid.level_786:.2f}")
            elif grid.current_zone == "EXPANSION":
                status = "WAITING"
                setup_name = "FIB_EXPANSION_WAIT_PULLBACK"
                reasons.append(f"Trading above 38.2% Fib (₹{grid.level_382:.2f}) — wait for pullback into Golden Pocket")
            else:
                status = "INVALIDATED"
                setup_name = "FIB_STRUCTURE_INVALIDATED"
                reasons.append(f"Broken below 78.6% invalidation level ₹{grid.level_786:.2f}")

            return FibonacciSetup(
                symbol=symbol,
                action="BUY CALL",
                direction="CALL",
                setup_name=setup_name,
                status=status,
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                target_3=round(t3, 2),
                risk_reward_ratio=round(max(0.1, rrr_t1), 2),
                extended_rrr=round(max(0.1, rrr_t2), 2),
                grid=grid,
                reasons=reasons,
                spot_price=current_close
            )

        else:  # PUT (Bearish)
            sl = grid.level_786 + (current_atr * 0.3)
            entry = current_close
            t1 = grid.level_0           # 0.0% Swing Low
            t2 = grid.ext_1272          # 127.2% Extension
            t3 = grid.ext_1618          # 161.8% Golden Extension

            risk = max(sl - entry, current_atr * 0.5)
            rrr_t1 = (entry - t1) / risk if risk > 0 else 1.5
            rrr_t2 = (entry - t2) / risk if risk > 0 else 2.5

            if grid.is_in_golden_pocket:
                status = "TRIGGERED"
                setup_name = "FIB_GOLDEN_POCKET_RETEST"
                reasons.append(f"Price currently in Golden Pocket (50.0% - 61.8% Fib: ₹{grid.level_500:.2f} - ₹{grid.level_618:.2f})")
                reasons.append(f"Optimal risk-defined bounce into resistance from Swing Low ₹{grid.swing_low:.2f}")
            elif grid.is_approaching:
                status = "APPROACHING"
                setup_name = "FIB_PULLBACK_APPROACHING"
                reasons.append(f"Bouncing towards 50.0% Equilibrium (₹{grid.level_500:.2f}) & 61.8% Golden Pocket")
            elif grid.current_zone == "DEEP_DISCOUNT":
                status = "TRIGGERED"
                setup_name = "FIB_DEEP_DISCOUNT_TAP"
                reasons.append(f"Testing Deep Premium 78.6% level at ₹{grid.level_786:.2f}")
            elif grid.current_zone == "EXPANSION":
                status = "WAITING"
                setup_name = "FIB_EXPANSION_WAIT_PULLBACK"
                reasons.append(f"Trading below 38.2% Fib (₹{grid.level_382:.2f}) — wait for bounce into Golden Pocket")
            else:
                status = "INVALIDATED"
                setup_name = "FIB_STRUCTURE_INVALIDATED"
                reasons.append(f"Broken above 78.6% invalidation level ₹{grid.level_786:.2f}")

            return FibonacciSetup(
                symbol=symbol,
                action="BUY PUT",
                direction="PUT",
                setup_name=setup_name,
                status=status,
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                target_3=round(t3, 2),
                risk_reward_ratio=round(max(0.1, rrr_t1), 2),
                extended_rrr=round(max(0.1, rrr_t2), 2),
                grid=grid,
                reasons=reasons,
                spot_price=current_close
            )
