"""
Flux Charts Volumized Order Blocks Indicator
============================================
Translated directly from the Pine Script:
"Volumized Order Blocks | Flux Charts" by Flux Charts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

@dataclass
class FluxOBInfo:
    """Represents a Volumized Order Block Zone."""
    top: float
    bottom: float
    ob_volume: float
    ob_type: str                  # "Bull" (Demand) or "Bear" (Supply)
    start_time: pd.Timestamp
    start_idx: int
    bb_volume: float = 0.0        # Volume at invalidation break
    ob_low_volume: float = 0.0
    ob_high_volume: float = 0.0
    breaker: bool = False         # True if zone has been breached/invalidated
    break_time: pd.Timestamp | None = None
    break_idx: int | None = None
    disabled: bool = False        # True if combined into another block
    combined: bool = False
    combined_timeframes_str: str | None = None

    def copy(self) -> FluxOBInfo:
        return FluxOBInfo(
            top=self.top,
            bottom=self.bottom,
            ob_volume=self.ob_volume,
            ob_type=self.ob_type,
            start_time=self.start_time,
            start_idx=self.start_idx,
            bb_volume=self.bb_volume,
            ob_low_volume=self.ob_low_volume,
            ob_high_volume=self.ob_high_volume,
            breaker=self.breaker,
            break_time=self.break_time,
            break_idx=self.break_idx,
            disabled=self.disabled,
            combined=self.combined,
            combined_timeframes_str=self.combined_timeframes_str,
        )

@dataclass
class FluxSwingPoint:
    x: int
    y: float
    volume: float
    crossed: bool = False

class FluxOrderBlockDetector:
    """
    Translates the Flux Charts Volumized Order Blocks logic to Python.
    Tracks swing pivots, identifies order block zones with volume profile splits,
    handles breaker mitigation, and merges overlapping zones.
    """

    def __init__(
        self,
        swing_length: int = 10,
        max_atr_mult: float = 3.5,
        max_order_blocks: int = 30,
        ob_end_method: str = "Wick",     # "Wick" or "Close"
        combine_obs: bool = True,
        num_render_blocks: int = 3,      # Equivalent to 'bullishOrderBlocks' in Pine Script
    ) -> None:
        self.swing_length = swing_length
        self.max_atr_mult = max_atr_mult
        self.max_order_blocks = max_order_blocks
        self.ob_end_method = ob_end_method
        self.combine_obs = combine_obs
        self.num_render_blocks = num_render_blocks

    def calculate(self, df: pd.DataFrame) -> list[list[FluxOBInfo]]:
        """
        Runs the indicator across the entire DataFrame bar-by-bar.
        Returns a list (matching length of df) where each element is the list of 
        currently active, combined, and unmitigated order blocks at that bar.
        """
        if df.empty or len(df) < self.swing_length + 2:
            return [[] for _ in range(len(df))]

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values
        times = pd.to_datetime(df["timestamp"]).values

        n = len(df)

        # ── Step 1: Precalculate ATR 10 ─────────────────────────────────────
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        tr[0] = highs[0] - lows[0]
        
        atr = np.zeros(n)
        atr[9] = np.mean(tr[:10])
        alpha = 1.0 / 10.0
        for i in range(10, n):
            atr[i] = tr[i] * alpha + atr[i - 1] * (1.0 - alpha)

        # ── Step 2: Bar-by-bar Simulation ───────────────────────────────────
        swing_types = [0] * n
        top_swing: FluxSwingPoint | None = None
        btm_swing: FluxSwingPoint | None = None

        bullish_obs_list: list[FluxOBInfo] = []
        bearish_obs_list: list[FluxOBInfo] = []

        history: list[list[FluxOBInfo]] = []

        for idx in range(n):
            if idx < self.swing_length + 1:
                history.append([])
                continue

            # ── 2a: Update Swing Types & Pivots ──────────────────────────────
            # FIX: Pine Script ta.highest(len) / ta.lowest(len) at bar idx refers
            # to the highest/lowest of bars [idx-len .. idx-1] (strictly past,
            # excluding the current bar).  The reference bar compared against
            # this window is bar [idx-len-1] — i.e. one bar *before* the window.
            #
            # Previous (buggy) code used [idx-len+1 .. idx] which included the
            # current bar and compared against the wrong anchor bar.
            look_start = idx - self.swing_length       # window start (inclusive)
            look_end   = idx                           # window end   (exclusive)
            sub_highs = highs[look_start:look_end]     # swing_length bars, NOT including current
            sub_lows  = lows[look_start:look_end]
            upper = np.max(sub_highs)
            lower = np.min(sub_lows)

            # Anchor bar: the bar just before the look-back window
            anchor_idx = look_start - 1
            high_anchor = highs[anchor_idx]
            low_anchor  = lows[anchor_idx]

            if high_anchor > upper:
                curr_swing_type = 0   # potential swing HIGH
            elif low_anchor < lower:
                curr_swing_type = 1   # potential swing LOW
            else:
                curr_swing_type = swing_types[idx - 1]
            swing_types[idx] = curr_swing_type

            # New swing high confirmed: type just flipped to 0
            if curr_swing_type == 0 and swing_types[idx - 1] != 0:
                top_swing = FluxSwingPoint(
                    x=anchor_idx,
                    y=high_anchor,
                    volume=volumes[anchor_idx],
                    crossed=False
                )
                LOGGER.debug(
                    "Swing HIGH @ bar %d (price %.2f) confirmed at bar %d",
                    anchor_idx, high_anchor, idx
                )

            # New swing low confirmed: type just flipped to 1
            if curr_swing_type == 1 and swing_types[idx - 1] != 1:
                btm_swing = FluxSwingPoint(
                    x=anchor_idx,
                    y=low_anchor,
                    volume=volumes[anchor_idx],
                    crossed=False
                )
                LOGGER.debug(
                    "Swing LOW @ bar %d (price %.2f) confirmed at bar %d",
                    anchor_idx, low_anchor, idx
                )

            # ── 2b: Update Invalidation / Breaker state ──────────────────────
            # Bullish zones invalidation
            # If price dips below bottom, mark as breaker.
            # If it is a breaker, check if it is completely run through (high > top), and remove it.
            for ob in bullish_obs_list[:]:
                if not ob.breaker:
                    trigger_price = lows[idx] if self.ob_end_method == "Wick" else min(opens[idx], closes[idx])
                    if trigger_price < ob.bottom:
                        ob.breaker = True
                        ob.break_time = pd.Timestamp(times[idx])
                        ob.break_idx = idx
                        ob.bb_volume = volumes[idx]
                else:
                    if highs[idx] > ob.top:
                        bullish_obs_list.remove(ob)

            # Bearish zones invalidation
            for ob in bearish_obs_list[:]:
                if not ob.breaker:
                    trigger_price = highs[idx] if self.ob_end_method == "Wick" else max(opens[idx], closes[idx])
                    if trigger_price > ob.top:
                        ob.breaker = True
                        ob.break_time = pd.Timestamp(times[idx])
                        ob.break_idx = idx
                        ob.bb_volume = volumes[idx]
                else:
                    if lows[idx] < ob.bottom:
                        bearish_obs_list.remove(ob)

            # ── 2c: Detect New Order Block Formations ────────────────────────
            current_atr = atr[idx] if atr[idx] > 0 else tr[idx]

            # Bullish OB: price crosses ABOVE the last confirmed swing HIGH
            # The OB origin = the last BEARISH candle in the base before the impulse.
            # (Institutions placed buy orders at that level; it is now a demand zone.)
            if top_swing is not None and not top_swing.crossed and closes[idx] > top_swing.y:
                top_swing.crossed = True

                start_range = top_swing.x + 1
                end_range = idx  # exclusive — do not include the breakout bar itself

                if start_range < end_range:
                    # FIX: Find the LAST bearish (red) candle in the base instead of
                    # the lowest-low candle.  The last bearish candle IS the order block
                    # because that is where smart-money absorption happened.
                    ob_idx = -1
                    for j in range(end_range - 1, start_range - 1, -1):  # scan backwards
                        if closes[j] < opens[j]:   # bearish (red) candle
                            ob_idx = j
                            break

                    # Fallback: if no bearish candle found, use lowest-low candle
                    if ob_idx == -1:
                        lowest_low_val = np.inf
                        for j in range(start_range, end_range):
                            if lows[j] < lowest_low_val:
                                lowest_low_val = lows[j]
                                ob_idx = j

                    if ob_idx != -1:
                        ob_size = abs(highs[ob_idx] - lows[ob_idx])
                        if ob_size <= current_atr * self.max_atr_mult:
                            new_ob = FluxOBInfo(
                                top=highs[ob_idx],
                                bottom=lows[ob_idx],
                                ob_volume=(
                                    volumes[idx]
                                    + (volumes[idx - 1] if idx >= 1 else 0.0)
                                    + (volumes[idx - 2] if idx >= 2 else 0.0)
                                ),
                                ob_type="Bull",
                                start_time=pd.Timestamp(times[ob_idx]),
                                start_idx=ob_idx,
                                ob_low_volume=volumes[idx - 2] if idx >= 2 else 0.0,
                                ob_high_volume=volumes[idx] + (volumes[idx - 1] if idx >= 1 else 0.0),
                            )
                            bullish_obs_list.insert(0, new_ob)
                            if len(bullish_obs_list) > self.max_order_blocks:
                                bullish_obs_list.pop()
                            LOGGER.debug(
                                "Bull OB created: bar=%d time=%s top=%.2f btm=%.2f | "
                                "breakout bar=%d swing_high bar=%d",
                                ob_idx, times[ob_idx], highs[ob_idx], lows[ob_idx],
                                idx, top_swing.x
                            )

            # Bearish OB: price crosses BELOW the last confirmed swing LOW
            # The OB origin = the last BULLISH candle in the base before the impulse.
            if btm_swing is not None and not btm_swing.crossed and closes[idx] < btm_swing.y:
                btm_swing.crossed = True

                start_range = btm_swing.x + 1
                end_range = idx

                if start_range < end_range:
                    # FIX: Find the LAST bullish (green) candle in the base.
                    ob_idx = -1
                    for j in range(end_range - 1, start_range - 1, -1):  # scan backwards
                        if closes[j] > opens[j]:   # bullish (green) candle
                            ob_idx = j
                            break

                    # Fallback: if no bullish candle found, use highest-high candle
                    if ob_idx == -1:
                        highest_high_val = -np.inf
                        for j in range(start_range, end_range):
                            if highs[j] > highest_high_val:
                                highest_high_val = highs[j]
                                ob_idx = j

                    if ob_idx != -1:
                        ob_size = abs(highs[ob_idx] - lows[ob_idx])
                        if ob_size <= current_atr * self.max_atr_mult:
                            new_ob = FluxOBInfo(
                                top=highs[ob_idx],
                                bottom=lows[ob_idx],
                                ob_volume=(
                                    volumes[idx]
                                    + (volumes[idx - 1] if idx >= 1 else 0.0)
                                    + (volumes[idx - 2] if idx >= 2 else 0.0)
                                ),
                                ob_type="Bear",
                                start_time=pd.Timestamp(times[ob_idx]),
                                start_idx=ob_idx,
                                ob_low_volume=volumes[idx] + (volumes[idx - 1] if idx >= 1 else 0.0),
                                ob_high_volume=volumes[idx - 2] if idx >= 2 else 0.0,
                            )
                            bearish_obs_list.insert(0, new_ob)
                            if len(bearish_obs_list) > self.max_order_blocks:
                                bearish_obs_list.pop()
                            LOGGER.debug(
                                "Bear OB created: bar=%d time=%s top=%.2f btm=%.2f | "
                                "breakout bar=%d swing_low bar=%d",
                                ob_idx, times[ob_idx], highs[ob_idx], lows[ob_idx],
                                idx, btm_swing.x
                            )

            # ── 2d: Combine & Filter Active Render Zones ────────────────────
            # Copy active list items up to render limit
            active_bullish = [ob.copy() for ob in bullish_obs_list[:self.num_render_blocks]]
            active_bearish = [ob.copy() for ob in bearish_obs_list[:self.num_render_blocks]]
            
            combined_obs = active_bullish + active_bearish

            if self.combine_obs and len(combined_obs) > 1:
                # Combine touching zones iteratively
                last_combinations = 999
                current_time = pd.Timestamp(times[idx])
                while last_combinations > 0:
                    last_combinations = 0
                    for i in range(len(combined_obs)):
                        ob1 = combined_obs[i]
                        if ob1.disabled:
                            continue
                        for j in range(len(combined_obs)):
                            if i == j:
                                continue
                            ob2 = combined_obs[j]
                            if ob2.disabled or ob1.ob_type != ob2.ob_type:
                                continue
                            
                            # Overlap checks
                            # Time bounds: [start_idx, break_idx] (or current idx if active)
                            t1_start = ob1.start_idx
                            t1_end = ob1.break_idx if ob1.break_idx is not None else idx + 1
                            t2_start = ob2.start_idx
                            t2_end = ob2.break_idx if ob2.break_idx is not None else idx + 1
                            
                            time_overlap = max(0, min(t1_end, t2_end) - max(t1_start, t2_start))
                            
                            # Price bounds: [bottom, top]
                            price_overlap = max(0.0, min(ob1.top, ob2.top) - max(ob1.bottom, ob2.bottom))
                            
                            if time_overlap > 0 and price_overlap > 0.0:
                                # Touch / Overlap detected! Combine them
                                ob1.disabled = True
                                ob2.disabled = True
                                
                                new_top = max(ob1.top, ob2.top)
                                new_bottom = min(ob1.bottom, ob2.bottom)
                                new_start_idx = min(ob1.start_idx, ob2.start_idx)
                                new_start_time = min(ob1.start_time, ob2.start_time)
                                
                                # Break index calculation
                                if ob1.break_idx is None or ob2.break_idx is None:
                                    new_break_idx = None
                                    new_break_time = None
                                else:
                                    new_break_idx = max(ob1.break_idx, ob2.break_idx)
                                    new_break_time = max(ob1.break_time, ob2.break_time)

                                new_ob = FluxOBInfo(
                                    top=new_top,
                                    bottom=new_bottom,
                                    ob_volume=ob1.ob_volume + ob2.ob_volume,
                                    ob_type=ob1.ob_type,
                                    start_time=new_start_time,
                                    start_idx=new_start_idx,
                                    bb_volume=ob1.bb_volume + ob2.bb_volume,
                                    ob_low_volume=ob1.ob_low_volume + ob2.ob_low_volume,
                                    ob_high_volume=ob1.ob_high_volume + ob2.ob_high_volume,
                                    breaker=ob1.breaker or ob2.breaker,
                                    break_time=new_break_time,
                                    break_idx=new_break_idx,
                                    combined=True
                                )
                                combined_obs.insert(0, new_ob)
                                last_combinations += 1
                                break # break inner loop to restart scanning

            # Keep only non-disabled zones
            active_list = [ob for ob in combined_obs if not ob.disabled]
            history.append(active_list)

        return history
