"""
Volumetric Order Flow Structure [LuxAlgo] Indicator
===================================================
Translated directly from the Pine Script:
"Volumetric Order Flow Structure [LuxAlgo]" by LuxAlgo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

@dataclass
class VolumetricOrderBlock:
    """Represents a Volumetric Order Block Zone."""
    start_idx: int
    start_time: pd.Timestamp
    high: float
    low: float
    volume: float
    is_bullish: bool
    row_weights: list[int]
    poc_level: float
    breakout_idx: int
    breakout_time: pd.Timestamp
    label_text: str               # "BOS" or "CHoCH"
    css_color: str                # Hex color string e.g. "#089981"
    manipulations: list[dict[str, Any]] = field(default_factory=list)

    def copy(self) -> VolumetricOrderBlock:
        return VolumetricOrderBlock(
            start_idx=self.start_idx,
            start_time=self.start_time,
            high=self.high,
            low=self.low,
            volume=self.volume,
            is_bullish=self.is_bullish,
            row_weights=list(self.row_weights),
            poc_level=self.poc_level,
            breakout_idx=self.breakout_idx,
            breakout_time=self.breakout_time,
            label_text=self.label_text,
            css_color=self.css_color,
            manipulations=[dict(m) for m in self.manipulations]
        )

class VolumetricOrderFlowDetector:
    """
    Identifies order blocks, maps volume profile row weights, calculates POC,
    and detects institutional liquidity sweeps (manipulation bubbles).
    """

    def __init__(
        self,
        pivot_length: int = 3,
        vol_lookback: int = 20,
        max_obs: int = 5,
        hide_overlapping: bool = False,
        show_manipulation: bool = True,
        manip_size: float = 1.0,
    ) -> None:
        self.pivot_length = pivot_length
        self.vol_lookback = vol_lookback
        self.max_obs = max_obs
        self.hide_overlapping = hide_overlapping
        self.show_manipulation = show_manipulation
        self.manip_size = manip_size

    def calculate(self, df: pd.DataFrame) -> list[list[VolumetricOrderBlock]]:
        """
        Runs the indicator across the entire DataFrame bar-by-bar.
        Returns a list of lists containing the active order blocks at each bar.
        """
        if df.empty or len(df) < 2 * self.pivot_length + 2:
            return [[] for _ in range(len(df))]

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values
        times = pd.to_datetime(df["timestamp"]).values

        n = len(df)
        active_obs: list[VolumetricOrderBlock] = []
        history: list[list[VolumetricOrderBlock]] = []

        # Color constants
        BULL_COLOR = "#089981"
        BEAR_COLOR = "#f23645"

        # Pivot High/Low state
        last_ph: float | None = None
        last_ph_idx: int | None = None
        last_pl: float | None = None
        last_pl_idx: int | None = None

        trend = 0

        # Precalculate pivots for fast lookups
        # A pivot high/low at idx - pivot_length is confirmed at idx
        p_highs = [None] * n
        p_lows = [None] * n
        k = self.pivot_length

        for i in range(2 * k, n):
            # Pivot high check at i - k
            val_h = highs[i - k]
            is_ph = True
            for j in range(i - 2 * k, i + 1):
                if highs[j] > val_h:
                    is_ph = False
                    break
                if highs[j] == val_h and j > i - k:
                    # prevent double pivots by checking strict inequality for right side
                    is_ph = False
                    break
            if is_ph:
                p_highs[i] = (i - k, val_h)

            # Pivot low check at i - k
            val_l = lows[i - k]
            is_pl = True
            for j in range(i - 2 * k, i + 1):
                if lows[j] < val_l:
                    is_pl = False
                    break
                if lows[j] == val_l and j > i - k:
                    is_pl = False
                    break
            if is_pl:
                p_lows[i] = (i - k, val_l)

        # Bar-by-bar simulation loop
        for idx in range(n):
            # 1. Update Pivot Points
            if p_highs[idx] is not None:
                last_ph_idx, last_ph = p_highs[idx]
            if p_lows[idx] is not None:
                last_pl_idx, last_pl = p_lows[idx]

            # 2. Mitigation checks on active order blocks
            # Bullish block mitigated: close < low
            # Bearish block mitigated: close > high
            for ob in active_obs[:]:
                close_under = closes[idx] < ob.low
                close_over = closes[idx] > ob.high
                mitigated = close_under if ob.is_bullish else close_over
                if mitigated:
                    active_obs.remove(ob)

            # 3. Detect Manipulation sweeps
            if self.show_manipulation and len(active_obs) > 0:
                # Find maximum volume in last 200 bars for scaling
                start_lookback = max(0, idx - 199)
                max_vol = np.max(volumes[start_lookback : idx + 1])
                if max_vol == 0:
                    max_vol = 1.0

                for ob in active_obs:
                    raid_high = not ob.is_bullish and highs[idx] > ob.high and closes[idx] <= ob.high
                    raid_low = ob.is_bullish and lows[idx] < ob.low and closes[idx] >= ob.low

                    if raid_high or raid_low:
                        v_scale = (volumes[idx] / max_vol) * self.manip_size
                        if v_scale > 0.8:
                            m_size = "huge"
                        elif v_scale > 0.6:
                            m_size = "large"
                        elif v_scale > 0.4:
                            m_size = "normal"
                        elif v_scale > 0.2:
                            m_size = "small"
                        else:
                            m_size = "tiny"

                        ob.manipulations.append({
                            "bar_idx": idx,
                            "timestamp": pd.Timestamp(times[idx]),
                            "price": highs[idx] if raid_high else lows[idx],
                            "volume": float(volumes[idx]),
                            "size": m_size,
                            "type": "Bearish Sweep" if raid_high else "Bullish Sweep"
                        })

            # 4. Check for Breakout crossover/crossunder
            # Only trigger check if we have active pivot values
            is_crossover = last_ph is not None and closes[idx] > last_ph and closes[idx - 1] <= last_ph
            is_crossunder = last_pl is not None and closes[idx] < last_pl and closes[idx - 1] >= last_pl

            if is_crossover:
                # Draw bullish order block from startIdx (last_ph_idx)
                p_high = highs[last_ph_idx]
                p_low = lows[last_ph_idx]
                p_open = opens[last_ph_idx]
                p_close = closes[last_ph_idx]
                p_vol = volumes[last_ph_idx]

                # Hide overlapping blocks if enabled
                should_draw = True
                overlapping = []
                if self.hide_overlapping:
                    for i, existing in enumerate(active_obs):
                        if p_low < existing.high and p_high > existing.low:
                            overlapping.append(i)
                    
                    for i in overlapping:
                        if p_vol <= active_obs[i].volume:
                            should_draw = False
                            break
                    if should_draw:
                        # Remove overlapping
                        for i in sorted(overlapping, reverse=True):
                            active_obs.pop(i)

                if should_draw:
                    # Calculate Volume Profile (15 rows)
                    rows = 15
                    row_step = (p_high - p_low) / rows if p_high > p_low else 1.0
                    body_max = max(p_open, p_close)
                    body_min = min(p_open, p_close)

                    weights = []
                    max_weight = 0
                    poc_idx = -1

                    for r in range(rows):
                        r_top = p_high - r * row_step
                        r_btm = r_top - row_step
                        dist_from_mid = abs(r - 7.5)
                        w = int(max(2, 12 - dist_from_mid * 1.5))
                        if (r_top <= body_max and r_top >= body_min) or (r_btm <= body_max and r_btm >= body_min):
                            w += 5
                        weights.append(w)
                        if w > max_weight:
                            max_weight = w
                            poc_idx = r

                    poc_level = p_high - (poc_idx + 0.5) * row_step

                    ob_label = "CHoCH" if trend == -1 else "BOS"
                    new_ob = VolumetricOrderBlock(
                        start_idx=last_ph_idx,
                        start_time=pd.Timestamp(times[last_ph_idx]),
                        high=p_high,
                        low=p_low,
                        volume=p_vol,
                        is_bullish=True,
                        row_weights=weights,
                        poc_level=poc_level,
                        breakout_idx=idx,
                        breakout_time=pd.Timestamp(times[idx]),
                        label_text=ob_label,
                        css_color=BULL_COLOR
                    )
                    active_obs.append(new_ob)
                    trend = 1
                    last_ph = None  # Consume the pivot high

                    # Maintain maximum blocks limit
                    if len(active_obs) > self.max_obs:
                        active_obs.pop(0)

            elif is_crossunder:
                # Draw bearish order block from startIdx (last_pl_idx)
                p_high = highs[last_pl_idx]
                p_low = lows[last_pl_idx]
                p_open = opens[last_pl_idx]
                p_close = closes[last_pl_idx]
                p_vol = volumes[last_pl_idx]

                # Hide overlapping blocks if enabled
                should_draw = True
                overlapping = []
                if self.hide_overlapping:
                    for i, existing in enumerate(active_obs):
                        if p_low < existing.high and p_high > existing.low:
                            overlapping.append(i)
                    
                    for i in overlapping:
                        if p_vol <= active_obs[i].volume:
                            should_draw = False
                            break
                    if should_draw:
                        for i in sorted(overlapping, reverse=True):
                            active_obs.pop(i)

                if should_draw:
                    # Calculate Volume Profile (15 rows)
                    rows = 15
                    row_step = (p_high - p_low) / rows if p_high > p_low else 1.0
                    body_max = max(p_open, p_close)
                    body_min = min(p_open, p_close)

                    weights = []
                    max_weight = 0
                    poc_idx = -1

                    for r in range(rows):
                        r_top = p_high - r * row_step
                        r_btm = r_top - row_step
                        dist_from_mid = abs(r - 7.5)
                        w = int(max(2, 12 - dist_from_mid * 1.5))
                        if (r_top <= body_max and r_top >= body_min) or (r_btm <= body_max and r_btm >= body_min):
                            w += 5
                        weights.append(w)
                        if w > max_weight:
                            max_weight = w
                            poc_idx = r

                    poc_level = p_high - (poc_idx + 0.5) * row_step

                    ob_label = "CHoCH" if trend == 1 else "BOS"
                    new_ob = VolumetricOrderBlock(
                        start_idx=last_pl_idx,
                        start_time=pd.Timestamp(times[last_pl_idx]),
                        high=p_high,
                        low=p_low,
                        volume=p_vol,
                        is_bullish=False,
                        row_weights=weights,
                        poc_level=poc_level,
                        breakout_idx=idx,
                        breakout_time=pd.Timestamp(times[idx]),
                        label_text=ob_label,
                        css_color=BEAR_COLOR
                    )
                    active_obs.append(new_ob)
                    trend = -1
                    last_pl = None  # Consume the pivot low

                    if len(active_obs) > self.max_obs:
                        active_obs.pop(0)

            # Store copy of currently active blocks for this bar
            history.append([ob.copy() for ob in active_obs])

        return history
