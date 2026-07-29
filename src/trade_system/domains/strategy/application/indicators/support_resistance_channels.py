"""
Support Resistance Channels Indicator
=======================================
Translated from Pine Script: "Support Resistance Channels" by LonesomeTheBlue.

Finds pivot highs/lows, groups nearby pivots into support/resistance channels,
ranks channels by strength (pivot count + price-touch density), and detects
when price breaks through a channel boundary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class SRChannel:
    """A single Support / Resistance channel."""
    high: float               # Upper boundary
    low: float                # Lower boundary
    strength: float           # Composite strength score
    channel_type: str         # "support", "resistance", or "inside"


@dataclass
class SRBreakEvent:
    """Records a support or resistance break event."""
    bar_idx: int
    timestamp: pd.Timestamp
    break_type: str           # "resistance_broken" or "support_broken"
    level_high: float
    level_low: float
    close: float


@dataclass
class SRSnapshot:
    """State of channels at a single bar."""
    channels: list[SRChannel]
    pivot_high: float | None  # If a pivot high was confirmed at this bar
    pivot_low: float | None   # If a pivot low was confirmed at this bar
    break_event: SRBreakEvent | None


class SupportResistanceChannelDetector:
    """
    Identifies Support/Resistance channels from pivot points and assigns
    strength rankings based on pivot density and historical price touches.
    """

    def __init__(
        self,
        pivot_period: int = 10,
        source: str = "High/Low",
        channel_width_pct: int = 5,
        min_strength: int = 1,
        max_num_sr: int = 6,
        loopback: int = 290,
    ) -> None:
        self.pivot_period = pivot_period
        self.source = source          # "High/Low" or "Close/Open"
        self.channel_width_pct = channel_width_pct
        self.min_strength = min_strength
        self.max_num_sr = min(max_num_sr, 10)
        self.loopback = loopback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate(self, df: pd.DataFrame) -> list[SRSnapshot]:
        """
        Process the full OHLCV DataFrame bar-by-bar and return a list of
        ``SRSnapshot`` objects, one per bar.
        """
        n = len(df)
        if n < 2 * self.pivot_period + 2:
            return [SRSnapshot([], None, None, None) for _ in range(n)]

        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        opens = df["open"].values.astype(float)
        closes = df["close"].values.astype(float)
        times = pd.to_datetime(df["timestamp"], format="mixed").values

        # Source selection
        if self.source == "High/Low":
            src1 = highs
            src2 = lows
        else:
            src1 = np.maximum(closes, opens)
            src2 = np.minimum(closes, opens)

        k = self.pivot_period

        # Pre-calculate pivot points ----------------------------------------
        # pivot_high[i] is confirmed at bar i, but the pivot itself sits at i-k
        p_highs: list[tuple[int, float] | None] = [None] * n
        p_lows: list[tuple[int, float] | None] = [None] * n

        for i in range(2 * k, n):
            # Pivot High at i - k
            val_h = src1[i - k]
            is_ph = True
            for j in range(i - 2 * k, i + 1):
                if j == i - k:
                    continue
                if src1[j] > val_h:
                    is_ph = False
                    break
                if src1[j] == val_h and j > i - k:
                    is_ph = False
                    break
            if is_ph:
                p_highs[i] = (i - k, float(val_h))

            # Pivot Low at i - k
            val_l = src2[i - k]
            is_pl = True
            for j in range(i - 2 * k, i + 1):
                if j == i - k:
                    continue
                if src2[j] < val_l:
                    is_pl = False
                    break
                if src2[j] == val_l and j > i - k:
                    is_pl = False
                    break
            if is_pl:
                p_lows[i] = (i - k, float(val_l))

        # Bar-by-bar simulation ---------------------------------------------
        pivot_vals: list[float] = []
        pivot_locs: list[int] = []
        sr_array: list[float] = [0.0] * 20   # 10 channels × 2 (hi, lo)
        results: list[SRSnapshot] = []

        for idx in range(n):
            cur_ph: float | None = None
            cur_pl: float | None = None
            new_pivot = False

            if p_highs[idx] is not None:
                cur_ph = p_highs[idx][1]
                new_pivot = True
            if p_lows[idx] is not None:
                cur_pl = p_lows[idx][1]
                new_pivot = True

            # Add new pivot(s) and prune old ones
            if new_pivot:
                val = cur_ph if cur_ph is not None else cur_pl
                pivot_vals.insert(0, val)
                pivot_locs.insert(0, idx)
                # If both a pivot high AND low are confirmed on the same bar,
                # only the high gets inserted (matches Pine's if/else priority)

                # Prune old pivots beyond loopback
                while len(pivot_vals) > 0 and idx - pivot_locs[-1] > self.loopback:
                    pivot_vals.pop()
                    pivot_locs.pop()

            # Recalculate channels whenever a new pivot arrives
            if new_pivot and len(pivot_vals) > 0:
                sr_array = self._recalculate_channels(
                    pivot_vals, highs, lows, closes, idx
                )

            # Build current channel list
            channels: list[SRChannel] = []
            cur_close = closes[idx]
            for c in range(min(10, self.max_num_sr)):
                hi = sr_array[c * 2]
                lo = sr_array[c * 2 + 1]
                if hi == 0.0 and lo == 0.0:
                    continue
                if hi > cur_close and lo > cur_close:
                    ctype = "resistance"
                elif hi < cur_close and lo < cur_close:
                    ctype = "support"
                else:
                    ctype = "inside"
                channels.append(SRChannel(hi, lo, 0.0, ctype))

            # Break detection
            break_event: SRBreakEvent | None = None
            if idx > 0:
                not_in_channel = True
                for c in range(min(10, self.max_num_sr)):
                    hi = sr_array[c * 2]
                    lo = sr_array[c * 2 + 1]
                    if hi == 0.0 and lo == 0.0:
                        continue
                    if closes[idx] <= hi and closes[idx] >= lo:
                        not_in_channel = False
                        break

                if not_in_channel:
                    for c in range(min(10, self.max_num_sr)):
                        hi = sr_array[c * 2]
                        lo = sr_array[c * 2 + 1]
                        if hi == 0.0 and lo == 0.0:
                            continue
                        if closes[idx - 1] <= hi and closes[idx] > hi:
                            break_event = SRBreakEvent(
                                bar_idx=idx,
                                timestamp=pd.Timestamp(times[idx]),
                                break_type="resistance_broken",
                                level_high=hi,
                                level_low=lo,
                                close=closes[idx],
                            )
                            break
                        if closes[idx - 1] >= lo and closes[idx] < lo:
                            break_event = SRBreakEvent(
                                bar_idx=idx,
                                timestamp=pd.Timestamp(times[idx]),
                                break_type="support_broken",
                                level_high=hi,
                                level_low=lo,
                                close=closes[idx],
                            )
                            break

            results.append(SRSnapshot(channels, cur_ph, cur_pl, break_event))

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _recalculate_channels(
        self,
        pivot_vals: list[float],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        cur_idx: int,
    ) -> list[float]:
        """Recalculate the 10-slot SR channel array."""
        num_pivots = len(pivot_vals)

        # Channel width based on 300-bar range
        start_300 = max(0, cur_idx - 299)
        prdhighest = float(np.max(highs[start_300: cur_idx + 1]))
        prdlowest = float(np.min(lows[start_300: cur_idx + 1]))
        cwidth = (prdhighest - prdlowest) * self.channel_width_pct / 100.0

        # --- Step 1: get_sr_vals for each pivot ---
        # supres stores triplets: [strength, hi, lo] for each pivot
        supres: list[float] = []
        for x in range(num_pivots):
            lo = pivot_vals[x]
            hi = lo
            numpp = 0
            for y in range(num_pivots):
                cpp = pivot_vals[y]
                wdth = hi - cpp if cpp <= hi else cpp - lo
                if wdth <= cwidth:
                    if cpp <= hi:
                        lo = min(lo, cpp)
                    else:
                        hi = max(hi, cpp)
                    numpp += 20
            supres.extend([float(numpp), hi, lo])

        # --- Step 2: add price-touch strength ---
        lb = min(self.loopback, cur_idx + 1)
        for x in range(num_pivots):
            h = supres[x * 3 + 1]
            l = supres[x * 3 + 2]
            s = 0
            for y in range(lb):
                bar = cur_idx - y
                if bar < 0:
                    break
                if (highs[bar] <= h and highs[bar] >= l) or (lows[bar] <= h and lows[bar] >= l):
                    s += 1
            supres[x * 3] += s

        # --- Step 3: pick strongest channels (up to 10) ---
        sr_out = [0.0] * 20
        stren = [0.0] * 10
        src = 0
        for _ in range(num_pivots):
            stv = -1.0
            stl = -1
            for y in range(num_pivots):
                if supres[y * 3] > stv and supres[y * 3] >= self.min_strength * 20:
                    stv = supres[y * 3]
                    stl = y
            if stl < 0:
                break

            hh = supres[stl * 3 + 1]
            ll = supres[stl * 3 + 2]
            sr_out[src * 2] = hh
            sr_out[src * 2 + 1] = ll
            stren[src] = supres[stl * 3]

            # Zero-out all overlapping pivots
            for y in range(num_pivots):
                y_hi = supres[y * 3 + 1]
                y_lo = supres[y * 3 + 2]
                if (y_hi <= hh and y_hi >= ll) or (y_lo <= hh and y_lo >= ll):
                    supres[y * 3] = -1.0

            src += 1
            if src >= 10:
                break

        # --- Step 4: sort by descending strength ---
        for x in range(min(src, 9)):
            for y in range(x + 1, min(src, 10)):
                if stren[y] > stren[x]:
                    stren[x], stren[y] = stren[y], stren[x]
                    sr_out[x * 2], sr_out[y * 2] = sr_out[y * 2], sr_out[x * 2]
                    sr_out[x * 2 + 1], sr_out[y * 2 + 1] = sr_out[y * 2 + 1], sr_out[x * 2 + 1]

        return sr_out
