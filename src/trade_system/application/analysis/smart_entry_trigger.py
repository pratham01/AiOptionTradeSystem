"""
SmartEntryTrigger — Identifies optimal micro-entry levels for high-scoring EdgeScore stocks.

Instead of entering blindly when a stock scores high, this module finds the exact
entry price, stop-loss, and target levels using structural analysis:

  1. VWAP Pullback Entry   — Price broke above VWAP, pulled back, now bouncing
  2. ORB Breakout Entry    — Price breaks above/below 15/30-min opening range with volume
  3. Supertrend Touch      — Price touches supertrend line and bounces

Each trigger produces entry/SL/target levels based on ATR-driven risk management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class EntryTrigger:
    """A specific entry opportunity with risk-defined levels."""
    trigger_type: str           # "VWAP_PULLBACK", "ORB_BREAKOUT", "ST_TOUCH"
    entry_price: float
    stop_loss: float
    target_1: float             # 1:1 R:R
    target_2: float             # 2:1 R:R
    confidence: str             # "HIGH", "MEDIUM"
    status: str                 # "TRIGGERED", "APPROACHING", "WAITING"
    detail: str = ""


class SmartEntryTrigger:
    """
    Evaluates an EdgeScore stock and determines if a precise entry opportunity exists.

    Usage:
        trigger = SmartEntryTrigger()
        entry = trigger.evaluate(
            direction="CALL", ltp=2450.0, atr=35.0,
            df_15m=candle_data, target_date=date.today()
        )
        if entry and entry.status == "TRIGGERED":
            print(f"BUY at {entry.entry_price}, SL={entry.stop_loss}")
    """

    # Risk management constants
    SL_ATR_MULTIPLIER = 1.0      # SL = 1x ATR from entry
    TARGET1_ATR_MULT = 1.5       # Target 1 = 1.5x ATR (1.5:1 R:R)
    TARGET2_ATR_MULT = 3.0       # Target 2 = 3x ATR (3:1 R:R)
    VWAP_ZONE_PCT = 0.003        # ±0.3% around VWAP = "at VWAP"

    def evaluate(
        self,
        direction: str,
        ltp: float,
        atr: float,
        df_15m: pd.DataFrame,
        target_date: date,
        vwap: float = 0.0,
        supertrend_value: float = 0.0,
    ) -> Optional[EntryTrigger]:
        """
        Evaluate all entry trigger types and return the best one (if any).

        Args:
            direction: "CALL" or "PUT"
            ltp: Last traded price
            atr: Average True Range (14-period on 15m)
            df_15m: 15-minute candle data for the symbol
            target_date: The trading day
            vwap: Current VWAP value (0 = auto-compute)
            supertrend_value: Current supertrend level (0 = auto-compute)

        Returns:
            An EntryTrigger if a setup is found, else None.
        """
        if atr <= 0 or ltp <= 0 or df_15m.empty:
            return None

        triggers = []

        # Try each trigger type
        vwap_trigger = self._check_vwap_pullback(direction, ltp, atr, df_15m, target_date, vwap)
        if vwap_trigger:
            triggers.append(vwap_trigger)

        orb_trigger = self._check_orb_breakout(direction, ltp, atr, df_15m, target_date)
        if orb_trigger:
            triggers.append(orb_trigger)

        st_trigger = self._check_supertrend_touch(direction, ltp, atr, df_15m, supertrend_value)
        if st_trigger:
            triggers.append(st_trigger)

        if not triggers:
            return None

        # Return the highest-confidence trigger (prefer TRIGGERED > APPROACHING)
        status_priority = {"TRIGGERED": 0, "APPROACHING": 1, "WAITING": 2}
        triggers.sort(key=lambda t: (status_priority.get(t.status, 3), -{"HIGH": 1, "MEDIUM": 0}.get(t.confidence, -1)))
        return triggers[0]

    # ── Trigger Implementations ────────────────────────────────────────────────

    def _check_vwap_pullback(
        self, direction: str, ltp: float, atr: float,
        df_15m: pd.DataFrame, target_date: date, vwap: float
    ) -> Optional[EntryTrigger]:
        """
        VWAP Pullback: Price was above VWAP, pulled back to VWAP zone, now bouncing.
        This is one of the highest-probability intraday entries.
        """
        if vwap <= 0:
            # Auto-compute VWAP
            try:
                from trade_system.application.indicators.vwap import VWAPIndicator
                vwap_df = VWAPIndicator().calculate(df_15m)
                today_vwap = vwap_df[vwap_df["timestamp"].dt.date == target_date]
                if today_vwap.empty or "vwap" not in today_vwap.columns:
                    return None
                vwap = float(today_vwap["vwap"].iloc[-1])
            except Exception:
                return None

        if vwap <= 0:
            return None

        vwap_dist = (ltp - vwap) / vwap

        if direction == "CALL":
            # Ideal: price is within +0.3% of VWAP (just bounced off it)
            if abs(vwap_dist) <= self.VWAP_ZONE_PCT:
                entry = ltp
                sl = vwap - atr * self.SL_ATR_MULTIPLIER
                t1 = entry + atr * self.TARGET1_ATR_MULT
                t2 = entry + atr * self.TARGET2_ATR_MULT
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK",
                    entry_price=round(entry, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    confidence="HIGH",
                    status="TRIGGERED",
                    detail=f"Price at VWAP ₹{vwap:.0f} (pullback entry)"
                )
            elif 0 < vwap_dist <= 0.008:  # Within 0.8% above VWAP
                entry = vwap  # Wait for pullback to VWAP
                sl = vwap - atr * self.SL_ATR_MULTIPLIER
                t1 = entry + atr * self.TARGET1_ATR_MULT
                t2 = entry + atr * self.TARGET2_ATR_MULT
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK",
                    entry_price=round(entry, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    confidence="MEDIUM",
                    status="APPROACHING",
                    detail=f"Above VWAP ₹{vwap:.0f}, waiting for pullback"
                )
        elif direction == "PUT":
            if abs(vwap_dist) <= self.VWAP_ZONE_PCT:
                entry = ltp
                sl = vwap + atr * self.SL_ATR_MULTIPLIER
                t1 = entry - atr * self.TARGET1_ATR_MULT
                t2 = entry - atr * self.TARGET2_ATR_MULT
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK",
                    entry_price=round(entry, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    confidence="HIGH",
                    status="TRIGGERED",
                    detail=f"Price at VWAP ₹{vwap:.0f} (rejection entry)"
                )
            elif -0.008 <= vwap_dist < 0:
                entry = vwap
                sl = vwap + atr * self.SL_ATR_MULTIPLIER
                t1 = entry - atr * self.TARGET1_ATR_MULT
                t2 = entry - atr * self.TARGET2_ATR_MULT
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK",
                    entry_price=round(entry, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    confidence="MEDIUM",
                    status="APPROACHING",
                    detail=f"Below VWAP ₹{vwap:.0f}, waiting for bounce to VWAP"
                )

        return None

    def _check_orb_breakout(
        self, direction: str, ltp: float, atr: float,
        df_15m: pd.DataFrame, target_date: date
    ) -> Optional[EntryTrigger]:
        """
        ORB Breakout: Price breaks above/below the 30-minute opening range.
        One of the most statistically reliable intraday patterns.
        """
        today_df = df_15m[df_15m["timestamp"].dt.date == target_date]
        if len(today_df) < 2:
            return None

        # First 2 candles = 30-minute opening range
        orb = today_df.head(2)
        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            return None

        if direction == "CALL" and ltp > orb_high:
            # Already broken out — check if it's a fresh breakout
            breakout_dist = (ltp - orb_high) / orb_high
            if breakout_dist <= 0.005:  # Within 0.5% of breakout = fresh
                status = "TRIGGERED"
                confidence = "HIGH"
            elif breakout_dist <= 0.015:
                status = "TRIGGERED"
                confidence = "MEDIUM"
            else:
                return None  # Too far from breakout level

            entry = ltp
            sl = orb_high - atr * 0.5  # Tight SL just below ORB high
            t1 = entry + atr * self.TARGET1_ATR_MULT
            t2 = entry + atr * self.TARGET2_ATR_MULT
            return EntryTrigger(
                trigger_type="ORB_BREAKOUT",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                confidence=confidence,
                status=status,
                detail=f"ORB High ₹{orb_high:.0f} broken (30m range: ₹{orb_range:.0f})"
            )
        elif direction == "PUT" and ltp < orb_low:
            breakout_dist = (orb_low - ltp) / orb_low
            if breakout_dist <= 0.005:
                status = "TRIGGERED"
                confidence = "HIGH"
            elif breakout_dist <= 0.015:
                status = "TRIGGERED"
                confidence = "MEDIUM"
            else:
                return None

            entry = ltp
            sl = orb_low + atr * 0.5
            t1 = entry - atr * self.TARGET1_ATR_MULT
            t2 = entry - atr * self.TARGET2_ATR_MULT
            return EntryTrigger(
                trigger_type="ORB_BREAKOUT",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                confidence=confidence,
                status=status,
                detail=f"ORB Low ₹{orb_low:.0f} broken (30m range: ₹{orb_range:.0f})"
            )
        elif direction == "CALL" and ltp < orb_high:
            # Approaching ORB high but not yet broken
            dist_to_orb = (orb_high - ltp) / orb_high
            if dist_to_orb <= 0.005:  # Within 0.5% of ORB high
                return EntryTrigger(
                    trigger_type="ORB_BREAKOUT",
                    entry_price=round(orb_high, 2),
                    stop_loss=round(orb_high - atr * 0.5, 2),
                    target_1=round(orb_high + atr * self.TARGET1_ATR_MULT, 2),
                    target_2=round(orb_high + atr * self.TARGET2_ATR_MULT, 2),
                    confidence="MEDIUM",
                    status="APPROACHING",
                    detail=f"Near ORB High ₹{orb_high:.0f} ({dist_to_orb:.1%} away)"
                )
        elif direction == "PUT" and ltp > orb_low:
            dist_to_orb = (ltp - orb_low) / orb_low
            if dist_to_orb <= 0.005:
                return EntryTrigger(
                    trigger_type="ORB_BREAKOUT",
                    entry_price=round(orb_low, 2),
                    stop_loss=round(orb_low + atr * 0.5, 2),
                    target_1=round(orb_low - atr * self.TARGET1_ATR_MULT, 2),
                    target_2=round(orb_low - atr * self.TARGET2_ATR_MULT, 2),
                    confidence="MEDIUM",
                    status="APPROACHING",
                    detail=f"Near ORB Low ₹{orb_low:.0f} ({dist_to_orb:.1%} away)"
                )

        return None

    def _check_supertrend_touch(
        self, direction: str, ltp: float, atr: float,
        df_15m: pd.DataFrame, supertrend_value: float
    ) -> Optional[EntryTrigger]:
        """
        Supertrend Touch: Price touches the supertrend line and bounces.
        Trend-following entry at the trend support/resistance.
        """
        if supertrend_value <= 0:
            # Auto-compute supertrend
            try:
                from trade_system.application.indicators.supertrend import SupertrendIndicator
                st_df = SupertrendIndicator(period=7, multiplier=3).calculate(df_15m)
                if "supertrend" not in st_df.columns or "supertrend_direction" not in st_df.columns:
                    return None
                supertrend_value = float(st_df["supertrend"].iloc[-1])
                st_dir = int(st_df["supertrend_direction"].iloc[-1])
            except Exception:
                return None
        else:
            st_dir = 1 if ltp > supertrend_value else -1

        if supertrend_value <= 0:
            return None

        st_dist_pct = abs(ltp - supertrend_value) / supertrend_value

        if direction == "CALL" and st_dir == 1 and st_dist_pct <= 0.005:
            # Price is touching bullish supertrend from above = bounce entry
            entry = ltp
            sl = supertrend_value - atr * 0.3  # Tight SL below supertrend
            t1 = entry + atr * self.TARGET1_ATR_MULT
            t2 = entry + atr * self.TARGET2_ATR_MULT
            return EntryTrigger(
                trigger_type="ST_TOUCH",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                confidence="HIGH",
                status="TRIGGERED",
                detail=f"Touching bullish ST ₹{supertrend_value:.0f}"
            )
        elif direction == "PUT" and st_dir == -1 and st_dist_pct <= 0.005:
            entry = ltp
            sl = supertrend_value + atr * 0.3
            t1 = entry - atr * self.TARGET1_ATR_MULT
            t2 = entry - atr * self.TARGET2_ATR_MULT
            return EntryTrigger(
                trigger_type="ST_TOUCH",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_1=round(t1, 2),
                target_2=round(t2, 2),
                confidence="HIGH",
                status="TRIGGERED",
                detail=f"Touching bearish ST ₹{supertrend_value:.0f}"
            )

        return None
