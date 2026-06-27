"""
SmartEntryTrigger — Identifies optimal micro-entry levels for high-scoring EdgeScore stocks.
Supports INTRADAY, WEEKLY, and MONTHLY horizons.

Instead of entering blindly when a stock scores high, this module finds the exact
entry price, stop-loss, and target levels using structural analysis:
  1. VWAP Pullback Entry   — Price bounces off the session/rolling VWAP line.
  2. Range Breakout Entry  — Price breaks out above/below the relevant range (ORB for Intraday, Prev Week for Weekly, Prev Month for Monthly).
  3. Supertrend Touch      — Price bounces off the Supertrend support/resistance line.
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
    trigger_type: str           # "VWAP_PULLBACK", "ORB_BREAKOUT", "RANGE_BREAKOUT", "ST_TOUCH"
    entry_price: float
    stop_loss: float
    target_1: float             # Target 1
    target_2: float             # Target 2 (extended target)
    confidence: str             # "HIGH", "MEDIUM"
    status: str                 # "TRIGGERED", "APPROACHING", "WAITING"
    detail: str = ""


class SmartEntryTrigger:
    """
    Evaluates an EdgeScore stock and determines if a precise entry opportunity exists.
    """

    def __init__(self, horizon: str = "INTRADAY") -> None:
        self.horizon = horizon.upper()

        # Scale multipliers based on horizon
        if self.horizon == "INTRADAY":
            self.sl_multiplier = 1.0
            self.target_1_multiplier = 1.5
            self.target_2_multiplier = 3.0
            self.zone_pct = 0.003
        elif self.horizon == "WEEKLY":
            self.sl_multiplier = 1.2
            self.target_1_multiplier = 2.0
            self.target_2_multiplier = 4.0
            self.zone_pct = 0.010
        else:  # MONTHLY
            self.sl_multiplier = 1.5
            self.target_1_multiplier = 2.5
            self.target_2_multiplier = 5.0
            self.zone_pct = 0.015

    def evaluate(
        self,
        direction: str,
        ltp: float,
        atr: float,
        df_base: pd.DataFrame | None = None,
        target_date: date | None = None,
        vwap: float = 0.0,
        supertrend_value: float = 0.0,
        df_15m: pd.DataFrame | None = None,
    ) -> Optional[EntryTrigger]:
        """
        Evaluate all entry triggers for the selected horizon and return the best.
        """
        if df_base is None:
            df_base = df_15m
            
        if atr <= 0 or ltp <= 0 or df_base is None or df_base.empty:
            return None

        triggers = []

        # 1. VWAP Pullback Trigger
        vwap_trig = self._check_vwap_pullback(direction, ltp, atr, df_base, target_date, vwap)
        if vwap_trig:
            triggers.append(vwap_trig)

        # 2. Breakout Trigger (ORB or Range Breakout)
        breakout_trig = self._check_breakout(direction, ltp, atr, df_base, target_date)
        if breakout_trig:
            triggers.append(breakout_trig)

        # 3. Supertrend Touch Trigger
        st_trig = self._check_supertrend_touch(direction, ltp, atr, df_base, supertrend_value)
        if st_trig:
            triggers.append(st_trig)

        if not triggers:
            return None

        status_priority = {"TRIGGERED": 0, "APPROACHING": 1, "WAITING": 2}
        triggers.sort(key=lambda t: (status_priority.get(t.status, 3), -{"HIGH": 1, "MEDIUM": 0}.get(t.confidence, -1)))
        return triggers[0]

    # ── Trigger Implementations ────────────────────────────────────────────────

    def _check_vwap_pullback(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame, target_date: date, vwap: float
    ) -> Optional[EntryTrigger]:
        """VWAP Pullback: Price pulls back to the session or rolling VWAP and bounces."""
        if vwap <= 0:
            try:
                if self.horizon == "INTRADAY":
                    from trade_system.application.indicators.vwap import VWAPIndicator
                    vwap_df = VWAPIndicator().calculate(df_base)
                    today_vwap = vwap_df[vwap_df["timestamp"].dt.date == target_date]
                    if not today_vwap.empty and "vwap" in today_vwap.columns:
                        vwap = float(today_vwap["vwap"].iloc[-1])
                else:
                    lookback = 20 if self.horizon == "WEEKLY" else 252
                    df = df_base.copy()
                    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
                    df["tp_vol"] = df["tp"] * df["volume"]
                    df["vwap"] = df["tp_vol"].rolling(lookback).sum() / df["volume"].rolling(lookback).sum()
                    target_df = df[df["timestamp"].dt.date <= target_date]
                    if not target_df.empty:
                        vwap = float(target_df["vwap"].iloc[-1])
            except Exception:
                return None

        if vwap <= 0 or pd.isna(vwap):
            return None

        vwap_dist = (ltp - vwap) / vwap

        if direction == "CALL":
            if abs(vwap_dist) <= self.zone_pct:
                entry = ltp
                sl = vwap - atr * self.sl_multiplier
                t1 = entry + atr * self.target_1_multiplier
                t2 = entry + atr * self.target_2_multiplier
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                    target_1=round(t1, 2), target_2=round(t2, 2), confidence="HIGH", status="TRIGGERED",
                    detail=f"Price near VWAP ₹{vwap:.1f} (pullback zone)"
                )
            elif 0 < vwap_dist <= self.zone_pct * 2.5:
                entry = vwap
                sl = vwap - atr * self.sl_multiplier
                t1 = entry + atr * self.target_1_multiplier
                t2 = entry + atr * self.target_2_multiplier
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                    target_1=round(t1, 2), target_2=round(t2, 2), confidence="MEDIUM", status="APPROACHING",
                    detail=f"Above VWAP ₹{vwap:.1f}, waiting for pullback"
                )
        elif direction == "PUT":
            if abs(vwap_dist) <= self.zone_pct:
                entry = ltp
                sl = vwap + atr * self.sl_multiplier
                t1 = entry - atr * self.target_1_multiplier
                t2 = entry - atr * self.target_2_multiplier
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                    target_1=round(t1, 2), target_2=round(t2, 2), confidence="HIGH", status="TRIGGERED",
                    detail=f"Price near VWAP ₹{vwap:.1f} (rejection zone)"
                )
            elif -self.zone_pct * 2.5 <= vwap_dist < 0:
                entry = vwap
                sl = vwap + atr * self.sl_multiplier
                t1 = entry - atr * self.target_1_multiplier
                t2 = entry - atr * self.target_2_multiplier
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                    target_1=round(t1, 2), target_2=round(t2, 2), confidence="MEDIUM", status="APPROACHING",
                    detail=f"Below VWAP ₹{vwap:.1f}, waiting for pullback bounce"
                )

        return None

    def _check_breakout(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame, target_date: date
    ) -> Optional[EntryTrigger]:
        """Breakout/Breakdown of range boundaries (Intraday ORB or Swing Prev High/Low)."""
        
        # 1. Intraday ORB Breakout
        if self.horizon == "INTRADAY":
            today_df = df_base[df_base["timestamp"].dt.date == target_date]
            if len(today_df) < 2:
                return None
            orb = today_df.head(2)
            orb_high = float(orb["high"].max())
            orb_low = float(orb["low"].min())
            
            if direction == "CALL" and ltp > orb_high:
                dist = (ltp - orb_high) / orb_high
                if dist > 0.015: return None
                return EntryTrigger(
                    trigger_type="ORB_BREAKOUT", entry_price=round(ltp, 2), stop_loss=round(orb_high - atr * 0.5, 2),
                    target_1=round(ltp + atr * self.target_1_multiplier, 2), target_2=round(ltp + atr * self.target_2_multiplier, 2),
                    confidence="HIGH" if dist <= 0.005 else "MEDIUM", status="TRIGGERED",
                    detail=f"ORB High ₹{orb_high:.1f} broken"
                )
            elif direction == "PUT" and ltp < orb_low:
                dist = (orb_low - ltp) / orb_low
                if dist > 0.015: return None
                return EntryTrigger(
                    trigger_type="ORB_BREAKOUT", entry_price=round(ltp, 2), stop_loss=round(orb_low + atr * 0.5, 2),
                    target_1=round(ltp - atr * self.target_1_multiplier, 2), target_2=round(ltp - atr * self.target_2_multiplier, 2),
                    confidence="HIGH" if dist <= 0.005 else "MEDIUM", status="TRIGGERED",
                    detail=f"ORB Low ₹{orb_low:.1f} broken"
                )
            elif direction == "CALL" and ltp < orb_high:
                dist = (orb_high - ltp) / orb_high
                if dist <= 0.005:
                    return EntryTrigger(
                        trigger_type="ORB_BREAKOUT", entry_price=round(orb_high, 2), stop_loss=round(orb_high - atr * 0.5, 2),
                        target_1=round(orb_high + atr * self.target_1_multiplier, 2), target_2=round(orb_high + atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching ORB High ₹{orb_high:.1f}"
                    )
            elif direction == "PUT" and ltp > orb_low:
                dist = (ltp - orb_low) / orb_low
                if dist <= 0.005:
                    return EntryTrigger(
                        trigger_type="ORB_BREAKOUT", entry_price=round(orb_low, 2), stop_loss=round(orb_low + atr * 0.5, 2),
                        target_1=round(orb_low - atr * self.target_1_multiplier, 2), target_2=round(orb_low - atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching ORB Low ₹{orb_low:.1f}"
                    )
            return None

        # 2. Weekly & Monthly Range Breakouts
        else:
            base_filtered = df_base[df_base["timestamp"].dt.date <= target_date].sort_values("timestamp")
            if len(base_filtered) < 2:
                return None
                
            prev_candle = base_filtered.iloc[-2]
            prev_high = float(prev_candle["high"])
            prev_low = float(prev_candle["low"])
            
            ref_label = "Prev Week" if self.horizon == "WEEKLY" else "Prev Month"
            
            if direction == "CALL" and ltp > prev_high:
                dist = (ltp - prev_high) / prev_high
                if dist > self.zone_pct * 3: return None
                return EntryTrigger(
                    trigger_type="RANGE_BREAKOUT", entry_price=round(ltp, 2), stop_loss=round(prev_high - atr * 0.5, 2),
                    target_1=round(ltp + atr * self.target_1_multiplier, 2), target_2=round(ltp + atr * self.target_2_multiplier, 2),
                    confidence="HIGH" if dist <= self.zone_pct else "MEDIUM", status="TRIGGERED",
                    detail=f"Broken {ref_label} High ₹{prev_high:.1f}"
                )
            elif direction == "PUT" and ltp < prev_low:
                dist = (prev_low - ltp) / prev_low
                if dist > self.zone_pct * 3: return None
                return EntryTrigger(
                    trigger_type="RANGE_BREAKOUT", entry_price=round(ltp, 2), stop_loss=round(prev_low + atr * 0.5, 2),
                    target_1=round(ltp - atr * self.target_1_multiplier, 2), target_2=round(ltp - atr * self.target_2_multiplier, 2),
                    confidence="HIGH" if dist <= self.zone_pct else "MEDIUM", status="TRIGGERED",
                    detail=f"Broken {ref_label} Low ₹{prev_low:.1f}"
                )
            elif direction == "CALL" and ltp < prev_high:
                dist = (prev_high - ltp) / prev_high
                if dist <= self.zone_pct:
                    return EntryTrigger(
                        trigger_type="RANGE_BREAKOUT", entry_price=round(prev_high, 2), stop_loss=round(prev_high - atr * 0.5, 2),
                        target_1=round(prev_high + atr * self.target_1_multiplier, 2), target_2=round(prev_high + atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching {ref_label} High ₹{prev_high:.1f}"
                    )
            elif direction == "PUT" and ltp > prev_low:
                dist = (ltp - prev_low) / prev_low
                if dist <= self.zone_pct:
                    return EntryTrigger(
                        trigger_type="RANGE_BREAKOUT", entry_price=round(prev_low, 2), stop_loss=round(prev_low + atr * 0.5, 2),
                        target_1=round(prev_low - atr * self.target_1_multiplier, 2), target_2=round(prev_low - atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching {ref_label} Low ₹{prev_low:.1f}"
                    )
            return None

    def _check_supertrend_touch(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame, supertrend_value: float
    ) -> Optional[EntryTrigger]:
        """Supertrend Touch: Price touches the Supertrend line and bounces."""
        if supertrend_value <= 0:
            try:
                from trade_system.application.indicators.supertrend import SupertrendIndicator
                st_df = SupertrendIndicator(period=7, multiplier=3).calculate(df_base)
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

        if direction == "CALL" and st_dir == 1 and st_dist_pct <= self.zone_pct * 1.5:
            entry = ltp
            sl = supertrend_value - atr * 0.3
            t1 = entry + atr * self.target_1_multiplier
            t2 = entry + atr * self.target_2_multiplier
            return EntryTrigger(
                trigger_type="ST_TOUCH", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                target_1=round(t1, 2), target_2=round(t2, 2), confidence="HIGH", status="TRIGGERED",
                detail=f"Touching Supertrend line ₹{supertrend_value:.1f}"
            )
        elif direction == "PUT" and st_dir == -1 and st_dist_pct <= self.zone_pct * 1.5:
            entry = ltp
            sl = supertrend_value + atr * 0.3
            t1 = entry - atr * self.target_1_multiplier
            t2 = entry - atr * self.target_2_multiplier
            return EntryTrigger(
                trigger_type="ST_TOUCH", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                target_1=round(t1, 2), target_2=round(t2, 2), confidence="HIGH", status="TRIGGERED",
                detail=f"Touching Supertrend line ₹{supertrend_value:.1f}"
            )

        return None
