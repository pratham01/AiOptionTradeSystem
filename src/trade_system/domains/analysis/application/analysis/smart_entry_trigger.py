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

from trade_system.domains.strategy.application.strategies.fibonacci_retracement import (
    FibonacciRetracementStrategy, FibonacciGrid, FibonacciSetup
)

LOGGER = logging.getLogger(__name__)


@dataclass
class EntryTrigger:
    """A specific entry opportunity with risk-defined levels."""
    trigger_type: str           # "FIB_GOLDEN_ZONE", "VWAP_PULLBACK", "ORB_BREAKOUT", "EMA20_PULLBACK", "ST_TOUCH"
    entry_price: float
    stop_loss: float
    target_1: float             # Target 1
    target_2: float             # Target 2 (extended target)
    confidence: str             # "HIGH", "MEDIUM"
    status: str                 # "TRIGGERED", "APPROACHING", "WAITING"
    detail: str = ""
    fib_grid: Optional[FibonacciGrid] = None
    trigger_time: str = "—"     # Exact time of the trigger bar, e.g. "09:45", "10:30"


def _extract_time_str(df: pd.DataFrame, target_date: Optional[date] = None) -> str:
    """Extracts formatted HH:MM time from dataframe timestamps."""
    if df.empty or "timestamp" not in df.columns:
        return "—"
    try:
        slice_df = df
        if target_date and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            sub = df[df["timestamp"].dt.date == target_date]
            if not sub.empty:
                slice_df = sub
        last_val = slice_df.iloc[-1]["timestamp"]
        if hasattr(last_val, "strftime"):
            return last_val.strftime("%H:%M")
        return str(last_val).split()[-1][:5]
    except Exception:
        return "—"


class SmartEntryTrigger:
    """
    Evaluates an EdgeScore stock and determines optimal entry opportunities across multiple strategies:
    1. Fibonacci Golden Pocket Retracement (50.0% - 61.8% Retracement)
    2. Session / Rolling VWAP Pullback
    3. 20 EMA Dynamic Pullback
    4. Range / ORB Breakout
    5. Supertrend Touch
    """

    def __init__(self, horizon: str = "INTRADAY") -> None:
        self.horizon = horizon.upper()
        self.fib_strategy = FibonacciRetracementStrategy(swing_lookback=25)

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
        Evaluate all entry triggers for the selected horizon and return the highest-priority trigger.
        If no specific trigger has fired yet, returns a planned trend-continuation pullback trigger
        with defined entry, stop-loss, and target levels in 'WAITING' status.
        """
        all_trigs = self.evaluate_all(
            direction=direction,
            ltp=ltp,
            atr=atr,
            df_base=df_base,
            target_date=target_date,
            vwap=vwap,
            supertrend_value=supertrend_value,
            df_15m=df_15m,
        )

        if all_trigs:
            triggers = list(all_trigs.values())
            status_priority = {"TRIGGERED": 0, "APPROACHING": 1, "WAITING": 2}
            triggers.sort(key=lambda t: (status_priority.get(t.status, 3), -{"HIGH": 1, "MEDIUM": 0}.get(t.confidence, -1)))
            return triggers[0]

        if df_base is None or df_base.empty:
            return None

        # If no specific trigger conditions are met, create a baseline planned limit entry
        if ltp > 0 and atr > 0:
            if direction == "CALL":
                entry = round(ltp - atr * 0.25, 2)
                sl = round(entry - atr * self.sl_multiplier, 2)
                t1 = round(entry + atr * self.target_1_multiplier, 2)
                t2 = round(entry + atr * self.target_2_multiplier, 2)
            else:
                entry = round(ltp + atr * 0.25, 2)
                sl = round(entry + atr * self.sl_multiplier, 2)
                t1 = round(entry - atr * self.target_1_multiplier, 2)
                t2 = round(entry - atr * self.target_2_multiplier, 2)

            time_str = "⏳ Pending"
            raw_time = _extract_time_str(df_base, target_date)
            if raw_time and raw_time != "—":
                time_str = f"⏳ Pending ({raw_time})"

            return EntryTrigger(
                trigger_type="TREND_CONTINUATION",
                entry_price=entry,
                stop_loss=sl,
                target_1=t1,
                target_2=t2,
                confidence="MEDIUM",
                status="WAITING",
                detail=f"Monitoring for {direction} pullback to optimal entry zone",
                trigger_time=time_str
            )

        return None

    def evaluate_all(
        self,
        direction: str,
        ltp: float,
        atr: float,
        df_base: pd.DataFrame | None = None,
        target_date: date | None = None,
        vwap: float = 0.0,
        supertrend_value: float = 0.0,
        df_15m: pd.DataFrame | None = None,
    ) -> dict[str, EntryTrigger]:
        """
        Evaluates and returns ALL entry strategies for comparative analysis.
        """
        if df_base is None:
            df_base = df_15m

        if atr <= 0 or ltp <= 0 or df_base is None or df_base.empty:
            return {}

        results: dict[str, EntryTrigger] = {}

        # 1. Fibonacci Golden Pocket Retracement
        fib_trig = self._check_fibonacci_retracement(direction, ltp, atr, df_base, target_date)
        if fib_trig:
            results["FIBONACCI"] = fib_trig

        # 2. VWAP Pullback Trigger
        vwap_trig = self._check_vwap_pullback(direction, ltp, atr, df_base, target_date, vwap)
        if vwap_trig:
            results["VWAP"] = vwap_trig

        # 3. 20 EMA Dynamic Pullback
        ema_trig = self._check_ema_pullback(direction, ltp, atr, df_base)
        if ema_trig:
            results["EMA20"] = ema_trig

        # 4. Breakout Trigger (ORB or Range Breakout)
        breakout_trig = self._check_breakout(direction, ltp, atr, df_base, target_date)
        if breakout_trig:
            results["BREAKOUT"] = breakout_trig

        # 5. Supertrend Touch Trigger
        st_trig = self._check_supertrend_touch(direction, ltp, atr, df_base, supertrend_value)
        if st_trig:
            results["SUPERTREND"] = st_trig

        return results

    def _check_fibonacci_retracement(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame, target_date: date | None
    ) -> Optional[EntryTrigger]:
        """Fibonacci Golden Pocket: Evaluates 50.0% - 61.8% pullback zone from recent impulse swing."""
        try:
            df_slice = df_base.copy()
            if target_date and "timestamp" in df_slice.columns:
                if pd.api.types.is_datetime64_any_dtype(df_slice["timestamp"]):
                    df_slice = df_slice[df_slice["timestamp"].dt.date <= target_date]
            
            fib_setup = self.fib_strategy.evaluate_setup(
                df=df_slice,
                symbol="MOMENTUM_STOCK",
                direction=direction,
                atr=atr
            )

            if not fib_setup:
                return None

            conf = "HIGH" if fib_setup.status == "TRIGGERED" else "MEDIUM"
            time_str = _extract_time_str(df_slice, target_date)
            return EntryTrigger(
                trigger_type="FIB_GOLDEN_ZONE",
                entry_price=fib_setup.entry_price,
                stop_loss=fib_setup.stop_loss,
                target_1=fib_setup.target_1,
                target_2=fib_setup.target_2,
                confidence=conf,
                status=fib_setup.status,
                detail=fib_setup.reasons[0] if fib_setup.reasons else "Fibonacci Golden Pocket",
                fib_grid=fib_setup.grid,
                trigger_time=time_str
            )
        except Exception as exc:
            LOGGER.debug("Fibonacci trigger check error: %s", exc)
            return None

    def _check_ema_pullback(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame
    ) -> Optional[EntryTrigger]:
        """20 EMA Dynamic Trend Pullback."""
        try:
            if len(df_base) < 20:
                return None
            
            close_series = df_base["close"]
            ema20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
            dist = (ltp - ema20) / ema20
            time_str = _extract_time_str(df_base)

            if direction == "CALL":
                if abs(dist) <= self.zone_pct:
                    return EntryTrigger(
                        trigger_type="EMA20_PULLBACK",
                        entry_price=round(ltp, 2),
                        stop_loss=round(ema20 - atr * self.sl_multiplier, 2),
                        target_1=round(ltp + atr * self.target_1_multiplier, 2),
                        target_2=round(ltp + atr * self.target_2_multiplier, 2),
                        confidence="HIGH",
                        status="TRIGGERED",
                        detail=f"At 20 EMA ₹{ema20:.1f} (Trend Support Pullback)",
                        trigger_time=time_str
                    )
                elif 0 < dist <= self.zone_pct * 2.5:
                    return EntryTrigger(
                        trigger_type="EMA20_PULLBACK",
                        entry_price=round(ema20, 2),
                        stop_loss=round(ema20 - atr * self.sl_multiplier, 2),
                        target_1=round(ema20 + atr * self.target_1_multiplier, 2),
                        target_2=round(ema20 + atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM",
                        status="APPROACHING",
                        detail=f"Above 20 EMA ₹{ema20:.1f}, approaching support",
                        trigger_time=time_str
                    )
            elif direction == "PUT":
                if abs(dist) <= self.zone_pct:
                    return EntryTrigger(
                        trigger_type="EMA20_PULLBACK",
                        entry_price=round(ltp, 2),
                        stop_loss=round(ema20 + atr * self.sl_multiplier, 2),
                        target_1=round(ltp - atr * self.target_1_multiplier, 2),
                        target_2=round(ltp - atr * self.target_2_multiplier, 2),
                        confidence="HIGH",
                        status="TRIGGERED",
                        detail=f"At 20 EMA ₹{ema20:.1f} (Trend Resistance Pullback)",
                        trigger_time=time_str
                    )
                elif -self.zone_pct * 2.5 <= dist < 0:
                    return EntryTrigger(
                        trigger_type="EMA20_PULLBACK",
                        entry_price=round(ema20, 2),
                        stop_loss=round(ema20 + atr * self.sl_multiplier, 2),
                        target_1=round(ema20 - atr * self.target_1_multiplier, 2),
                        target_2=round(ema20 - atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM",
                        status="APPROACHING",
                        detail=f"Below 20 EMA ₹{ema20:.1f}, approaching resistance",
                        trigger_time=time_str
                    )
            return None
        except Exception:
            return None

    # ── Trigger Implementations ────────────────────────────────────────────────

    def _check_vwap_pullback(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame, target_date: date, vwap: float
    ) -> Optional[EntryTrigger]:
        """VWAP Pullback: Price pulls back to the session or rolling VWAP and bounces."""
        if vwap <= 0:
            try:
                if self.horizon == "INTRADAY":
                    from trade_system.domains.strategy.application.indicators.vwap import VWAPIndicator
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

        time_str = _extract_time_str(df_base, target_date)
        if direction == "CALL":
            if abs(vwap_dist) <= self.zone_pct:
                entry = ltp
                sl = vwap - atr * self.sl_multiplier
                t1 = entry + atr * self.target_1_multiplier
                t2 = entry + atr * self.target_2_multiplier
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                    target_1=round(t1, 2), target_2=round(t2, 2), confidence="HIGH", status="TRIGGERED",
                    detail=f"Price near VWAP ₹{vwap:.1f} (pullback zone)",
                    trigger_time=time_str
                )
            elif 0 < vwap_dist <= self.zone_pct * 2.5:
                entry = vwap
                sl = vwap - atr * self.sl_multiplier
                t1 = entry + atr * self.target_1_multiplier
                t2 = entry + atr * self.target_2_multiplier
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                    target_1=round(t1, 2), target_2=round(t2, 2), confidence="MEDIUM", status="APPROACHING",
                    detail=f"Above VWAP ₹{vwap:.1f}, waiting for pullback",
                    trigger_time=time_str
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
                    detail=f"Price near VWAP ₹{vwap:.1f} (rejection zone)",
                    trigger_time=time_str
                )
            elif -self.zone_pct * 2.5 <= vwap_dist < 0:
                entry = vwap
                sl = vwap + atr * self.sl_multiplier
                t1 = entry - atr * self.target_1_multiplier
                t2 = entry - atr * self.target_2_multiplier
                return EntryTrigger(
                    trigger_type="VWAP_PULLBACK", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                    target_1=round(t1, 2), target_2=round(t2, 2), confidence="MEDIUM", status="APPROACHING",
                    detail=f"Below VWAP ₹{vwap:.1f}, waiting for pullback bounce",
                    trigger_time=time_str
                )

        return None

    def _check_breakout(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame, target_date: date
    ) -> Optional[EntryTrigger]:
        """Breakout/Breakdown of range boundaries (Intraday ORB or Swing Prev High/Low)."""
        time_str = _extract_time_str(df_base, target_date)
        
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
                # Locate earliest bar breaking orb_high
                exact_time = time_str
                for b_idx in range(2, len(today_df)):
                    bar = today_df.iloc[b_idx]
                    if float(bar["high"]) > orb_high:
                        exact_time = bar["timestamp"].strftime("%H:%M") if hasattr(bar["timestamp"], "strftime") else str(bar["timestamp"]).split()[-1][:5]
                        break
                return EntryTrigger(
                    trigger_type="ORB_BREAKOUT", entry_price=round(ltp, 2), stop_loss=round(orb_high - atr * 0.5, 2),
                    target_1=round(ltp + atr * self.target_1_multiplier, 2), target_2=round(ltp + atr * self.target_2_multiplier, 2),
                    confidence="HIGH" if dist <= 0.005 else "MEDIUM", status="TRIGGERED",
                    detail=f"ORB High ₹{orb_high:.1f} broken",
                    trigger_time=exact_time
                )
            elif direction == "PUT" and ltp < orb_low:
                dist = (orb_low - ltp) / orb_low
                if dist > 0.015: return None
                # Locate earliest bar breaking orb_low
                exact_time = time_str
                for b_idx in range(2, len(today_df)):
                    bar = today_df.iloc[b_idx]
                    if float(bar["low"]) < orb_low:
                        exact_time = bar["timestamp"].strftime("%H:%M") if hasattr(bar["timestamp"], "strftime") else str(bar["timestamp"]).split()[-1][:5]
                        break
                return EntryTrigger(
                    trigger_type="ORB_BREAKOUT", entry_price=round(ltp, 2), stop_loss=round(orb_low + atr * 0.5, 2),
                    target_1=round(ltp - atr * self.target_1_multiplier, 2), target_2=round(ltp - atr * self.target_2_multiplier, 2),
                    confidence="HIGH" if dist <= 0.005 else "MEDIUM", status="TRIGGERED",
                    detail=f"ORB Low ₹{orb_low:.1f} broken",
                    trigger_time=exact_time
                )
            elif direction == "CALL" and ltp < orb_high:
                dist = (orb_high - ltp) / orb_high
                if dist <= 0.005:
                    return EntryTrigger(
                        trigger_type="ORB_BREAKOUT", entry_price=round(orb_high, 2), stop_loss=round(orb_high - atr * 0.5, 2),
                        target_1=round(orb_high + atr * self.target_1_multiplier, 2), target_2=round(orb_high + atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching ORB High ₹{orb_high:.1f}",
                        trigger_time=time_str
                    )
            elif direction == "PUT" and ltp > orb_low:
                dist = (ltp - orb_low) / orb_low
                if dist <= 0.005:
                    return EntryTrigger(
                        trigger_type="ORB_BREAKOUT", entry_price=round(orb_low, 2), stop_loss=round(orb_low + atr * 0.5, 2),
                        target_1=round(orb_low - atr * self.target_1_multiplier, 2), target_2=round(orb_low - atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching ORB Low ₹{orb_low:.1f}",
                        trigger_time=time_str
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
                    detail=f"Broken {ref_label} High ₹{prev_high:.1f}",
                    trigger_time=time_str
                )
            elif direction == "PUT" and ltp < prev_low:
                dist = (prev_low - ltp) / prev_low
                if dist > self.zone_pct * 3: return None
                return EntryTrigger(
                    trigger_type="RANGE_BREAKOUT", entry_price=round(ltp, 2), stop_loss=round(prev_low + atr * 0.5, 2),
                    target_1=round(ltp - atr * self.target_1_multiplier, 2), target_2=round(ltp - atr * self.target_2_multiplier, 2),
                    confidence="HIGH" if dist <= self.zone_pct else "MEDIUM", status="TRIGGERED",
                    detail=f"Broken {ref_label} Low ₹{prev_low:.1f}",
                    trigger_time=time_str
                )
            elif direction == "CALL" and ltp < prev_high:
                dist = (prev_high - ltp) / prev_high
                if dist <= self.zone_pct:
                    return EntryTrigger(
                        trigger_type="RANGE_BREAKOUT", entry_price=round(prev_high, 2), stop_loss=round(prev_high - atr * 0.5, 2),
                        target_1=round(prev_high + atr * self.target_1_multiplier, 2), target_2=round(prev_high + atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching {ref_label} High ₹{prev_high:.1f}",
                        trigger_time=time_str
                    )
            elif direction == "PUT" and ltp > prev_low:
                dist = (ltp - prev_low) / prev_low
                if dist <= self.zone_pct:
                    return EntryTrigger(
                        trigger_type="RANGE_BREAKOUT", entry_price=round(prev_low, 2), stop_loss=round(prev_low + atr * 0.5, 2),
                        target_1=round(prev_low - atr * self.target_1_multiplier, 2), target_2=round(prev_low - atr * self.target_2_multiplier, 2),
                        confidence="MEDIUM", status="APPROACHING", detail=f"Approaching {ref_label} Low ₹{prev_low:.1f}",
                        trigger_time=time_str
                    )
            return None

    def _check_supertrend_touch(
        self, direction: str, ltp: float, atr: float,
        df_base: pd.DataFrame, supertrend_value: float
    ) -> Optional[EntryTrigger]:
        """Supertrend Touch: Price touches the Supertrend line and bounces."""
        if supertrend_value <= 0:
            try:
                from trade_system.domains.strategy.application.indicators.supertrend import SupertrendIndicator
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
        time_str = _extract_time_str(df_base)

        if direction == "CALL" and st_dir == 1 and st_dist_pct <= self.zone_pct * 1.5:
            entry = ltp
            sl = supertrend_value - atr * 0.3
            t1 = entry + atr * self.target_1_multiplier
            t2 = entry + atr * self.target_2_multiplier
            return EntryTrigger(
                trigger_type="ST_TOUCH", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                target_1=round(t1, 2), target_2=round(t2, 2), confidence="HIGH", status="TRIGGERED",
                detail=f"Touching Supertrend line ₹{supertrend_value:.1f}",
                trigger_time=time_str
            )
        elif direction == "PUT" and st_dir == -1 and st_dist_pct <= self.zone_pct * 1.5:
            entry = ltp
            sl = supertrend_value + atr * 0.3
            t1 = entry - atr * self.target_1_multiplier
            t2 = entry - atr * self.target_2_multiplier
            return EntryTrigger(
                trigger_type="ST_TOUCH", entry_price=round(entry, 2), stop_loss=round(sl, 2),
                target_1=round(t1, 2), target_2=round(t2, 2), confidence="HIGH", status="TRIGGERED",
                detail=f"Touching Supertrend line ₹{supertrend_value:.1f}",
                trigger_time=time_str
            )

        return None
