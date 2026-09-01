"""
Wyckoff Method & Volume Spread Analysis (VSA) Strategy Engine
=============================================================
Institutional Trading Methodology based on:
1. Richard Wyckoff's Accumulation & Distribution Schematics:
   - Phase C Spring (Terminal Shakeout below Trading Range Support)
   - Phase C UTAD (Upthrust After Distribution above Trading Range Resistance)
   - Sign of Strength (SOS) & Sign of Weakness (SOW) breakout thrusts
   - Last Point of Support (LPS) & Last Point of Supply (LPSY) retests
2. Volume Spread Analysis (VSA):
   - Stopping Volume: Ultra-high volume absorption halting a trend
   - No Supply / No Demand Tests: Low volume pullbacks confirming exhaustion
   - Effort vs. Result Anomalies: High volume on narrow spread indicating passive absorption

Produces institutional trade setups with exact Entry, Stop Loss, Targets, and RRR.
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
class WyckoffSetup:
    """Represents a validated Wyckoff / VSA trade setup."""
    symbol: str
    action: str                   # "BUY CALL" (LONG) or "BUY PUT" (SHORT)
    direction: int                # 1 for Bullish, -1 for Bearish
    pattern_type: str             # "WYCKOFF_SPRING", "WYCKOFF_UTAD", "WYCKOFF_SOS", "VSA_STOPPING_VOLUME", "NO_SUPPLY_TEST"
    phase: str                    # "Phase C (Shakeout)", "Phase D (Markup)", "Phase C (Distribution)"
    confluence_score: float       # 0 - 100
    entry_price: float            # Recommended entry level
    stop_loss: float              # Invalidation level
    target_1: float               # Initial Target (1:2 RRR)
    target_2: float               # Extended Target (Opposite Range Bound / 1:3 RRR)
    risk_reward_ratio: float      # Calculated RRR
    trading_range_high: float     # Resistance of accumulation/distribution box
    trading_range_low: float      # Support of accumulation/distribution box
    volume_surge_ratio: float     # Volume vs 20 SMA
    reasons: List[str] = field(default_factory=list)
    timestamp: str = ""
    spot_price: float = 0.0

    def format_summary(self) -> str:
        icon = "🟢" if self.direction == 1 else "🔴"
        sym_short = self.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
        return (
            f"{icon} <b>{sym_short}</b> | <b>{self.action}</b> [{self.pattern_type}] (Score: {self.confluence_score:.0f}/100)\n"
            f"   Phase: <code>{self.phase}</code> | Range: ₹{self.trading_range_low:.2f} - ₹{self.trading_range_high:.2f}\n"
            f"   Entry: ₹{self.entry_price:.2f} | SL: ₹{self.stop_loss:.2f} | T1: ₹{self.target_1:.2f} (1:{self.risk_reward_ratio:.1f} RRR)\n"
            f"   Institutional Rationale: {', '.join(self.reasons)}"
        )


class WyckoffVsaStrategy:
    """
    Wyckoff Method & Volume Spread Analysis Strategy Engine.
    Scales dynamically to individual stock price and ATR volatility.
    """

    def __init__(
        self,
        range_lookback: int = 20,
        volume_ma_period: int = 20,
        spring_penetration_max_pct: float = 3.5,
        min_rrr: float = 1.5,
    ) -> None:
        self.range_lookback = range_lookback
        self.volume_ma_period = volume_ma_period
        self.spring_penetration_max_pct = spring_penetration_max_pct
        self.min_rrr = min_rrr

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates Average True Range (ATR)."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    def analyze_setup(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Optional[WyckoffSetup]:
        """
        Analyzes a candle sequence to detect Wyckoff Accumulation/Distribution or VSA patterns.
        """
        if df.empty or len(df) < (self.range_lookback + 10):
            return None

        clean_df = df.copy().sort_values("timestamp").reset_index(drop=True)
        clean_df["vol_sma"] = clean_df["volume"].rolling(self.volume_ma_period, min_periods=1).mean()
        clean_df["spread"] = clean_df["high"] - clean_df["low"]
        clean_df["spread_sma"] = clean_df["spread"].rolling(self.volume_ma_period, min_periods=1).mean()
        clean_df["atr"] = self.calculate_atr(clean_df, period=14)

        n = len(clean_df)
        curr = clean_df.iloc[-1]
        prev = clean_df.iloc[-2]
        
        current_close = float(curr["close"])
        current_open = float(curr["open"])
        current_high = float(curr["high"])
        current_low = float(curr["low"])
        current_vol = float(curr["volume"])
        current_spread = float(curr["spread"])
        current_atr = float(curr["atr"])
        avg_vol = float(curr["vol_sma"])
        avg_spread = float(curr["spread_sma"])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        bar_date = str(curr["timestamp"])

        # ── 1. Define Trading Range (TR) from prior consolidation ───────────
        # Exclude last 3 bars to determine established Range High (TRH) and Range Low (TRL)
        range_window = clean_df.iloc[-self.range_lookback - 3 : -3]
        if range_window.empty:
            return None

        tr_high = float(range_window["high"].max())
        tr_low = float(range_window["low"].min())
        tr_range = tr_high - tr_low

        if tr_range <= 0:
            return None

        # ── 2. Detect Wyckoff Spring (Phase C Accumulation) ───────────────────
        # Criteria:
        # a) Current or previous candle poked below TR Low (liquidity sweep of retail stops).
        # b) Candle closed back ABOVE TR Low with bullish rejection wick (lower wick > 40% of spread).
        # c) Penetration is not a full breakdown (< spring_penetration_max_pct).
        # d) Volume can be low (No Supply Test) or high (Stopping Volume absorption).
        is_spring = False
        spring_score = 0.0
        spring_reasons = []

        lower_wick = min(current_open, current_close) - current_low
        has_bull_rejection = lower_wick >= (current_spread * 0.35) or current_close > current_open

        if current_low < tr_low and current_close >= tr_low and has_bull_rejection:
            penetration_pct = ((tr_low - current_low) / tr_low) * 100
            if penetration_pct <= self.spring_penetration_max_pct:
                is_spring = True
                spring_score = 75.0
                spring_reasons.append(f"Wyckoff Spring: False breakdown of TRL (₹{tr_low:.2f}) with swift buyer reclaim")

                if vol_ratio >= 1.3:
                    spring_score += 15.0
                    spring_reasons.append(f"Stopping Volume absorption confirmed ({vol_ratio:.1f}x avg volume)")
                elif vol_ratio < 0.8:
                    spring_score += 10.0
                    spring_reasons.append("No Supply Test: Sellers exhausted on low volume")

        # Also check previous candle spring with current candle confirming (LPS - Last Point of Support)
        elif float(prev["low"]) < tr_low and float(prev["close"]) >= tr_low and current_close > float(prev["high"]):
            is_spring = True
            spring_score = 80.0
            spring_reasons.append(f"Wyckoff Phase C Spring confirmed with Phase D markup follow-through")

        if is_spring:
            sl = min(current_low, float(prev["low"])) - (current_atr * 0.3)
            risk = max(current_close - sl, current_atr * 0.5)
            t1 = current_close + (risk * 2.0)
            t2 = tr_high + (current_atr * 0.5)
            rrr = (t1 - current_close) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr:
                return WyckoffSetup(
                    symbol=symbol,
                    action="BUY CALL",
                    direction=1,
                    pattern_type="WYCKOFF_SPRING_ACCUMULATION",
                    phase="Phase C (Shakeout)",
                    confluence_score=min(100.0, spring_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(rrr, 2),
                    trading_range_high=round(tr_high, 2),
                    trading_range_low=round(tr_low, 2),
                    volume_surge_ratio=round(vol_ratio, 2),
                    reasons=spring_reasons,
                    timestamp=bar_date,
                    spot_price=current_close
                )

        # ── 3. Detect Wyckoff UTAD (Upthrust After Distribution - Phase C) ────
        # Criteria:
        # a) Current or previous candle spiked above TR High.
        # b) Candle closed back BELOW TR High with bearish rejection wick (upper wick > 40% of spread).
        # c) High volume indicating heavy institutional distribution / selling into retail buy stops.
        is_utad = False
        utad_score = 0.0
        utad_reasons = []

        upper_wick = current_high - max(current_open, current_close)
        has_bear_rejection = upper_wick >= (current_spread * 0.35) or current_close < current_open

        if current_high > tr_high and current_close <= tr_high and has_bear_rejection:
            penetration_pct = ((current_high - tr_high) / tr_high) * 100
            if penetration_pct <= self.spring_penetration_max_pct:
                is_utad = True
                utad_score = 75.0
                utad_reasons.append(f"Wyckoff UTAD: Upthrust trap above TRH (₹{tr_high:.2f}) with aggressive seller rejection")

                if vol_ratio >= 1.3:
                    utad_score += 15.0
                    utad_reasons.append(f"High-Volume Distribution / Supply Dump ({vol_ratio:.1f}x volume)")

        elif float(prev["high"]) > tr_high and float(prev["close"]) <= tr_high and current_close < float(prev["low"]):
            is_utad = True
            utad_score = 80.0
            utad_reasons.append(f"Wyckoff Phase C UTAD confirmed with Phase D markdown follow-through")

        if is_utad:
            sl = max(current_high, float(prev["high"])) + (current_atr * 0.3)
            risk = max(sl - current_close, current_atr * 0.5)
            t1 = current_close - (risk * 2.0)
            t2 = tr_low - (current_atr * 0.5)
            rrr = (current_close - t1) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr:
                return WyckoffSetup(
                    symbol=symbol,
                    action="BUY PUT",
                    direction=-1,
                    pattern_type="WYCKOFF_UTAD_DISTRIBUTION",
                    phase="Phase C (Distribution)",
                    confluence_score=min(100.0, utad_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(rrr, 2),
                    trading_range_high=round(tr_high, 2),
                    trading_range_low=round(tr_low, 2),
                    volume_surge_ratio=round(vol_ratio, 2),
                    reasons=utad_reasons,
                    timestamp=bar_date,
                    spot_price=current_close
                )

        # ── 4. Detect Sign of Strength (SOS) Breakout (Phase D Markup) ────────
        # Breakout candle closing decisively above TR High with Volume Expansion > 1.5x
        if current_close > tr_high and current_close > current_open and vol_ratio >= 1.4:
            sos_score = 70.0 + min(20.0, (vol_ratio - 1.0) * 15.0)
            sl = tr_high - (current_atr * 0.5)
            risk = max(current_close - sl, current_atr * 0.5)
            t1 = current_close + (risk * 2.0)
            t2 = current_close + (risk * 3.5)
            rrr = (t1 - current_close) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr:
                return WyckoffSetup(
                    symbol=symbol,
                    action="BUY CALL",
                    direction=1,
                    pattern_type="WYCKOFF_SOS_EXPANSION",
                    phase="Phase D (Sign of Strength)",
                    confluence_score=min(100.0, sos_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(rrr, 2),
                    trading_range_high=round(tr_high, 2),
                    trading_range_low=round(tr_low, 2),
                    volume_surge_ratio=round(vol_ratio, 2),
                    reasons=[
                        f"Sign of Strength (SOS): Decisive close above TRH (₹{tr_high:.2f})",
                        f"Institutional Expansion Volume: {vol_ratio:.1f}x vs 20 SMA",
                    ],
                    timestamp=bar_date,
                    spot_price=current_close
                )

        # ── 5. Detect Sign of Weakness (SOW) Breakdown (Phase D Markdown) ─────
        # Breakdown candle closing decisively below TR Low with Volume Expansion > 1.4x
        if current_close < tr_low and current_close < current_open and vol_ratio >= 1.4:
            sow_score = 70.0 + min(20.0, (vol_ratio - 1.0) * 15.0)
            sl = tr_low + (current_atr * 0.5)
            risk = max(sl - current_close, current_atr * 0.5)
            t1 = current_close - (risk * 2.0)
            t2 = current_close - (risk * 3.5)
            rrr = (current_close - t1) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr:
                return WyckoffSetup(
                    symbol=symbol,
                    action="BUY PUT",
                    direction=-1,
                    pattern_type="WYCKOFF_SOW_BREAKDOWN",
                    phase="Phase D (Sign of Weakness)",
                    confluence_score=min(100.0, sow_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(rrr, 2),
                    trading_range_high=round(tr_high, 2),
                    trading_range_low=round(tr_low, 2),
                    volume_surge_ratio=round(vol_ratio, 2),
                    reasons=[
                        f"Sign of Weakness (SOW): Decisive close below TRL (₹{tr_low:.2f})",
                        f"Institutional Distribution Volume: {vol_ratio:.1f}x vs 20 SMA",
                    ],
                    timestamp=bar_date,
                    spot_price=current_close
                )

        return None
