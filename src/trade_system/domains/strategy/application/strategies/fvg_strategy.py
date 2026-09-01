"""
Fair Value Gap (FVG) Strategy & Imbalance Analysis Engine
=========================================================
Institutional Price Inefficiency Engine analyzing Daily timeframe candles:
1. BISI (Buy-Side Imbalance Sell-Side Inefficiency — Bullish FVG):
   - Candle 1 High < Candle 3 Low (Gap between C1 High and C3 Low)
   - Consequent Encroachment (CE): 50% midpoint of the imbalance zone
2. SIBI (Sell-Side Imbalance Buy-Side Inefficiency — Bearish FVG):
   - Candle 1 Low > Candle 3 High (Gap between C3 High and C1 Low)
   - Consequent Encroachment (CE): 50% midpoint
3. Mitigation & Retest Tracking:
   - Untested / Virgin FVGs acting as institutional magnets
   - Consequent Encroachment tests with price rejection (Taps)
   - Inversion FVGs (IFVG): Imbalances broken through that flip from resistance to support

Produces actionable FVG trade setups with exact Entry, Stop Loss, Target 1, Target 2, and RRR.
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
class DailyFairValueGap:
    """Represents a single Fair Value Gap on daily candles."""
    fvg_type: str                 # "BULLISH" (BISI) or "BEARISH" (SIBI)
    top: float                    # Upper price bound
    bottom: float                 # Lower price bound
    ce_level: float               # Consequent Encroachment (50% midpoint)
    bar_idx: int                  # Index of middle candle (Candle 2)
    creation_date: str            # Date of creation
    gap_size: float               # Absolute price gap size
    gap_size_pct: float           # Gap as percentage of stock price
    volume_ratio: float           # Impulse candle volume vs 20 SMA
    is_mitigated: bool = False    # True if price has entered the zone
    is_fully_filled: bool = False # True if price has completely crossed through
    tested_ce: bool = False       # True if price wicked into 50% CE level
    mitigation_date: Optional[str] = None
    inversion: bool = False       # True if flipped from support/resistance


@dataclass
class FvgTradeSetup:
    """Actionable trade setup generated from Daily FVG analysis."""
    symbol: str
    action: str                   # "BUY CALL" (Long) or "BUY PUT" (Short)
    direction: int                # 1 for Bullish, -1 for Bearish
    setup_type: str               # "FVG_CE_RETEST", "FVG_DISCOUNT_TAP", "INVERSION_FVG_SUPPORT", "FRESH_FVG_BREAKAWAY"
    confluence_score: float       # 0 - 100
    entry_price: float            # Recommended Entry price / zone
    stop_loss: float              # Invalidation level below FVG bottom / above FVG top
    target_1: float               # Target 1 (1:2 RRR)
    target_2: float               # Target 2 (Opposing liquidity / 1:3 RRR)
    risk_reward_ratio: float      # Calculated RRR
    fvg_top: float                # FVG upper bound
    fvg_bottom: float             # FVG lower bound
    consequent_encroachment: float# 50% CE level
    gap_size_pct: float           # Imbalance size %
    reasons: List[str] = field(default_factory=list)
    timestamp: str = ""
    spot_price: float = 0.0

    def format_summary(self) -> str:
        icon = "🟢" if self.direction == 1 else "🔴"
        sym_short = self.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
        return (
            f"{icon} <b>{sym_short}</b> | <b>{self.action}</b> [{self.setup_type}] (Score: {self.confluence_score:.0f}/100)\n"
            f"   FVG Zone: ₹{self.fvg_bottom:.2f} - ₹{self.fvg_top:.2f} | 50% CE: ₹{self.consequent_encroachment:.2f}\n"
            f"   Entry: ₹{self.entry_price:.2f} | SL: ₹{self.stop_loss:.2f} | T1: ₹{self.target_1:.2f} (1:{self.risk_reward_ratio:.1f} RRR)\n"
            f"   Institutional Rationale: {', '.join(self.reasons)}"
        )


class FairValueGapStrategy:
    """
    Fair Value Gap (FVG) Strategy Engine for Daily Timeframe Stocks.
    Scales dynamically to individual stock price and ATR volatility.
    """

    def __init__(
        self,
        min_gap_atr_mult: float = 0.25,
        lookback_bars: int = 40,
        min_rrr: float = 1.5,
    ) -> None:
        self.min_gap_atr_mult = min_gap_atr_mult
        self.lookback_bars = lookback_bars
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

    def detect_all_fvgs(self, df: pd.DataFrame) -> List[DailyFairValueGap]:
        """
        Scans candle history to detect all 3-candle Fair Value Gaps and tracks their mitigation.
        """
        fvgs: List[DailyFairValueGap] = []
        n = len(df)
        if n < 3:
            return fvgs

        clean_df = df.copy().sort_values("timestamp").reset_index(drop=True)
        if "vol_sma" not in clean_df.columns:
            clean_df["vol_sma"] = clean_df["volume"].rolling(20, min_periods=1).mean()

        atr = self.calculate_atr(clean_df, period=14)
        dates = clean_df["timestamp"].astype(str).values
        highs = clean_df["high"].values
        lows = clean_df["low"].values
        closes = clean_df["close"].values
        volumes = clean_df["volume"].values
        vol_smas = clean_df["vol_sma"].values
        atr_vals = atr.values

        for i in range(2, n):
            c1_high = highs[i - 2]
            c1_low = lows[i - 2]
            c2_vol = volumes[i - 1]
            c2_avg_v = vol_smas[i - 1] if vol_smas[i - 1] > 0 else 1.0
            v_ratio = c2_vol / c2_avg_v

            c3_high = highs[i]
            c3_low = lows[i]
            current_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else (highs[i] - lows[i])
            min_gap = current_atr * self.min_gap_atr_mult

            # 1. Bullish FVG (BISI): Candle 3 Low > Candle 1 High
            if c3_low > c1_high and (c3_low - c1_high) >= min_gap:
                gap_size = c3_low - c1_high
                gap_pct = (gap_size / closes[i]) * 100
                ce = (c3_low + c1_high) / 2.0

                fvg = DailyFairValueGap(
                    fvg_type="BULLISH",
                    top=round(c3_low, 2),
                    bottom=round(c1_high, 2),
                    ce_level=round(ce, 2),
                    bar_idx=i - 1,
                    creation_date=dates[i - 1],
                    gap_size=round(gap_size, 2),
                    gap_size_pct=round(gap_pct, 2),
                    volume_ratio=round(v_ratio, 2),
                    is_mitigated=False,
                    is_fully_filled=False,
                    tested_ce=False,
                )

                # Check forward price action to track mitigation
                for k in range(i + 1, n):
                    # Mitigated if low enters FVG
                    if lows[k] <= fvg.top:
                        fvg.is_mitigated = True
                        fvg.mitigation_date = dates[k]
                    if lows[k] <= fvg.ce_level:
                        fvg.tested_ce = True
                    # Fully filled / invalidated if close or low breaks below bottom
                    if lows[k] < fvg.bottom:
                        fvg.is_fully_filled = True
                        # If price closed below and stays below, it becomes an Inversion FVG
                        if closes[k] < fvg.bottom:
                            fvg.inversion = True
                        break

                fvgs.append(fvg)

            # 2. Bearish FVG (SIBI): Candle 1 Low > Candle 3 High
            elif c1_low > c3_high and (c1_low - c3_high) >= min_gap:
                gap_size = c1_low - c3_high
                gap_pct = (gap_size / closes[i]) * 100
                ce = (c1_low + c3_high) / 2.0

                fvg = DailyFairValueGap(
                    fvg_type="BEARISH",
                    top=round(c1_low, 2),
                    bottom=round(c3_high, 2),
                    ce_level=round(ce, 2),
                    bar_idx=i - 1,
                    creation_date=dates[i - 1],
                    gap_size=round(gap_size, 2),
                    gap_size_pct=round(gap_pct, 2),
                    volume_ratio=round(v_ratio, 2),
                    is_mitigated=False,
                    is_fully_filled=False,
                    tested_ce=False,
                )

                for k in range(i + 1, n):
                    if highs[k] >= fvg.bottom:
                        fvg.is_mitigated = True
                        fvg.mitigation_date = dates[k]
                    if highs[k] >= fvg.ce_level:
                        fvg.tested_ce = True
                    if highs[k] > fvg.top:
                        fvg.is_fully_filled = True
                        if closes[k] > fvg.top:
                            fvg.inversion = True
                        break

                fvgs.append(fvg)

        return fvgs

    def analyze_setup(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Optional[FvgTradeSetup]:
        """
        Analyzes the latest price action relative to active daily Fair Value Gaps.
        Identifies high-probability FVG Retest / Consequent Encroachment setups.
        """
        if df.empty or len(df) < 10:
            return None

        clean_df = df.copy().sort_values("timestamp").reset_index(drop=True)
        fvgs = self.detect_all_fvgs(clean_df)
        if not fvgs:
            return None

        atr_series = self.calculate_atr(clean_df, period=14)
        curr_bar = clean_df.iloc[-1]
        prev_bar = clean_df.iloc[-2] if len(clean_df) >= 2 else curr_bar

        current_close = float(curr_bar["close"])
        current_open = float(curr_bar["open"])
        current_high = float(curr_bar["high"])
        current_low = float(curr_bar["low"])
        current_atr = float(atr_series.iloc[-1])
        bar_date = str(curr_bar["timestamp"])

        # Filter recent active FVGs (created within last 30 bars and not fully violated)
        active_bull_fvgs = [f for f in fvgs if f.fvg_type == "BULLISH" and not f.is_fully_filled and (len(clean_df) - 1 - f.bar_idx) <= 30]
        active_bear_fvgs = [f for f in fvgs if f.fvg_type == "BEARISH" and not f.is_fully_filled and (len(clean_df) - 1 - f.bar_idx) <= 30]

        # ── 1. Evaluate Bullish FVG Retest (BUY CALL / LONG) ─────────────────
        for fvg in reversed(active_bull_fvgs):
            # Condition A: Price has entered the Bullish FVG zone or tapped 50% CE
            in_zone = fvg.bottom <= current_low <= (fvg.top + current_atr * 0.2)
            closed_above_bottom = current_close >= fvg.bottom
            bull_rejection = (current_close > current_open) or ((current_close - current_low) >= (current_high - current_low) * 0.35)

            if in_zone and closed_above_bottom and bull_rejection:
                score = 75.0
                reasons = [
                    f"Retest & Respect of Bullish FVG (₹{fvg.bottom:.2f} - ₹{fvg.top:.2f})",
                    f"Consequent Encroachment (50% CE) defended at ₹{fvg.ce_level:.2f}",
                ]

                if current_low <= fvg.ce_level:
                    score += 15.0
                    reasons.append("Exact 50% CE Imbalance Fill & Buyer Absorption")

                if fvg.volume_ratio >= 1.3:
                    score += 10.0
                    reasons.append(f"Originating Impulse Volume: {fvg.volume_ratio:.1f}x SMA")

                sl = fvg.bottom - (current_atr * 0.3)
                risk = max(current_close - sl, current_atr * 0.5)
                t1 = current_close + (risk * 2.0)
                t2 = current_close + (risk * 3.5)
                rrr = (t1 - current_close) / risk if risk > 0 else 2.0

                if rrr >= self.min_rrr:
                    return FvgTradeSetup(
                        symbol=symbol,
                        action="BUY CALL",
                        direction=1,
                        setup_type="FVG_CE_RETEST" if current_low <= fvg.ce_level else "FVG_DISCOUNT_TAP",
                        confluence_score=min(100.0, score),
                        entry_price=round(current_close, 2),
                        stop_loss=round(sl, 2),
                        target_1=round(t1, 2),
                        target_2=round(t2, 2),
                        risk_reward_ratio=round(rrr, 2),
                        fvg_top=fvg.top,
                        fvg_bottom=fvg.bottom,
                        consequent_encroachment=fvg.ce_level,
                        gap_size_pct=fvg.gap_size_pct,
                        reasons=reasons,
                        timestamp=bar_date,
                        spot_price=current_close
                    )

        # ── 2. Evaluate Bearish FVG Retest (BUY PUT / SHORT) ─────────────────
        for fvg in reversed(active_bear_fvgs):
            in_zone = (fvg.bottom - current_atr * 0.2) <= current_high <= fvg.top
            closed_below_top = current_close <= fvg.top
            bear_rejection = (current_close < current_open) or ((current_high - current_close) >= (current_high - current_low) * 0.35)

            if in_zone and closed_below_top and bear_rejection:
                score = 75.0
                reasons = [
                    f"Retest & Respect of Bearish FVG (₹{fvg.bottom:.2f} - ₹{fvg.top:.2f})",
                    f"Consequent Encroachment (50% CE) defended at ₹{fvg.ce_level:.2f}",
                ]

                if current_high >= fvg.ce_level:
                    score += 15.0
                    reasons.append("Exact 50% CE Imbalance Fill & Seller Rejection")

                if fvg.volume_ratio >= 1.3:
                    score += 10.0
                    reasons.append(f"Originating Impulse Volume: {fvg.volume_ratio:.1f}x SMA")

                sl = fvg.top + (current_atr * 0.3)
                risk = max(sl - current_close, current_atr * 0.5)
                t1 = current_close - (risk * 2.0)
                t2 = current_close - (risk * 3.5)
                rrr = (current_close - t1) / risk if risk > 0 else 2.0

                if rrr >= self.min_rrr:
                    return FvgTradeSetup(
                        symbol=symbol,
                        action="BUY PUT",
                        direction=-1,
                        setup_type="FVG_CE_RETEST" if current_high >= fvg.ce_level else "FVG_PREMIUM_TAP",
                        confluence_score=min(100.0, score),
                        entry_price=round(current_close, 2),
                        stop_loss=round(sl, 2),
                        target_1=round(t1, 2),
                        target_2=round(t2, 2),
                        risk_reward_ratio=round(rrr, 2),
                        fvg_top=fvg.top,
                        fvg_bottom=fvg.bottom,
                        consequent_encroachment=fvg.ce_level,
                        gap_size_pct=fvg.gap_size_pct,
                        reasons=reasons,
                        timestamp=bar_date,
                        spot_price=current_close
                    )

        return None
