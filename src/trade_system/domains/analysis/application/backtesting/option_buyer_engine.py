"""
Option Buyer Protocol & Dynamic Quantitative Strategy Engine.

Engineered specifically for Indian Index and Stock Option Buyers (Nifty, BankNifty, Sensex, F&O Universe).
Enforces:
1. Time-of-Day Gates: Exploits Morning Opening Momentum (09:30-11:15 IST) and Afternoon Breakout /
   Gamma Squeeze (13:15-15:00 IST), while strictly VETOING the Midday Churn Trap (11:15-13:15 IST)
   where option writers harvest theta.
2. Dynamic Support & Resistance (S/R) Zones: Prevents buying Calls directly under a Resistance ceiling
   or buying Puts directly above a Support floor.
3. Market Regime Adaptation: Vetoes trend-following breakout attempts during Choppy Rangebound markets.
4. India VIX & Volatility Dynamics: Protects against violent IV crush (falling VIX) and confirms
   explosive expansion impulses.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from trade_system.domains.analysis.application.backtesting.modular_backtester import (
    TradeSide,
    EntrySignal,
)

LOGGER = logging.getLogger(__name__)


class OptionBuyerFilter:
    """
    Quantitative Gatekeeper for Option Buyers.
    Filters out high-failure trades to preserve capital and maximize expectancy.
    """

    # Prime Option Buyer Windows (IST)
    MORNING_START = dtime(9, 30)
    MORNING_END = dtime(11, 15)
    AFTERNOON_START = dtime(13, 15)
    AFTERNOON_END = dtime(15, 0)
    MIDDAY_START = dtime(11, 15)
    MIDDAY_END = dtime(13, 15)

    @classmethod
    def is_prime_time_window(cls, timestamp: Any) -> Tuple[bool, str]:
        """
        Determines if current candle falls within favorable Option Buying windows.
        Option buyers suffer severe theta decay during midday consolidation (11:15-13:15 IST).
        """
        if timestamp is None:
            return True, "NO_TIMESTAMP"

        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", ""))
            except Exception:
                return True, "PARSING_FALLBACK"
        elif isinstance(timestamp, (datetime, pd.Timestamp)):
            dt = timestamp
        else:
            return True, "UNKNOWN_TYPE"

        t = dt.time()

        # Morning Opening Impulse Window
        if cls.MORNING_START <= t <= cls.MORNING_END:
            return True, "MORNING_IMPULSE_WINDOW"

        # Afternoon Gamma Blast & Breakout Window
        if cls.AFTERNOON_START <= t <= cls.AFTERNOON_END:
            return True, "AFTERNOON_GAMMA_WINDOW"

        # The Midday Churn Trap (Theta Harvesting by Sellers)
        if cls.MIDDAY_START < t < cls.AFTERNOON_START:
            return False, "MIDDAY_CHURN_TRAP_VETO"

        # Post-15:00 Close Window
        if t > cls.AFTERNOON_END:
            return False, "LATE_DAY_LOCKOUT_VETO"

        # Pre-09:30 Opening Unsettled Window
        return False, "OPENING_SETTLEMENT_VETO"

    @classmethod
    def has_sr_clearance(
        cls,
        side: TradeSide,
        price: float,
        nearest_res: float,
        nearest_sup: float,
        min_room_pct: float = 0.35,
    ) -> Tuple[bool, str]:
        """
        Ensures the trade has adequate 'room to run' before hitting an opposing institutional zone:
        - Long (Call Buy): Must NOT enter directly under Resistance ceiling.
        - Short (Put Buy): Must NOT enter directly above Support floor.
        """
        if price <= 0:
            return True, "OK"

        if side == TradeSide.LONG:
            dist_res_pct = ((nearest_res - price) / price) * 100.0
            if 0 < dist_res_pct < min_room_pct:
                return False, f"RESISTANCE_CEILING_VETO (Room {dist_res_pct:.2f}% < {min_room_pct}%)"
        elif side == TradeSide.SHORT:
            dist_sup_pct = ((price - nearest_sup) / price) * 100.0
            if 0 < dist_sup_pct < min_room_pct:
                return False, f"SUPPORT_FLOOR_VETO (Room {dist_sup_pct:.2f}% < {min_room_pct}%)"

        return True, "SR_CLEARANCE_OK"

    @classmethod
    def is_regime_favorable(cls, regime: str, strategy_style: str = "TREND") -> Tuple[bool, str]:
        """
        Vetoes trend breakout entries during choppy consolidation regimes.
        """
        if strategy_style == "TREND" and regime == "CHOPPY_RANGEBOUND":
            return False, "CHOPPY_REGIME_VETO"
        return True, "REGIME_FAVORABLE"

    @classmethod
    def is_vix_favorable(cls, side: TradeSide, vix_trend: str, vix_regime: str) -> Tuple[bool, str]:
        """
        Guards against IV crush and favorable volatility tailwinds.
        If VIX is crashing violently, option premiums lose value rapidly even if price moves.
        """
        if vix_trend == "CRUSHING":
            return False, "IV_CRUSH_HAZARD_VETO"
        return True, "VIX_FAVORABLE"

    @classmethod
    def is_200dma_aligned(
        cls, side: TradeSide, close: float, sma_200: float
    ) -> Tuple[bool, str]:
        """
        Paul Tudor Jones 200-DMA Trend Gate.
        The most powerful single filter used by institutional traders:
        - Long (Call): Only when close > SMA(200) → trading WITH the macro trend
        - Short (Put): Only when close < SMA(200) → trading WITH the macro trend
        Never fight the primary trend.
        """
        if sma_200 <= 0 or np.isnan(sma_200):
            return True, "SMA200_UNAVAILABLE"

        if side == TradeSide.LONG and close < sma_200:
            return False, f"200DMA_TREND_VETO_LONG (Close {close:.2f} < SMA200 {sma_200:.2f})"
        if side == TradeSide.SHORT and close > sma_200:
            return False, f"200DMA_TREND_VETO_SHORT (Close {close:.2f} > SMA200 {sma_200:.2f})"

        return True, "200DMA_TREND_ALIGNED"

    @classmethod
    def is_iv_rank_favorable(
        cls, iv_rank: float, trade_type: str = "BUY"
    ) -> Tuple[bool, str]:
        """
        Tom Sosnoff (tastytrade) / Sheldon Natenberg IV Rank Gate.
        Prevents buying expensive options that will suffer IV crush:
        - IV Rank < 40: Favorable for option BUYING (options are cheap)
        - IV Rank 40-60: Neutral zone (use spreads ideally)
        - IV Rank > 60: VETO for buying (options are expensive, IV crush likely)
        """
        if np.isnan(iv_rank):
            return True, "IV_RANK_UNAVAILABLE"

        if trade_type == "BUY":
            if iv_rank > 60.0:
                return False, f"IV_RANK_EXPENSIVE_VETO ({iv_rank:.1f} > 60)"
            if iv_rank > 40.0:
                return True, f"IV_RANK_NEUTRAL_CAUTION ({iv_rank:.1f})"
        return True, f"IV_RANK_FAVORABLE ({iv_rank:.1f})"

    @classmethod
    def apply_full_gate(
        cls,
        side: TradeSide,
        row: "pd.Series",
        strategy_style: str = "TREND",
    ) -> Tuple[bool, str]:
        """
        Unified gate chain applying ALL filters in sequence.
        Returns (passed, reason) — use this in all new strategies for consistency.
        """
        ts = row.get("timestamp")
        close = float(row["close"])

        # 1. Time-of-Day
        time_ok, time_reason = cls.is_prime_time_window(ts)
        if not time_ok:
            return False, time_reason

        # 2. VIX
        vix_trend = row.get("vix_trend", "STABLE")
        vix_regime = row.get("vix_regime", "NORMAL (11.5-16)")
        vix_ok, vix_reason = cls.is_vix_favorable(side, vix_trend, vix_regime)
        if not vix_ok:
            return False, vix_reason

        # 3. Regime
        regime = row.get("market_regime", "TRENDING_BULL")
        regime_ok, regime_reason = cls.is_regime_favorable(regime, strategy_style)
        if not regime_ok:
            return False, regime_reason

        # 4. S/R Clearance
        nearest_res = float(row.get("nearest_res", close * 1.01))
        nearest_sup = float(row.get("nearest_sup", close * 0.99))
        sr_ok, sr_reason = cls.has_sr_clearance(side, close, nearest_res, nearest_sup, 0.35)
        if not sr_ok:
            return False, sr_reason

        # 5. 200-DMA Trend Gate (Paul Tudor Jones)
        sma_200 = float(row.get("sma_200", 0))
        dma_ok, dma_reason = cls.is_200dma_aligned(side, close, sma_200)
        if not dma_ok:
            return False, dma_reason

        # 6. IV Rank Gate (Sosnoff / Natenberg)
        iv_rank = float(row.get("iv_rank", 30.0))
        iv_ok, iv_reason = cls.is_iv_rank_favorable(iv_rank, "BUY")
        if not iv_ok:
            return False, iv_reason

        return True, "ALL_GATES_PASSED"


# ─────────────────────────────────────────────────────────────────────────────
# Upgraded Option-Buyer Strategies
# ─────────────────────────────────────────────────────────────────────────────

def strategy_option_buyer_momentum_burst(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    🚀 Option Buyer: Momentum Burst & Squeeze Expansion.
    Specially engineered for Indian Index Option Buyers.
    
    Confluence triggers:
    1. Time Window: Within Morning Impulse (09:30-11:15) or Afternoon Expansion (13:15-15:00).
    2. Volatility Ignition: Squeeze release OR candle body > 0.8x ATR.
    3. Volume Acceleration: Volume > 1.25x 20-period Volume SMA.
    4. S/R Clearance: Minimum 0.35% room before opposing zone.
    5. EMA Alignment: Price above/below both 9 EMA and 21 EMA.
    6. VIX: Volatility not crushing.
    """
    if i < 25:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")

    # 1. Time-of-Day Filter
    time_ok, time_reason = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    # 2. VIX Filter
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL (11.5-16)")

    # 3. Indicator reads
    close = float(row["close"])
    open_p = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    vol = float(row.get("volume", 0))
    vol_sma = float(row.get("vol_sma_20", vol))
    atr = float(row.get("atr", 15.0))
    ema9 = float(row.get("ema_9", close))
    ema21 = float(row.get("ema_21", close))
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))
    sq_rel = bool(row.get("squeeze_release", False))

    candle_body = abs(close - open_p)
    vol_surge = (vol >= 1.2 * vol_sma) if vol_sma > 0 else True
    impulse_bar = (candle_body >= 0.7 * atr) or sq_rel

    # 200-DMA and IV Rank (World-Class Filters)
    sma_200 = float(row.get("sma_200", 0))
    iv_rank = float(row.get("iv_rank", 30.0))

    # Long (Call Buy) Signal
    if (close > open_p) and (close > ema9 > ema21) and impulse_bar and vol_surge:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if sr_ok and vix_ok and dma_ok and iv_ok:
            stop_price = max(low - (0.2 * atr), close - (1.2 * atr))
            target_price = close + (2.0 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=85.0,
                notes="Option Buyer Momentum Burst Long (Call)",
            )

    # Short (Put Buy) Signal
    if (close < open_p) and (close < ema9 < ema21) and impulse_bar and vol_surge:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if sr_ok and vix_ok and dma_ok and iv_ok:
            stop_price = min(high + (0.2 * atr), close + (1.2 * atr))
            target_price = close - (2.0 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=85.0,
                notes="Option Buyer Momentum Burst Short (Put)",
            )

    return None


def strategy_smc_liquidity_sweep_opt(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    🏛️ SMC: Liquidity Sweep + Reclaim (Option-Buyer Optimized).
    Filters out midday traps and verifies S/R bounce clearance and VIX favorability.
    """
    if i < 20:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")

    # Time & Midday Veto
    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    atr = float(row.get("atr", 15.0))
    ema50 = float(row.get("ema_50", close))
    bull_sweep = bool(row.get("bull_sweep", False))
    bear_sweep = bool(row.get("bear_sweep", False))
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL (11.5-16)")

    sma_200 = float(row.get("sma_200", 0))
    iv_rank = float(row.get("iv_rank", 30.0))

    # Long Sweep Reclaim
    if bull_sweep and close > ema50:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.3)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if sr_ok and vix_ok and dma_ok and iv_ok:
            stop_price = low - (0.3 * atr)
            target_price = close + (2.5 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=88.0,
                notes="SMC Bullish Sweep Reclaim (Opt Buyer Filtered)",
            )

    # Short Sweep Reject
    if bear_sweep and close < ema50:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.3)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if sr_ok and vix_ok and dma_ok and iv_ok:
            stop_price = high + (0.3 * atr)
            target_price = close - (2.5 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=88.0,
                notes="SMC Bearish Sweep Reject (Opt Buyer Filtered)",
            )

    return None


def strategy_ict_fvg_mitigation_opt(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    ⚡ ICT: Fair Value Gap Mitigation (Option-Buyer Optimized).
    Gated by Market Regime (only executed during TRENDING or SQUEEZE, never in CHOPPY_RANGEBOUND).
    """
    if i < 20:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")

    # Time Filter
    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    # Regime Gate: Veto in Choppy Consolidation
    regime = row.get("market_regime", "TRENDING_BULL")
    regime_ok, _ = OptionBuyerFilter.is_regime_favorable(regime, "TREND")
    if not regime_ok:
        return None

    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    atr = float(row.get("atr", 15.0))
    ema50 = float(row.get("ema_50", close))
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL (11.5-16)")

    sma_200 = float(row.get("sma_200", 0))
    iv_rank = float(row.get("iv_rank", 30.0))

    # Check recent 4 bars for FVG
    recent = df.iloc[max(0, i - 4): i]
    recent_bull_fvg = recent[recent["bull_fvg"] == True]
    recent_bear_fvg = recent[recent["bear_fvg"] == True]

    if not recent_bull_fvg.empty and close > ema50:
        last_fvg = recent_bull_fvg.iloc[-1]
        fvg_mid = float(last_fvg.get("fvg_mid", 0))
        if low <= fvg_mid <= high and close > row["open"]:
            sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.35)
            vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
            dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
            iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
            if sr_ok and vix_ok and dma_ok and iv_ok:
                stop_price = float(last_fvg.get("fvg_bottom", low - atr))
                target_price = close + (2.0 * (close - stop_price))
                return EntrySignal(
                    side=TradeSide.LONG,
                    price=close,
                    stop_loss=stop_price,
                    target=target_price,
                    confidence=82.0,
                    notes="ICT FVG 50% CE Bounce (Regime Gated)",
                )

    if not recent_bear_fvg.empty and close < ema50:
        last_fvg = recent_bear_fvg.iloc[-1]
        fvg_mid = float(last_fvg.get("fvg_mid", 0))
        if low <= fvg_mid <= high and close < row["open"]:
            sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.35)
            vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
            dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
            iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
            if sr_ok and vix_ok and dma_ok and iv_ok:
                stop_price = float(last_fvg.get("fvg_top", high + atr))
                target_price = close - (2.0 * (stop_price - close))
                return EntrySignal(
                    side=TradeSide.SHORT,
                    price=close,
                    stop_loss=stop_price,
                    target=target_price,
                    confidence=82.0,
                    notes="ICT FVG 50% CE Rejection (Regime Gated)",
                )

    return None


def strategy_supertrend_sr_opt(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    ⚡ SuperTrend + S/R & Time Protocol (Option-Buyer Optimized).
    Takes SuperTrend flips ONLY during prime hours, with 200-DMA alignment,
    IV Rank check, and S/R clearance.
    """
    if i < 20:
        return None

    row = df.iloc[i]
    prev_row = df.iloc[i - 1]
    ts = row.get("timestamp")

    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    close = float(row["close"])
    ema200 = float(row.get("ema_200", close))
    sma_200 = float(row.get("sma_200", ema200))
    iv_rank = float(row.get("iv_rank", 30.0))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL")
    st_dir = int(row.get("supertrend_dir", 0))
    prev_st_dir = int(prev_row.get("supertrend_dir", 0))
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))
    atr = float(row.get("atr", 15.0))

    # Bullish flip
    if prev_st_dir == -1 and st_dir == 1 and close > ema200:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if sr_ok and vix_ok and dma_ok and iv_ok:
            stop_price = float(row.get("supertrend", close - (1.5 * atr)))
            target_price = close + (2.0 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=80.0,
                notes="SuperTrend Bullish Flip (S/R Gated)",
            )

    # Bearish flip
    if prev_st_dir == 1 and st_dir == -1 and close < ema200:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if sr_ok and vix_ok and dma_ok and iv_ok:
            stop_price = float(row.get("supertrend", close + (1.5 * atr)))
            target_price = close - (2.0 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=80.0,
                notes="SuperTrend Bearish Flip (S/R Gated)",
            )

    return None


def strategy_wyckoff_amd_opt(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    🌀 Wyckoff: AMD Judas Swing (Option-Buyer Optimized).
    Filters out midday traps, confirms S/R room-to-run, 200-DMA alignment,
    IV rank, and invalidates stop at manipulation wick.
    """
    if i < 16:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")
    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    range_high = df["high"].iloc[i - 16 : i - 2].max()
    range_low = df["low"].iloc[i - 16 : i - 2].min()
    range_span = range_high - range_low
    atr = float(row.get("atr", 15.0))

    if range_span < (2.5 * atr):
        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])
        open_p = float(row["open"])
        nearest_res = float(row.get("nearest_res", close * 1.01))
        nearest_sup = float(row.get("nearest_sup", close * 0.99))
        vix_trend = row.get("vix_trend", "STABLE")
        vix_regime = row.get("vix_regime", "NORMAL")
        sma_200 = float(row.get("sma_200", 0))
        iv_rank = float(row.get("iv_rank", 30.0))

        # Spring Reclaim (Long / Call)
        if low < range_low and close > range_low and close > open_p:
            sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.3)
            vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
            dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
            iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
            if sr_ok and vix_ok and dma_ok and iv_ok:
                sl = low - (0.2 * atr)
                tp = close + (2.5 * (close - sl))
                return EntrySignal(
                    side=TradeSide.LONG,
                    price=close,
                    stop_loss=sl,
                    target=tp,
                    confidence=90.0,
                    notes="Wyckoff Spring Reclaim (Opt Buyer Filtered)",
                )

        # UTAD Rejection (Short / Put)
        elif high > range_high and close < range_high and close < open_p:
            sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.3)
            vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
            dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
            iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
            if sr_ok and vix_ok and dma_ok and iv_ok:
                sl = high + (0.2 * atr)
                tp = close - (2.5 * (sl - close))
                return EntrySignal(
                    side=TradeSide.SHORT,
                    price=close,
                    stop_loss=sl,
                    target=tp,
                    confidence=90.0,
                    notes="Wyckoff UTAD Rejection (Opt Buyer Filtered)",
                )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6 New World-Class Option Buyer Strategies
# ─────────────────────────────────────────────────────────────────────────────

def strategy_connors_rsi_mean_reversion(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    🧘 Larry Connors: RSI(2) + SMA(200) High-Probability Mean Reversion.
    World-class quantitative edge:
    - Long (Call Buy): Price > SMA(200), RSI(2) < 10 (extreme oversold pull-in within primary uptrend).
    - Short (Put Buy): Price < SMA(200), RSI(2) > 90 (extreme overbought pop within primary downtrend).
    - Stop: 2.5 x ATR from entry.
    - Target: 2.0R or SMA(5) reclaim.
    - Filters: Prime Time Window, S/R Clearance, VIX, IV Rank.
    """
    if i < 20:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")
    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    close = float(row["close"])
    sma_200 = float(row.get("sma_200", 0))
    rsi_2 = float(row.get("rsi_2", 50.0))
    atr = float(row.get("atr", 15.0))
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL")
    iv_rank = float(row.get("iv_rank", 30.0))

    # Long Setup: Trend is UP (>200 SMA), but short-term pullback (RSI(2) < 10)
    if rsi_2 < 10.0:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.30)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = close - (2.5 * atr)
            target_price = close + (2.0 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=88.0,
                notes="Connors RSI(2) Mean Reversion Long (Call)",
            )

    # Short Setup: Trend is DOWN (<200 SMA), but short-term bounce (RSI(2) > 90)
    if rsi_2 > 90.0:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.30)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = close + (2.5 * atr)
            target_price = close - (2.0 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=88.0,
                notes="Connors RSI(2) Mean Reversion Short (Put)",
            )

    return None


def strategy_opening_range_breakout(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    ⚡ Oliver Velez: Opening Range Breakout (ORB).
    First-Hour (09:15-10:15 IST) Range Expansion with volume confirmation.
    - Long (Call Buy): Price expands above 1st-hour high with volume > 1.3x 20-bar avg.
    - Short (Put Buy): Price breaks below 1st-hour low with volume > 1.3x 20-bar avg.
    - Stop: Opposite side of opening range (or 1.5 ATR for tight control).
    - Target: 2.0R projection.
    """
    if i < 5:
        return None

    row = df.iloc[i]
    close = float(row["close"])
    atr = float(row.get("atr", 15.0))
    sma_200 = float(row.get("sma_200", 0))
    iv_rank = float(row.get("iv_rank", 30.0))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL")
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))

    orb_long = bool(row.get("orb_breakout_long", False))
    orb_short = bool(row.get("orb_breakout_short", False))
    orb_high = float(row.get("orb_high", close * 1.01))
    orb_low = float(row.get("orb_low", close * 0.99))

    # Long Breakout
    if orb_long:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = max(orb_low, close - (1.5 * atr))
            target_price = close + (2.0 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=85.0,
                notes="Oliver Velez ORB Breakout Long (Call)",
            )

    # Short Breakdown
    if orb_short:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = min(orb_high, close + (1.5 * atr))
            target_price = close - (2.0 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=85.0,
                notes="Oliver Velez ORB Breakdown Short (Put)",
            )

    return None


def strategy_momentum_pinball(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    🎯 Linda Bradford Raschke: Momentum Pinball.
    Uses RSI(3) of ROC(1) to catch 1-2 day swing bursts after exhaustion pinball extremes.
    - Buy Trigger: Pinball < 30 on previous bar or current bar, confirmed by green candle.
    - Sell Trigger: Pinball > 70 on previous bar or current bar, confirmed by red candle.
    - Stop: 1.5 x ATR from entry.
    - Target: 2.0R.
    """
    if i < 10:
        return None

    row = df.iloc[i]
    prev_row = df.iloc[i - 1]
    ts = row.get("timestamp")
    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    close = float(row["close"])
    open_p = float(row["open"])
    atr = float(row.get("atr", 15.0))
    sma_200 = float(row.get("sma_200", 0))
    iv_rank = float(row.get("iv_rank", 30.0))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL")
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))

    prev_pinball = float(prev_row.get("pinball", 50.0))
    curr_pinball = float(row.get("pinball", 50.0))

    # Buy Setup: Prior bar was oversold (<30) and current bar turns green
    if (prev_pinball < 30.0 or curr_pinball < 30.0) and close > open_p:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.30)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = close - (1.5 * atr)
            target_price = close + (2.0 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=84.0,
                notes="Linda Raschke Momentum Pinball Buy (Call)",
            )

    # Sell Setup: Prior bar was overbought (>70) and current bar turns red
    if (prev_pinball > 70.0 or curr_pinball > 70.0) and close < open_p:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.30)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = close + (1.5 * atr)
            target_price = close - (2.0 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=84.0,
                notes="Linda Raschke Momentum Pinball Sell (Put)",
            )

    return None


def strategy_vcp_breakout(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    📈 Mark Minervini: Volatility Contraction Pattern (VCP) Breakout.
    Detects progressive range contraction (3+ contractions) with volume dry-up,
    followed by an explosive expansion breakout.
    - Long (Call Buy): Price > SMA(200), VCP breakout triggered with volume surge.
    - Stop: 1.5 x ATR below entry.
    - Target: 2.5R projection (institutional multi-day run).
    """
    if i < 20:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")
    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    vcp_breakout = bool(row.get("vcp_breakout", False))
    if not vcp_breakout:
        return None

    close = float(row["close"])
    atr = float(row.get("atr", 15.0))
    sma_200 = float(row.get("sma_200", 0))
    iv_rank = float(row.get("iv_rank", 30.0))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL")
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))

    dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
    sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.35)
    vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
    iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")

    if dma_ok and sr_ok and vix_ok and iv_ok:
        stop_price = close - (1.5 * atr)
        target_price = close + (2.5 * (close - stop_price))
        return EntrySignal(
            side=TradeSide.LONG,
            price=close,
            stop_loss=stop_price,
            target=target_price,
            confidence=89.0,
            notes="Minervini VCP Range Contraction Breakout Long (Call)",
        )

    return None


def strategy_expiry_day_gamma_edge(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    💥 Jeff Augen: Expiry Day Gamma Scalping & Expansion Edge.
    Triggers exclusively on Indian Index Expiry Days (Wednesdays/Thursdays) during
    afternoon gamma expansion windows (12:30-14:45 IST).
    - Long (Call Buy): Price expands > 0.4 ATR above VWAP with volume surge > 1.3x SMA.
    - Short (Put Buy): Price breaks > 0.4 ATR below VWAP with volume surge > 1.3x SMA.
    - Stop: 1.0 x ATR (tight expiry stop to protect premium decay).
    - Target: 2.5R (gamma acceleration multiplier).
    """
    if i < 15:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None

    dt = pd.to_datetime(ts)
    # Check expiry day (Wednesday=2 or Thursday=3)
    if dt.weekday() not in (2, 3):
        return None

    # Afternoon Gamma Expansion window: 12:30 to 14:45 IST
    from datetime import time as dtime
    cur_time = dt.time()
    if not (dtime(12, 30) <= cur_time <= dtime(14, 45)):
        return None

    close = float(row["close"])
    open_p = float(row["open"])
    vwap = float(row.get("vwap", close))
    atr = float(row.get("atr", 15.0))
    vol = float(row.get("volume", 0))
    vol_sma = float(row.get("vol_sma_20", vol))
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL")

    vol_surge = (vol >= 1.3 * vol_sma) if vol_sma > 0 else True
    if not vol_surge:
        return None

    # Expiry Call Expansion
    if close > vwap + (0.4 * atr) and close > open_p:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.25)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        if sr_ok and vix_ok:
            stop_price = close - (1.0 * atr)
            target_price = close + (2.5 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=86.0,
                notes="Jeff Augen Expiry Day Gamma Expansion Call (Long)",
            )

    # Expiry Put Expansion
    if close < vwap - (0.4 * atr) and close < open_p:
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.25)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        if sr_ok and vix_ok:
            stop_price = close + (1.0 * atr)
            target_price = close - (2.5 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=86.0,
                notes="Jeff Augen Expiry Day Gamma Expansion Put (Short)",
            )

    return None


def strategy_elephant_bar_continuation(
    df: pd.DataFrame, i: int
) -> Optional[EntrySignal]:
    """
    🐘 Oliver Velez: Elephant Bar Continuation (87% Win-Rate Setup).
    Identifies institutional ignition bars whose body engulfs / clears 3+ prior bars
    with volume confirmation.
    - Long: Bullish Elephant Bar clearing 3+ prior red candles.
    - Short: Bearish Elephant Bar clearing 3+ prior green candles.
    - Stop: 0.2 ATR beyond the ignition bar opposite extreme.
    - Target: 2.0R.
    """
    if i < 10:
        return None

    row = df.iloc[i]
    ts = row.get("timestamp")
    time_ok, _ = OptionBuyerFilter.is_prime_time_window(ts)
    if not time_ok:
        return None

    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    atr = float(row.get("atr", 15.0))
    sma_200 = float(row.get("sma_200", 0))
    iv_rank = float(row.get("iv_rank", 30.0))
    vix_trend = row.get("vix_trend", "STABLE")
    vix_regime = row.get("vix_regime", "NORMAL")
    nearest_res = float(row.get("nearest_res", close * 1.01))
    nearest_sup = float(row.get("nearest_sup", close * 0.99))

    is_bull = bool(row.get("elephant_bar_bull", False))
    is_bear = bool(row.get("elephant_bar_bear", False))

    if is_bull:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.LONG, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.LONG, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = low - (0.2 * atr)
            target_price = close + (2.0 * (close - stop_price))
            return EntrySignal(
                side=TradeSide.LONG,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=87.0,
                notes="Oliver Velez Elephant Bar Continuation Long (Call)",
            )

    if is_bear:
        dma_ok, _ = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close, sma_200)
        sr_ok, _ = OptionBuyerFilter.has_sr_clearance(TradeSide.SHORT, close, nearest_res, nearest_sup, 0.35)
        vix_ok, _ = OptionBuyerFilter.is_vix_favorable(TradeSide.SHORT, vix_trend, vix_regime)
        iv_ok, _ = OptionBuyerFilter.is_iv_rank_favorable(iv_rank, "BUY")
        if dma_ok and sr_ok and vix_ok and iv_ok:
            stop_price = high + (0.2 * atr)
            target_price = close - (2.0 * (stop_price - close))
            return EntrySignal(
                side=TradeSide.SHORT,
                price=close,
                stop_loss=stop_price,
                target=target_price,
                confidence=87.0,
                notes="Oliver Velez Elephant Bar Continuation Short (Put)",
            )

    return None


