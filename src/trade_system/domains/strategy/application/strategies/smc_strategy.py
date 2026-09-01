"""
Smart Money Concept (SMC) Strategy & Indicator Engine
=====================================================
Comprehensive institutional price action engine that analyzes:
1. Market Structure: Swing Highs/Lows, Break of Structure (BOS), Change of Character (CHoCH).
2. Order Blocks (OB): Volumetric institutional Demand & Supply zones with mitigation tracking.
3. Fair Value Gaps (FVG): Imbalance voids (BISI / SIBI) with dynamic ATR filtering.
4. Liquidity Pools & Sweeps: Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL) stop runs.
5. Premium vs. Discount Equilibrium: 50% dealing range analysis.

Produces high-conviction trade setups with exact Entry, Stop Loss, Targets, RRR, and Confluence Scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from trade_system.shared.exceptions import DataSanityError

LOGGER = logging.getLogger(__name__)


@dataclass
class SmcOrderBlock:
    """Represents an active or historical Order Block."""
    ob_type: str                  # "BULLISH" (Demand) or "BEARISH" (Supply)
    top: float                    # Upper price bound
    bottom: float                 # Lower price bound
    origin_bar_idx: int           # Index of base candle
    breakout_bar_idx: int         # Index of BOS bar
    origin_date: str              # Date/time of base candle
    volume: float                 # Volume of base candle
    volume_ratio: float           # Volume vs 20 SMA
    is_mitigated: bool = False    # True if tested
    mitigated_date: Optional[str] = None
    is_valid: bool = True         # False if fully broken through

    @property
    def mid(self) -> float:
        return round((self.top + self.bottom) / 2.0, 2)


@dataclass
class SmcFairValueGap:
    """Represents a 3-candle Fair Value Gap (Imbalance)."""
    fvg_type: str                 # "BULLISH" (BISI) or "BEARISH" (SIBI)
    top: float                    # Upper bound of imbalance
    bottom: float                 # Lower bound of imbalance
    bar_idx: int                  # Middle candle index
    date_str: str                 # Timestamp
    size_pct: float               # Size as % of stock price
    is_filled: bool = False

    @property
    def mid(self) -> float:
        return round((self.top + self.bottom) / 2.0, 2)


@dataclass
class SmcLiquiditySweep:
    """Represents a detected liquidity sweep (stop hunt / turtle soup)."""
    sweep_type: str               # "SSL_SWEEP" (Bullish reversal) or "BSL_SWEEP" (Bearish reversal)
    level: float                  # Swept price level
    bar_idx: int                  # Sweep candle index
    date_str: str                 # Timestamp
    wick_penetration_pct: float   # How deep the wick went past the level


@dataclass
class SmcTradeSetup:
    """Complete actionable trade setup produced by Smart Money Concept analysis."""
    symbol: str
    action: str                   # "BUY CALL" (or LONG) vs "BUY PUT" (or SHORT)
    direction: int                # 1 for Bullish, -1 for Bearish
    setup_type: str               # e.g., "DISCOUNT_OB_RETEST", "SSL_SWEEP_REVERSAL", "BULLISH_CHOCH_EXPANSION"
    confluence_score: float       # 0 - 100
    entry_price: float            # Recommended Entry price / zone
    stop_loss: float              # Invalidation level
    target_1: float               # First liquidity target (1:2 RRR target)
    target_2: float               # Extended target (Major opposing OB/BSL)
    risk_reward_ratio: float      # Potential RRR (e.g. 2.5)
    market_structure: str         # "BULLISH_BOS", "BULLISH_CHOCH", "BEARISH_BOS", "BEARISH_CHOCH", "RANGING"
    equilibrium_status: str       # "DISCOUNT" (<50%), "PREMIUM" (>50%), "EQUILIBRIUM" (~50%)
    reasons: List[str] = field(default_factory=list)
    active_ob: Optional[SmcOrderBlock] = None
    active_fvg: Optional[SmcFairValueGap] = None
    recent_sweep: Optional[SmcLiquiditySweep] = None
    timestamp: str = ""
    spot_price: float = 0.0

    def format_summary(self) -> str:
        icon = "🟢" if self.direction == 1 else "🔴"
        sym_short = self.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
        return (
            f"{icon} <b>{sym_short}</b> | <b>{self.action}</b> (Score: {self.confluence_score:.0f}/100)\n"
            f"   Structure: <code>{self.market_structure}</code> | Zone: <code>{self.equilibrium_status}</code>\n"
            f"   Entry: ₹{self.entry_price:.2f} | SL: ₹{self.stop_loss:.2f} | T1: ₹{self.target_1:.2f} (RRR: 1:{self.risk_reward_ratio:.1f})\n"
            f"   Rationale: {', '.join(self.reasons)}"
        )


class SmartMoneyConceptStrategy:
    """
    Institutional Smart Money Concept (SMC) & ICT Strategy Engine.
    
    Dynamically scales thresholds to the stock's ATR and price volatility,
    making it equally effective on penny stocks, large caps, and indices.
    """

    def __init__(
        self,
        swing_length: int = 5,
        fvg_min_atr_mult: float = 0.3,
        ob_vol_multiplier: float = 1.1,
        min_rrr: float = 1.5,
    ) -> None:
        self.swing_length = swing_length
        self.fvg_min_atr_mult = fvg_min_atr_mult
        self.ob_vol_multiplier = ob_vol_multiplier
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

    def find_swings(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Detect swing highs and lows with lookback/lookforward pivots."""
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        k = self.swing_length

        swing_highs = np.zeros(n)
        swing_lows = np.zeros(n)

        for i in range(k, n - k):
            window_high = highs[i - k : i + k + 1]
            if highs[i] == window_high.max() and highs[i] > highs[i - 1]:
                swing_highs[i] = highs[i]

            window_low = lows[i - k : i + k + 1]
            if lows[i] == window_low.min() and lows[i] < lows[i - 1]:
                swing_lows[i] = lows[i]

        return swing_highs, swing_lows

    def detect_fair_value_gaps(self, df: pd.DataFrame, atr: pd.Series) -> List[SmcFairValueGap]:
        """Detect 3-candle Fair Value Gaps (BISI & SIBI)."""
        fvgs: List[SmcFairValueGap] = []
        n = len(df)
        if n < 3:
            return fvgs

        dates = df["timestamp"].astype(str).values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atr_vals = atr.values

        for i in range(2, n):
            c1_high = highs[i - 2]
            c1_low = lows[i - 2]
            c3_high = highs[i]
            c3_low = lows[i]
            min_gap = atr_vals[i] * self.fvg_min_atr_mult

            # Bullish FVG: Candle 3 Low > Candle 1 High
            if c3_low > c1_high and (c3_low - c1_high) >= min_gap:
                gap_size = c3_low - c1_high
                gap_pct = (gap_size / closes[i]) * 100
                fvg = SmcFairValueGap(
                    fvg_type="BULLISH",
                    top=round(c3_low, 2),
                    bottom=round(c1_high, 2),
                    bar_idx=i - 1,
                    date_str=dates[i - 1],
                    size_pct=round(gap_pct, 2),
                    is_filled=False
                )
                # Check if subsequent candles filled it
                for j in range(i + 1, n):
                    if lows[j] <= fvg.bottom:
                        fvg.is_filled = True
                        break
                fvgs.append(fvg)

            # Bearish FVG: Candle 1 Low > Candle 3 High
            elif c1_low > c3_high and (c1_low - c3_high) >= min_gap:
                gap_size = c1_low - c3_high
                gap_pct = (gap_size / closes[i]) * 100
                fvg = SmcFairValueGap(
                    fvg_type="BEARISH",
                    top=round(c1_low, 2),
                    bottom=round(c3_high, 2),
                    bar_idx=i - 1,
                    date_str=dates[i - 1],
                    size_pct=round(gap_pct, 2),
                    is_filled=False
                )
                for j in range(i + 1, n):
                    if highs[j] >= fvg.top:
                        fvg.is_filled = True
                        break
                fvgs.append(fvg)

        return fvgs

    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        swing_highs: np.ndarray,
        swing_lows: np.ndarray,
    ) -> List[SmcOrderBlock]:
        """Detects institutional Order Blocks based on market structure breaks."""
        obs: List[SmcOrderBlock] = []
        n = len(df)
        if n < 10:
            return obs

        if "vol_sma" not in df.columns:
            df["vol_sma"] = df["volume"].rolling(20, min_periods=1).mean()

        dates = df["timestamp"].astype(str).values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values
        vol_sma = df["vol_sma"].values

        last_swing_high_idx = -1
        last_swing_low_idx = -1

        for i in range(n):
            conf_idx = i - self.swing_length
            if conf_idx >= 0:
                if swing_highs[conf_idx] > 0:
                    last_swing_high_idx = conf_idx
                if swing_lows[conf_idx] > 0:
                    last_swing_low_idx = conf_idx

            # 1. Bullish BOS / CHoCH -> Create Demand Order Block
            if last_swing_high_idx != -1 and closes[i] > highs[last_swing_high_idx]:
                origin_idx = -1
                for j in range(i - 1, max(0, last_swing_high_idx - 2), -1):
                    if closes[j] <= opens[j]:  # Last bearish candle
                        origin_idx = j
                        break
                if origin_idx == -1:
                    origin_idx = last_swing_high_idx

                vol = volumes[origin_idx]
                avg_v = vol_sma[origin_idx] if vol_sma[origin_idx] > 0 else 1.0
                ratio = vol / avg_v

                ob = SmcOrderBlock(
                    ob_type="BULLISH",
                    top=round(highs[origin_idx], 2),
                    bottom=round(lows[origin_idx], 2),
                    origin_bar_idx=origin_idx,
                    breakout_bar_idx=i,
                    origin_date=dates[origin_idx],
                    volume=vol,
                    volume_ratio=round(ratio, 2),
                    is_mitigated=False
                )

                # Check historical mitigation
                for k in range(i + 1, n):
                    if lows[k] < ob.bottom:
                        ob.is_valid = False
                        break
                    elif lows[k] <= ob.top:
                        ob.is_mitigated = True
                        ob.mitigated_date = dates[k]

                obs.append(ob)
                last_swing_high_idx = -1

            # 2. Bearish BOS / CHoCH -> Create Supply Order Block
            elif last_swing_low_idx != -1 and closes[i] < lows[last_swing_low_idx]:
                origin_idx = -1
                for j in range(i - 1, max(0, last_swing_low_idx - 2), -1):
                    if closes[j] >= opens[j]:  # Last bullish candle
                        origin_idx = j
                        break
                if origin_idx == -1:
                    origin_idx = last_swing_low_idx

                vol = volumes[origin_idx]
                avg_v = vol_sma[origin_idx] if vol_sma[origin_idx] > 0 else 1.0
                ratio = vol / avg_v

                ob = SmcOrderBlock(
                    ob_type="BEARISH",
                    top=round(highs[origin_idx], 2),
                    bottom=round(lows[origin_idx], 2),
                    origin_bar_idx=origin_idx,
                    breakout_bar_idx=i,
                    origin_date=dates[origin_idx],
                    volume=vol,
                    volume_ratio=round(ratio, 2),
                    is_mitigated=False
                )

                for k in range(i + 1, n):
                    if highs[k] > ob.top:
                        ob.is_valid = False
                        break
                    elif highs[k] >= ob.bottom:
                        ob.is_mitigated = True
                        ob.mitigated_date = dates[k]

                obs.append(ob)
                last_swing_low_idx = -1

        return obs

    def detect_liquidity_sweeps(
        self,
        df: pd.DataFrame,
        swing_highs: np.ndarray,
        swing_lows: np.ndarray,
        lookback: int = 30
    ) -> List[SmcLiquiditySweep]:
        """Detects Buy-Side (BSL) and Sell-Side (SSL) liquidity sweeps."""
        sweeps: List[SmcLiquiditySweep] = []
        n = len(df)
        if n < 10:
            return sweeps

        dates = df["timestamp"].astype(str).values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Track recent swing levels
        active_highs = []
        active_lows = []

        for i in range(max(0, n - lookback), n):
            conf_idx = i - self.swing_length
            if conf_idx >= 0:
                if swing_highs[conf_idx] > 0:
                    active_highs.append(swing_highs[conf_idx])
                if swing_lows[conf_idx] > 0:
                    active_lows.append(swing_lows[conf_idx])

            # Check BSL Sweep (Wick above swing high, close below)
            for sh in active_highs[-5:]:
                if highs[i] > sh and closes[i] < sh and closes[i] < opens[i]:
                    pen_pct = ((highs[i] - sh) / sh) * 100
                    sweeps.append(SmcLiquiditySweep(
                        sweep_type="BSL_SWEEP",
                        level=round(sh, 2),
                        bar_idx=i,
                        date_str=dates[i],
                        wick_penetration_pct=round(pen_pct, 2)
                    ))

            # Check SSL Sweep (Wick below swing low, close above)
            for sl in active_lows[-5:]:
                if lows[i] < sl and closes[i] > sl and closes[i] > opens[i]:
                    pen_pct = ((sl - lows[i]) / sl) * 100
                    sweeps.append(SmcLiquiditySweep(
                        sweep_type="SSL_SWEEP",
                        level=round(sl, 2),
                        bar_idx=i,
                        date_str=dates[i],
                        wick_penetration_pct=round(pen_pct, 2)
                    ))

        return sweeps

    def analyze_symbol(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Optional[SmcTradeSetup]:
        """
        Runs the full Smart Money Concepts analysis on a stock/index DataFrame.
        Evaluates confluence, determines entry/SL/targets, and outputs trade setup.
        """
        if df.empty or len(df) < 30:
            return None

        clean_df = df.copy().sort_values("timestamp").reset_index(drop=True)
        clean_df["vol_sma"] = clean_df["volume"].rolling(20, min_periods=1).mean()
        atr = self.calculate_atr(clean_df, period=14)
        clean_df["atr"] = atr

        swing_highs, swing_lows = self.find_swings(clean_df)
        fvgs = self.detect_fair_value_gaps(clean_df, atr)
        obs = self.detect_order_blocks(clean_df, swing_highs, swing_lows)
        sweeps = self.detect_liquidity_sweeps(clean_df, swing_highs, swing_lows, lookback=15)

        last_bar = clean_df.iloc[-1]
        current_close = float(last_bar["close"])
        current_high = float(last_bar["high"])
        current_low = float(last_bar["low"])
        current_atr = float(atr.iloc[-1])
        current_vol = float(last_bar["volume"])
        avg_vol = float(clean_df["vol_sma"].iloc[-1])
        rvol = current_vol / avg_vol if avg_vol > 0 else 1.0
        bar_date = str(last_bar["timestamp"])

        # ── 1. Calculate Dealing Range & Equilibrium ─────────────────────────
        recent_window = clean_df.tail(40)
        range_high = float(recent_window["high"].max())
        range_low = float(recent_window["low"].min())
        equilibrium = (range_high + range_low) / 2.0

        if current_close < equilibrium - (current_atr * 0.2):
            eq_status = "DISCOUNT"  # Institutional Buy Zone
        elif current_close > equilibrium + (current_atr * 0.2):
            eq_status = "PREMIUM"   # Institutional Sell Zone
        else:
            eq_status = "EQUILIBRIUM"

        # ── 2. Market Structure Analysis ─────────────────────────────────────
        recent_obs = [ob for ob in obs if ob.is_valid]
        active_bull_obs = [ob for ob in recent_obs if ob.ob_type == "BULLISH" and not ob.is_mitigated]
        active_bear_obs = [ob for ob in recent_obs if ob.ob_type == "BEARISH" and not ob.is_mitigated]

        unfilled_bull_fvgs = [f for f in fvgs if f.fvg_type == "BULLISH" and not f.is_filled]
        unfilled_bear_fvgs = [f for f in fvgs if f.fvg_type == "BEARISH" and not f.is_filled]

        # Recent structure trend
        is_bullish_structure = len(active_bull_obs) >= len(active_bear_obs) and current_close > clean_df["close"].rolling(20).mean().iloc[-1]
        market_structure = "BULLISH" if is_bullish_structure else "BEARISH"

        # ── 3. Evaluate Bullish Setups (BUY CALL / LONG) ─────────────────────
        bullish_score = 0.0
        bullish_reasons = []
        selected_bull_ob: Optional[SmcOrderBlock] = None
        selected_bull_fvg: Optional[SmcFairValueGap] = None
        selected_sweep: Optional[SmcLiquiditySweep] = None

        # Check if price is reacting/retesting a Demand Order Block in Discount
        for ob in active_bull_obs:
            # Within 1.5 ATR of the Order Block top
            if ob.bottom <= current_low <= ob.top * 1.02 or abs(current_close - ob.top) <= current_atr * 0.8:
                bullish_score += 35
                bullish_reasons.append(f"Price retesting unmitigated Demand OB (₹{ob.bottom:.2f} - ₹{ob.top:.2f})")
                selected_bull_ob = ob
                break

        # Check Discount Zone
        if eq_status == "DISCOUNT":
            bullish_score += 20
            bullish_reasons.append("Trading in Institutional Discount Zone (<50% dealing range)")

        # Check for recent SSL Sweep (Stop hunt of previous lows)
        recent_ssl_sweeps = [s for s in sweeps if s.sweep_type == "SSL_SWEEP" and (len(clean_df) - 1 - s.bar_idx) <= 3]
        if recent_ssl_sweeps:
            selected_sweep = recent_ssl_sweeps[-1]
            bullish_score += 25
            bullish_reasons.append(f"Recent SSL Liquidity Sweep at ₹{selected_sweep.level:.2f} with buyer rejection")

        # Check for Unfilled Bullish FVG in Discount
        for fvg in unfilled_bull_fvgs:
            if fvg.bottom <= current_low <= fvg.top or abs(current_close - fvg.top) <= current_atr * 0.5:
                bullish_score += 20
                bullish_reasons.append(f"Confluence with Bullish FVG (₹{fvg.bottom:.2f} - ₹{fvg.top:.2f})")
                selected_bull_fvg = fvg
                break

        # Volume expansion
        if rvol >= self.ob_vol_multiplier:
            bullish_score += 15
            bullish_reasons.append(f"Volume Surge {rvol:.1f}x vs 20 SMA")

        # ── 4. Evaluate Bearish Setups (BUY PUT / SHORT) ─────────────────────
        bearish_score = 0.0
        bearish_reasons = []
        selected_bear_ob: Optional[SmcOrderBlock] = None
        selected_bear_fvg: Optional[SmcFairValueGap] = None

        # Check if price is reacting/retesting a Supply Order Block in Premium
        for ob in active_bear_obs:
            if ob.bottom * 0.98 <= current_high <= ob.top or abs(current_close - ob.bottom) <= current_atr * 0.8:
                bearish_score += 35
                bearish_reasons.append(f"Price retesting unmitigated Supply OB (₹{ob.bottom:.2f} - ₹{ob.top:.2f})")
                selected_bear_ob = ob
                break

        # Check Premium Zone
        if eq_status == "PREMIUM":
            bearish_score += 20
            bearish_reasons.append("Trading in Institutional Premium Zone (>50% dealing range)")

        # Check for recent BSL Sweep (Stop hunt of previous highs)
        recent_bsl_sweeps = [s for s in sweeps if s.sweep_type == "BSL_SWEEP" and (len(clean_df) - 1 - s.bar_idx) <= 3]
        if recent_bsl_sweeps:
            selected_sweep = recent_bsl_sweeps[-1]
            bearish_score += 25
            bearish_reasons.append(f"Recent BSL Liquidity Sweep at ₹{selected_sweep.level:.2f} with seller rejection")

        # Check for Unfilled Bearish FVG in Premium
        for fvg in unfilled_bear_fvgs:
            if fvg.bottom <= current_high <= fvg.top or abs(current_close - fvg.bottom) <= current_atr * 0.5:
                bearish_score += 20
                bearish_reasons.append(f"Confluence with Bearish FVG (₹{fvg.bottom:.2f} - ₹{fvg.top:.2f})")
                selected_bear_fvg = fvg
                break

        if rvol >= self.ob_vol_multiplier:
            bearish_score += 15
            bearish_reasons.append(f"Volume Surge {rvol:.1f}x vs 20 SMA")

        # ── 5. Generate Trade Setup based on Highest Conviction ───────────────
        if bullish_score >= 50 and bullish_score > bearish_score:
            sl = (selected_bull_ob.bottom if selected_bull_ob else range_low) - (current_atr * 0.3)
            risk = max(current_close - sl, current_atr * 0.5)
            t1 = current_close + (risk * 2.0)
            t2 = range_high if range_high > t1 else current_close + (risk * 3.5)
            rrr = (t1 - current_close) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr:
                return SmcTradeSetup(
                    symbol=symbol,
                    action="BUY CALL",
                    direction=1,
                    setup_type="BULLISH_DEMAND_EXPANSION" if selected_bull_ob else "SSL_SWEEP_REVERSAL",
                    confluence_score=min(100.0, bullish_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(rrr, 2),
                    market_structure="BULLISH_STRUCTURE" if is_bullish_structure else "REVERSAL_CHONCH",
                    equilibrium_status=eq_status,
                    reasons=bullish_reasons,
                    active_ob=selected_bull_ob,
                    active_fvg=selected_bull_fvg,
                    recent_sweep=selected_sweep,
                    timestamp=bar_date,
                    spot_price=current_close
                )

        elif bearish_score >= 50:
            sl = (selected_bear_ob.top if selected_bear_ob else range_high) + (current_atr * 0.3)
            risk = max(sl - current_close, current_atr * 0.5)
            t1 = current_close - (risk * 2.0)
            t2 = range_low if range_low < t1 else current_close - (risk * 3.5)
            rrr = (current_close - t1) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr:
                return SmcTradeSetup(
                    symbol=symbol,
                    action="BUY PUT",
                    direction=-1,
                    setup_type="BEARISH_SUPPLY_EXPANSION" if selected_bear_ob else "BSL_SWEEP_REVERSAL",
                    confluence_score=min(100.0, bearish_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(rrr, 2),
                    market_structure="BEARISH_STRUCTURE" if not is_bullish_structure else "REVERSAL_CHONCH",
                    equilibrium_status=eq_status,
                    reasons=bearish_reasons,
                    active_ob=selected_bear_ob,
                    active_fvg=selected_bear_fvg,
                    recent_sweep=selected_sweep,
                    timestamp=bar_date,
                    spot_price=current_close
                )

        return None
