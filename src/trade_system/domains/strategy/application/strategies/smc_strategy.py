"""
Smart Money Concept (SMC) & Supply/Demand Strategy Engine
=========================================================
Institutional price action engine integrating:
1. Photon Mechanical Market Structure:
   - Swing Structure (Macro institutional trend) vs Internal Structure (Pullback legs).
   - Strong Protected Lows/Highs (Stop Loss invalidations).
   - Weak Target Highs/Lows (Liquidity take-profit targets).
   - Swing BOS vs CHoCH and Internal CHoCH realignment triggers.
2. JeaFx Supply & Demand Engine:
   - Imbalance / FVG Validation Rule (only valid if followed by open imbalance).
   - Extreme vs Decisional zone classification.
   - Mitigation & touch count tracking (Fresh vs Mitigated vs Exhausted).
   - Range-to-Range (Demand-to-Supply) target discovery.
3. Option Chain Confluence:
   - Put/Call OI Walls, Max Pain drift, and PCR sentiment confluence.

Produces high-conviction trade setups with exact Entry, Stop Loss, Targets, RRR, and Confluence Scores.
Conforms fully to BaseStrategy for backtesting and live execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from trade_system.domains.strategy.application.strategies.base import (
    BaseStrategy, StrategyContext, TradeSignal
)
from trade_system.domains.strategy.application.indicators.smc_structure import (
    SMCStructureEngine, SMCStructureState, SMCStructurePoint, SMCStructureBreak,
    StructureTrend, PointType, BreakType
)
from trade_system.domains.strategy.application.indicators.supply_demand import (
    SupplyDemandEngine, SupplyDemandZone, SDZoneType
)

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
    setup_type: str               # e.g., "PRO_TREND_DECISIONAL_DEMAND", "EXTREME_DEMAND_REVERSAL"
    confluence_score: float       # 0 - 100
    entry_price: float            # Recommended Entry price / zone
    stop_loss: float              # Invalidation level (Behind Strong Protected pivot)
    target_1: float               # Range-to-Range Target (Opposing S/D zone)
    target_2: float               # Liquidity Run Target (Weak Target High/Low)
    risk_reward_ratio: float      # Potential RRR (e.g. 2.5)
    market_structure: str         # "BULLISH_STRUCTURE", "BULLISH_REALIGNED", etc.
    equilibrium_status: str       # "DISCOUNT" (<50%), "PREMIUM" (>50%), "EQUILIBRIUM" (~50%)
    zone_classification: str = "NONE"      # "EXTREME_DEMAND", "DECISIONAL_DEMAND", etc.
    strong_protected_level: Optional[float] = None
    weak_target_level: Optional[float] = None
    opposing_target_zone: Optional[float] = None
    is_internal_realigned: bool = False
    reasons: List[str] = field(default_factory=list)
    active_ob: Optional[SmcOrderBlock] = None
    active_fvg: Optional[SmcFairValueGap] = None
    recent_sweep: Optional[SmcLiquiditySweep] = None
    active_sd_zone: Optional[SupplyDemandZone] = None
    structure_state: Optional[SMCStructureState] = None
    timestamp: str = ""
    spot_price: float = 0.0

    def format_summary(self) -> str:
        icon = "🟢" if self.direction == 1 else "🔴"
        sym_short = self.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
        zone_info = f" | Zone: <code>{self.zone_classification}</code>" if self.zone_classification != "NONE" else ""
        realigned_tag = " ⚡<b>REALIGNED</b>" if self.is_internal_realigned else ""
        return (
            f"{icon} <b>{sym_short}</b> | <b>{self.action}</b> (Score: {self.confluence_score:.0f}/100){realigned_tag}\n"
            f"   Structure: <code>{self.market_structure}</code>{zone_info} | Eq: <code>{self.equilibrium_status}</code>\n"
            f"   Entry: ₹{self.entry_price:.2f} | Protected SL: ₹{self.stop_loss:.2f} | T1 (Range): ₹{self.target_1:.2f} | T2 (Target): ₹{self.target_2:.2f} (RRR: 1:{self.risk_reward_ratio:.1f})\n"
            f"   Rationale: {', '.join(self.reasons)}"
        )


class SmartMoneyConceptStrategy(BaseStrategy):
    """
    Institutional Smart Money Concept (SMC) & Supply/Demand Strategy.
    
    Combines Photon Trading market structure (Swing vs Internal pivots,
    Strong vs Weak swings, Internal CHoCH realignment) with JeaFx
    imbalance-validated Supply & Demand zones (Extreme vs Decisional).
    """
    name: str = "smc"
    version: str = "3.0"
    supported_timeframes: List[str] = ["5m", "15m", "1h", "D"]

    def __init__(
        self,
        swing_length: int = 5,
        internal_length: int = 2,
        fvg_min_atr_mult: float = 0.25,
        ob_vol_multiplier: float = 1.1,
        min_rrr: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.fvg_min_atr_mult = fvg_min_atr_mult
        self.ob_vol_multiplier = ob_vol_multiplier
        self.min_rrr = min_rrr

        # Specialized Sub-Engines
        self.structure_engine = SMCStructureEngine(
            swing_length=self.swing_length,
            internal_length=self.internal_length,
        )
        self.supply_demand_engine = SupplyDemandEngine(
            swing_length=self.swing_length,
            fvg_min_atr_mult=self.fvg_min_atr_mult,
        )

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
        """Detect swing highs and lows for backward compatibility."""
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
        """Detects Order Blocks for backward compatibility."""
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

            if last_swing_high_idx != -1 and closes[i] > highs[last_swing_high_idx]:
                origin_idx = -1
                for j in range(i - 1, max(0, last_swing_high_idx - 2), -1):
                    if closes[j] <= opens[j]:
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
                for k in range(i + 1, n):
                    if lows[k] < ob.bottom:
                        ob.is_valid = False
                        break
                    elif lows[k] <= ob.top:
                        ob.is_mitigated = True
                        ob.mitigated_date = dates[k]

                obs.append(ob)
                last_swing_high_idx = -1

            elif last_swing_low_idx != -1 and closes[i] < lows[last_swing_low_idx]:
                origin_idx = -1
                for j in range(i - 1, max(0, last_swing_low_idx - 2), -1):
                    if closes[j] >= opens[j]:
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

        active_highs = []
        active_lows = []

        for i in range(max(0, n - lookback), n):
            conf_idx = i - self.swing_length
            if conf_idx >= 0:
                if swing_highs[conf_idx] > 0:
                    active_highs.append(swing_highs[conf_idx])
                if swing_lows[conf_idx] > 0:
                    active_lows.append(swing_lows[conf_idx])

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

    def analyze_symbol(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        option_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SmcTradeSetup]:
        """
        Runs the refined SMC Market Structure + Supply/Demand Analysis.
        
        Evaluates:
        - Structural trend (Photon): Swing Trend + Internal Pullback/Realignment
        - Protected Swing Point (SL) & Weak Swing Target (TP2)
        - Supply/Demand Zone Tap (JeaFx): Extreme vs Decisional, Freshness, FVG validation
        - Range-to-Range Target (TP1): Nearest opposing unmitigated zone
        - Option Chain confluence (Max pain drift, Put/Call OI walls)
        """
        if df.empty or len(df) < 25:
            return None

        clean_df = df.copy().sort_values("timestamp").reset_index(drop=True)
        clean_df["vol_sma"] = clean_df["volume"].rolling(20, min_periods=1).mean()
        atr = self.calculate_atr(clean_df, period=14)
        clean_df["atr"] = atr

        # Run Sub-Engines
        structure_state = self.structure_engine.analyze(clean_df)
        sd_zones = self.supply_demand_engine.detect_zones(clean_df, structure_state)
        swing_highs, swing_lows = self.find_swings(clean_df)
        fvgs = self.detect_fair_value_gaps(clean_df, atr)
        legacy_obs = self.detect_order_blocks(clean_df, swing_highs, swing_lows)
        sweeps = self.detect_liquidity_sweeps(clean_df, swing_highs, swing_lows, lookback=15)

        last_bar = clean_df.iloc[-1]
        current_close = float(last_bar["close"])
        current_high = float(last_bar["high"])
        current_low = float(last_bar["low"])
        current_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else (current_close * 0.01)
        current_vol = float(last_bar["volume"])
        avg_vol = float(clean_df["vol_sma"].iloc[-1]) if not np.isnan(clean_df["vol_sma"].iloc[-1]) else 1.0
        rvol = current_vol / avg_vol if avg_vol > 0 else 1.0
        bar_date = str(last_bar["timestamp"])

        # Dealing Range & Equilibrium
        recent_window = clean_df.tail(min(45, len(clean_df)))
        range_high = float(recent_window["high"].max())
        range_low = float(recent_window["low"].min())
        equilibrium = (range_high + range_low) / 2.0

        if current_close < equilibrium - (current_atr * 0.2):
            eq_status = "DISCOUNT"
        elif current_close > equilibrium + (current_atr * 0.2):
            eq_status = "PREMIUM"
        else:
            eq_status = "EQUILIBRIUM"

        # ── 1. EVALUATE BULLISH SETUPS ────────────────────────────────────────
        bullish_score = 0.0
        bullish_reasons = []
        selected_sd_zone: Optional[SupplyDemandZone] = None
        selected_fvg: Optional[SmcFairValueGap] = None
        selected_sweep: Optional[SmcLiquiditySweep] = None

        # Filter active demand zones
        active_demand_zones = [z for z in sd_zones if z.is_demand and z.is_valid]
        for z in active_demand_zones:
            # Price within zone or within 0.8 ATR of zone top
            if (z.bottom <= current_low <= z.top * 1.01) or (abs(current_close - z.top) <= current_atr * 0.8):
                selected_sd_zone = z
                if z.zone_type == SDZoneType.EXTREME_DEMAND:
                    bullish_score += 35
                    bullish_reasons.append(f"Retesting Extreme Demand Zone (₹{z.bottom:.2f} - ₹{z.top:.2f})")
                else:
                    bullish_score += 30
                    bullish_reasons.append(f"Retesting Decisional Demand Zone (₹{z.bottom:.2f} - ₹{z.top:.2f})")

                if z.is_fresh:
                    bullish_score += 15
                    bullish_reasons.append("Pristine unmitigated institutional zone (0 touches)")
                elif z.touch_count == 1:
                    bullish_score += 8
                    bullish_reasons.append("First test of demand zone")

                if z.has_valid_imbalance:
                    bullish_score += 10
                    bullish_reasons.append(f"Verified by Imbalance/FVG gap (₹{z.imbalance_size:.2f})")
                break

        # Fallback to legacy OB if no SD zone found
        selected_legacy_ob = None
        if not selected_sd_zone:
            active_bull_obs = [ob for ob in legacy_obs if ob.ob_type == "BULLISH" and ob.is_valid and not ob.is_mitigated]
            for ob in active_bull_obs:
                if ob.bottom <= current_low <= ob.top * 1.02 or abs(current_close - ob.top) <= current_atr * 0.8:
                    bullish_score += 30
                    bullish_reasons.append(f"Retesting Demand Order Block (₹{ob.bottom:.2f} - ₹{ob.top:.2f})")
                    selected_legacy_ob = ob
                    break

        # Market Structure Alignment (Photon)
        if structure_state.swing_trend == StructureTrend.BULLISH:
            bullish_score += 20
            bullish_reasons.append("Macro Swing Structure is Bullish (Higher Highs & Higher Lows)")
            if structure_state.is_internal_realigned:
                bullish_score += 25
                bullish_reasons.append("⚡ Internal CHoCH Trigger: Pullback ended, realigned with Swing Trend")
            elif structure_state.is_pullback:
                bullish_reasons.append("Pullback into Discount Demand in progress")

        # Dealing Range Discount
        if eq_status == "DISCOUNT":
            bullish_score += 15
            bullish_reasons.append("Trading in Institutional Discount Zone (<50% dealing range)")

        # Liquidity Sweep
        recent_ssl_sweeps = [s for s in sweeps if s.sweep_type == "SSL_SWEEP" and (len(clean_df) - 1 - s.bar_idx) <= 3]
        if recent_ssl_sweeps:
            selected_sweep = recent_ssl_sweeps[-1]
            bullish_score += 20
            bullish_reasons.append(f"Sell-Side Liquidity (SSL) swept at ₹{selected_sweep.level:.2f} with swift buyer reclaim")

        # Unfilled FVG
        unfilled_bull_fvgs = [f for f in fvgs if f.fvg_type == "BULLISH" and not f.is_filled]
        for fvg in unfilled_bull_fvgs:
            if fvg.bottom <= current_low <= fvg.top or abs(current_close - fvg.top) <= current_atr * 0.5:
                bullish_score += 10
                bullish_reasons.append(f"Confluence with Bullish FVG (₹{fvg.bottom:.2f} - ₹{fvg.top:.2f})")
                selected_fvg = fvg
                break

        # Volume expansion
        if rvol >= self.ob_vol_multiplier:
            bullish_score += 10
            bullish_reasons.append(f"Volume Surge {rvol:.1f}x vs 20 SMA")

        # Option Chain Confluence
        if option_context:
            put_wall = option_context.get("put_wall", 0.0)
            max_pain = option_context.get("max_pain", 0.0)
            pcr = option_context.get("pcr", 1.0)
            if put_wall > 0 and abs(current_close - put_wall) <= current_atr * 1.5:
                bullish_score += 10
                bullish_reasons.append(f"Derivative Confluence: Strong Put OI Wall Support at ₹{put_wall:.2f}")
            if max_pain > 0 and current_close < max_pain:
                bullish_score += 5
                bullish_reasons.append(f"Max Pain Gravitational Pull upward toward ₹{max_pain:.2f}")
            if pcr > 1.1:
                bullish_score += 5
                bullish_reasons.append(f"Bullish Option PCR Sentiment ({pcr:.2f})")

        # ── 2. EVALUATE BEARISH SETUPS ────────────────────────────────────────
        bearish_score = 0.0
        bearish_reasons = []
        selected_bear_sd_zone: Optional[SupplyDemandZone] = None
        selected_bear_fvg: Optional[SmcFairValueGap] = None

        active_supply_zones = [z for z in sd_zones if z.is_supply and z.is_valid]
        for z in active_supply_zones:
            if (z.bottom * 0.99 <= current_high <= z.top) or (abs(current_close - z.bottom) <= current_atr * 0.8):
                selected_bear_sd_zone = z
                if z.zone_type == SDZoneType.EXTREME_SUPPLY:
                    bearish_score += 35
                    bearish_reasons.append(f"Retesting Extreme Supply Zone (₹{z.bottom:.2f} - ₹{z.top:.2f})")
                else:
                    bearish_score += 30
                    bearish_reasons.append(f"Retesting Decisional Supply Zone (₹{z.bottom:.2f} - ₹{z.top:.2f})")

                if z.is_fresh:
                    bearish_score += 15
                    bearish_reasons.append("Pristine unmitigated institutional zone (0 touches)")
                elif z.touch_count == 1:
                    bearish_score += 8
                    bearish_reasons.append("First test of supply zone")

                if z.has_valid_imbalance:
                    bearish_score += 10
                    bearish_reasons.append(f"Verified by Imbalance/FVG gap (₹{z.imbalance_size:.2f})")
                break

        if not selected_bear_sd_zone:
            active_bear_obs = [ob for ob in legacy_obs if ob.ob_type == "BEARISH" and ob.is_valid and not ob.is_mitigated]
            for ob in active_bear_obs:
                if ob.bottom * 0.98 <= current_high <= ob.top or abs(current_close - ob.bottom) <= current_atr * 0.8:
                    bearish_score += 30
                    bearish_reasons.append(f"Retesting Supply Order Block (₹{ob.bottom:.2f} - ₹{ob.top:.2f})")
                    selected_legacy_ob = ob
                    break

        if structure_state.swing_trend == StructureTrend.BEARISH:
            bearish_score += 20
            bearish_reasons.append("Macro Swing Structure is Bearish (Lower Lows & Lower Highs)")
            if structure_state.is_internal_realigned:
                bearish_score += 25
                bearish_reasons.append("⚡ Internal CHoCH Trigger: Pullback ended, realigned with Swing Trend")
            elif structure_state.is_pullback:
                bearish_reasons.append("Pullback into Premium Supply in progress")

        if eq_status == "PREMIUM":
            bearish_score += 15
            bearish_reasons.append("Trading in Institutional Premium Zone (>50% dealing range)")

        recent_bsl_sweeps = [s for s in sweeps if s.sweep_type == "BSL_SWEEP" and (len(clean_df) - 1 - s.bar_idx) <= 3]
        if recent_bsl_sweeps:
            selected_sweep = recent_bsl_sweeps[-1]
            bearish_score += 20
            bearish_reasons.append(f"Buy-Side Liquidity (BSL) swept at ₹{selected_sweep.level:.2f} with swift seller rejection")

        unfilled_bear_fvgs = [f for f in fvgs if f.fvg_type == "BEARISH" and not f.is_filled]
        for fvg in unfilled_bear_fvgs:
            if fvg.bottom <= current_high <= fvg.top or abs(current_close - fvg.bottom) <= current_atr * 0.5:
                bearish_score += 10
                bearish_reasons.append(f"Confluence with Bearish FVG (₹{fvg.bottom:.2f} - ₹{fvg.top:.2f})")
                selected_bear_fvg = fvg
                break

        if rvol >= self.ob_vol_multiplier:
            bearish_score += 10
            bearish_reasons.append(f"Volume Surge {rvol:.1f}x vs 20 SMA")

        if option_context:
            call_wall = option_context.get("call_wall", 0.0)
            max_pain = option_context.get("max_pain", 0.0)
            pcr = option_context.get("pcr", 1.0)
            if call_wall > 0 and abs(current_close - call_wall) <= current_atr * 1.5:
                bearish_score += 10
                bearish_reasons.append(f"Derivative Confluence: Strong Call OI Wall Resistance at ₹{call_wall:.2f}")
            if max_pain > 0 and current_close > max_pain:
                bearish_score += 5
                bearish_reasons.append(f"Max Pain Gravitational Pull downward toward ₹{max_pain:.2f}")
            if pcr < 0.7:
                bearish_score += 5
                bearish_reasons.append(f"Bearish Option PCR Sentiment ({pcr:.2f})")

        # ── 3. BUILD AND RETURN HIGHEST CONVICTION SETUP ──────────────────────
        # Bullish Setup
        if bullish_score >= 50 and bullish_score > bearish_score:
            base_sl = selected_sd_zone.bottom if selected_sd_zone else (
                selected_legacy_ob.bottom if selected_legacy_ob else range_low
            )
            strong_low_val = structure_state.strong_protected_level if (
                structure_state.swing_trend == StructureTrend.BULLISH and structure_state.strong_protected_level is not None
            ) else None

            if strong_low_val is not None:
                sl = min(base_sl, strong_low_val) - (current_atr * 0.25)
                prot_lvl = strong_low_val
            else:
                sl = base_sl - (current_atr * 0.25)
                prot_lvl = base_sl

            risk = max(current_close - sl, current_atr * 0.4)

            # Target 1: JeaFx Range-to-Range target (nearest opposing supply zone)
            opposing_supply = self.supply_demand_engine.find_opposing_target_zone(
                current_price=current_close,
                is_bullish=True,
                active_zones=sd_zones,
            )
            t1 = opposing_supply.bottom if opposing_supply else (current_close + risk * 2.0)
            if t1 <= current_close:
                t1 = current_close + risk * 2.0

            # Target 2: Photon Weak Target High
            weak_target = structure_state.weak_target_level if structure_state.swing_trend == StructureTrend.BULLISH else None
            t2 = weak_target if (weak_target and weak_target > t1) else (current_close + risk * 3.5)

            rrr = (t1 - current_close) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr or (t2 - current_close) / risk >= self.min_rrr:
                z_name = selected_sd_zone.zone_type.value if selected_sd_zone else (
                    "DEMAND_OB" if selected_legacy_ob else "DISCOUNT_SUPPORT"
                )
                setup_label = f"PRO_TREND_{z_name}" if structure_state.swing_trend == StructureTrend.BULLISH else f"REVERSAL_{z_name}"
                return SmcTradeSetup(
                    symbol=symbol,
                    action="BUY CALL",
                    direction=1,
                    setup_type=setup_label,
                    confluence_score=min(100.0, bullish_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(max(rrr, (t2 - current_close) / risk), 2),
                    market_structure=structure_state.alignment_description,
                    equilibrium_status=eq_status,
                    zone_classification=z_name,
                    strong_protected_level=round(prot_lvl, 2) if prot_lvl else None,
                    weak_target_level=round(t2, 2) if t2 else None,
                    opposing_target_zone=round(opposing_supply.bottom, 2) if opposing_supply else None,
                    is_internal_realigned=structure_state.is_internal_realigned,
                    reasons=bullish_reasons,
                    active_ob=selected_legacy_ob,
                    active_fvg=selected_fvg,
                    recent_sweep=selected_sweep,
                    active_sd_zone=selected_sd_zone,
                    structure_state=structure_state,
                    timestamp=bar_date,
                    spot_price=current_close,
                )

        # Bearish Setup
        elif bearish_score >= 50:
            base_sl = selected_bear_sd_zone.top if selected_bear_sd_zone else (
                selected_legacy_ob.top if selected_legacy_ob else range_high
            )
            strong_high_val = structure_state.strong_protected_level if (
                structure_state.swing_trend == StructureTrend.BEARISH and structure_state.strong_protected_level is not None
            ) else None

            if strong_high_val is not None:
                sl = max(base_sl, strong_high_val) + (current_atr * 0.25)
                prot_lvl = strong_high_val
            else:
                sl = base_sl + (current_atr * 0.25)
                prot_lvl = base_sl

            risk = max(sl - current_close, current_atr * 0.4)

            # Target 1: JeaFx Range-to-Range target (nearest opposing demand zone)
            opposing_demand = self.supply_demand_engine.find_opposing_target_zone(
                current_price=current_close,
                is_bullish=False,
                active_zones=sd_zones,
            )
            t1 = opposing_demand.top if opposing_demand else (current_close - risk * 2.0)
            if t1 >= current_close:
                t1 = current_close - risk * 2.0

            weak_target = structure_state.weak_target_level if structure_state.swing_trend == StructureTrend.BEARISH else None
            t2 = weak_target if (weak_target and weak_target < t1) else (current_close - risk * 3.5)

            rrr = (current_close - t1) / risk if risk > 0 else 2.0

            if rrr >= self.min_rrr or (current_close - t2) / risk >= self.min_rrr:
                z_name = selected_bear_sd_zone.zone_type.value if selected_bear_sd_zone else (
                    "SUPPLY_OB" if selected_legacy_ob else "PREMIUM_RESISTANCE"
                )
                setup_label = f"PRO_TREND_{z_name}" if structure_state.swing_trend == StructureTrend.BEARISH else f"REVERSAL_{z_name}"
                return SmcTradeSetup(
                    symbol=symbol,
                    action="BUY PUT",
                    direction=-1,
                    setup_type=setup_label,
                    confluence_score=min(100.0, bearish_score),
                    entry_price=round(current_close, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(t1, 2),
                    target_2=round(t2, 2),
                    risk_reward_ratio=round(max(rrr, (current_close - t2) / risk), 2),
                    market_structure=structure_state.alignment_description,
                    equilibrium_status=eq_status,
                    zone_classification=z_name,
                    strong_protected_level=round(prot_lvl, 2) if prot_lvl else None,
                    weak_target_level=round(t2, 2) if t2 else None,
                    opposing_target_zone=round(opposing_demand.top, 2) if opposing_demand else None,
                    is_internal_realigned=structure_state.is_internal_realigned,
                    reasons=bearish_reasons,
                    active_ob=selected_legacy_ob,
                    active_fvg=selected_bear_fvg,
                    recent_sweep=selected_sweep,
                    active_sd_zone=selected_bear_sd_zone,
                    structure_state=structure_state,
                    timestamp=bar_date,
                    spot_price=current_close,
                )

        return None

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        BaseStrategy requirement: Generates historical signal dataframe.
        """
        if data is None or data.empty:
            return pd.DataFrame()

        df = data.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["stop_loss"] = np.nan
        df["target_1"] = np.nan
        df["target_2"] = np.nan

        # Rolling window evaluation across dataset
        n = len(df)
        window = max(30, self.swing_length * 4)
        for i in range(window, n, 5):  # Step by 5 bars for performance
            sub_df = df.iloc[: i + 1]
            setup = self.analyze_symbol(sub_df, symbol="HISTORICAL")
            if setup:
                df.iloc[i, df.columns.get_loc("signal")] = setup.direction
                df.iloc[i, df.columns.get_loc("entry_price")] = setup.entry_price
                df.iloc[i, df.columns.get_loc("stop_loss")] = setup.stop_loss
                df.iloc[i, df.columns.get_loc("target_1")] = setup.target_1
                df.iloc[i, df.columns.get_loc("target_2")] = setup.target_2

        return df

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        """Returns formatted signal message for notifications."""
        sig = row.get("signal", 0)
        action = "BUY CALL" if sig == 1 else ("BUY PUT" if sig == -1 else "HOLD")
        entry = row.get("entry_price", row.get("close", 0.0))
        sl = row.get("stop_loss", 0.0)
        t1 = row.get("target_1", 0.0)
        return (
            f"🏛️ <b>SMC Alert: {symbol}</b>\n"
            f"Action: <b>{action}</b> @ ₹{entry:.2f}\n"
            f"Protected SL: ₹{sl:.2f} | T1 (Range): ₹{t1:.2f}"
        )

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        """
        Real-time evaluation hook conforming to BaseStrategy.
        """
        if context.history_df is None or context.history_df.empty:
            return None

        # Build optional option context
        option_ctx = None
        if context.option_chain_df is not None and not context.option_chain_df.empty:
            oc_df = context.option_chain_df
            # Compute call / put wall
            call_wall = float(oc_df.loc[oc_df["call_oi"].idxmax()]["strike_price"]) if "call_oi" in oc_df.columns else 0.0
            put_wall = float(oc_df.loc[oc_df["put_oi"].idxmax()]["strike_price"]) if "put_oi" in oc_df.columns else 0.0
            tot_call_oi = oc_df["call_oi"].sum() if "call_oi" in oc_df.columns else 1.0
            tot_put_oi = oc_df["put_oi"].sum() if "put_oi" in oc_df.columns else 1.0
            pcr = float(tot_put_oi / tot_call_oi) if tot_call_oi > 0 else 1.0
            option_ctx = {
                "call_wall": call_wall,
                "put_wall": put_wall,
                "pcr": pcr,
            }

        setup = self.analyze_symbol(context.history_df, symbol=context.symbol, option_context=option_ctx)
        if not setup:
            return None

        now = datetime.now()
        return TradeSignal(
            symbol=context.symbol,
            timestamp=now,
            direction="CALL" if setup.direction == 1 else "PUT",
            action=setup.action.replace(" ", "_"),
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            target_1=setup.target_1,
            target_2=setup.target_2,
            confidence=round(setup.confluence_score / 100.0, 2),
            strategy_name=self.name,
            timeframe=context.timeframe,
            confluence_factors=setup.reasons,
            metadata={
                "setup_type": setup.setup_type,
                "zone_classification": setup.zone_classification,
                "market_structure": setup.market_structure,
                "equilibrium_status": setup.equilibrium_status,
                "strong_protected_level": setup.strong_protected_level,
                "weak_target_level": setup.weak_target_level,
                "opposing_target_zone": setup.opposing_target_zone,
                "is_internal_realigned": setup.is_internal_realigned,
            }
        )
