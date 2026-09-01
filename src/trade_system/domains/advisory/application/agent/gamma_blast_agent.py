"""
GammaBlastAgent — Pro Option Buyer's Gamma Explosion Detector.

Identifies when conditions converge for explosive 5-15% option premium moves:
1. Expiry Proximity (gamma is highest on expiry day / day before)
2. Negative GEX Zone (market makers amplify moves instead of dampening)
3. Breakout + Volume Surge (the catalyst)
4. IV Expansion opportunity (cheap options about to explode)
5. OI Wall proximity (institutional fuel for the move)

This agent thinks like a professional gamma scalper who buys ATM/slightly-OTM
weekly options and rides the explosive move when the market breaks a key level
in a negative-gamma environment.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, time as dt_time, timedelta
from typing import Any, Optional, List

import pandas as pd
import numpy as np

from trade_system.shared import (
    TradeSuggestion, TradeDirection, TradeHorizon, OptionParams,
    OptionChainAnalysis,
)
from trade_system.domains.market_data.infrastructure.data.fo_universe import FO_METADATA
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import log_agent_thought
from sqlalchemy.orm import Session

LOGGER = logging.getLogger(__name__)


@dataclass
class GammaContext:
    """Contextual data for gamma blast evaluation."""
    days_to_expiry: int = 5
    is_expiry_day: bool = False
    is_expiry_eve: bool = False
    gex_label: str = "NEUTRAL"
    total_gex: float = 0.0
    gamma_wall: float = 0.0
    iv_percentile: float = 50.0  # 0-100
    pcr: float = 1.0


class GammaBlastAgent:
    """
    Detects setups where option premiums are about to explode due to
    gamma acceleration — the professional option buyer's edge.
    """

    # Tunable thresholds
    VWAP_STRETCH_MIN = 1.0        # Minimum % stretch from VWAP for momentum
    VOLUME_SURGE_MIN = 1.8        # Minimum volume surge multiplier
    RSI_MOMENTUM_CALL = 55        # RSI must be above this for bullish gamma
    RSI_MOMENTUM_PUT = 45         # RSI must be below this for bearish gamma
    IV_CHEAP_PERCENTILE = 40      # IV below this = cheap gamma (buy opportunity)
    OI_WALL_PROXIMITY_PCT = 2.0   # Max % distance to nearest OI wall

    def __init__(self) -> None:
        self.engine = get_engine()
        self._daily_signals: dict[str, int] = {}  # symbol -> count today
        self._last_signal_time: dict[str, datetime] = {}

    def _thought(self, message: str, symbol: str | None = None, action: str | None = None):
        LOGGER.info(f"[GammaBlastAgent] {message}")
        try:
            with Session(self.engine) as session:
                log_agent_thought(session, "GammaBlastAgent", message, symbol, action)
        except:
            pass

    def get_gamma_context(self, oc_analysis: Optional[OptionChainAnalysis] = None) -> GammaContext:
        """Build gamma context from option chain data and calendar."""
        ctx = GammaContext()

        # 1. Expiry Calendar (NSE weekly expiry = Thursday)
        today = date.today()
        weekday = today.weekday()  # 0=Mon, 3=Thu
        if weekday <= 3:
            days_to_expiry = 3 - weekday
        else:
            days_to_expiry = 7 - weekday + 3  # Next Thursday

        ctx.days_to_expiry = days_to_expiry
        ctx.is_expiry_day = (days_to_expiry == 0)
        ctx.is_expiry_eve = (days_to_expiry == 1)

        # 2. GEX from Option Chain
        if oc_analysis and hasattr(oc_analysis, 'metadata') and oc_analysis.metadata:
            gex = oc_analysis.metadata.get('gex', {})
            ctx.gex_label = gex.get('gex_label', 'NEUTRAL')
            ctx.total_gex = gex.get('total_gex', 0.0)
            ctx.gamma_wall = gex.get('gamma_wall', 0.0)
            ctx.pcr = oc_analysis.pcr
            ctx.iv_percentile = oc_analysis.iv_percentile or 50.0

        return ctx

    def analyze_for_gamma_blast(
        self,
        symbol: str,
        df: pd.DataFrame,
        oc_analysis: Optional[OptionChainAnalysis] = None,
        gamma_ctx: Optional[GammaContext] = None,
    ) -> Optional[TradeSuggestion]:
        """
        The core gamma blast detection engine.

        Logic flow:
        1. Is gamma environment favorable? (near expiry + negative GEX)
        2. Is there a breakout happening? (price breaking key level)
        3. Is volume confirming? (institutional participation)
        4. Is IV cheap? (maximum leverage on premium)
        5. Is price near an OI wall? (fuel for the move)
        """
        if df.empty or len(df) < 30:
            return None

        # Rate limit: max 2 signals per symbol per day
        today_key = f"{symbol}_{pd.to_datetime(df.iloc[-1]['timestamp']).date()}"
        if self._daily_signals.get(today_key, 0) >= 2:
            return None

        # Debounce: minimum 30 minutes between signals for same symbol
        last_time = self._last_signal_time.get(symbol)
        if last_time and (pd.to_datetime(df.iloc[-1]["timestamp"]) - last_time).total_seconds() < 1800:
            return None

        latest = df.iloc[-1]
        price = float(latest['close'])

        # Build gamma context if not provided
        if gamma_ctx is None:
            gamma_ctx = self.get_gamma_context(oc_analysis)

        # ═══════════════════════════════════════════════════════
        # LAYER 1: GAMMA ENVIRONMENT CHECK
        # ═══════════════════════════════════════════════════════
        gamma_multiplier = 1.0

        # Expiry proximity boosts gamma sensitivity
        if gamma_ctx.is_expiry_day:
            gamma_multiplier = 2.5  # Maximum gamma on expiry day
            self._thought(f"EXPIRY DAY: Gamma multiplier 2.5x for {symbol}", symbol, "GAMMA_BOOST")
        elif gamma_ctx.is_expiry_eve:
            gamma_multiplier = 1.8
        elif gamma_ctx.days_to_expiry <= 2:
            gamma_multiplier = 1.4

        # Negative GEX = market makers amplify moves (the blast zone)
        is_negative_gex = "NEGATIVE" in gamma_ctx.gex_label
        if is_negative_gex:
            gamma_multiplier *= 1.3
            self._thought(f"NEGATIVE GEX detected for {symbol}: Market makers are SHORT gamma", symbol, "GEX_ALERT")

        # If gamma environment is completely unfavorable, skip
        if gamma_multiplier < 1.2 and not is_negative_gex:
            return None  # Not enough gamma juice

        # ═══════════════════════════════════════════════════════
        # LAYER 2: BREAKOUT DETECTION
        # ═══════════════════════════════════════════════════════
        breakout_direction = None
        breakout_level = None
        breakout_type = ""

        # Check VWAP for momentum direction
        vwap = latest.get('vwap')
        if not vwap or vwap <= 0:
            return None

        dist_vwap_pct = ((price - vwap) / vwap) * 100

        # Previous day high/low breakout (most powerful for gamma)
        if 'date' not in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['timestamp'], format="mixed").dt.date

        today_mask = df['date'] == df['date'].iloc[-1]
        prev_day_df = df[~today_mask]

        if len(prev_day_df) >= 10:
            prev_day_high = prev_day_df['high'].max()
            prev_day_low = prev_day_df['low'].min()

            if price > prev_day_high and dist_vwap_pct > self.VWAP_STRETCH_MIN:
                breakout_direction = TradeDirection.CALL
                breakout_level = prev_day_high
                breakout_type = "PREV_DAY_HIGH_BREAKOUT"

            elif price < prev_day_low and dist_vwap_pct < -self.VWAP_STRETCH_MIN:
                breakout_direction = TradeDirection.PUT
                breakout_level = prev_day_low
                breakout_type = "PREV_DAY_LOW_BREAKDOWN"

        # If no prev-day breakout, check intraday range breakout
        if breakout_direction is None:
            today_df = df[today_mask]
            if len(today_df) >= 6:
                # Check for 30-minute high/low breakout
                first_30m = today_df.head(6)  # First 6 bars of 5m = 30 minutes
                orb_high = first_30m['high'].max()
                orb_low = first_30m['low'].min()

                if price > orb_high * 1.002 and dist_vwap_pct > self.VWAP_STRETCH_MIN:
                    breakout_direction = TradeDirection.CALL
                    breakout_level = orb_high
                    breakout_type = "30M_ORB_BREAKOUT"

                elif price < orb_low * 0.998 and dist_vwap_pct < -self.VWAP_STRETCH_MIN:
                    breakout_direction = TradeDirection.PUT
                    breakout_level = orb_low
                    breakout_type = "30M_ORB_BREAKDOWN"

        if breakout_direction is None:
            return None  # No breakout detected

        # ═══════════════════════════════════════════════════════
        # LAYER 3: VOLUME CONFIRMATION
        # ═══════════════════════════════════════════════════════
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        if vol_ma <= 0:
            return None
        vol_surge = float(latest['volume']) / vol_ma

        # Near expiry, we accept slightly lower volume thresholds
        adjusted_vol_min = self.VOLUME_SURGE_MIN
        if gamma_ctx.is_expiry_day:
            adjusted_vol_min = 1.3  # Lower bar on expiry day
        elif gamma_ctx.is_expiry_eve:
            adjusted_vol_min = 1.5

        if vol_surge < adjusted_vol_min:
            return None

        # ═══════════════════════════════════════════════════════
        # LAYER 4: RSI MOMENTUM CONFIRMATION
        # ═══════════════════════════════════════════════════════
        rsi_val = latest.get('rsi')
        if rsi_val is None:
            # Calculate RSI manually
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_val = float(rsi_series.iloc[-1])

        if breakout_direction == TradeDirection.CALL and rsi_val < self.RSI_MOMENTUM_CALL:
            return None  # Weak momentum for bullish gamma
        if breakout_direction == TradeDirection.PUT and rsi_val > self.RSI_MOMENTUM_PUT:
            return None  # Weak momentum for bearish gamma

        # ═══════════════════════════════════════════════════════
        # LAYER 5: OI WALL PROXIMITY (FUEL CHECK)
        # ═══════════════════════════════════════════════════════
        oi_wall_msg = ""
        if oc_analysis:
            if breakout_direction == TradeDirection.CALL:
                # For bullish gamma, we want price approaching the Call OI wall
                # (market makers will be forced to buy to hedge = amplification)
                ce_wall = oc_analysis.highest_ce_oi_strike
                if ce_wall > 0:
                    dist_to_wall = ((ce_wall - price) / price) * 100
                    if 0 < dist_to_wall <= self.OI_WALL_PROXIMITY_PCT:
                        gamma_multiplier *= 1.2
                        oi_wall_msg = f" Approaching Call Wall at {ce_wall} ({dist_to_wall:.1f}% away)."
            else:
                pe_wall = oc_analysis.highest_pe_oi_strike
                if pe_wall > 0:
                    dist_to_wall = ((price - pe_wall) / price) * 100
                    if 0 < dist_to_wall <= self.OI_WALL_PROXIMITY_PCT:
                        gamma_multiplier *= 1.2
                        oi_wall_msg = f" Approaching Put Wall at {pe_wall} ({dist_to_wall:.1f}% away)."

        # ═══════════════════════════════════════════════════════
        # LAYER 6: IV CHECK (CHEAP GAMMA?)
        # ═══════════════════════════════════════════════════════
        iv_msg = ""
        if gamma_ctx.iv_percentile <= self.IV_CHEAP_PERCENTILE:
            gamma_multiplier *= 1.15
            iv_msg = f" IV at {gamma_ctx.iv_percentile:.0f}th percentile (CHEAP GAMMA)."

        # ═══════════════════════════════════════════════════════
        # BUILD THE GAMMA BLAST SIGNAL
        # ═══════════════════════════════════════════════════════
        confidence = min(0.95, 0.55 + (gamma_multiplier - 1.0) * 0.15 + (vol_surge - 1.0) * 0.05)

        # Dynamic SL/Target based on ATR
        atr = self._calc_atr(df)
        if atr is None or atr <= 0:
            atr = price * 0.015  # Fallback: 1.5% of price

        # Gamma trades: tight SL (1x ATR), aggressive target (3x ATR on expiry, 2x otherwise)
        target_multiplier = 3.0 if gamma_ctx.is_expiry_day else 2.5 if gamma_ctx.is_expiry_eve else 2.0

        if breakout_direction == TradeDirection.CALL:
            stop_loss = breakout_level * 0.998  # Just below breakout level
            target = price + (atr * target_multiplier)
        else:
            stop_loss = breakout_level * 1.002  # Just above breakdown level
            target = price - (atr * target_multiplier)

        # Horizon: BTST if after 2 PM, else INTRADAY
        now = pd.to_datetime(df.iloc[-1]["timestamp"])
        horizon = TradeHorizon.SWING if now.hour >= 14 else TradeHorizon.INTRADAY

        # Expiry context for narrative
        expiry_ctx = ""
        if gamma_ctx.is_expiry_day:
            expiry_ctx = "🔥 EXPIRY DAY GAMMA. "
        elif gamma_ctx.is_expiry_eve:
            expiry_ctx = "⚡ EXPIRY EVE GAMMA. "

        gex_ctx = ""
        if is_negative_gex:
            gex_ctx = "Market Makers SHORT GAMMA (amplifying moves). "

        narrative = (
            f"💥 GAMMA BLAST: {breakout_type} with {vol_surge:.1f}x volume surge. "
            f"{expiry_ctx}{gex_ctx}"
            f"RSI: {rsi_val:.0f}. VWAP Stretch: {dist_vwap_pct:+.1f}%. "
            f"Gamma Multiplier: {gamma_multiplier:.1f}x."
            f"{oi_wall_msg}{iv_msg}"
        )

        self._thought(narrative, symbol, "GAMMA_BLAST")

        # Record the signal
        self._daily_signals[today_key] = self._daily_signals.get(today_key, 0) + 1
        self._last_signal_time[symbol] = pd.to_datetime(df.iloc[-1]["timestamp"])

        return TradeSuggestion(
            id=str(uuid.uuid4()),
            symbol=symbol,
            timestamp=pd.to_datetime(df.iloc[-1]["timestamp"]),
            direction=breakout_direction,
            horizon=horizon,
            entry_zone_low=price * 0.998,
            entry_zone_high=price * 1.002,
            target=round(target, 2),
            stop_loss=round(stop_loss, 2),
            option_params=OptionParams(
                direction=breakout_direction,
                expiry_type="weekly",
            ),
            confidence=round(confidence, 3),
            narrative=narrative,
            setup_features=None,
            sector=FO_METADATA.get(symbol, "Other"),
            tags=["gamma_blast", breakout_type.lower(), "pro_option_buyer"],
        )

    def scan_gamma_potential_postmarket(
        self,
        symbols: List[str],
        oc_analysis: Optional[OptionChainAnalysis] = None,
    ) -> List[dict]:
        """
        Post-market scan: Rank stocks by their gamma blast potential for tomorrow.
        Returns a sorted list of dicts with gamma scores.
        """
        gamma_ctx = self.get_gamma_context(oc_analysis)
        results = []

        with Session(self.engine) as session:
            from trade_system.domains.market_data.infrastructure.database.repository import get_market_data

            for symbol in symbols:
                try:
                    data = get_market_data(session, symbol, "5", limit=78)  # ~1 day
                    if not data or len(data) < 30:
                        continue

                    df = pd.DataFrame([
                        {"timestamp": d.timestamp, "open": d.open, "high": d.high,
                         "low": d.low, "close": d.close, "volume": d.volume}
                        for d in data
                    ])

                    price = float(df['close'].iloc[-1])
                    day_high = df['high'].max()
                    day_low = df['low'].min()
                    day_range = day_high - day_low

                    # 1. Compression Score (tight range = coiling for breakout)
                    atr = self._calc_atr(df)
                    compression = 1.0 - min(1.0, (day_range / (atr * 2)) if atr and atr > 0 else 0.5)

                    # 2. Close Position (near HOD = bullish, near LOD = bearish)
                    if day_range > 0:
                        close_position = (price - day_low) / day_range  # 0=LOD, 1=HOD
                    else:
                        close_position = 0.5

                    # 3. Volume Profile (was today's volume high?)
                    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
                    vol_ratio = float(df['volume'].iloc[-1]) / vol_ma if vol_ma > 0 else 1.0

                    # 4. OI Wall proximity
                    oi_proximity = 0.0
                    if oc_analysis:
                        ce_dist = abs(price - oc_analysis.highest_ce_oi_strike) / price * 100 if oc_analysis.highest_ce_oi_strike > 0 else 99
                        pe_dist = abs(price - oc_analysis.highest_pe_oi_strike) / price * 100 if oc_analysis.highest_pe_oi_strike > 0 else 99
                        nearest_wall_dist = min(ce_dist, pe_dist)
                        if nearest_wall_dist <= 3.0:
                            oi_proximity = 1.0 - (nearest_wall_dist / 3.0)

                    # 5. Gamma Potential Score
                    gamma_score = (
                        compression * 0.30 +           # Coiling = high potential
                        abs(close_position - 0.5) * 0.25 +  # Strong close = momentum
                        min(1.0, vol_ratio / 2.0) * 0.20 +  # Volume interest
                        oi_proximity * 0.15 +           # Near institutional walls
                        (1.0 if gamma_ctx.days_to_expiry <= 2 else 0.5) * 0.10  # Expiry proximity
                    )

                    direction = "CALL" if close_position > 0.6 else "PUT" if close_position < 0.4 else "NEUTRAL"

                    results.append({
                        "symbol": symbol,
                        "gamma_score": round(gamma_score, 3),
                        "direction": direction,
                        "price": round(price, 2),
                        "compression": round(compression, 3),
                        "close_position": round(close_position, 3),
                        "vol_ratio": round(vol_ratio, 2),
                        "oi_proximity": round(oi_proximity, 3),
                        "days_to_expiry": gamma_ctx.days_to_expiry,
                    })

                except Exception as e:
                    LOGGER.debug(f"Gamma scan error for {symbol}: {e}")

        # Sort by gamma score descending
        results.sort(key=lambda x: x['gamma_score'], reverse=True)
        return results

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        if len(df) < period + 1:
            return None
        try:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            val = tr.rolling(period).mean().iloc[-1]
            return round(float(val), 3) if pd.notna(val) else None
        except:
            return None
