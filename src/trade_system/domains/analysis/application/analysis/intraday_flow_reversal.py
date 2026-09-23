"""
Intraday Smart Flow Reversal Engine (ISFRE).

Quantifies intraday momentum, price thrust from session extremes (Low/High),
relative volume surges (RVol), VWAP reclaims, and derivatives Open Interest (OI)
dynamics (Long Buildup, Short Covering, Short Buildup, Long Unwinding) across
the F&O stock universe.
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
class IntradayReversalSetup:
    """Structured container for an individual stock's intraday reversal/momentum metrics."""
    symbol: str
    clean_symbol: str
    sector: str
    ltp: float
    pchange: float
    open_today: float
    high_today: float
    low_today: float
    move_from_low_pct: float      # ((LTP - Low) / Low) * 100
    drop_from_high_pct: float     # ((High - LTP) / High) * 100
    range_pos_pct: float          # ((LTP - Low) / (High - Low)) * 100 (0 to 100)
    vol_surge: float              # Relative volume vs time-adjusted historical average
    volume_today: int
    vwap: Optional[float]
    vs_vwap: Optional[float]      # ((LTP - VWAP) / VWAP) * 100
    daily_rsi: Optional[float]
    pcr_oi: Optional[float]
    pcr_volume: Optional[float]
    sentiment_state: str          # OVERBOUGHT, OVERSOLD, NEUTRAL, etc.
    flow_quadrant: str            # LONG_BUILDUP, SHORT_COVERING, SHORT_BUILDUP, LONG_UNWINDING, NEUTRAL
    direction: str                # BULLISH, BEARISH, NEUTRAL
    reversal_score: int           # 0 to 100 composite institutional conviction score
    key_catalyst: str             # Explanatory narrative of the setup
    entry_trigger: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    risk_reward: Optional[float] = None


class IntradayFlowReversalEngine:
    """
    Quantitative engine analyzing intraday recovery, exhaustion, and smart money flow.
    """

    @staticmethod
    def classify_derivative_quadrant(
        price_change: float,
        move_from_low: float,
        drop_from_high: float,
        pcr_oi: Optional[float] = None,
        sentiment_state: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Classifies stock into institutional derivatives quadrant based on price thrust & PCR/OI:
        - LONG_BUILDUP: Price expanding from low + healthy/rising PCR (fresh accumulation)
        - SHORT_COVERING: Reversal from extreme oversold conditions (PCR <= 0.55 / trapped call writers covering)
        - SHORT_BUILDUP: Price breaking down from high + bearish PCR/OI addition
        - LONG_UNWINDING: Reversal from extreme overbought conditions (PCR >= 0.85 / bulls dumping)
        """
        # Rebound from low (Bullish thrust)
        if move_from_low >= 1.2:
            if pcr_oi is not None and (pcr_oi <= 0.55 or sentiment_state in ["OVERSOLD", "EXTREME_OVERSOLD"]):
                return "⚡ SHORT COVERING", "BULLISH"
            elif pcr_oi is not None and pcr_oi >= 0.70:
                return "🟢 LONG BUILD-UP", "BULLISH"
            elif price_change >= 0.5:
                return "🟢 LONG BUILD-UP", "BULLISH"
            else:
                return "⚡ SHORT COVERING", "BULLISH"

        # Pullback from high (Bearish rejection)
        elif drop_from_high >= 1.2:
            if pcr_oi is not None and (pcr_oi >= 0.85 or sentiment_state in ["OVERBOUGHT", "EXTREME_OVERBOUGHT"]):
                return "🩸 LONG UNWINDING", "BEARISH"
            elif pcr_oi is not None and pcr_oi <= 0.65:
                return "🔴 SHORT BUILD-UP", "BEARISH"
            elif price_change <= -0.5:
                return "🔴 SHORT BUILD-UP", "BEARISH"
            else:
                return "🩸 LONG UNWINDING", "BEARISH"

        return "⚪ BALANCED CHURN", "NEUTRAL"

    @staticmethod
    def compute_composite_score(
        thrust_pct: float,
        range_pos: float,
        vol_surge: float,
        vs_vwap: Optional[float],
        pcr_oi: Optional[float],
        direction: str,
    ) -> int:
        """
        Computes an institutional confluence score from 0 to 100:
        - Intraday Thrust magnitude (up to 30 pts)
        - Range Position conviction (up to 25 pts)
        - Volume Surge multiplier (up to 25 pts)
        - VWAP Alignment (up to 10 pts)
        - Derivative OI/PCR Confluence (up to 10 pts)
        """
        score = 0

        # 1. Thrust magnitude (0 to 30 pts)
        if thrust_pct >= 3.0:
            score += 30
        elif thrust_pct >= 2.0:
            score += 25
        elif thrust_pct >= 1.5:
            score += 20
        elif thrust_pct >= 1.0:
            score += 15
        else:
            score += int(thrust_pct * 10)

        # 2. Range Position conviction (0 to 25 pts)
        if direction == "BULLISH":
            if range_pos >= 85:
                score += 25
            elif range_pos >= 70:
                score += 20
            elif range_pos >= 55:
                score += 12
        elif direction == "BEARISH":
            if range_pos <= 15:
                score += 25
            elif range_pos <= 30:
                score += 20
            elif range_pos <= 45:
                score += 12
        else:
            score += 5

        # 3. Volume Surge multiplier (0 to 25 pts)
        if vol_surge >= 3.0:
            score += 25
        elif vol_surge >= 2.0:
            score += 20
        elif vol_surge >= 1.5:
            score += 15
        elif vol_surge >= 1.0:
            score += 10
        elif vol_surge > 0:
            score += 5

        # 4. VWAP alignment (0 to 10 pts)
        if vs_vwap is not None:
            if direction == "BULLISH" and vs_vwap > 0:
                score += 10 if vs_vwap >= 0.3 else 7
            elif direction == "BEARISH" and vs_vwap < 0:
                score += 10 if vs_vwap <= -0.3 else 7

        # 5. Derivative OI / PCR Confluence (0 to 10 pts)
        if pcr_oi is not None:
            if direction == "BULLISH":
                if pcr_oi <= 0.55:  # Oversold short squeeze confluence
                    score += 10
                elif pcr_oi >= 0.75:  # Strong put writer floor
                    score += 8
            elif direction == "BEARISH":
                if pcr_oi >= 0.90:  # Overbought long liquidation confluence
                    score += 10
                elif pcr_oi <= 0.60:  # Strong call writer ceiling
                    score += 8

        return min(100, max(0, score))

    @classmethod
    def evaluate_universe(
        cls,
        merged_closes: pd.DataFrame,
        quotes: Optional[Dict[str, Any]] = None,
        pcr_cache: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> List[IntradayReversalSetup]:
        """
        Scan merged stock dataset and extract all intraday flow & reversal candidates.
        """
        if merged_closes is None or merged_closes.empty:
            return []

        quotes = quotes or {}
        indicators = indicators or {}

        # Build fast lookup for PCR cache (handles both StockPCRInfo objects and dicts)
        pcr_lookup: Dict[str, Any] = {}
        if pcr_cache:
            for item in pcr_cache.get("all", []):
                sym = getattr(item, "symbol", None) or (item.get("symbol", "") if isinstance(item, dict) else "")
                clean = getattr(item, "clean_symbol", None) or (item.get("clean_symbol", "") if isinstance(item, dict) else "")
                if not clean and sym:
                    clean = sym.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-INDEX", "")
                if sym:
                    pcr_lookup[sym] = item
                if clean:
                    pcr_lookup[clean] = item
                    pcr_lookup[f"NSE:{clean}-EQ"] = item

        setups: List[IntradayReversalSetup] = []

        for _, row in merged_closes.iterrows():
            symbol = row.get("symbol", "")
            clean = symbol.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-INDEX", "")
            sector = row.get("sector", "GENERAL")
            ltp = float(row.get("close_last", 0.0))
            pchange = float(row.get("pChange", 0.0))
            vol_surge = float(row.get("vol_surge", 0.0))
            volume_today = int(row.get("volume_today", 0))

            if ltp <= 0:
                continue

            # Determine Intraday High / Low
            high_today = float(row.get("high_today", 0.0))
            low_today = float(row.get("low_today", 0.0))
            open_today = float(row.get("open_today", 0.0))

            # Quote fallback if available
            if quotes and symbol in quotes:
                q = quotes[symbol]
                q_high = float(getattr(q, "high", 0.0) or 0.0)
                q_low = float(getattr(q, "low", 0.0) or 0.0)
                q_open = float(getattr(q, "open", 0.0) or 0.0)
                if q_high > 0:
                    high_today = max(high_today, q_high)
                if q_low > 0:
                    low_today = min(low_today, q_low) if low_today > 0 else q_low
                if q_open > 0 and open_today <= 0:
                    open_today = q_open

            # Safety clamps for single-candle or uninitialized extremes
            if high_today <= 0 or high_today < ltp:
                high_today = ltp
            if low_today <= 0 or low_today > ltp:
                low_today = ltp
            if open_today <= 0:
                open_today = ltp

            # Compute Thrusts
            move_from_low = ((ltp - low_today) / low_today * 100) if low_today > 0 else 0.0
            drop_from_high = ((high_today - ltp) / high_today * 100) if high_today > 0 else 0.0

            span = high_today - low_today
            range_pos = ((ltp - low_today) / span * 100) if span > 0 else 50.0

            # Indicators lookup
            ind = indicators.get(symbol) or indicators.get(clean) or {}
            vwap = ind.get("vwap")
            vs_vwap = ind.get("vs_vwap")
            daily_rsi = ind.get("daily_rsi")

            # PCR lookup (safely handle StockPCRInfo or dict)
            pcr_info = pcr_lookup.get(symbol) or pcr_lookup.get(clean)
            if pcr_info is not None:
                if isinstance(pcr_info, dict):
                    pcr_oi = pcr_info.get("pcr_oi")
                    pcr_volume = pcr_info.get("pcr_volume")
                    sentiment_state = pcr_info.get("sentiment_state", "NEUTRAL")
                else:
                    pcr_oi = getattr(pcr_info, "pcr_oi", None)
                    pcr_volume = getattr(pcr_info, "pcr_volume", None)
                    sentiment_state = getattr(pcr_info, "sentiment_state", "NEUTRAL")
            else:
                pcr_oi = None
                pcr_volume = None
                sentiment_state = "NEUTRAL"

            # Derivative Quadrant & Direction
            flow_quadrant, direction = cls.classify_derivative_quadrant(
                price_change=pchange,
                move_from_low=move_from_low,
                drop_from_high=drop_from_high,
                pcr_oi=pcr_oi,
                sentiment_state=sentiment_state,
            )

            thrust_to_score = move_from_low if direction == "BULLISH" else (drop_from_high if direction == "BEARISH" else max(move_from_low, drop_from_high))

            reversal_score = cls.compute_composite_score(
                thrust_pct=thrust_to_score,
                range_pos=range_pos,
                vol_surge=vol_surge,
                vs_vwap=vs_vwap,
                pcr_oi=pcr_oi,
                direction=direction,
            )

            # Generate narrative
            if direction == "BULLISH":
                vwap_str = f"reclaiming VWAP (₹{vwap:.1f})" if (vs_vwap is not None and vs_vwap >= 0) else "approaching VWAP"
                pcr_desc = f", PCR {pcr_oi:.2f} ({sentiment_state})" if pcr_oi else ""
                narrative = (
                    f"Surged +{move_from_low:.2f}% from Day Low (₹{low_today:.2f}) with {vol_surge:.1f}x volume surge, "
                    f"now trading in top {range_pos:.0f}% of range {vwap_str}{pcr_desc}."
                )
                entry_trigger = round(ltp * 1.002, 2)
                stop_loss = round(low_today * 0.997, 2)
                risk = entry_trigger - stop_loss
                target_1 = round(entry_trigger + 1.5 * risk, 2)
                target_2 = round(entry_trigger + 2.5 * risk, 2)
                rr = 2.0
            elif direction == "BEARISH":
                vwap_str = f"slipping below VWAP (₹{vwap:.1f})" if (vs_vwap is not None and vs_vwap <= 0) else "dropping"
                pcr_desc = f", PCR {pcr_oi:.2f} ({sentiment_state})" if pcr_oi else ""
                narrative = (
                    f"Dumped -{drop_from_high:.2f}% from Day High (₹{high_today:.2f}) with {vol_surge:.1f}x volume surge, "
                    f"now trading in bottom {100-range_pos:.0f}% of range {vwap_str}{pcr_desc}."
                )
                entry_trigger = round(ltp * 0.998, 2)
                stop_loss = round(high_today * 1.003, 2)
                risk = stop_loss - entry_trigger
                target_1 = round(entry_trigger - 1.5 * risk, 2)
                target_2 = round(entry_trigger - 2.5 * risk, 2)
                rr = 2.0
            else:
                narrative = f"Consolidating inside session range (₹{low_today:.2f}–₹{high_today:.2f})."
                entry_trigger = None
                stop_loss = None
                target_1 = None
                target_2 = None
                rr = None

            setup = IntradayReversalSetup(
                symbol=symbol,
                clean_symbol=clean,
                sector=sector,
                ltp=ltp,
                pchange=pchange,
                open_today=open_today,
                high_today=high_today,
                low_today=low_today,
                move_from_low_pct=round(move_from_low, 2),
                drop_from_high_pct=round(drop_from_high, 2),
                range_pos_pct=round(range_pos, 1),
                vol_surge=round(vol_surge, 2),
                volume_today=volume_today,
                vwap=round(vwap, 2) if vwap is not None else None,
                vs_vwap=round(vs_vwap, 2) if vs_vwap is not None else None,
                daily_rsi=round(daily_rsi, 1) if daily_rsi is not None else None,
                pcr_oi=round(pcr_oi, 2) if pcr_oi is not None else None,
                pcr_volume=round(pcr_volume, 2) if pcr_volume is not None else None,
                sentiment_state=sentiment_state,
                flow_quadrant=flow_quadrant,
                direction=direction,
                reversal_score=reversal_score,
                key_catalyst=narrative,
                entry_trigger=entry_trigger,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                risk_reward=rr,
            )
            setups.append(setup)

        # Sort primarily by composite reversal score descending
        setups.sort(key=lambda s: s.reversal_score, reverse=True)
        return setups
