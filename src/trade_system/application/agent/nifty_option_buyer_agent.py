"""
NiftyOptionBuyerAgent — Specialized agent for high-conviction Nifty 50 option buying.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, time
from typing import Any, List, Optional
import pandas as pd

from trade_system.core import (
    TradeSuggestion, 
    MarketContext,
    OptionChainAnalysis,
    TradeDirection,
    TradeHorizon,
    OptionParams,
    Signal,
    SignalType
)
from trade_system.application.indicators.rsi_divergence import RsiDivergence
from trade_system.application.indicators.trendlines_breaks import TrendlinesWithBreaks
from trade_system.application.indicators.sr_yata import SandRYata
from trade_system.application.indicators.supertrend import SupertrendIndicator

LOGGER = logging.getLogger(__name__)

from trade_system.infrastructure.data.storage import CsvDataCatalog

class NiftyOptionBuyerAgent:
    """
    Focused specifically on Nifty 50 Intraday Option Buying.
    Uses Supertrend (7,3) on 3m timeframe for primary trend check.
    """

    def __init__(self, broker: Any, settings: any) -> None:
        self.broker = broker
        self.settings = settings
        self.catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
        # Initialize indicators
        self.rsi_div = RsiDivergence(len_fast=5, len_slow=14)
        self.trendlines = TrendlinesWithBreaks(length=14, mult=1.0)
        self.sr_yata = SandRYata(length=21)
        self.supertrend = SupertrendIndicator(period=7, multiplier=3)
        self._last_signal_time = None

    async def analyze_and_suggest(
        self, 
        symbol: str,
        market_context: MarketContext,
        oc_analysis: OptionChainAnalysis,
        current_price: float
    ) -> Optional[TradeSuggestion]:
        """
        Runs the full technical checklist including Supertrend confirmation.
        """
        LOGGER.info(f"NiftyOptionBuyerAgent: Analyzing {symbol} with Supertrend (7,3) confluence...")

        # 1. Broad Bias Filter
        if market_context.bias == "NEUTRAL":
            return None

        # 2. Load Local Data (3-minute timeframe for Supertrend)
        try:
            # We use 5m as a base for technicals but Supertrend usually runs on 3m
            # Attempt to fetch/load 3m data
            path_3m = self.catalog.historical_path(symbol, "3")
            if not path_3m.exists():
                # Fallback to 5m if 3m not available
                path_3m = self.catalog.historical_path(symbol, "5")
            
            if not path_3m.exists():
                LOGGER.warning(f"No local intraday data for {symbol}. Sync required.")
                return None
            
            df = pd.read_csv(path_3m, parse_dates=['timestamp'])
            if len(df) < 50:
                return None
        except Exception as e:
            LOGGER.error(f"Failed to read local data: {e}")
            return None

        # 3. Calculate Confluence Indicators
        df = self.supertrend.calculate(df)
        df = self.rsi_div.calculate(df)
        df = self.trendlines.calculate(df)
        
        latest = df.iloc[-1]
        
        st_dir = latest.get('supertrend_direction', 0) # 1 for Bullish, -1 for Bearish
        rsi_signal = latest.get('rsi_signal', 0) 
        trendline_signal = latest.get('trendline_signal', 0)
        
        reasons = []
        confidence = 0.70 # Baseline for trend follower

        # --- BULLISH CASE (CALL) ---
        if market_context.bias == "BULLISH" and st_dir == 1:
            reasons.append("Supertrend Bullish")
            
            # Conviction Multipliers
            if rsi_signal == 1:
                reasons.append("💎 Bullish RSI Divergence (Confirm)")
                confidence += 0.20
            elif rsi_signal == -1:
                reasons.append("⚠️ Bearish RSI Divergence (Exhaustion Warning)")
                confidence -= 0.15 # Downgrade due to potential reversal
            
            if trendline_signal == 1: # Trendline break
                reasons.append("Bullish Trendline Break")
                confidence += 0.05
            
            if confidence >= 0.65:
                direction = TradeDirection.CALL
        
        # --- BEARISH CASE (PUT) ---
        elif market_context.bias == "BEARISH" and st_dir == -1:
            reasons.append("Supertrend Bearish")
            
            if rsi_signal == -1:
                reasons.append("💎 Bearish RSI Divergence (Confirm)")
                confidence += 0.20
            elif rsi_signal == 1:
                reasons.append("⚠️ Bullish RSI Divergence (Bottom-Fishing Warning)")
                confidence -= 0.15
            
            if trendline_signal == -1:
                reasons.append("Bearish Trendline Break")
                confidence += 0.05

            if confidence >= 0.65:
                direction = TradeDirection.PUT
        else:
            direction = None

        if not direction:
            return None

        # 5. Suggest RR Ratio (Dynamic)
        rr_ratio = 3.0 if market_context.regime.value in ["TRENDING_BULL", "TRENDING_BEAR"] else 2.0
        
        risk_points = 40.0
        target = current_price + (risk_points * rr_ratio) if direction == TradeDirection.CALL else current_price - (risk_points * rr_ratio)
        stop_loss = current_price - risk_points if direction == TradeDirection.CALL else current_price + risk_points

        # --- NEW: Swarm Institutional Merge ---
        is_unified = self._verify_institutional_merge(direction, market_context, oc_analysis)
        if is_unified:
            reasons.append("💎 UNIFIED SWARM CONFLUENCE")
            confidence = min(1.0, confidence + 0.10)

        # Rate Limiting: Max 1 signal per 10 minutes
        current_time = datetime.now()
        if self._last_signal_time:
            time_since_last = (current_time - self._last_signal_time).total_seconds() / 60.0
            if time_since_last < 10.0:
                LOGGER.info(f"NiftyOptionBuyerAgent: Cooldown active. Skipping {direction.value} signal.")
                return None
                
        self._last_signal_time = current_time

        suggestion = TradeSuggestion(
            id=str(uuid.uuid4()),
            timestamp=current_time,
            symbol=symbol,
            direction=direction,
            horizon=TradeHorizon.INTRADAY,
            entry_zone_low=current_price - 10,
            entry_zone_high=current_price + 10,
            target=target,
            stop_loss=stop_loss,
            option_params=OptionParams(
                direction=direction,
                suggested_strike=oc_analysis.atm_strike,
                expiry_type="weekly"
            ),
            confidence=round(confidence, 3),
            narrative=f"{', '.join(reasons)}. RR: 1:{rr_ratio}",
            setup_features=None,
            is_nifty=True,
            tags=reasons
        )
        if is_unified: suggestion.tags.append("unified_swarm_setup")
        return suggestion

    def _verify_institutional_merge(self, direction: TradeDirection, ctx: MarketContext, oc: OptionChainAnalysis) -> bool:
        """Verifies if the technical setup is backed by the full institutional matrix."""
        try:
            m = ctx.metadata
            oc_m = getattr(oc, 'metadata', {})
            
            # 1. Hurst Alignment
            hurst = m.get("hurst_exponent", 0.5)
            # Breakouts need trending memory
            if hurst < 0.45: return False
            
            # 2. Breadth Alignment
            breadth = m.get("market_breadth", {}).get("label", "NEUTRAL")
            if direction == TradeDirection.CALL and breadth not in ["BULLISH", "STRONGLY_BULLISH"]: return False
            if direction == TradeDirection.PUT and breadth not in ["BEARISH", "STRONGLY_BEARISH"]: return False
            
            # 3. Order Flow (ATM Volume)
            vd = oc_m.get("vol_delta", {})
            if direction == TradeDirection.CALL and vd.get("label") != "AGGRESSIVE_BULLISH_VOLUME": return False
            if direction == TradeDirection.PUT and vd.get("label") != "AGGRESSIVE_BEARISH_VOLUME": return False
            
            # 4. GEX Safety
            gex = oc_m.get("gex", {})
            if gex.get("gex_label") == "NEGATIVE_GEX (Volatile)":
                # In negative GEX, we are even more strict
                pass
                
            return True
        except: return False
import numpy as np
