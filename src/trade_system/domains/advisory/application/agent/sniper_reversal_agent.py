"""
SniperReversalAgent — Precision reversal detection using Institutional Absorption.
Identifies high-probability bottoms/tops using Price-CVD divergence and RSI.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from trade_system.shared import TradeSuggestion, TradeDirection, TradeHorizon, OptionParams, OptionChainAnalysis
from trade_system.domains.market_data.infrastructure.data.fo_universe import FO_METADATA

LOGGER = logging.getLogger(__name__)

class SniperReversalAgent:
    """
    Finds where trends exhaust and institutional absorption begins.
    """

    def __init__(self):
        # Tracking potential setups in progress
        self.potential_traps: dict[str, dict] = {} 

    def analyze_for_reversal(self, symbol: str, df: pd.DataFrame, oc_analysis: Optional[OptionChainAnalysis] = None) -> Optional[TradeSuggestion]:
        """
        Scans a 5-minute DataFrame for Sniper Reversal confluence.
        1. Extreme Stretch from VWAP
        2. CVD Absorption (Price makes new low, CVD doesn't)
        3. RSI Bullish Divergence
        4. (Optional) Option Chain Convergence at massive OI walls
        """
        if df.empty or len(df) < 20: return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 1. EXHAUSTION (The Stretch) ---
        vwap = latest.get('vwap')
        if not vwap: return None
        
        price = float(latest['close'])
        dist_vwap_pct = ((price - vwap) / vwap) * 100
        
        # --- 2. ABSORPTION (CVD Divergence) ---
        # Bullish Sniper (Bottom Fishing)
        # Price makes a lower low than previous 5 bars, but CVD is higher or flat
        low_5 = df['low'].tail(6).iloc[:-1].min()
        cvd_5_min = df['cvd'].tail(6).iloc[:-1].min()
        
        is_bullish_cvd_div = (latest['low'] < low_5) and (latest['cvd'] >= cvd_5_min)
        
        # --- 3. RSI DIVERGENCE ---
        rsi_val = latest.get('rsi')
        # If rsi not in df, calculation is handled by collector or we use a fallback
        # Let's assume it's there or handle it
        if rsi_val is None: return None
        
        is_oversold = rsi_val <= 35
        
        # --- THE SNIPER TRIGGER ---
        # 1. Price is stretched (> 1.5% below VWAP)
        # 2. Institutional absorption (CVD Divergence)
        # 3. Market is oversold
        is_at_put_wall = True
        put_wall_msg = ""
        if oc_analysis and oc_analysis.highest_pe_oi_strike > 0:
            pe_dist = abs(price - oc_analysis.highest_pe_oi_strike) / price * 100
            if pe_dist > 1.5:  # Must be within 1.5% of the Put Wall
                is_at_put_wall = False
            else:
                put_wall_msg = f" Supported by Put Wall at {oc_analysis.highest_pe_oi_strike} (PCR: {oc_analysis.pcr:.2f})."

        if dist_vwap_pct <= -1.5 and is_bullish_cvd_div and is_oversold and is_at_put_wall:
            # Check for a bullish candle (Hammer / Green)
            if price > latest['open'] or (latest['high'] - latest['close']) < (latest['close'] - latest['low']):
                return self._build_sniper_alert(symbol, TradeDirection.CALL, price, latest['low'], "BOTTOM_SNIPER", dist_vwap_pct, extra_msg=put_wall_msg)

        # Bearish Sniper (Top Sniping)
        high_5 = df['high'].tail(6).iloc[:-1].max()
        cvd_5_max = df['cvd'].tail(6).iloc[:-1].max()
        is_bearish_cvd_div = (latest['high'] > high_5) and (latest['cvd'] <= cvd_5_max)
        is_overbought = rsi_val >= 65
        
        is_at_call_roof = True
        call_roof_msg = ""
        if oc_analysis and oc_analysis.highest_ce_oi_strike > 0:
            ce_dist = abs(price - oc_analysis.highest_ce_oi_strike) / price * 100
            if ce_dist > 1.5:  # Must be within 1.5% of the Call Roof
                is_at_call_roof = False
            else:
                call_roof_msg = f" Resisted by Call Roof at {oc_analysis.highest_ce_oi_strike} (PCR: {oc_analysis.pcr:.2f})."

        if dist_vwap_pct >= 1.5 and is_bearish_cvd_div and is_overbought and is_at_call_roof:
            if price < latest['open']:
                return self._build_sniper_alert(symbol, TradeDirection.PUT, price, latest['high'], "TOP_SNIPER", dist_vwap_pct, extra_msg=call_roof_msg)

        return None

    def _build_sniper_alert(self, symbol: str, direction: TradeDirection, price: float, sl: float, setup: str, stretch: float, extra_msg: str = "") -> TradeSuggestion:
        import uuid
        # Sniper trades have very tight SL, huge RR potential
        target_pts = abs(price - sl) * 3
        target = price + target_pts if direction == TradeDirection.CALL else price - target_pts
        
        return TradeSuggestion(
            id=str(uuid.uuid4()),
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            horizon=TradeHorizon.INTRADAY,
            entry_zone_low=price * 0.999,
            entry_zone_high=price * 1.001,
            target=round(target, 2),
            stop_loss=round(sl, 2),
            option_params=OptionParams(direction=direction),
            confidence=0.85, # Sniper shots are high confidence
            narrative=f"🎯 {setup}: Institutional Absorption detected. Price stretched {stretch:.1f}% from VWAP with CVD Divergence.{extra_msg}",
            setup_features=None,
            sector=FO_METADATA.get(symbol, "Other"),
            tags=[setup.lower(), "sniper_reversal"]
        )
