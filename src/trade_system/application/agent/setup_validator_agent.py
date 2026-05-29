"""
SetupValidatorAgent — validates that a candidate has a viable option-buying setup.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from trade_system.core import (
    TradeSuggestion,
    TradeDirection,
    TradeHorizon,
    SetupFeatures,
    OptionParams,
    MarketContext,
)
from trade_system.application.agent.candidate_screener_agent import CandidateScore
from trade_system.core.ports.weights import WeightsProvider
from trade_system.application.advisory.llm import LlmAdvisorClient

LOGGER = logging.getLogger(__name__)

# Risk management constants
DEFAULT_RISK_REWARD_MIN = 1.5
DEFAULT_ATR_MULTIPLIER_TARGET = 1.5
DEFAULT_ATR_MULTIPLIER_SL = 0.7

NIFTY_STRIKE_STEP = 50
STOCK_STRIKE_STEP = 50

class SetupValidatorAgent:
    """
    Validates trading setups and produces TradeSuggestion objects.
    """

    CONFIDENCE_THRESHOLD = 0.50

    def __init__(
        self,
        broker: Any = None,
        weight_evolver: WeightsProvider | None = None,
        llm_client: LlmAdvisorClient | None = None,
        min_rr: float = DEFAULT_RISK_REWARD_MIN,
    ) -> None:
        self.broker = broker
        if weight_evolver is None:
            from trade_system.application.evolution.weight_evolver import WeightEvolver
            self.weight_evolver = WeightEvolver()
        else:
            self.weight_evolver = weight_evolver
        self.llm = llm_client or LlmAdvisorClient()
        self.min_rr = min_rr

        self.weights = self.weight_evolver.load_weights("setup_validator")
        LOGGER.info("SetupValidatorAgent weights: %s", self.weights)

    async def validate(
        self,
        candidate: CandidateScore,
        market_context: MarketContext,
        current_bars: pd.DataFrame | None = None,
    ) -> TradeSuggestion | None:
        if current_bars is None and self.broker is not None:
            from datetime import date, timedelta
            try:
                current_bars = self.broker.fetch_history(
                    symbol=candidate.symbol,
                    resolution="15",
                    range_from=(date.today() - timedelta(days=5)).isoformat(),
                    range_to=date.today().isoformat(),
                )
            except: pass

        atr = self._get_atr(current_bars, candidate.atr_pct, candidate.entry_price)
        entry_low, entry_high, target, sl = self._compute_levels(candidate, atr, market_context)

        if entry_high <= 0 or target <= 0 or sl <= 0:
            return None

        risk = abs(entry_high - sl)
        reward = abs(target - entry_high)
        rr = reward / risk if risk > 0 else 0.0

        if rr < self.min_rr:
            return None

        confidence = self._compute_confidence(candidate, market_context, rr)
        
        # --- Institutional Strategy Merge (High-Probability Filter) ---
        confidence, is_unified = self._apply_institutional_merge(candidate, market_context, confidence)

        if confidence < self.CONFIDENCE_THRESHOLD:
            return None

        direction = TradeDirection.CALL if candidate.direction == "CALL" else TradeDirection.PUT
        option_params = self._suggest_option_params(direction, candidate.entry_price, candidate.symbol)

        setup_features = SetupFeatures(
            rsi_daily=candidate.rsi_daily,
            adx=candidate.adx,
            volume_surge=candidate.volume_surge,
            is_compressed=candidate.is_compressed,
            vol_delta_positive=candidate.vol_delta_positive,
            above_poc=candidate.above_poc,
            breakout_type=candidate.breakout_type,
            market_regime=market_context.regime.value,
            confidence=confidence,
            metadata=market_context.metadata
        )

        narrative = self._build_narrative(candidate, market_context, rr, confidence)
        if is_unified: narrative = "🌟 **UNIFIED SWARM SETUP:** " + narrative

        if self.llm.configured():
            try: narrative = await self._llm_narrative(candidate, market_context, rr, confidence, narrative)
            except: pass

        tags = self._build_tags(candidate, market_context, confidence)
        if is_unified: tags.append("unified_swarm_setup")

        suggestion = TradeSuggestion(
            id=str(uuid.uuid4()),
            symbol=candidate.symbol,
            timestamp=datetime.now(),
            direction=direction,
            horizon=candidate.horizon,
            entry_zone_low=round(entry_low, 2),
            entry_zone_high=round(entry_high, 2),
            target=round(target, 2),
            stop_loss=round(sl, 2),
            option_params=option_params,
            confidence=round(confidence, 3),
            narrative=narrative,
            setup_features=setup_features,
            is_nifty="NIFTY" in candidate.symbol.upper(),
            sector=candidate.sector,
            tags=tags,
        )

        LOGGER.info(f"✓ Valid setup: {candidate.symbol} {direction.value} | Conf: {confidence:.2f}")
        return suggestion

    def _apply_institutional_merge(self, candidate: CandidateScore, ctx: MarketContext, current_conf: float) -> tuple[float, bool]:
        try:
            m = ctx.metadata
            is_nifty = "NIFTY" in candidate.symbol.upper()
            
            # Individual Stock Heat (Performance from CandidateScore)
            stock_heat = candidate.component_scores.get("performance", 0.5)
            
            # 1. Regime Alignment (Hurst)
            hurst_val = m.get("hurst_exponent", 0.5)
            regime_ok = True
            if candidate.breakout_type and hurst_val < 0.45:
                # If Nifty is ranging, only allow breakouts with EXTREME individual heat
                regime_ok = (stock_heat >= 0.8) 
            
            # 2. Breadth Alignment (Advance-Decline)
            breadth = m.get("market_breadth", {})
            breadth_label = breadth.get("label", "NEUTRAL")
            breadth_ok = True
            if is_nifty: # Strict for Nifty
                if candidate.direction == "CALL" and breadth_label in ["BEARISH", "STRONGLY_BEARISH"]: breadth_ok = False
                elif candidate.direction == "PUT" and breadth_label in ["BULLISH", "STRONGLY_BULLISH"]: breadth_ok = False
            else: # Lean for F&O Stocks (Alpha hunters)
                if stock_heat > 0.7: breadth_ok = True # Follow the alpha

            # 3. Order Flow (ATM Volume) - Only for Nifty
            order_flow_ok = True
            if is_nifty and ctx.option_chain and hasattr(ctx.option_chain, 'metadata'):
                vd = ctx.option_chain.metadata.get("vol_delta", {})
                if candidate.direction == "CALL" and vd.get("label") != "AGGRESSIVE_BULLISH_VOLUME": order_flow_ok = False
                elif candidate.direction == "PUT" and vd.get("label") != "AGGRESSIVE_BEARISH_VOLUME": order_flow_ok = False

            is_unified = regime_ok and breadth_ok and order_flow_ok
            if is_unified: return min(1.0, current_conf + 0.12), True
            return current_conf, False
        except: return current_conf, False

    def _compute_levels(self, candidate, atr, market_context):
        price = candidate.entry_price
        if atr:
            entry_low, entry_high = price - atr*0.1, price + atr*0.1
            if candidate.direction == "CALL":
                target, sl = price + atr*1.5, price - atr*0.7
            else:
                target, sl = price - atr*1.5, price + atr*0.7
        else:
            entry_low, entry_high = price*0.998, price*1.002
            target, sl = (price*1.015, price*0.993) if candidate.direction=="CALL" else (price*0.985, price*1.007)
        return entry_low, entry_high, target, sl

    def _get_atr(self, bars, atr_pct, price):
        if bars is not None and len(bars) >= 14:
            tr = pd.concat([bars["high"]-bars["low"], (bars["high"]-bars["close"].shift()).abs(), (bars["low"]-bars["close"].shift()).abs()], axis=1).max(axis=1)
            return tr.rolling(14).mean().iloc[-1]
        return (atr_pct/100.0)*price if atr_pct else None

    def _compute_confidence(self, candidate, market_context, rr):
        w = self.weights
        score = candidate.component_scores.get("volume_delta", 0) * w.get("volume_delta", 0.25)
        score += candidate.component_scores.get("value_area", 0) * w.get("price_action", 0.30)
        score += candidate.component_scores.get("momentum", 0) * w.get("momentum", 0.20)
        score += candidate.component_scores.get("pre_breakout", 0) * 0.25
        return round(min(max(score, 0.0), 1.0), 4)

    def _suggest_option_params(self, direction, spot, symbol):
        step = 50 if "NIFTY" in symbol.upper() else 50
        atm = round(spot / step) * step
        return OptionParams(direction=direction, suggested_strike=float(atm))

    def _build_narrative(self, candidate, market_context, rr, confidence):
        return f"{candidate.direction} setup on {candidate.symbol}. R:R={rr:.1f}, Conf={confidence:.0%}"

    async def _llm_narrative(self, candidate, market_context, rr, confidence, rule_narrative):
        prompt = f"Write a 2-sentence rationale for {candidate.direction} on {candidate.symbol} with {confidence:.0%} confidence."
        return await self.llm.complete(prompt)

    def _build_tags(self, candidate, market_context, confidence):
        tags = [candidate.horizon.value.lower()]
        if confidence >= 0.75: tags.append("high_confidence")
        return tags
