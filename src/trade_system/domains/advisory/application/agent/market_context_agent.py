"""
MarketContextAgent — LLM-powered market regime and bias classifier.

Synthesizes VIX, PCR, global indices, news, and option chain skew into
a structured MarketContext that all downstream agents consume.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from trade_system.shared import (
    MarketContext,
    MarketRegime,
    OptionChainSnapshot,
    GlobalContext,
)
from trade_system.shared.events.bus import EventBus, EventType
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import get_market_data
from trade_system.domains.advisory.application.agent.advance_decline_agent import AdvanceDeclineAgent
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

# VIX thresholds for regime classification
VIX_LOW = 13.0
VIX_MEDIUM = 18.0
VIX_HIGH = 22.0
VIX_EXTREME = 28.0

# PCR thresholds
PCR_OVERSOLD_PUTS = 1.3    # Too many puts → contrarian bullish
PCR_OVERSOLD_CALLS = 0.7   # Too many calls → contrarian bearish


class MarketContextAgent:
    """
    Determines overall market bias, regime, and tradability for the session.

    Inputs:
      - India VIX
      - PCR (Put-Call Ratio, OI-based)
      - Global indices (GIFT Nifty, SGX Nifty, S&P, etc.)
      - Top news headlines
      - Option chain snapshot (max pain, OI skew)

    Outputs:
      - MarketContext with regime, bias, risk_level, narrative, tradeable flag
    """

    def __init__(self, llm_client: LlmAdvisorClient | None = None, event_bus: EventBus | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()
        self.engine = get_engine()
        self.breadth_agent = AdvanceDeclineAgent()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.event_bus = event_bus
        
        if self.event_bus:
            self.event_bus.subscribe(EventType.SESSION_STARTED, self.on_session_started)

    async def on_session_started(self, payload: dict[str, Any]) -> None:
        """Autonomous handler for when a new session begins."""
        LOGGER.info("MarketContextAgent received SESSION_STARTED. Beginning analysis...")
        # In a real swarm, it might fetch vix/pcr itself here or use payload
        vix = payload.get("vix")
        pcr = payload.get("pcr")
        option_chain = payload.get("option_chain")
        global_ctx = payload.get("global_ctx")
        extra_context = payload.get("extra_context")
        
        context = await self.analyze(vix, pcr, option_chain, global_ctx, extra_context)
        await self.event_bus.emit(EventType.MARKET_CONTEXT_READY, context)

    async def analyze(
        self,
        vix: float | None = None,
        pcr: float | None = None,
        option_chain: OptionChainSnapshot | None = None,
        global_ctx: GlobalContext | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> MarketContext:
        """
        Run full market context analysis.
        """
        # Step 1: Mathematical Regime Detection (Hurst Exponent)
        hurst_val, hurst_label = self._calculate_hurst_regime("NSE:NIFTY50-INDEX")
        
        # Step 2: Market Breadth Check
        breadth = self.breadth_agent.calculate_breadth()

        # Step 3: Rule-based pre-analysis
        rule_context = self._rule_based_analysis(vix, pcr, option_chain, global_ctx, hurst_val, breadth)

        # Step 4: Try LLM enrichment
        if extra_context is None: extra_context = {}
        extra_context.update({
            "hurst_exponent": hurst_val,
            "hurst_regime": hurst_label,
            "market_breadth": breadth
        })
        rule_context.metadata = extra_context

        if self.llm.configured():
            try:
                narrative = await self._llm_narrative(
                    vix, pcr, option_chain, global_ctx, rule_context, extra_context or {}
                )
                rule_context.narrative = narrative
            except Exception as exc:
                LOGGER.warning("LLM market context failed: %s. Using rule-based narrative.", exc)
                rule_context.narrative = self._build_rule_narrative(rule_context, vix, pcr)
        else:
            rule_context.narrative = self._build_rule_narrative(rule_context, vix, pcr)

        LOGGER.info(
            "Market Context: regime=%s bias=%s risk=%s tradeable=%s",
            rule_context.regime.value,
            rule_context.bias,
            rule_context.risk_level,
            rule_context.tradeable,
        )
        return rule_context

    def _rule_based_analysis(
        self,
        vix: float | None,
        pcr: float | None,
        option_chain: OptionChainSnapshot | None,
        global_ctx: GlobalContext | None,
        hurst_val: float = 0.5,
        breadth: dict[str, Any] | None = None
    ) -> MarketContext:
        """Pure rule-based context — no LLM."""
        score = 0.0
        risk_level = "MEDIUM"
        tradeable = True

        # --- Institutional Regime Adjustments ---
        if hurst_val > 0.55: score += 0.1 # Trending
        elif hurst_val < 0.45: score -= 0.1 # Ranging

        if breadth and breadth.get("label") == "STRONGLY_BULLISH": score += 0.2
        elif breadth and breadth.get("label") == "STRONGLY_BEARISH": score -= 0.2

        # --- VIX analysis ---
        if vix is not None:
            if vix < VIX_LOW:
                score += 0.3
                risk_level = "LOW"
            elif vix < VIX_MEDIUM:
                score += 0.1
                risk_level = "MEDIUM"
            elif vix < VIX_HIGH:
                score -= 0.2
                risk_level = "HIGH"
            else:
                score -= 0.5
                risk_level = "EXTREME"
                tradeable = False   # Too volatile for option buyers

        # --- PCR analysis ---
        if pcr is not None:
            if pcr > PCR_OVERSOLD_PUTS:
                score += 0.3    # Too many puts = contrarian bullish
            elif 0.9 <= pcr <= 1.2:
                score += 0.1    # Balanced
            elif pcr < PCR_OVERSOLD_CALLS:
                score -= 0.3    # Too many calls = contrarian bearish

        # --- Global context ---
        if global_ctx:
            gift_pct = global_ctx.gift_nifty_pct or 0.0
            sp500_pct = global_ctx.sp500_pct or 0.0
            if gift_pct > 0.3:
                score += 0.2
            elif gift_pct < -0.3:
                score -= 0.2
            if sp500_pct > 0.5:
                score += 0.1
            elif sp500_pct < -0.5:
                score -= 0.1

        # --- Option chain skew ---
        if option_chain:
            skew = option_chain.iv_skew
            if skew == "put_heavy":
                score -= 0.1
            elif skew == "call_heavy":
                score += 0.1

            # If call OI building up > puts: bearish wall
            coi = option_chain.call_oi_change or 0.0
            poi = option_chain.put_oi_change or 0.0
            if coi > poi * 1.5:
                score -= 0.1   # Strong call writing = resistance
            elif poi > coi * 1.5:
                score += 0.1   # Strong put writing = support

        # --- Determine bias ---
        if score >= 0.4:
            bias = "BULLISH"
        elif score <= -0.4:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # --- Determine regime ---
        regime = self._classify_regime(vix, pcr, global_ctx, hurst_val, score)

        ctx = MarketContext(
            timestamp=datetime.now(),
            regime=regime,
            bias=bias,
            risk_level=risk_level,
            vix=vix,
            pcr=pcr,
            score=round(max(-1.0, min(1.0, score)), 3),
            tradeable=tradeable,
            option_chain=option_chain,
            global_ctx=global_ctx,
        )
        return ctx

    def _classify_regime(
        self,
        vix: float | None,
        pcr: float | None,
        global_ctx: GlobalContext | None,
        hurst_val: float = 0.5,
        score: float = 0.0
    ) -> MarketRegime:
        """Classify the current market regime using VIX and Hurst."""
        if vix is not None and vix >= VIX_HIGH:
            return MarketRegime.VOLATILE

        # Hurst-based classification
        if hurst_val > 0.55:
            # Trending
            return MarketRegime.TRENDING_BULL if score > 0 else MarketRegime.TRENDING_BEAR
        
        if hurst_val < 0.45:
            return MarketRegime.RANGING

        return MarketRegime.UNKNOWN

    def _calculate_hurst_regime(self, symbol: str) -> tuple[float, str]:
        """
        Calculates the Hurst Exponent to identify price 'Memory'.
        H > 0.5: Persistent (Trend following)
        H < 0.5: Anti-persistent (Mean reverting)
        """
        try:
            with Session(self.engine) as session:
                data = get_market_data(session, symbol, "D", limit=100)
                if not data or len(data) < 50: return 0.5, "RANDOM"
                
                prices = np.array([d.close for d in data])
                lags = range(2, 20)
                tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = poly[0] * 2.0
                
                label = "TRENDING" if hurst > 0.55 else ("RANGING" if hurst < 0.45 else "RANDOM")
                return round(float(hurst), 3), label
        except Exception as e:
            LOGGER.error(f"Hurst calculation failed: {e}")
            return 0.5, "UNKNOWN"

    async def _llm_narrative(
        self,
        vix: float | None,
        pcr: float | None,
        option_chain: OptionChainSnapshot | None,
        global_ctx: GlobalContext | None,
        rule_context: MarketContext,
        extra_context: dict[str, Any],
    ) -> str:
        """Generate a one-paragraph market narrative via LLM."""
        oc_dict: dict[str, Any] = {}
        if option_chain:
            oc_dict = {
                "pcr": option_chain.pcr,
                "vix": option_chain.vix,
                "max_pain": option_chain.max_pain,
                "atm_strike": option_chain.atm_strike,
                "call_oi_change_pct": option_chain.call_oi_change,
                "put_oi_change_pct": option_chain.put_oi_change,
                "iv_skew": option_chain.iv_skew,
            }

        gc_dict: dict[str, Any] = {}
        if global_ctx:
            gc_dict = {
                "gift_nifty_pct": global_ctx.gift_nifty_pct,
                "sgx_nifty_pct": global_ctx.sgx_nifty_pct,
                "sp500_pct": global_ctx.sp500_pct,
                "nasdaq_pct": global_ctx.nasdaq_pct,
                "dxy_pct": global_ctx.dxy_pct,
                "crude_oil_pct": global_ctx.crude_oil_pct,
                "top_headlines": global_ctx.top_headlines[:5],
            }

        prompt_data = {
            "task": "market_context_narrative",
            "vix": vix,
            "pcr": pcr,
            "rule_regime": rule_context.regime.value,
            "rule_bias": rule_context.bias,
            "rule_risk": rule_context.risk_level,
            "rule_score": rule_context.score,
            "option_chain": oc_dict,
            "global_context": gc_dict,
            "extra": extra_context,
        }

        prompt = (
            "You are an expert Indian equity derivatives trader and macro analyst.\n\n"
            "Based on the data below, write a concise (2-3 sentence) market context narrative "
            "for today's Nifty options trading session. Focus on:\n"
            "1. Overall market bias (bullish/bearish/sideways) and the key reason\n"
            "2. Key risk factor or event to watch\n"
            "3. Whether it's a favorable day for option buyers\n\n"
            "Be direct and actionable. Use simple language a trader would understand.\n\n"
            f"Data:\n{json.dumps(prompt_data, indent=2, default=str)}\n\n"
            "Narrative (2-3 sentences only):"
        )

        return await self.llm.complete(prompt)

    def _build_rule_narrative(self, ctx: MarketContext, vix: float | None, pcr: float | None) -> str:
        """Build a simple narrative without LLM."""
        parts = []

        bias_str = {
            "BULLISH": "Market bias is bullish",
            "BEARISH": "Market bias is bearish",
            "NEUTRAL": "Market is in a balanced/sideways phase",
        }.get(ctx.bias, "Market bias is unclear")
        parts.append(bias_str)

        if vix is not None:
            if vix < VIX_LOW:
                parts.append(f"with low VIX ({vix:.1f}) — favorable for option buyers")
            elif vix < VIX_MEDIUM:
                parts.append(f"with moderate VIX ({vix:.1f}) — normal conditions")
            elif vix < VIX_HIGH:
                parts.append(f"but VIX is elevated ({vix:.1f}) — options are expensive, be selective")
            else:
                parts.append(f"but VIX is very high ({vix:.1f}) — avoid option buying today")

        if pcr is not None:
            if pcr > PCR_OVERSOLD_PUTS:
                parts.append(f"PCR at {pcr:.2f} suggests excessive put writing — support likely")
            elif pcr < PCR_OVERSOLD_CALLS:
                parts.append(f"PCR at {pcr:.2f} suggests excessive call writing — resistance likely")

        if not ctx.tradeable:
            parts.append("Extreme volatility detected — recommend standing aside today")

        return ". ".join(parts) + "."
