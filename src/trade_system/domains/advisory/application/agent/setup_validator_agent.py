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

from trade_system.shared import (
    TradeSuggestion,
    TradeDirection,
    TradeHorizon,
    SetupFeatures,
    OptionParams,
    MarketContext,
)
from trade_system.domains.advisory.application.agent.candidate_screener_agent import CandidateScore
from trade_system.shared.ports.weights import WeightsProvider
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient

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
        settings: Any = None,
    ) -> None:
        self.broker = broker
        self.settings = settings
        if weight_evolver is None:
            from trade_system.domains.analysis.application.evolution.weight_evolver import WeightEvolver
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

        # Check for User-defined AI Directive and analyze alignment
        user_directive = None
        if market_context and hasattr(market_context, "metadata") and market_context.metadata:
            user_directive = market_context.metadata.get("user_directive")

        if user_directive and self.llm.configured():
            alignment_prompt = (
                f"You are a trading strategy auditor.\n"
                f"Candidate Trade: {candidate.direction} on {candidate.symbol} ({candidate.pattern or 'Unknown'})\n"
                f"Strategy Directive: '{user_directive}'\n\n"
                f"Rate the alignment of this trade with the directive on a scale of 0.0 to 1.0 "
                f"(1.0 = perfect alignment/satisfies it, 0.0 = complete violation or mismatch).\n"
                f"Respond with a single number only (e.g. 0.85)."
            )
            try:
                align_str = await self.llm.complete(alignment_prompt)
                alignment_score = float(align_str.strip())
                if alignment_score < 0.4:
                    confidence *= 0.5  # Heavy penalty for mismatch
                    LOGGER.info(f"SetupValidatorAgent: Candidate {candidate.symbol} penalized for directive mismatch (score {alignment_score})")
                else:
                    confidence = min(1.0, confidence + (alignment_score - 0.5) * 0.15)
                    LOGGER.info(f"SetupValidatorAgent: Candidate {candidate.symbol} adjusted by directive (score {alignment_score}, new conf {confidence})")
            except Exception as e:
                LOGGER.warning(f"Failed to check directive alignment: {e}")

        # --- General Agentic AI Critique & Validation (suggestion then validation happens everytime) ---
        ai_critique = ""
        if self.llm.configured():
            critique_prompt = (
                f"You are a professional option trading specialist evaluating a trade proposal for an option buyer.\n\n"
                f"Candidate Trade:\n"
                f"- Symbol: {candidate.symbol}\n"
                f"- Direction: {candidate.direction} (Buy Call if CALL, Buy Put if PUT)\n"
                f"- Entry Zone: {entry_low:.2f} - {entry_high:.2f}\n"
                f"- Target: {target:.2f} (Reward: {reward:.2f})\n"
                f"- Stop Loss: {sl:.2f} (Risk: {risk:.2f})\n"
                f"- Risk/Reward Ratio: {rr:.2f}\n"
                f"- Volume Surge Ratio: {candidate.volume_surge:.2f}x\n"
                f"- RSI Daily: {candidate.rsi_daily if candidate.rsi_daily else 'N/A'}\n"
                f"- Sector: {candidate.sector}\n\n"
                f"Provide a JSON response containing:\n"
                f"1. 'ai_score': A float between 0.0 and 1.0 representing your conviction in this trade setup based on volatility, momentum, and risk/reward.\n"
                f"2. 'critique': A 2-sentence explanation summarizing the primary setup strength and the main risk (e.g., theta decay, resistance levels).\n\n"
                f"Respond ONLY with a valid JSON response. Do not include markdown formatting or backticks. Example: {{\"ai_score\": 0.85, \"critique\": \"This is a good setup because...\"}}"
            )
            try:
                critique_str = await self.llm.complete(critique_prompt)
                critique_str_clean = critique_str.strip().replace("```json", "").replace("```", "").strip()
                critique_data = json.loads(critique_str_clean)
                ai_score = float(critique_data.get("ai_score", 0.5))
                ai_critique = critique_data.get("critique", "")
                
                # Blend rules-based and AI-based confidence
                old_conf = confidence
                confidence = round(0.4 * confidence + 0.6 * ai_score, 3)
                LOGGER.info(f"SetupValidatorAgent: LLM Agentic validation for {candidate.symbol}. Rule Conf: {old_conf:.2f} -> AI Conf: {confidence:.2f} (AI Score: {ai_score:.2f})")
            except Exception as e:
                LOGGER.warning(f"Failed to perform LLM agentic validation: {e}")


        # --- Institutional Movement Alignment Filter ---
        is_inst_buildup = False
        inst_score = 0.5
        reason = ""
        try:
            oc_df = await self._fetch_latest_option_chain(candidate.symbol)
            if not oc_df.empty:
                total_pe_oi = oc_df[oc_df["option_type"] == "PE"]["oi"].sum()
                total_ce_oi = oc_df[oc_df["option_type"] == "CE"]["oi"].sum()
                pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
                
                spot = candidate.entry_price
                step = 50 if "NIFTY" in candidate.symbol.upper() else 10
                strikes = oc_df["strike"].unique()
                if len(strikes) > 1:
                    sorted_strikes = sorted(strikes)
                    avg_step = sum(sorted_strikes[j] - sorted_strikes[j-1] for j in range(1, len(sorted_strikes))) / (len(sorted_strikes) - 1)
                    step = avg_step if avg_step > 0 else step
                    
                atm_strike = round(spot / step) * step
                near_oc = oc_df[(oc_df["strike"] >= atm_strike - 2 * step) & (oc_df["strike"] <= atm_strike + 2 * step)]
                
                near_pe_oi = near_oc[near_oc["option_type"] == "PE"]["oi"].sum()
                near_ce_oi = near_oc[near_oc["option_type"] == "CE"]["oi"].sum()
                
                near_pe_change = near_oc[near_oc["option_type"] == "PE"]["oi_change"].sum()
                near_ce_change = near_oc[near_oc["option_type"] == "CE"]["oi_change"].sum()
                
                if candidate.direction == "CALL":
                    is_put_writing = near_pe_oi > near_ce_oi * 1.1 or near_pe_change > near_ce_change * 1.2
                    is_ce_unwinding = near_ce_change < 0
                    
                    if is_put_writing or is_ce_unwinding:
                        inst_score = 0.8
                        reason = f"Institutional Put writing (PE near ATM: {near_pe_oi:,.0f} vs CE: {near_ce_oi:,.0f}) and/or CE unwinding."
                    else:
                        inst_score = 0.3
                        reason = "Weak institutional backing. Heavy Call resistance near ATM."
                else:
                    is_call_writing = near_ce_oi > near_pe_oi * 1.1 or near_ce_change > near_pe_change * 1.2
                    is_pe_unwinding = near_pe_change < 0
                    
                    if is_call_writing or is_pe_unwinding:
                        inst_score = 0.8
                        reason = f"Institutional Call writing (CE near ATM: {near_ce_oi:,.0f} vs PE: {near_pe_oi:,.0f}) and/or PE unwinding."
                    else:
                        inst_score = 0.3
                        reason = "Weak institutional backing. Heavy Put support near ATM."
                        
                if inst_score >= 0.7:
                    confidence = min(1.0, confidence + 0.15)
                    is_inst_buildup = True
                    LOGGER.info(f"SetupValidatorAgent: Boosted confidence for {candidate.symbol} due to institutional alignment: {reason}")
                else:
                    confidence *= 0.5
                    LOGGER.info(f"SetupValidatorAgent: Penalized confidence for {candidate.symbol} due to weak institutional alignment: {reason}")
        except Exception as exc:
            LOGGER.warning(f"SetupValidatorAgent: Institutional check failed for {candidate.symbol}: {exc}")

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
        if is_inst_buildup: narrative = "🛡️ **INSTITUTIONAL ALIGNED:** " + narrative

        if ai_critique:
            narrative = f"🤖 **AI CRITIQUE:** {ai_critique}\n\n" + narrative
        elif self.llm.configured():
            try: narrative = await self._llm_narrative(candidate, market_context, rr, confidence, narrative, user_directive=user_directive)
            except: pass

        tags = self._build_tags(candidate, market_context, confidence)
        if is_unified: tags.append("unified_swarm_setup")
        if is_inst_buildup: tags.append("institutional_buildup")

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

    async def _llm_narrative(self, candidate, market_context, rr, confidence, rule_narrative, user_directive=None):
        prompt = f"Write a 2-sentence rationale for {candidate.direction} on {candidate.symbol} with {confidence:.0%} confidence."
        if user_directive:
            prompt += f" Respect this strategy directive: '{user_directive}'."
        return await self.llm.complete(prompt)

    def _build_tags(self, candidate, market_context, confidence):
        tags = [candidate.horizon.value.lower()]
        if confidence >= 0.75: tags.append("high_confidence")
        return tags

    async def _fetch_latest_option_chain(self, symbol: str) -> pd.DataFrame:
        try:
            from trade_system.domains.market_data.infrastructure.database.connection import get_engine
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                ts_query = text("""
                    SELECT MAX(timestamp) as max_ts
                    FROM option_chain_data
                    WHERE underlying_symbol = :symbol
                """)
                ts_res = conn.execute(ts_query, {"symbol": symbol}).first()
                if not ts_res or not ts_res[0]:
                    return pd.DataFrame()
                
                data_query = text("""
                    SELECT strike, option_type, oi, oi_change, ltp, volume
                    FROM option_chain_data
                    WHERE underlying_symbol = :symbol
                      AND timestamp = :ts
                """)
                return pd.read_sql(data_query, conn, params={"symbol": symbol, "ts": ts_res[0]})
        except Exception as e:
            LOGGER.debug(f"Failed to fetch option chain for {symbol}: {e}")
            return pd.DataFrame()
