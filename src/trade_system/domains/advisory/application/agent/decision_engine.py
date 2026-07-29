"""Decision engine for AI-powered trade decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from trade_system.shared import Signal, SignalType, SignalEvent
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager, RiskAssessment

LOGGER = logging.getLogger(__name__)


class DecisionAction(Enum):
    EXECUTE = "EXECUTE"
    HOLD = "HOLD"
    MODIFY = "MODIFY"
    REJECT = "REJECT"


@dataclass(slots=True)
class Decision:
    """A decision produced by the agent."""

    signal: Signal
    action: DecisionAction
    confidence: float
    reasoning: str
    adjusted_params: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal.to_dict(),
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "adjusted_params": self.adjusted_params,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMClient(Protocol):
    """Protocol for LLM integration."""

    async def analyze(
        self,
        context: dict[str, Any],
        signal: Signal,
        risk: RiskAssessment,
    ) -> dict[str, Any]:
        """Analyze signal and return decision parameters."""
        ...


class DecisionEngine:
    """Main decision engine that combines rule-based and LLM analysis."""

    def __init__(
        self,
        risk_manager: RiskManager | None = None,
        llm_client: LLMClient | None = None,
        min_confidence: float = 0.6,
        enable_llm: bool = False,
    ) -> None:
        self.risk_manager = risk_manager or RiskManager()
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.enable_llm = enable_llm

    async def decide(self, signal: Signal, context: dict[str, Any]) -> Decision:
        """Make a decision about a signal."""
        # First: Risk assessment
        risk = self.risk_manager.assess(signal, context)

        if not risk.approved:
            return Decision(
                signal=signal,
                action=DecisionAction.REJECT,
                confidence=1.0,
                reasoning=f"Risk check failed: {risk.reason}",
            )

        # Second: Rule-based validation
        rule_confidence = self._apply_rules(signal, context)
        if rule_confidence < self.min_confidence:
            return Decision(
                signal=signal,
                action=DecisionAction.HOLD,
                confidence=rule_confidence,
                reasoning="Rule-based confidence below threshold",
            )

        # Third: Optional LLM analysis
        if self.enable_llm and self.llm_client:
            try:
                llm_result = await self.llm_client.analyze(context, signal, risk)
                llm_confidence = llm_result.get("confidence", rule_confidence)

                if llm_confidence < self.min_confidence:
                    return Decision(
                        signal=signal,
                        action=DecisionAction.HOLD,
                        confidence=llm_confidence,
                        reasoning=f"LLM analysis: {llm_result.get('reasoning', 'Low confidence')}",
                    )

                return Decision(
                    signal=signal,
                    action=DecisionAction.EXECUTE,
                    confidence=llm_confidence,
                    reasoning=llm_result.get("reasoning", "LLM approved"),
                    adjusted_params=llm_result.get("adjustments", {}),
                )
            except Exception as e:
                LOGGER.error("LLM analysis failed: %s", e)
                # Fall back to rule-based

        return Decision(
            signal=signal,
            action=DecisionAction.EXECUTE,
            confidence=rule_confidence,
            reasoning="Rule-based approval",
            adjusted_params={},
        )

    def _apply_rules(self, signal: Signal, context: dict[str, Any]) -> float:
        """Apply high-conviction rules for option buying. Returns confidence 0-1."""
        score = signal.confidence

        # 1. Market Regime Alignment
        # Only trade in direction of the broad bias (BULLISH for CALL, BEARISH for PUT)
        bias = context.get("market_bias", "NEUTRAL")
        direction = signal.metadata.get("direction") # "CALL" or "PUT"
        
        if (bias == "BULLISH" and direction == "CALL") or (bias == "BEARISH" and direction == "PUT"):
            score += 0.2
        elif bias != "NEUTRAL":
            score -= 0.3 # Penalize counter-trend trades

        # 2. Momentum Confirmation (RSI Divergence)
        rsi_div = context.get("rsi_divergence", 0.0)
        if direction == "CALL" and rsi_div > 5.0:
            score += 0.15
        elif direction == "PUT" and rsi_div < -5.0:
            score += 0.15

        # 3. Structural Confirmation (Trendline Breaks & S&R)
        if context.get("trendline_break", False):
            score += 0.15
            
        if context.get("near_confirmed_support", False) and direction == "CALL":
            score += 0.1
        elif context.get("near_confirmed_resistance", False) and direction == "PUT":
            score += 0.1

        # 4. Volatility (VIX) Sweet Spot
        # Option buyers need momentum but not extreme fear (which makes premiums too high)
        vix = context.get("vix", 15.0)
        if 14.0 <= vix <= 22.0:
            score += 0.1
        elif vix > 25.0:
            score -= 0.1 # High slippage risk

        # 5. Time Filter (Best momentum windows)
        now_time = datetime.now().time()
        # Morning surge (9:15 - 10:45) or Afternoon move (1:30 - 3:15)
        if (datetime.strptime("09:15", "%H:%M").time() <= now_time <= datetime.strptime("10:45", "%H:%M").time()) or \
           (datetime.strptime("13:30", "%H:%M").time() <= now_time <= datetime.strptime("15:15", "%H:%M").time()):
            score += 0.05
        else:
            score -= 0.1 # Low volume chop risk

        return min(max(score, 0.0), 1.0)
