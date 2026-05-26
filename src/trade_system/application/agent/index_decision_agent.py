"""
IndexDecisionAgent — AI Agent specifically for live indices trading decisions.
"""
from __future__ import annotations

import logging
from typing import Any, List

from trade_system.core import (
    TradeSuggestion, 
    PreMarketBrief, 
    OptionChainAnalysis,
    TradeDirection,
    TradeHorizon,
    OptionParams
)
from trade_system.application.advisory.llm import LlmAdvisorClient

LOGGER = logging.getLogger(__name__)

class IndexDecisionAgent:
    """
    Takes live market decisions for indices (Nifty, BankNifty) by combining
    Pre-Market macro data, live Option Chain data, and technical indicators.
    """

    def __init__(self, llm_client: LlmAdvisorClient | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()

    async def generate_decisions(
        self, 
        symbol: str, 
        premarket_brief: PreMarketBrief, 
        oc_analysis: OptionChainAnalysis,
        current_price: float
    ) -> List[TradeSuggestion]:
        """
        Evaluate conditions and generate trade suggestions for the index.
        """
        LOGGER.info(f"IndexDecisionAgent evaluating {symbol} at ₹{current_price}...")
        suggestions = []

        # Example logic: Align macro sentiment with option chain bias
        if premarket_brief.overall_sentiment == "BULLISH" and oc_analysis.trend_bias == "BULLISH":
            # Strong bullish alignment
            direction = TradeDirection.CALL
            target = oc_analysis.highest_ce_oi_strike - 50  # Target just below resistance
            stop_loss = current_price - 100
        elif premarket_brief.overall_sentiment == "BEARISH" and oc_analysis.trend_bias == "BEARISH":
            # Strong bearish alignment
            direction = TradeDirection.PUT
            target = oc_analysis.highest_pe_oi_strike + 50 # Target just above support
            stop_loss = current_price + 100
        else:
            LOGGER.info(f"Mixed signals for {symbol}. Pre-market: {premarket_brief.overall_sentiment}, Options: {oc_analysis.trend_bias}. Standing aside.")
            return suggestions

        import uuid
        from datetime import datetime
        
        # Construct the suggestion
        suggestion = TradeSuggestion(
            id=str(uuid.uuid4()),
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            horizon=TradeHorizon.INTRADAY,
            entry_zone_low=current_price - 20,
            entry_zone_high=current_price + 20,
            target=target,
            stop_loss=stop_loss,
            option_params=OptionParams(
                direction=direction,
                suggested_strike=oc_analysis.atm_strike,
                expiry_type="weekly"
            ),
            confidence=0.85,
            narrative=f"Strong {direction.value} alignment between macro sentiment and option chain OI buildup.",
            setup_features=None, # In a real implementation, populate with actual technicals
            is_nifty=("NIFTY" in symbol.upper())
        )
        
        suggestions.append(suggestion)
        return suggestions
