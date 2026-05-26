"""
ConvictionFuserAgent — The Brain that merges, scores, and culls trades to enforce discipline.
"""
from __future__ import annotations

import logging
from typing import List

from trade_system.core import SessionPlan, TradeSuggestion, MarketContext, TradeDirection
from trade_system.core.models.domain import MarketRegime, TradeOutcome

LOGGER = logging.getLogger(__name__)

class ConvictionFuserAgent:
    """
    Final layer of the Swarm. Evaluates all suggestions from the SessionPlan,
    upgrades confidence based on confluence, removes duplicates, and strictly
    limits the session to the Top 3 best setups.
    """

    def __init__(self, max_trades_per_day: int = 3):
        self.max_trades = max_trades_per_day

    def fuse_and_filter(self, plan: SessionPlan) -> SessionPlan:
        LOGGER.info("ConvictionFuserAgent: Fusing and filtering signals...")
        
        all_suggs = plan.all_suggestions()
        if not all_suggs:
            return plan
            
        # 1. Confluence Scoring
        for sugg in all_suggs:
            self._apply_confluence(sugg, plan.market_context)
            
        # 2. Merge and Deduplicate
        merged = self._merge_duplicates(all_suggs)
        
        # 3. Sort by confidence descending
        merged.sort(key=lambda s: s.confidence, reverse=True)
        
        # 4. Enforce strict budget (Top N)
        top_picks = merged[:self.max_trades]
        rejected = merged[self.max_trades:]
        
        for r in rejected:
            r.narrative += " [REJECTED: Budget Limit Reached]"
            LOGGER.info(f"ConvictionFuserAgent: Rejecting {r.symbol} (Confidence: {r.confidence:.2f}) due to daily budget.")
            
        for t in top_picks:
            t.tags.append("swarm_brain_approved")
            LOGGER.info(f"ConvictionFuserAgent: 🚀 APPROVED High-Conviction Setup: {t.symbol} (Confidence: {t.confidence:.2f})")
            
        # 5. Rebuild plan
        plan.nifty_suggestions = [s for s in top_picks if s.is_nifty]
        plan.fo_suggestions = [s for s in top_picks if not s.is_nifty]
        
        return plan
        
    def _apply_confluence(self, sugg: TradeSuggestion, ctx: MarketContext | None):
        """Boosts confidence if the trade aligns with macro regime."""
        if not ctx:
            return
            
        boost = 0.0
        
        # Macro alignment boost
        is_bullish_trade = (sugg.direction == TradeDirection.CALL)
        
        if is_bullish_trade and ctx.regime == MarketRegime.TRENDING_BULL:
            boost += 0.15
            sugg.tags.append("macro_aligned")
        elif (not is_bullish_trade) and ctx.regime == MarketRegime.TRENDING_BEAR:
            boost += 0.15
            sugg.tags.append("macro_aligned")
            
        if boost > 0:
            old_conf = sugg.confidence
            sugg.confidence = min(1.0, sugg.confidence + boost)
            LOGGER.debug(f"Confluence boost applied to {sugg.symbol}: {old_conf:.2f} -> {sugg.confidence:.2f}")

    def _merge_duplicates(self, suggestions: List[TradeSuggestion]) -> List[TradeSuggestion]:
        """Keep only the highest confidence setup per symbol."""
        best_by_symbol = {}
        for s in suggestions:
            if s.symbol not in best_by_symbol:
                best_by_symbol[s.symbol] = s
            else:
                existing = best_by_symbol[s.symbol]
                if s.confidence > existing.confidence:
                    best_by_symbol[s.symbol] = s
        return list(best_by_symbol.values())
