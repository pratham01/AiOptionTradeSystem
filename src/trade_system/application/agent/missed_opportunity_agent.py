"""
MissedOpportunityAgent — Analyzes historical data post-market to find missed profitable trades.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, List

from trade_system.core.ports.broker import DataBroker
from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.infrastructure.database.connection import get_engine
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

class MissedOpportunityAgent:
    """
    Scans the market data at the end of the day to find high-PnL moves that the agents missed.
    Also analyzes the outcomes of suggested trades.
    """

    def __init__(self, broker: DataBroker, llm_client: LlmAdvisorClient | None = None) -> None:
        self.broker = broker
        self.llm = llm_client or LlmAdvisorClient()
        self.engine = get_engine()

    async def analyze_today_performance(self) -> dict[str, Any]:
        """
        Analyze today's trade suggestions vs actual outcomes.
        """
        LOGGER.info("MissedOpportunityAgent: Analyzing today's executed trades...")
        stats = {"total": 0, "wins": 0, "losses": 0, "details": []}
        
        try:
            with self.engine.connect() as conn:
                query = text("SELECT id, symbol, direction, entry_zone_low, target, stop_loss FROM suggested_trades WHERE date = date('now')")
                trades = conn.execute(query).fetchall()
                
            for t in trades:
                stats["total"] += 1
                outcome = await self._analyze_single_trade_outcome(t.symbol, t.direction, t.entry_zone_low, t.target, t.stop_loss)
                stats["details"].append({"symbol": t.symbol, "outcome": outcome})
                if outcome == "TARGET_HIT": stats["wins"] += 1
                elif outcome == "SL_HIT": stats["losses"] += 1
                
        except Exception as e:
            LOGGER.error(f"Failed to analyze trade performance: {e}")
            
        return stats

    async def _analyze_single_trade_outcome(self, symbol: str, direction: str, entry: float, target: float, sl: float) -> str:
        # Fetch intraday data to see if target or SL was hit first
        try:
            df = self.broker.get_historical_data(symbol, date.today(), date.today(), "5")
            if df is None or df.empty: return "NO_DATA"
            
            for _, row in df.iterrows():
                if direction == "CALL":
                    if row['high'] >= target: return "TARGET_HIT"
                    if row['low'] <= sl: return "SL_HIT"
                else:
                    if row['low'] <= target: return "TARGET_HIT"
                    if row['high'] >= sl: return "SL_HIT"
            return "EXPIRED"
        except: return "ERROR"

    async def analyze_missed_setups(self, top_gainers: List[str]) -> List[str]:
        """
        Compare top gainers against our suggestions to find missed momentum.
        """
        LOGGER.info("MissedOpportunityAgent: Scanning for missed momentum in top gainers...")
        insights = []
        
        # Get list of symbols we ALREADY suggested today
        with self.engine.connect() as conn:
            suggested = [r[0] for r in conn.execute(text("SELECT symbol FROM suggested_trades WHERE date = date('now')")).fetchall()]

        for symbol in top_gainers:
            if symbol in suggested: continue
            
            # This stock was a top gainer but we didn't suggest it. Why?
            insight = f"Momentum Gap: {symbol} was a top performer today but was not in our session plan."
            
            if self.llm.configured():
                explanation = await self._llm_explain_miss(symbol)
                insights.append(f"{insight} Reason: {explanation}")
            else:
                insights.append(insight)
                
        return insights[:5] # Return top 5 gaps

    async def _llm_explain_miss(self, symbol: str) -> str:
        prompt = f"The stock {symbol} was a top gainer today, but our automated screener missed it. Provide a technical reason why a momentum-based breakout screener might deprioritize this stock in the first hour of trade (e.g., volume profile, gap size, or sector rotation)."
        return await self.llm.complete(prompt)
