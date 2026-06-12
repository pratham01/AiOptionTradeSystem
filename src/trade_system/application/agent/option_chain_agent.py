"""
OptionChainAgent — Dedicated AI Agent for derivatives data analysis.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trade_system.core import OptionChainAnalysis, DataBroker
from trade_system.core.events.bus import EventBus, EventType
from trade_system.application.analysis.option_chain_analyzer import OptionChainAnalyzer

LOGGER = logging.getLogger(__name__)

class OptionChainAgent:
    """
    Analyzes Option Chain data (PCR, GEX, Volume Delta) for institutional positioning.
    """

    def __init__(self, broker: DataBroker, event_bus: EventBus | None = None) -> None:
        self.broker = broker
        self.event_bus = event_bus
        
        if self.event_bus:
            self.event_bus.subscribe(EventType.SESSION_STARTED, self.on_session_started)

    async def on_session_started(self, payload: dict[str, Any]) -> None:
        """Autonomous handler for when a new session begins."""
        LOGGER.info("OptionChainAgent received SESSION_STARTED. Beginning analysis...")
        symbol = payload.get("nifty_symbol", "NSE:NIFTY50-INDEX")
        analysis = await self.analyze(symbol)
        
        if analysis and self.event_bus:
            await self.event_bus.emit(EventType.OPTION_CHAIN_READY, analysis)

    async def analyze(self, symbol: str) -> OptionChainAnalysis | None:
        """
        Analyze the option chain for a given symbol.
        """
        LOGGER.info(f"Analyzing option chain for {symbol}...")
        
        try:
            # Initialize the heavy-lifting analyzer
            analyzer = OptionChainAnalyzer(self.broker.fyers, symbol=symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", ""))
            result = analyzer.analyze()
            
            if not result: return None

            # Map the rich result to the core OptionChainAnalysis model
            metrics = result['metrics']
            pain = result['max_pain']
            walls = result['oi_walls']
            
            # Create analysis object and enrich with extra data
            analysis = OptionChainAnalysis(
                timestamp=datetime.now(),
                symbol=symbol,
                pcr=metrics['pcr_oi'],
                max_pain=float(pain['max_pain_strike']),
                atm_strike=float(pain['max_pain_strike']), # Approximation
                highest_ce_oi_strike=float(walls['resistance']),
                highest_pe_oi_strike=float(walls['support']),
                ce_oi_change_pct=metrics.get('ce_oi_change', 0.0),
                pe_oi_change_pct=metrics.get('pe_oi_change', 0.0),
                trend_bias=result['market_nature']['label']
            )
            
            # Attach institutional forensics for downstream agents
            # Using extra_context style or metadata
            analysis.metadata = {
                "gex": result['gex'],
                "vol_delta": result['vol_delta'],
                "market_nature": result['market_nature']
            }

            LOGGER.info(f"Derivatives Forensics for {symbol} complete. GEX: {result['gex']['gex_label']}")
            return analysis
        except Exception as e:
            LOGGER.error(f"OptionChainAgent failed for {symbol}: {e}")
            return None
