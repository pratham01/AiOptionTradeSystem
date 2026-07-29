from __future__ import annotations

import sqlite3
import pandas as pd
from typing import List, Dict

from trade_system.domains.strategy.application.strategies.prajwal_price_action import PrajwalPriceActionStrategy, TradeSuggestion
from trade_system.domains.analysis.application.research.llm import ResearchLlmClient

class PrajwalPriceActionAgent:
    def __init__(self, db_path: str = "data/trade_system.db", llm_client: ResearchLlmClient | None = None):
        self.db_path = db_path
        self.strategy = PrajwalPriceActionStrategy(pivot_period=10, proximity_pct=0.2)
        self.llm_client = llm_client or ResearchLlmClient()

    def scan_symbols(self, symbols: List[str], timeframe: str = "15m", limit: int = 1000) -> List[Dict]:
        """Scans the DB for the given symbols and returns trade suggestions."""
        table = "ohlcv_15m" if timeframe == "15m" else "ohlcv_daily"
        
        suggestions_found = []
        
        with sqlite3.connect(self.db_path) as conn:
            for symbol in symbols:
                # Fetch historical data
                query = f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM {table}
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """
                df = pd.read_sql(query, conn, params=(symbol, limit))
                
                if df.empty or len(df) < 50:
                    continue
                    
                # Sort chronological
                df = df.sort_values("timestamp").reset_index(drop=True)
                
                suggestion = self.strategy.analyze_setup(df)
                if suggestion:
                    # Get recent price action context
                    context = self._build_context(df.tail(10), suggestion)
                    
                    # Call LLM for narrative
                    prompt = (
                        "You are an AI trained in the Prajwal Price Action trading style. "
                        "Your goal is to identify pure price action setups based on Support/Resistance "
                        "zones, candlestick rejections (Hammers, Shooting Stars), and volume breakouts.\n\n"
                        f"Context for {symbol}:\n{context}\n\n"
                        "Provide a very concise 2-sentence narrative on why this is a high probability setup "
                        "and what the trader should watch out for. Focus only on pure price action."
                    )
                    
                    provider, llm_note = self.llm_client.review(prompt)
                    
                    suggestions_found.append({
                        "symbol": symbol,
                        "action": suggestion.action,
                        "entry": suggestion.entry,
                        "stop_loss": suggestion.stop_loss,
                        "target": suggestion.target,
                        "reason": suggestion.reason,
                        "llm_narrative": llm_note if provider != "error" else "No AI narrative available.",
                        "timestamp": df.iloc[-1]['timestamp']
                    })
                    
        return suggestions_found
        
    def _build_context(self, df: pd.DataFrame, suggestion: TradeSuggestion) -> str:
        text = f"Action: {suggestion.action}\n"
        text += f"Reason: {suggestion.reason}\n"
        text += f"Entry: {suggestion.entry:.2f}, Stop Loss: {suggestion.stop_loss:.2f}, Target: {suggestion.target:.2f}\n"
        text += "Recent S/R Channels:\n"
        for ch in suggestion.channels:
            text += f" - {ch.channel_type.capitalize()}: {ch.low:.2f} to {ch.high:.2f}\n"
            
        text += "\nRecent Price Action:\n"
        for _, row in df.iterrows():
            text += f" {row['timestamp']}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}\n"
            
        return text
