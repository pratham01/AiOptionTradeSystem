import logging
import asyncio
from typing import Dict, Any
from sqlalchemy import text
from datetime import datetime

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
from trade_system.domains.advisory.application.agent.web_agent import WebScrapingAgent

LOGGER = logging.getLogger(__name__)

class MoversAnalysisAgent:
    """
    Agent that performs deep textual analysis on top movers.
    It analyzes intraday volume, previous day technicals, and optionally
    scrapes web news to understand exactly *why* a stock moved and how to predict it.
    """
    def __init__(self, use_web_news: bool = True):
        self.llm = LlmAdvisorClient()
        self.use_web_news = use_web_news
        self.engine = get_engine()
        
    def _fetch_intraday_context(self, symbol: str, date_str: str) -> str:
        """Fetch basic 15m OHLCV for the day to give the LLM volume context."""
        query = text("""
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv_15m 
            WHERE symbol = :symbol AND date(timestamp) = :date
            ORDER BY timestamp ASC
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"symbol": symbol, "date": date_str}).fetchall()
            
        if not rows:
            return "No intraday data available."
            
        context = []
        for r in rows:
            ts = r.timestamp
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    pass
            
            ts_str = ts.strftime('%H:%M') if hasattr(ts, 'strftime') else str(ts)
            context.append(f"{ts_str} | O:{r.open} H:{r.high} L:{r.low} C:{r.close} Vol:{r.volume}")
            
        return "\n".join(context)

    def _scrape_news(self, symbol: str) -> str:
        """Scrape Yahoo Finance or Google for recent news headlines on the symbol."""
        if not self.use_web_news:
            return "Web scraping disabled."
            
        base_symbol = symbol.split(":")[1].split("-")[0] if ":" in symbol else symbol
        url = f"https://finance.yahoo.com/quote/{base_symbol}.NS/news"
        
        try:
            with WebScrapingAgent(headless=True) as agent:
                html = agent.fetch_page_content(url, timeout=15000)
                # Yahoo finance news headlines usually in h3 or specific lists
                headlines = agent.extract_text(html, "h3")
                # Filter out generic noise
                clean = [h for h in headlines if len(h) > 20][:5]
                if clean:
                    return "Recent News Headlines:\n- " + "\n- ".join(clean)
                return "No major news headlines found."
        except Exception as e:
            LOGGER.warning(f"Failed to scrape news for {symbol}: {e}")
            return "Failed to fetch news."

    async def analyze_mover(self, stock_record: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate detailed analysis for a specific stock record.
        stock_record must contain: id, symbol, name, change_pct, volume_divergence, etc.
        """
        symbol = stock_record["symbol"]
        date_str = stock_record["timestamp"].split(" ")[0] if isinstance(stock_record["timestamp"], str) else stock_record["timestamp"].strftime("%Y-%m-%d")
        
        # Gather Context
        intraday_context = self._fetch_intraday_context(symbol, date_str)
        news_context = await asyncio.to_thread(self._scrape_news, symbol)
        
        # Prepare Prompt
        prompt = f"""
You are an expert quantitative and technical trading AI. Analyze this top mover for {date_str}.

STOCK: {symbol} ({stock_record.get('name', '')})
CHANGE: {stock_record.get('change_pct', 0)}%
VOLUME SURGE: {stock_record.get('volume_ratio', 1.0)}x avg volume
DIVERGENCE: {stock_record.get('volume_divergence', 'None')}

--- NEWS CONTEXT ---
{news_context}

--- INTRADAY PRICE & VOLUME ACTION ---
{intraday_context[:1000]} # Truncated to save tokens if too long

Based on this data, provide a structured JSON response with exactly these three keys (no markdown formatting, just raw JSON):
1. "catalyst_analysis": A short paragraph explaining exactly *why* it moved today (fundamental news + volume confirmation).
2. "volume_behavior": A short paragraph on how volume behaved (e.g. 'massive buying at 9:15 AM' or 'consistent accumulation').
3. "predictive_factors": A short paragraph on how a trader could have predicted or caught this move (e.g. 'watch for pre-market news + 5min opening range breakout').
"""
        
        # Combine the system prompt into the main prompt for simple complete()
        full_prompt = f"You are a professional financial analyst. Always output raw valid JSON.\n\n{prompt}"
        
        # Add a small delay to avoid LLM rate limits
        await asyncio.sleep(2)
        response = await self.llm.complete(full_prompt)
        
        import json
        try:
            # Clean possible markdown
            clean_json = response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:-3]
            elif clean_json.startswith("```"):
                clean_json = clean_json[3:-3]
                
            result = json.loads(clean_json)
            return {
                "catalyst_analysis": result.get("catalyst_analysis", "Analysis unavailable."),
                "volume_behavior": result.get("volume_behavior", "Analysis unavailable."),
                "predictive_factors": result.get("predictive_factors", "Analysis unavailable."),
                "technical_setup": f"Volume ratio: {stock_record.get('volume_ratio', 1.0)}x. Divergence: {stock_record.get('volume_divergence', 'None')}"
            }
        except Exception as e:
            LOGGER.error(f"Failed to parse LLM output for {symbol}: {e}\nOutput: {response}")
            return {
                "catalyst_analysis": "Error generating analysis.",
                "volume_behavior": "Error generating analysis.",
                "predictive_factors": "Error generating analysis.",
                "technical_setup": "Error generating analysis."
            }
