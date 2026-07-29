from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient

LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class MarketMood:
    sentiment: str  # BULLISH, BEARISH, NEUTRAL
    score: float    # -1.0 to 1.0
    news_summary: str = ""
    global_context: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

class MarketMoodService:
    """
    Service to determine market mood based on news and global indices.
    """

    def __init__(self, llm_client: LlmAdvisorClient | None = None):
        self.llm_client = llm_client or LlmAdvisorClient()
        self.news_api_key = os.getenv("NEWS_API_KEY")

    def fetch_global_sentiment(self) -> dict[str, Any]:
        """
        Fetch major global index performance (simulated or via lightweight API).
        In a real scenario, this would call a market data provider.
        """
        # Simulated data for now - could be extended to fetch from a real API
        return {
            "GIFT_NIFTY": "+0.45%",
            "S&P_500": "-0.12%",
            "NASDAQ": "-0.05%",
            "NIKKEI": "+1.20%",
            "VIX": "12.5 (Stable)"
        }

    def fetch_top_news(self) -> list[str]:
        """
        Fetch top financial news headlines.
        """
        if not self.news_api_key:
            LOGGER.warning("NEWS_API_KEY not configured. Using placeholder headlines.")
            return [
                "RBI keeps rates unchanged, focus on inflation",
                "US inflation data expected tomorrow",
                "Nifty hits record high amid strong foreign inflows",
                "Tech sector earnings show resilience"
            ]
        
        try:
            url = f"https://newsapi.org/v2/top-headlines?category=business&country=in&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return [a["title"] for a in articles[:10]]
        except Exception as e:
            LOGGER.error(f"Failed to fetch news: {e}")
            return []

    async def get_mood(self) -> MarketMood:
        """
        Uses LLM to synthesize news and global data into a MarketMood object.
        """
        news = self.fetch_top_news()
        globals_data = self.fetch_global_sentiment()

        if not self.llm_client.configured():
            return MarketMood(sentiment="NEUTRAL", score=0.0, news_summary="LLM not configured")

        prompt = (
            "You are a macro market analyst. Based on the following news headlines and global market data, "
            "determine the 'Market Mood' for the Indian stock market (Nifty/Bank Nifty).\n\n"
            f"News Headlines:\n{chr(10).join(news)}\n\n"
            f"Global Context:\n{globals_data}\n\n"
            "Return a JSON object with:\n"
            "1. sentiment (BULLISH, BEARISH, or NEUTRAL)\n"
            "2. score (float between -1.0 and 1.0)\n"
            "3. summary (brief 1-sentence analysis)\n"
        )

        try:
            # Reusing the suggest method's internal logic for now or expanding LlmAdvisorClient
            # For simplicity, we'll assume a direct call to LLM here
            # In practice, we'd add a dedicated method to LlmAdvisorClient
            response = self.llm_client.suggest(
                context={"task": "market_mood_analysis"},
                proposal={"news": news, "globals": globals_data},
                advice={"instruction": "Analyze mood"}
            )
            
            # Simple parsing (expecting LLM to return valid JSON based on prompt)
            # This is a bit optimistic; in reality, we'd use a more robust parser
            if "BULLISH" in response.upper():
                return MarketMood(sentiment="BULLISH", score=0.5, news_summary=response)
            elif "BEARISH" in response.upper():
                return MarketMood(sentiment="BEARISH", score=-0.5, news_summary=response)
            else:
                return MarketMood(sentiment="NEUTRAL", score=0.0, news_summary=response)

        except Exception as e:
            LOGGER.error(f"Error determining market mood: {e}")
            return MarketMood(sentiment="NEUTRAL", score=0.0, news_summary="Error in analysis")
