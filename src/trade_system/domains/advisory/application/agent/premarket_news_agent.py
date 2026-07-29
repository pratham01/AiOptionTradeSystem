"""
PreMarketNewsAgent — AI Agent for pre-market macroeconomic analysis.

Fetches data on USD/INR, crude oil, global market sentiment, and 
top business/economic news impacting the Indian market.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trade_system.shared import PreMarketBrief
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
from trade_system.shared.utils.news_scraper import BusinessNewsScraper

LOGGER = logging.getLogger(__name__)

class PreMarketNewsAgent:
    """
    Analyzes pre-market indicators and news to set the macro context for the day.
    """

    def __init__(self, llm_client: LlmAdvisorClient | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()
        self.scraper = BusinessNewsScraper()

    async def analyze(
        self,
        usd_inr: float | None = None,
        crude_oil: float | None = None,
        fii_dii_net: float | None = None,
        raw_news: list[str] | None = None,
    ) -> PreMarketBrief:
        """
        Synthesize pre-market context.
        """
        # If no news provided, scrape directly from MoneyControl/ET
        if not raw_news:
            LOGGER.info("No news provided to agent. Scraping direct business headlines...")
            raw_news = self.scraper.get_all_headlines()
        
        raw_news = raw_news or []
        
        brief = PreMarketBrief(
            timestamp=datetime.now(),
            usd_inr=usd_inr,
            crude_oil=crude_oil,
            fii_dii_net=fii_dii_net,
            top_news=raw_news,
        )

        if not self.llm.configured():
            LOGGER.warning("LLM not configured for PreMarketNewsAgent. Returning raw data.")
            brief.agent_summary = "Pre-market data aggregated without LLM analysis."
            return brief

        # Build prompt for LLM
        prompt = (
            "You are an expert Indian macro-economic analyst.\n"
            "Analyze the following pre-market data and provide a concise summary "
            "of how these factors will impact the Nifty50 and BankNifty today.\n\n"
            f"USD/INR: {usd_inr}\n"
            f"Crude Oil: {crude_oil}\n"
            f"FII/DII Net (Cr): {fii_dii_net}\n"
            f"Top News: {raw_news}\n\n"
            "Provide:\n"
            "1. Overall Sentiment (BULLISH, BEARISH, NEUTRAL, VOLATILE)\n"
            "2. A short paragraph explaining the key drivers.\n"
            "Format your response EXACTLY as:\n"
            "SENTIMENT: <sentiment>\n"
            "SUMMARY: <summary>"
        )

        try:
            llm_text = await self.llm.complete(prompt)
            
            # Parse response
            for line in llm_text.split('\n'):
                if line.startswith("SENTIMENT:"):
                    brief.overall_sentiment = line.replace("SENTIMENT:", "").strip()
                elif line.startswith("SUMMARY:"):
                    brief.agent_summary = line.replace("SUMMARY:", "").strip()

            LOGGER.info(f"Pre-market analysis complete. Sentiment: {brief.overall_sentiment}")

        except Exception as e:
            LOGGER.error(f"Failed to generate pre-market LLM analysis: {e}")
            brief.agent_summary = "Error generating pre-market summary."

        return brief
