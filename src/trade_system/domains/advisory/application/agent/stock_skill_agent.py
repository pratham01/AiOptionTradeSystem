"""
StockSkillAgent — Specialized AI Agent for deep stock-specific analysis and rule generation.
Links technicals, OI data, news, and macro factors to create durable trading skills.
"""
from __future__ import annotations

import logging
import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from pathlib import Path

from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
from trade_system.domains.advisory.application.agent.option_chain_agent import OptionChainAgent
from trade_system.domains.advisory.application.agent.skill_creator_agent import SkillCreatorAgent
from trade_system.shared.utils.news_scraper import BusinessNewsScraper
from trade_system.domains.market_data.infrastructure.database.repository import get_market_data
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from sqlalchemy.orm import Session
import pandas as pd

LOGGER = logging.getLogger(__name__)

class StockSkillAgent:
    """
    Agent that performs an exhaustive deep-dive on a single stock to generate a trading skill.
    """

    def __init__(
        self, 
        broker: Any, 
        llm_client: Optional[LlmAdvisorClient] = None
    ) -> None:
        self.broker = broker
        self.llm = llm_client or LlmAdvisorClient()
        self.oc_agent = OptionChainAgent(broker=broker)
        self.news_scraper = BusinessNewsScraper()
        self.skill_creator = SkillCreatorAgent(llm_client=self.llm)
        self.engine = get_engine()

    async def analyze_and_create_skill(self, symbol: str) -> str:
        """
        Performs deep analysis and generates a new durable skill for the stock.
        """
        LOGGER.info(f"StockSkillAgent: Starting deep-dive analysis for {symbol}...")
        
        try:
            # 1. Gather Technical Data (Last 3 days 5m + Daily)
            with Session(self.engine) as session:
                df_daily = self._to_df(get_market_data(session, symbol, "D", limit=10))
                df_5m = self._to_df(get_market_data(session, symbol, "5", limit=300))
            
            if df_5m.empty:
                return f"Error: No intraday data available for {symbol} to perform analysis."

            # 2. Gather Derivatives Data (If applicable)
            # F&O stocks only
            oc_analysis = await self.oc_agent.analyze(symbol)
            
            # 3. Gather News Context
            all_news = self.news_scraper.get_all_headlines()
            relevant_news = [h for h in all_news if symbol.split(':')[-1].replace('-EQ', '') in h.upper()]
            
            # 4. Construct the "Story" for the LLM
            story = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price_action": {
                    "current_price": float(df_5m['close'].iloc[-1]),
                    "day_change_pct": ((df_5m['close'].iloc[-1] - df_5m['close'].iloc[0]) / df_5m['close'].iloc[0]) * 100,
                    "rsi": self._calc_rsi(df_5m),
                    "volume_surge": float(df_5m['volume'].iloc[-1] / df_5m['volume'].mean()) if not df_5m.empty else 1.0
                },
                "derivatives": oc_analysis.__dict__ if oc_analysis else "N/A",
                "news_mentions": relevant_news or ["No direct headlines found today."]
            }

            # 5. Generate Skill via LLM
            prompt = f"""
            Act as an elite quantitative institutional trader. Analyze the following 'Story' for {symbol}:
            
            DATA:
            {json.dumps(story, indent=2)}
            
            TASK:
            1. Identify if today's move was driven by 'Smart Money' (high relative volume + positive OI skew) or 'Retail Noise'.
            2. Link the technical Supertrend/RSI signals with the News/Macro context.
            3. Create a DURABLE SKILL (behavioral rule) for trading this stock in the future.
            
            Format the output as a Markdown Skill (# Skill: ...) including Entry Catalyst and Risk Management.
            """
            
            skill_markdown = await self.llm.complete(prompt)
            
            # Save via SkillCreator
            # We bypass the prompt in SkillCreator and just use the markdown we got
            file_name = f"skill-{symbol.lower().replace(':', '-')}.md"
            file_path = Path("data/skills") / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(skill_markdown)
            
            LOGGER.info(f"StockSkillAgent: Successfully generated durable skill for {symbol} at {file_path}")
            return skill_markdown

        except Exception as e:
            LOGGER.error(f"StockSkillAgent analysis failed for {symbol}: {e}", exc_info=True)
            return f"Analysis Error: {str(e)}"

    def _to_df(self, models) -> pd.DataFrame:
        if not models: return pd.DataFrame()
        return pd.DataFrame([
            {"timestamp": m.timestamp, "open": m.open, "high": m.high, "low": m.low, "close": m.close, "volume": m.volume}
            for m in models
        ])

    def _calc_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1: return 50.0
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
