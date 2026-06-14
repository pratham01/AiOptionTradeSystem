from __future__ import annotations

import logging
from datetime import datetime
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.application.agent.top_gainers_agent import BrokerTopGainersAgent, create_fyers_broker, format_top_gainers_table
from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.infrastructure.data.fo_universe import get_fo_universe

from trade_system.config import Settings
from trade_system.infrastructure.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)

class Nifty500TopStocksAgent:
    """
    An AI agent that provides the top 10 stocks from the Nifty 500 universe
    from Monday to Friday. It fetches data via the broker and can use the LLM
    to generate an analysis.
    """

    def __init__(
        self,
        broker: FyersBroker | None = None,
        llm_client: LlmAdvisorClient | None = None,
        settings: Settings | None = None
    ) -> None:
        self.broker = broker
        self.llm = llm_client or LlmAdvisorClient()
        self.settings = settings or Settings.load()

    async def get_top_stocks_analysis(self) -> str:
        # Check if today is between Monday (0) and Friday (4)
        today = datetime.now()
        if today.weekday() > 4:
            return "Market is closed today (Weekend). I can only provide real-time top stocks from Monday to Friday."

        # Fetch broker dynamically if not provided
        try:
            active_broker = self.broker or create_fyers_broker()
        except Exception as e:
            LOGGER.error(f"Failed to initialize broker: {e}")
            return f"Error initializing broker connection: {e}"

        from trade_system.infrastructure.data.nse_universe import NSE_UNIVERSE

        # Create agent for Nifty 500 universe segment
        agent_n500 = BrokerTopGainersAgent(broker=active_broker, symbols=NSE_UNIVERSE)

        LOGGER.info("Fetching top 10 stocks from Nifty 500...")
        import asyncio
        loop = asyncio.get_running_loop()

        top_500 = await loop.run_in_executor(None, agent_n500.top_gainers, 10)

        if not top_500:
            return "No data retrieved from broker for the Nifty 500 universe."

        # Format as table
        table_str = format_top_gainers_table(top_500, title_prefix="Top (Nifty 500)")

        result_str = table_str
        # Let the AI Brain (LLM) synthesize a quick narrative if configured
        if self.llm.configured():
            prompt = (
                "You are an AI Trading Assistant. Here are the top 10 gainers in the Nifty 500 today:\n\n"
                f"{table_str}\n\n"
                "Provide a brief market narrative (2-3 sentences max) summarizing the sentiment, sector leadership, and strength based on these stocks. "
                "Output just the table and your summary."
            )
            try:
                analysis = await self.llm.complete(prompt)
                if analysis and "Error connecting to" in analysis:
                    LOGGER.warning("LLM returned an error message. Using fallback raw table.")
                    result_str = table_str + f"\n\n({analysis})"
                else:
                    result_str = analysis
            except Exception as e:
                LOGGER.warning(f"LLM analysis failed, returning raw table. Error: {e}")
                result_str = table_str + "\n\n(AI Brain analysis unavailable)"
        
        # Send only to the dedicated Nifty 500 top gainer stocks Telegram subgroup
        try:
            bot_token = self.settings.top_gainer_telegram.bot_token or self.settings.telegram.bot_token
            chat_id = self.settings.top_gainer_telegram.chat_id or self.settings.telegram.chat_id
            if bot_token and chat_id:
                LOGGER.info("Sending top stocks analysis to Telegram (Nifty 500 Top Gainers)...")
                notifier = TelegramNotifier(token=bot_token, chat_id=chat_id)
                if not notifier.send(result_str, parse_mode="Markdown"):
                    notifier.send(result_str, parse_mode=None)
        except Exception as e:
            LOGGER.warning(f"Failed to send top stocks analysis to Telegram: {e}")
            
        return result_str

if __name__ == "__main__":
    import asyncio
    async def main():
        agent = Nifty500TopStocksAgent()
        print(await agent.get_top_stocks_analysis())

    asyncio.run(main())
