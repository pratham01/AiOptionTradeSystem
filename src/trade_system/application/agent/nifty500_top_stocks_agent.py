from __future__ import annotations

import logging
from datetime import datetime
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.application.agent.top_gainers_agent import BrokerTopGainersAgent, create_fyers_broker, format_top_gainers_table
from trade_system.application.advisory.llm import LlmAdvisorClient

LOGGER = logging.getLogger(__name__)

class Nifty500TopStocksAgent:
    """
    An AI agent that provides the top 10 stocks from the Nifty 500 universe
    from Monday to Friday. It fetches data via the broker and can use the LLM
    to generate an analysis.
    """

    def __init__(self, broker: FyersBroker | None = None, llm_client: LlmAdvisorClient | None = None) -> None:
        self.broker = broker
        self.llm = llm_client or LlmAdvisorClient()

    async def get_top_stocks_analysis(self) -> str:
        # Check if today is between Monday (0) and Friday (4)
        today = datetime.now()
        if today.weekday() > 4:
            return "Market is closed today (Weekend). I can only provide real-time top Nifty 500 stocks from Monday to Friday."

        # Fetch broker dynamically if not provided
        try:
            active_broker = self.broker or create_fyers_broker()
        except Exception as e:
            LOGGER.error(f"Failed to initialize broker: {e}")
            return f"Error initializing broker connection: {e}"

        # Use the underlying top gainers logic which runs against NSE_UNIVERSE (Nifty 500+)
        agent = BrokerTopGainersAgent(broker=active_broker)
        LOGGER.info("Fetching top 10 stocks from Nifty 500 universe...")
        top_10_gainers = agent.top_gainers(top_n=10)

        if not top_10_gainers:
            return "No data retrieved from broker for the Nifty 500 universe."

        # Format as table
        table_str = format_top_gainers_table(top_10_gainers)

        # Let the AI Brain (LLM) synthesize a quick narrative if configured
        if self.llm.configured():
            prompt = (
                "You are an AI Trading Assistant. Here are the top 10 gainers in the Nifty 500 today:\n\n"
                f"{table_str}\n\n"
                "Provide a very brief 2-sentence market narrative based on these top gainers (e.g., Which sectors seem to be leading? Are these defensive or high-beta?). "
                "Output just the table and your 2-sentence summary."
            )
            try:
                analysis = await self.llm.complete(prompt)
                return analysis
            except Exception as e:
                LOGGER.warning(f"LLM analysis failed, returning raw table. Error: {e}")
                return table_str + "\n\n(AI Brain analysis unavailable)"
        
        return table_str

if __name__ == "__main__":
    import asyncio
    async def main():
        agent = Nifty500TopStocksAgent()
        print(await agent.get_top_stocks_analysis())

    asyncio.run(main())
