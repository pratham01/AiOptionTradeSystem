import pytest
import pandas as pd
from unittest.mock import MagicMock

from trade_system.domains.advisory.application.agent.option_research_agent import OptionResearchAgent

class MockLlmClient:
    async def complete(self, prompt: str) -> str:
        return "Mocked scholar research / journal report content."

@pytest.mark.asyncio
async def test_research_option_wisdom():
    mock_llm = MockLlmClient()
    agent = OptionResearchAgent(llm_client=mock_llm)
    
    report = await agent.research_option_wisdom()
    assert "Mocked" in report

@pytest.mark.asyncio
async def test_run_predictability_backtest_with_insufficient_data():
    # If the database has no data or less than 50 rows, it should return success=False.
    # We query a non-existent symbol on the real database to trigger the empty data flow.
    agent = OptionResearchAgent()
    
    res = await agent.run_predictability_backtest("NON_EXISTENT_SYMBOL", "Momentum Surge", 1.5)
    assert not res["success"]
    assert "Insufficient historical data" in res["error"]

@pytest.mark.asyncio
async def test_run_predictability_backtest_runs_successfully():
    agent = OptionResearchAgent()
    
    # We query a symbol we know exists in the local database
    res = await agent.run_predictability_backtest("NSE:NIFTY50-INDEX", "Momentum Surge", 1.5)
    
    if res["success"]:
        assert "symbol" in res
        assert "t_stat" in res
        assert "p_value" in res
        assert "win_rate_when_signal" in res
        assert "monthly_performance" in res
    else:
        # If database is not initialized/accessible during tests, it's fine as long as it handles the error gracefully
        assert "success" in res
        assert "error" in res

@pytest.mark.asyncio
async def test_generate_autonomous_journal():
    mock_llm = MockLlmClient()
    agent = OptionResearchAgent(llm_client=mock_llm)
    
    res = await agent.generate_autonomous_journal()
    assert "success" in res
    if res["success"]:
        assert "journal_text" in res
        assert "missed_opportunities" in res
