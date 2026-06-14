import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from trade_system.application.agent.nifty500_top_stocks_agent import Nifty500TopStocksAgent
from trade_system.application.agent.top_gainers_agent import GainerRow

@pytest.fixture
def mock_gainer_rows():
    return [
        GainerRow(symbol="NSE:STOCK1-EQ", close=100.0, change_pct=5.5, volume=10000),
        GainerRow(symbol="NSE:STOCK2-EQ", close=200.0, change_pct=4.2, volume=20000),
    ]

@pytest.fixture
def mock_fo_worst_rows():
    return [
        GainerRow(symbol="NSE:STOCK3-EQ", close=50.0, change_pct=-5.0, volume=15000),
    ]

@pytest.mark.asyncio
@patch("trade_system.application.agent.nifty500_top_stocks_agent.TelegramNotifier")
@patch("trade_system.application.agent.nifty500_top_stocks_agent.create_fyers_broker")
@patch("trade_system.application.agent.nifty500_top_stocks_agent.BrokerTopGainersAgent")
@patch("trade_system.application.agent.nifty500_top_stocks_agent.datetime")
async def test_get_top_stocks_analysis_success(mock_dt, MockAgent, mock_create_broker, MockTelegramNotifier, mock_gainer_rows, mock_fo_worst_rows):
    # Mock date to a weekday (Monday, June 15th, 2026)
    mock_dt.now.return_value = datetime(2026, 6, 15, 12, 0, 0)
    
    mock_broker = MagicMock()
    mock_create_broker.return_value = mock_broker
    
    # Mock the BrokerTopGainersAgent methods
    mock_agent_instance = MagicMock()
    mock_agent_instance.top_gainers.return_value = mock_gainer_rows
    mock_agent_instance.top_and_worst.return_value = (mock_gainer_rows[:1], mock_fo_worst_rows)
    MockAgent.return_value = mock_agent_instance
    
    # Mock LLM Client
    mock_llm = MagicMock()
    mock_llm.configured.return_value = True
    from unittest.mock import AsyncMock
    mock_llm.complete = AsyncMock(return_value="Mocked LLM market analysis narrative.")
    
    # Mock TelegramNotifier instance
    mock_notifier_instance = MagicMock()
    MockTelegramNotifier.return_value = mock_notifier_instance
    
    agent = Nifty500TopStocksAgent(broker=mock_broker, llm_client=mock_llm)
    
    # Execute analysis
    result = await agent.get_top_stocks_analysis()
    
    # Asserts
    assert result == "Mocked LLM market analysis narrative."
    assert mock_llm.complete.call_count == 1
    assert MockTelegramNotifier.call_count == 1
    assert mock_notifier_instance.send.call_count == 1
    
    # Verify the prompt contained all the index names
    prompt_sent = mock_llm.complete.call_args[0][0]
    assert "Top (Nifty 500)" in prompt_sent
    assert "Top (Nifty Next 50)" in prompt_sent
    assert "Top (Nifty Midcap 100)" in prompt_sent
    assert "Top (Nifty Smallcap 100)" in prompt_sent
    assert "Top F&O" in prompt_sent
    assert "Worst F&O" in prompt_sent

@pytest.mark.asyncio
@patch("trade_system.application.agent.nifty500_top_stocks_agent.TelegramNotifier")
@patch("trade_system.application.agent.nifty500_top_stocks_agent.create_fyers_broker")
@patch("trade_system.application.agent.nifty500_top_stocks_agent.BrokerTopGainersAgent")
@patch("trade_system.application.agent.nifty500_top_stocks_agent.datetime")
async def test_get_top_stocks_analysis_no_llm(mock_dt, MockAgent, mock_create_broker, MockTelegramNotifier, mock_gainer_rows, mock_fo_worst_rows):
    # Mock date to a weekday
    mock_dt.now.return_value = datetime(2026, 6, 15, 12, 0, 0)
    
    mock_broker = MagicMock()
    mock_create_broker.return_value = mock_broker
    
    # Mock the BrokerTopGainersAgent methods
    mock_agent_instance = MagicMock()
    mock_agent_instance.top_gainers.return_value = mock_gainer_rows
    mock_agent_instance.top_and_worst.return_value = (mock_gainer_rows[:1], mock_fo_worst_rows)
    MockAgent.return_value = mock_agent_instance
    
    # Mock LLM Client not configured
    mock_llm = MagicMock()
    mock_llm.configured.return_value = False
    
    # Mock TelegramNotifier instance
    mock_notifier_instance = MagicMock()
    MockTelegramNotifier.return_value = mock_notifier_instance
    
    agent = Nifty500TopStocksAgent(broker=mock_broker, llm_client=mock_llm)
    
    result = await agent.get_top_stocks_analysis()
    
    # Should output raw formatted tables
    assert "Top (Nifty 500)" in result
    assert "Top (Nifty Next 50)" in result
    assert "Top (Nifty Midcap 100)" in result
    assert "Top (Nifty Smallcap 100)" in result
    assert "Top F&O" in result
    assert "Worst F&O" in result
    assert "STOCK1" in result
    assert MockTelegramNotifier.call_count == 1
    assert mock_notifier_instance.send.call_count == 1

@pytest.mark.asyncio
@patch("trade_system.application.agent.nifty500_top_stocks_agent.datetime")
async def test_get_top_stocks_analysis_weekend(mock_dt):
    # Mock date to a Sunday (June 14th, 2026)
    mock_dt.now.return_value = datetime(2026, 6, 14, 12, 0, 0)
    
    agent = Nifty500TopStocksAgent()
    result = await agent.get_top_stocks_analysis()
    
    assert "Market is closed today (Weekend)" in result
