import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock, patch
from datetime import datetime, date
from trade_system.domains.advisory.application.agent.early_morning_agent import EarlyMorningAgent
from trade_system.domains.advisory.application.agent.missed_opportunity_agent import MissedOpportunityAgent
from trade_system.shared import TradeDirection
from trade_system.domains.market_data.infrastructure.database.models import AgentThought

@pytest.fixture
def sample_today_df():
    # Construct a DataFrame with 3-minute bars for today
    base_time = pd.Timestamp(date.today()).replace(hour=9, minute=15)
    timestamps = [base_time + pd.Timedelta(minutes=3 * i) for i in range(10)]
    
    # 09:15 to 09:30 are the first 6 bars (index 0 to 5)
    # Range high is 101.5, range low is 98.0
    opens =  [99.0, 99.5, 98.5, 99.2, 99.8, 101.0, 100.5, 102.0, 97.0, 96.0]
    highs =  [99.5, 100.0, 99.0, 99.8, 100.0, 101.5, 101.0, 102.5, 98.0, 97.0]
    lows =   [98.5, 99.0, 98.0, 98.5, 99.0,  100.0, 99.5, 101.0, 96.5, 95.0]
    closes = [99.5, 98.5, 99.2, 99.8, 99.0,  101.2, 100.5, 102.0, 97.0, 95.5]
    volumes =[100, 120, 90, 110, 100,  80,   95,    200,   150,  100]
    
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "symbol": "NSE:SBIN-EQ"
    }, index=timestamps)
    return df

@pytest.mark.asyncio
async def test_orb_filters_rejections(sample_today_df):
    agent = EarlyMorningAgent(broker=MagicMock())
    
    # At index 7 (09:36), close is 102.0, which is a breakout above range high of 101.5 (index 0 to 5 range)
    df_breakout = sample_today_df.iloc[:8]
    
    # 1. Test Rejection: Index Trend is BEARISH on CALL Breakout
    with patch.object(agent, "_get_nifty_trend", return_value="BEARISH"):
        with patch.object(agent, "_calculate_daily_atr", return_value=5.0):
            with patch.object(agent, "_thought") as mock_thought:
                sugg = await agent._check_orb("NSE:SBIN-EQ", df_breakout, set(), set())
                assert sugg is None
                mock_thought.assert_called_once()
                args, kwargs = mock_thought.call_args
                logged_msg = args[0]
                assert "ORB_REJECTED" in logged_msg
                assert "Index trend is not BULLISH" in logged_msg
                agent._opening_ranges.clear()

    # 2. Test Rejection: ATR Exhaustion (Range size > 35% ATR)
    # Range size is 3.5 (101.5 - 98.0). ATR is 5.0. 3.5 > 0.35 * 5.0 (1.75) -> should reject
    with patch.object(agent, "_get_nifty_trend", return_value="BULLISH"):
        with patch.object(agent, "_calculate_daily_atr", return_value=5.0):
            with patch.object(agent, "_thought") as mock_thought:
                sugg = await agent._check_orb("NSE:SBIN-EQ", df_breakout, set(), set())
                assert sugg is None
                mock_thought.assert_called_once()
                args, kwargs = mock_thought.call_args
                logged_msg = args[0]
                assert "exceeds 35% of daily ATR" in logged_msg
                agent._opening_ranges.clear()

    # 3. Test Rejection: Relative Volume (RVOL < 1.5x)
    # Range avg volume is 100. We modify the volume of the breakout bar to 80 (RVOL = 80/100 = 0.8x < 1.5x) -> should reject
    df_rvol_reject = df_breakout.copy()
    df_rvol_reject.loc[df_rvol_reject.index[-1], 'volume'] = 80
    with patch.object(agent, "_get_nifty_trend", return_value="BULLISH"):
        with patch.object(agent, "_calculate_daily_atr", return_value=20.0): # 3.5 / 20.0 = 17.5% ATR (fine)
            with patch.object(agent, "_thought") as mock_thought:
                sugg = await agent._check_orb("NSE:SBIN-EQ", df_rvol_reject, set(), set())
                assert sugg is None
                mock_thought.assert_called_once()
                args, kwargs = mock_thought.call_args
                logged_msg = args[0]
                assert "volume expansion is too low" in logged_msg.lower()
                agent._opening_ranges.clear()

    # 4. Test Rejection: Price below VWAP
    # We copy breakout df and inject a huge spike at index 6 (09:33) which pushes VWAP high,
    # then index 7 (09:36) breakout is at 102.0 (above orb_high of 101.5 but below VWAP)
    # Set volume of breakout bar to 200 so RVOL filter passes (RVOL = 200/100 = 2.0x >= 1.5x)
    df_vwap_reject = df_breakout.copy()
    df_vwap_reject.loc[df_vwap_reject.index[6], ['high', 'low', 'close', 'volume']] = [109.0, 108.0, 109.0, 10000]
    df_vwap_reject.loc[df_vwap_reject.index[-1], ['high', 'low', 'close', 'volume']] = [102.5, 101.5, 102.0, 200]
    
    with patch.object(agent, "_get_nifty_trend", return_value="BULLISH"):
        with patch.object(agent, "_calculate_daily_atr", return_value=20.0):
            with patch.object(agent, "_thought") as mock_thought:
                sugg = await agent._check_orb("NSE:SBIN-EQ", df_vwap_reject, set(), set())
                assert sugg is None
                mock_thought.assert_called_once()
                args, kwargs = mock_thought.call_args
                logged_msg = args[0]
                assert "is below VWAP" in logged_msg
                agent._opening_ranges.clear()

    # 5. Test Rejection: Strict Sector Alignment
    # Leaders = {"IT", "Banking"}. Since SBIN's mapping is not mocked, it defaults to "Other" or "Unknown".
    # Since "Other"/"Unknown" is not in leaders, it should reject.
    with patch.object(agent, "_get_nifty_trend", return_value="BULLISH"):
        with patch.object(agent, "_calculate_daily_atr", return_value=20.0):
            with patch.object(agent, "_thought") as mock_thought:
                sugg = await agent._check_orb("NSE:SBIN-EQ", df_breakout, {"IT", "Banking"}, set())
                assert sugg is None
                mock_thought.assert_called_once()
                args, kwargs = mock_thought.call_args
                logged_msg = args[0]
                assert "is not leading" in logged_msg
                agent._opening_ranges.clear()

@pytest.mark.asyncio
@patch("trade_system.domains.advisory.application.agent.missed_opportunity_agent.Session")
@patch("trade_system.domains.advisory.application.agent.missed_opportunity_agent.get_market_data")
async def test_true_negative_eod_simulation(mock_get_market_data, mock_session):
    broker = MagicMock()
    agent = MissedOpportunityAgent(broker=broker)
    
    # 1. Setup rejected thought JSON string
    rejection_msg = json.dumps({
        "type": "ORB_REJECTED",
        "symbol": "NSE:SBIN-EQ",
        "direction": "CALL",
        "entry": 100.0,
        "sl": 98.0,
        "target": 104.0,
        "reason": "Volume expansion is too low (RVOL: 0.8x)",
        "timestamp": datetime.now().isoformat()
    })
    
    mock_thought = AgentThought(
        agent_name="EarlyMorningAgent",
        action="ORB_REJECTED",
        message=rejection_msg,
        timestamp=datetime.now()
    )
    
    # Mock query thoughts returned by session
    mock_query = MagicMock()
    mock_query.filter.return_value.filter.return_value.all.return_value = [mock_thought]
    mock_session.return_value.__enter__.return_value.query.return_value = mock_query
    
    # Mock intraday bars showing price hit SL (98.0) before Target (104.0) -> TRUE NEGATIVE (avoided loss)
    bar1 = MagicMock()
    bar1.high = 101.0
    bar1.low = 98.5
    bar2 = MagicMock()
    bar2.high = 100.5
    bar2.low = 97.5 # Hit SL!
    
    mock_get_market_data.return_value = [bar1, bar2]
    
    # Run audit
    results = await agent.analyze_true_negatives()
    
    assert len(results) == 1
    assert results[0]["symbol"] == "NSE:SBIN-EQ"
    assert results[0]["outcome"] == "SL_HIT" # Correctly identified as avoided loss!
    
    # Mock intraday bars showing price hit Target (104.0) first -> FALSE NEGATIVE (missed winner)
    bar3 = MagicMock()
    bar3.high = 104.5 # Hit Target!
    bar3.low = 99.0
    
    mock_get_market_data.return_value = [bar1, bar3]
    results_fn = await agent.analyze_true_negatives()
    assert results_fn[0]["outcome"] == "TARGET_HIT"


@pytest.mark.asyncio
async def test_nr7_inside_bar_detectors():
    from trade_system.domains.advisory.application.agent.next_day_predictor_agent import NextDayPredictorAgent
    # Create a dummy DataFrame with 10 rows
    dates = pd.date_range("2026-05-01", periods=10)
    df = pd.DataFrame({
        "open":  [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "high":  [105, 105, 104, 104, 103, 103, 102, 102, 101.5, 100.5],
        "low":   [95,  96,  96,  97,  97,  98,  98,  99,  99.5,  99.5],
        "close": [101, 101, 101, 101, 101, 101, 101, 101, 101.2, 101.4],
        "volume":[100]*10
    }, index=dates)
    
    # Ranges: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    # The last range is 1.0, which is the narrowest of the last 7 ranges [7, 6, 5, 4, 3, 2, 1]
    # EMA 20: close is > EMA 20 since close is 101.4
    res_nr7 = NextDayPredictorAgent._detect_nr7(df)
    assert res_nr7 is not None
    assert res_nr7["pattern"] == "NR7 Compression"
    assert res_nr7["direction"] == "CALL"

    # Test Inside Bar:
    # 9th bar: high=101.5, low=99.0.
    # 10th bar: high=100.5, low=99.5. This is strictly inside!
    df.loc[df.index[-2], ['high', 'low']] = [101.5, 99.0]
    df.loc[df.index[-1], ['high', 'low']] = [100.5, 99.5]
    
    res_ib = NextDayPredictorAgent._detect_inside_bar(df)
    assert res_ib is not None
    assert res_ib["pattern"] == "Inside Bar"


@pytest.mark.asyncio
async def test_early_morning_compression_bracket(sample_today_df):
    agent = EarlyMorningAgent(broker=MagicMock())
    
    # Mock watchlist to return SBIN as a compression stock
    watchlist_mock = {
        "NSE:SBIN-EQ": {
            "symbol": "NSE:SBIN-EQ",
            "patterns": ["NR7 Compression", "Inside Bar"]
        }
    }
    
    # Mock yesterday's daily bar with high=100.0, low=95.0
    mock_daily_bar = MagicMock()
    mock_daily_bar.high = 100.0
    mock_daily_bar.low = 95.0
    
    # Modify sample_today_df to break out above yesterday's high of 100.0
    # Let's say latest bar has close = 101.0 (exceeding 100.0)
    df_breakout = sample_today_df.copy()
    df_breakout.loc[df_breakout.index[-1], ['close', 'high', 'volume']] = [101.0, 101.5, 300]
    
    with patch.object(agent, "_load_watchlist", return_value=watchlist_mock):
        with patch("trade_system.domains.advisory.application.agent.early_morning_agent.get_market_data", return_value=[mock_daily_bar]):
            with patch.object(agent, "_get_nifty_trend", return_value="BULLISH"):
                with patch.object(agent, "_calculate_daily_atr", return_value=20.0):
                    # We expect check_orb to detect breakout of yesterday's high (100.0) with yesterday's low (95.0) as SL.
                    # Sector is set to IT/Banking/etc., let's mock the leaders/laggards to empty set so sector alignment is skipped
                    sugg = await agent._check_orb("NSE:SBIN-EQ", df_breakout, set(), set())
                    assert sugg is not None
                    assert sugg.symbol == "NSE:SBIN-EQ"
                    assert sugg.direction == TradeDirection.CALL
                    # SL should be yesterday's low (95.0)
                    assert sugg.stop_loss == 95.0
                    # Target should be price + 2 * risk = 101.0 + 2 * 6.0 = 113.0
                    assert sugg.target == 113.0

