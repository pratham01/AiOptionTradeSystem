import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, date

from trade_system.domains.analysis.application.analysis.breakout_screener import BreakoutScreener

@pytest.fixture
def mock_breakout_screener():
    screener = BreakoutScreener()
    screener.fo_metadata = {"NSE:TESTSTOCK-EQ": "TEST_SECTOR"}
    return screener

@pytest.fixture
def base_candles():
    # Construct 20 bars yesterday (June 11), 20 bars today (June 12) = 40 bars total
    timestamps = []
    for i in range(20):
        timestamps.append(pd.Timestamp("2026-06-11 10:00:00") + pd.Timedelta(minutes=15 * i))
    for i in range(20):
        timestamps.append(pd.Timestamp("2026-06-12 09:15:00") + pd.Timedelta(minutes=15 * i))
    
    opens =   [100.0] * 40
    highs =   [101.0] * 40
    lows =    [99.0] * 40
    closes =  [100.0] * 40
    volumes = [100.0] * 40
    
    # Add a trend
    for i in range(40):
        closes[i] = 100.0 + (0.05 * i)
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
        volumes[i] = 100.0 + i
        
    df = pd.DataFrame({
        "symbol": ["NSE:TESTSTOCK-EQ"] * 40,
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    
    # Add index candles
    df_nifty = pd.DataFrame({
        "symbol": ["NSE:NIFTY50-INDEX"] * 40,
        "timestamp": timestamps,
        "open": [18000.0] * 40,
        "high": [18010.0] * 40,
        "low": [17990.0] * 40,
        "close": [18000.0] * 40,
        "volume": [1000] * 40
    })
    
    return pd.concat([df, df_nifty], ignore_index=True)

@pytest.fixture
def mock_daily_df():
    # Mock daily history of past 30 days
    timestamps = [pd.Timestamp(date(2026, 5, 1)) + pd.Timedelta(days=i) for i in range(30)]
    return pd.DataFrame({
        "symbol": ["NSE:TESTSTOCK-EQ"] * 30,
        "timestamp": timestamps,
        "open": [100.0] * 30,
        "high": [105.0] * 30,
        "low": [95.0] * 30,
        "close": [100.0] * 30,
        "volume": [5000.0] * 30
    })

@patch("trade_system.domains.analysis.application.analysis.breakout_screener.pd.read_sql")
@patch("trade_system.domains.analysis.application.analysis.breakout_screener.get_engine")
def test_gap_fill_detection(mock_engine, mock_read_sql, mock_breakout_screener, base_candles, mock_daily_df):
    # Setup mock data for Gap Fill
    base_15m = base_candles.copy()
    
    # Separate today's candles (June 12th) and yesterday's candles (June 11th)
    timestamps = []
    # 20 bars yesterday
    for i in range(20):
        timestamps.append(pd.Timestamp("2026-06-11 10:00:00") + pd.Timedelta(minutes=15 * i))
    # 20 bars today
    for i in range(20):
        timestamps.append(pd.Timestamp("2026-06-12 09:15:00") + pd.Timedelta(minutes=15 * i))
        
    df_stock = base_15m[base_15m['symbol'] == "NSE:TESTSTOCK-EQ"].copy()
    df_stock['timestamp'] = timestamps
    
    # Yesterday's last close is 100.0
    for i in range(20):
        df_stock.loc[df_stock.index[i], 'close'] = 100.0
        
    # Today's first candle opens at 102.0 (2% gap up)
    df_stock.loc[df_stock.index[20], 'open'] = 102.0
    
    # Today's latest candle close is 100.8 (filled (102.0 - 100.8) / (102.0 - 100.0) = 60% of gap)
    df_stock.loc[df_stock.index[-1], 'close'] = 100.8
    
    # Reassemble all mock data
    df_index = base_15m[base_15m['symbol'].str.contains("INDEX")].copy()
    df_index['timestamp'] = timestamps
    mock_15m_data = pd.concat([df_stock, df_index], ignore_index=True)
    
    mock_read_sql.side_effect = [mock_15m_data, mock_daily_df]
    
    alerts = mock_breakout_screener.scan_for_breakouts(target_date=date(2026, 6, 12))
    
    gap_fill_alerts = [a for a in alerts if a.get("alert_type") == "GAP_FILL"]
    assert len(gap_fill_alerts) == 1
    assert gap_fill_alerts[0]["symbol"] == "NSE:TESTSTOCK-EQ"
    assert gap_fill_alerts[0]["direction"] == "SHORT"
    assert gap_fill_alerts[0]["gap_pct"] >= 2.0

@patch("trade_system.domains.analysis.application.analysis.breakout_screener.pd.read_sql")
@patch("trade_system.domains.analysis.application.analysis.breakout_screener.get_engine")
def test_consolidation_breakout_on_the_fly(mock_engine, mock_read_sql, mock_breakout_screener, base_candles, mock_daily_df):
    # Previous 15 candles before latest are tight consolidation around 100.0
    base_15m = base_candles.copy()
    
    df_stock = base_15m[base_15m['symbol'] == "NSE:TESTSTOCK-EQ"].copy()
    # Fill range 0 to 38 (39 elements) with tight consolidation
    for i in range(39):
        df_stock.loc[df_stock.index[i], 'high'] = 100.5
        df_stock.loc[df_stock.index[i], 'low'] = 99.5
        df_stock.loc[df_stock.index[i], 'close'] = 100.0
        df_stock.loc[df_stock.index[i], 'volume'] = 100.0
        
    # Latest candle breakout
    df_stock.loc[df_stock.index[-1], 'close'] = 101.5
    df_stock.loc[df_stock.index[-1], 'high'] = 101.6
    df_stock.loc[df_stock.index[-1], 'volume'] = 500.0
    
    df_index = base_15m[base_15m['symbol'].str.contains("INDEX")].copy()
    mock_15m_data = pd.concat([df_stock, df_index], ignore_index=True)
    
    mock_read_sql.side_effect = [mock_15m_data, mock_daily_df]
    
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
    
    alerts = mock_breakout_screener.scan_for_breakouts(target_date=date(2026, 6, 12))
    
    cons_alerts = [a for a in alerts if a.get("alert_type") == "CONSOLIDATION_BREAKOUT"]
    assert len(cons_alerts) == 1
    assert cons_alerts[0]["symbol"] == "NSE:TESTSTOCK-EQ"
    assert cons_alerts[0]["direction"] == "LONG"
    assert cons_alerts[0]["close"] == 101.5

@patch("trade_system.domains.analysis.application.analysis.breakout_screener.pd.read_sql")
@patch("trade_system.domains.analysis.application.analysis.breakout_screener.get_engine")
def test_mean_reversion_detection(mock_engine, mock_read_sql, mock_breakout_screener, base_candles, mock_daily_df):
    # Ascending close values to force RSI above 75, then spike last one
    base_15m = base_candles.copy()
    
    df_stock = base_15m[base_15m['symbol'] == "NSE:TESTSTOCK-EQ"].copy()
    for i in range(40):
        df_stock.loc[df_stock.index[i], 'close'] = 100.0 + (i * 0.5)
        df_stock.loc[df_stock.index[i], 'volume'] = 100.0
        
    df_stock.loc[df_stock.index[-1], 'close'] = 135.0
    
    df_index = base_15m[base_15m['symbol'].str.contains("INDEX")].copy()
    mock_15m_data = pd.concat([df_stock, df_index], ignore_index=True)
    
    mock_read_sql.side_effect = [mock_15m_data, mock_daily_df]
    
    alerts = mock_breakout_screener.scan_for_breakouts(target_date=date(2026, 6, 12))
    
    mr_alerts = [a for a in alerts if a.get("alert_type") == "MEAN_REVERSION"]
    assert len(mr_alerts) == 1
    assert mr_alerts[0]["symbol"] == "NSE:TESTSTOCK-EQ"
    assert mr_alerts[0]["direction"] == "SHORT"
    assert mr_alerts[0]["rsi"] >= 75

@patch("trade_system.domains.analysis.application.analysis.breakout_screener.pd.read_sql")
@patch("trade_system.domains.analysis.application.analysis.breakout_screener.get_engine")
def test_vwap_pullback_detection(mock_engine, mock_read_sql, mock_breakout_screener, base_candles, mock_daily_df):
    # Setup VWAP and daily close to verify pullback
    base_15m = base_candles.copy()
    
    df_stock = base_15m[base_15m['symbol'] == "NSE:TESTSTOCK-EQ"].copy()
    # Set yesterday's last close to 100.0
    for i in range(20):
         df_stock.loc[df_stock.index[i], 'close'] = 100.0
         
    for i in range(20, 39):
        df_stock.loc[df_stock.index[i], 'close'] = 102.0
        df_stock.loc[df_stock.index[i], 'open'] = 100.0
        df_stock.loc[df_stock.index[i], 'volume'] = 100.0
        
    # Pullback on last candle
    df_stock.loc[df_stock.index[-1], 'low'] = 101.0
    df_stock.loc[df_stock.index[-1], 'close'] = 102.0
    df_stock.loc[df_stock.index[-1], 'volume'] = 120.0
    
    df_index = base_15m[base_15m['symbol'].str.contains("INDEX")].copy()
    mock_15m_data = pd.concat([df_stock, df_index], ignore_index=True)
    
    # Mock daily df to reflect a strong trend (latest close 102.0 vs yesterday close 100.0)
    df_daily = mock_daily_df.copy()
    df_daily.loc[df_daily.index[-1], 'close'] = 102.0
    df_daily.loc[df_daily.index[-1], 'open'] = 100.0
    
    mock_read_sql.side_effect = [mock_15m_data, df_daily]
    
    alerts = mock_breakout_screener.scan_for_breakouts(target_date=date(2026, 6, 12))
    
    pullback_alerts = [a for a in alerts if a.get("alert_type") == "VWAP_PULLBACK"]
    assert len(pullback_alerts) == 1
    assert pullback_alerts[0]["symbol"] == "NSE:TESTSTOCK-EQ"
    assert pullback_alerts[0]["direction"] == "LONG"
