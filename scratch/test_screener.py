import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, date

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from unittest.mock import MagicMock, patch
from trade_system.application.analysis.breakout_screener import BreakoutScreener

def debug():
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
    
    df_nifty = pd.DataFrame({
        "symbol": ["NSE:NIFTY50-INDEX"] * 40,
        "timestamp": timestamps,
        "open": [18000.0] * 40,
        "high": [18010.0] * 40,
        "low": [17990.0] * 40,
        "close": [18000.0] * 40,
        "volume": [1000] * 40
    })
    
    mock_15m_data = pd.concat([df, df_nifty], ignore_index=True)
    
    # Consolidation edits
    df_stock = mock_15m_data[mock_15m_data['symbol'] == "NSE:TESTSTOCK-EQ"].copy()
    for i in range(39):
        df_stock.loc[df_stock.index[i], 'high'] = 100.5
        df_stock.loc[df_stock.index[i], 'low'] = 99.5
        df_stock.loc[df_stock.index[i], 'close'] = 100.0
        df_stock.loc[df_stock.index[i], 'volume'] = 100.0
        
    df_stock.loc[df_stock.index[-1], 'close'] = 101.5
    df_stock.loc[df_stock.index[-1], 'high'] = 101.6
    df_stock.loc[df_stock.index[-1], 'volume'] = 500.0
    
    mock_15m_data = pd.concat([df_stock, df_nifty], ignore_index=True)
    
    mock_daily_df = pd.DataFrame({
        "symbol": ["NSE:TESTSTOCK-EQ"] * 30,
        "timestamp": [pd.Timestamp(date(2026, 5, 1)) + pd.Timedelta(days=i) for i in range(30)],
        "open": [100.0] * 30,
        "high": [105.0] * 30,
        "low": [95.0] * 30,
        "close": [100.0] * 30,
        "volume": [5000.0] * 30
    })
    
    screener = BreakoutScreener()
    screener.fo_metadata = {"NSE:TESTSTOCK-EQ": "TEST_SECTOR"}
    
    with patch("trade_system.application.analysis.breakout_screener.pd.read_sql") as mock_read_sql:
        with patch("trade_system.application.analysis.breakout_screener.get_engine") as mock_engine:
            mock_read_sql.side_effect = [mock_15m_data, mock_daily_df]
            
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            
            alerts = screener.scan_for_breakouts(target_date=date(2026, 6, 12))
            print("Alerts count:", len(alerts))
            print("Alerts:", alerts)

if __name__ == "__main__":
    debug()
