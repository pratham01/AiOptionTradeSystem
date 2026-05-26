from datetime import date

import pandas as pd

from trade_system.interfaces.live.helpers import get_gap_adjusted_data


def test_gap_adjustment_bridges_previous_close_to_today_open():
    index = pd.to_datetime(
        [
            "2026-03-19 15:27:00",
            "2026-03-19 15:28:00",
            "2026-03-19 15:29:00",
            "2026-03-20 09:15:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 110],
            "high": [101, 102, 103, 112],
            "low": [99, 100, 101, 109],
            "close": [100, 101, 102, 111],
            "volume": [1, 1, 1, 1],
            "symbol": ["NSE:NIFTY50-INDEX"] * 4,
        },
        index=index,
    )

    adjusted = get_gap_adjusted_data(df, supertrend_period=7, current_date=date(2026, 3, 20))

    assert adjusted.iloc[-2]["close"] == 110
    assert adjusted.iloc[-2]["high"] >= 110
    assert adjusted.iloc[-2]["low"] <= 110
