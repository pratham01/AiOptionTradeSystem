from pathlib import Path
import pandas as pd
import numpy as np

from trade_system.domains.analysis.application.backtesting.zone_engine_research import ZoneEngineResearch


def test_zone_engine_run_2026_smoke(tmp_path: Path):
    # Generate mock 3-minute Nifty-like data for 2026
    timestamps = pd.date_range("2026-05-26 09:15:00", periods=300, freq="3min")
    closes = []
    price = 22000.0
    for i in range(300):
        # Introduce some trends to potential triggers
        if i < 100:
            price += 2.0
        elif i < 200:
            price -= 3.0
        else:
            price += 1.5
        closes.append(round(price, 2))
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": pd.Series(closes).shift(1).fillna(22000.0).round(2),
        "high": [val + 10.0 for val in closes],
        "low": [val - 10.0 for val in closes],
        "close": closes,
        "volume": [1000 + (idx % 10) * 100 for idx in range(300)]
    })
    
    csv_path = tmp_path / "NSE_NIFTY50-INDEX_3min_2026.csv"
    df.to_csv(csv_path, index=False)

    artifacts = ZoneEngineResearch().run_year(
        csv_path,
        tmp_path / "output",
    )

    assert artifacts.summary_path.exists()
    assert artifacts.trades_path.exists()
    assert artifacts.report_path.exists()
    assert {"trades", "win_rate", "net_points", "avg_points"}.issubset(artifacts.summary.columns)

