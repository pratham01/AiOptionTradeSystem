"""
Unit tests for StockMojo Smart OI Engine.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from trade_system.domains.analysis.application.analysis.stockmojo_smart_oi_engine import (
    aggregate_option_snapshots,
    detect_price_volume_divergences,
    format_indian_number,
    resample_intraday_data,
)


def test_format_indian_number():
    assert format_indian_number(38_700_000) == "+3.87 Cr"
    assert format_indian_number(-5_564_000) == "-55.64 L"
    assert format_indian_number(13_130) == "+13.13 K"
    assert format_indian_number(-450) == "-450"
    assert format_indian_number(0) == "0"


def test_aggregate_option_snapshots():
    base_time = datetime(2026, 9, 23, 9, 15)
    records = []
    
    # Snapshot 1: 9:15
    records.append({"timestamp": base_time, "strike": 23300, "option_type": "CE", "oi": 100000, "oi_change": 5000, "volume": 10000})
    records.append({"timestamp": base_time, "strike": 23300, "option_type": "PE", "oi": 80000, "oi_change": 2000, "volume": 8000})
    
    # Snapshot 2: 9:18 (Put writing surge)
    t2 = base_time + timedelta(minutes=3)
    records.append({"timestamp": t2, "strike": 23300, "option_type": "CE", "oi": 102000, "oi_change": 2000, "volume": 15000})
    records.append({"timestamp": t2, "strike": 23300, "option_type": "PE", "oi": 110000, "oi_change": 30000, "volume": 25000})

    df = pd.DataFrame(records)
    aggregated = aggregate_option_snapshots(df)

    assert len(aggregated) == 2
    assert "smart_oi_delta" in aggregated.columns
    assert "oi_pcr" in aggregated.columns
    assert "vol_pcr" in aggregated.columns

    # In Snapshot 2: PE added 30,000, CE added 2,000 -> smart_oi_delta = +28,000
    assert aggregated.iloc[1]["smart_oi_delta"] == 28000
    assert aggregated.iloc[1]["oi_pcr"] == pytest.approx(110000 / 102000, rel=1e-3)


def test_resample_intraday_data():
    base_time = datetime(2026, 9, 23, 9, 15)
    price_records = []
    
    # Generate 15 1-minute price bars
    for i in range(15):
        t = base_time + timedelta(minutes=i)
        price_records.append({
            "timestamp": t,
            "open": 23300 + i,
            "high": 23305 + i,
            "low": 23295 + i,
            "close": 23302 + i,
            "volume": 1000 * (i + 1)
        })
    price_df = pd.DataFrame(price_records)

    # Resample to 3m and 5m
    res_3m = resample_intraday_data(price_df, pd.DataFrame(), timeframe="3m")
    assert len(res_3m) == 5
    assert "candle_volume_delta" in res_3m.columns
    assert "cvd" in res_3m.columns
    assert "vwap" in res_3m.columns

    res_5m = resample_intraday_data(price_df, pd.DataFrame(), timeframe="5m")
    assert len(res_5m) == 3


def test_detect_price_volume_divergences():
    base_time = datetime(2026, 9, 23, 9, 15)
    bars = []
    # Create price pattern with Bearish Divergence:
    # First high at index 4 (close=23350, delta=+50000)
    # Pullback at index 8 (close=23320, delta=-10000)
    # Second higher high at index 12 (close=23380, delta=-25000) -> BEARISH DIVERGENCE!
    closes = [
        23300, 23320, 23340, 23348, 23350, 23345, 23330, 23325, 23320,
        23335, 23360, 23375, 23380, 23370, 23360, 23355, 23350
    ]
    deltas = [
        10000, 25000, 40000, 48000, 50000, 30000, 10000, -5000, -10000,
        -5000, 5000, 1000, -25000, -30000, -20000, -10000, -5000
    ]

    for i in range(len(closes)):
        bars.append({
            "timestamp": base_time + timedelta(minutes=i * 3),
            "close": closes[i],
            "open": closes[i] - 2,
            "high": closes[i] + 5,
            "low": closes[i] - 5,
            "volume": 5000,
            "smart_oi_delta": deltas[i]
        })
    df = pd.DataFrame(bars)

    signals = detect_price_volume_divergences(df, delta_col="smart_oi_delta", window=2)
    assert len(signals) > 0
    assert any(s.divergence_type == "BEARISH_DIVERGENCE" for s in signals)


def test_detect_bullish_divergence():
    base_time = datetime(2026, 9, 23, 9, 15)
    bars = []
    # Create price pattern with Bullish Divergence:
    # First low at index 4 (close=23300, delta=-40000)
    # Rebound at index 8 (close=23340, delta=+20000)
    # Second lower low at index 12 (close=23270, delta=+30000) -> BULLISH DIVERGENCE!
    closes = [
        23350, 23330, 23310, 23305, 23300, 23315, 23330, 23335, 23340,
        23325, 23300, 23280, 23270, 23285, 23300, 23310, 23320
    ]
    deltas = [
        -10000, -20000, -35000, -38000, -40000, -20000, 5000, 15000, 20000,
        10000, 15000, 25000, 30000, 35000, 40000, 30000, 20000
    ]

    for i in range(len(closes)):
        bars.append({
            "timestamp": base_time + timedelta(minutes=i * 3),
            "close": closes[i],
            "open": closes[i] + 2,
            "high": closes[i] + 5,
            "low": closes[i] - 5,
            "volume": 5000,
            "smart_oi_delta": deltas[i]
        })
    df = pd.DataFrame(bars)

    signals = detect_price_volume_divergences(df, delta_col="smart_oi_delta", window=2)
    assert len(signals) > 0
    assert any(s.divergence_type == "BULLISH_DIVERGENCE" for s in signals)
