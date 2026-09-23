"""
Unit tests for Intraday Smart Flow Reversal Engine (ISFRE).
"""
import pytest
import pandas as pd
from trade_system.domains.analysis.application.analysis.intraday_flow_reversal import (
    IntradayFlowReversalEngine,
    IntradayReversalSetup,
)


def test_classify_derivative_quadrant():
    # 1. Bullish rebound from low with oversold PCR -> Short covering / squeeze
    q1, d1 = IntradayFlowReversalEngine.classify_derivative_quadrant(
        price_change=-0.2,
        move_from_low=2.5,
        drop_from_high=0.2,
        pcr_oi=0.45,
        sentiment_state="OVERSOLD",
    )
    assert q1 == "⚡ SHORT COVERING"
    assert d1 == "BULLISH"

    # 2. Bullish rebound from low with strong PCR -> Long buildup
    q2, d2 = IntradayFlowReversalEngine.classify_derivative_quadrant(
        price_change=1.5,
        move_from_low=2.0,
        drop_from_high=0.1,
        pcr_oi=0.95,
        sentiment_state="NEUTRAL",
    )
    assert q2 == "🟢 LONG BUILD-UP"
    assert d2 == "BULLISH"

    # 3. Bearish dump from high with overbought PCR -> Long unwinding
    q3, d3 = IntradayFlowReversalEngine.classify_derivative_quadrant(
        price_change=0.5,
        move_from_low=0.2,
        drop_from_high=3.0,
        pcr_oi=1.15,
        sentiment_state="EXTREME_OVERBOUGHT",
    )
    assert q3 == "🩸 LONG UNWINDING"
    assert d3 == "BEARISH"

    # 4. Bearish dump from high with bearish PCR -> Short buildup
    q4, d4 = IntradayFlowReversalEngine.classify_derivative_quadrant(
        price_change=-1.5,
        move_from_low=0.1,
        drop_from_high=2.2,
        pcr_oi=0.58,
        sentiment_state="NEUTRAL",
    )
    assert q4 == "🔴 SHORT BUILD-UP"
    assert d4 == "BEARISH"


def test_evaluate_universe_synthetic():
    data = [
        {
            "symbol": "NSE:RELIANCE-EQ",
            "sector": "ENERGY",
            "close_last": 1250.0,
            "close_prev": 1245.0,
            "pChange": 0.4,
            "volume_today": 5000000,
            "avg_vol_time_adj": 2000000,
            "vol_surge": 2.5,
            "open_today": 1240.0,
            "high_today": 1252.0,
            "low_today": 1220.0,
            "move_from_low": 2.46,
            "drop_from_high": 0.16,
            "range_pos": 93.8,
        },
        {
            "symbol": "NSE:INFY-EQ",
            "sector": "IT",
            "close_last": 1800.0,
            "close_prev": 1810.0,
            "pChange": -0.55,
            "volume_today": 3000000,
            "avg_vol_time_adj": 1500000,
            "vol_surge": 2.0,
            "open_today": 1835.0,
            "high_today": 1840.0,
            "low_today": 1795.0,
            "move_from_low": 0.28,
            "drop_from_high": 2.17,
            "range_pos": 11.1,
        },
    ]
    df = pd.DataFrame(data)

    pcr_cache = {
        "all": [
            {
                "symbol": "NSE:RELIANCE-EQ",
                "clean_symbol": "RELIANCE",
                "pcr_oi": 0.50,
                "sentiment_state": "OVERSOLD",
            },
            {
                "symbol": "NSE:INFY-EQ",
                "clean_symbol": "INFY",
                "pcr_oi": 0.95,
                "sentiment_state": "OVERBOUGHT",
            },
        ]
    }

    indicators = {
        "NSE:RELIANCE-EQ": {"vwap": 1235.0, "vs_vwap": 1.21, "daily_rsi": 42.0},
        "NSE:INFY-EQ": {"vwap": 1820.0, "vs_vwap": -1.10, "daily_rsi": 68.0},
    }

    setups = IntradayFlowReversalEngine.evaluate_universe(
        merged_closes=df,
        quotes={},
        pcr_cache=pcr_cache,
        indicators=indicators,
    )

    assert len(setups) == 2

    # Check RELIANCE setup (Bullish Rebound & Squeeze)
    rel = [s for s in setups if s.clean_symbol == "RELIANCE"][0]
    assert rel.direction == "BULLISH"
    assert rel.move_from_low_pct == 2.46
    assert rel.flow_quadrant == "⚡ SHORT COVERING"
    assert rel.reversal_score >= 70
    assert rel.entry_trigger > rel.ltp
    assert rel.stop_loss < rel.low_today

    # Check INFY setup (Bearish Dump & Long Unwinding)
    infy = [s for s in setups if s.clean_symbol == "INFY"][0]
    assert infy.direction == "BEARISH"
    assert infy.drop_from_high_pct == 2.17
    assert infy.flow_quadrant == "🩸 LONG UNWINDING"
    assert infy.reversal_score >= 70
    assert infy.stop_loss > infy.high_today
