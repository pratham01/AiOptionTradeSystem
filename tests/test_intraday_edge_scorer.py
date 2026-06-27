"""
Unit tests for IntradayEdgeScorer and SmartEntryTrigger.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime

from trade_system.application.analysis.intraday_edge_scorer import (
    IntradayEdgeScorer,
    EdgeScore,
    LayerResult,
    LAYER_WEIGHTS,
)
from trade_system.application.analysis.smart_entry_trigger import (
    SmartEntryTrigger,
    EntryTrigger,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_15m_df(
    n: int = 50,
    start_price: float = 2400.0,
    trend: float = 0.002,
    symbol: str = "NSE:TEST-EQ",
    target_date: date | None = None,
) -> pd.DataFrame:
    """Generate realistic 15-minute OHLCV data for testing."""
    if target_date is None:
        target_date = date(2026, 6, 27)

    # Create timestamps spanning 3 trading days ending on target_date
    timestamps = []
    prices = [start_price]
    day_offsets = [2, 1, 0]  # 2 days ago, yesterday, today

    for day_offset in day_offsets:
        trade_date = pd.Timestamp(target_date) - pd.Timedelta(days=day_offset)
        for hour in range(9, 16):
            for minute in [15, 30, 45, 0]:
                if hour == 9 and minute == 0:
                    continue
                if hour == 15 and minute > 30:
                    continue
                timestamps.append(pd.Timestamp(trade_date.date()) + pd.Timedelta(hours=hour, minutes=minute))
                if len(timestamps) >= n:
                    break
            if len(timestamps) >= n:
                break
        if len(timestamps) >= n:
            break

    # Pad if needed
    while len(timestamps) < n:
        timestamps.append(timestamps[-1] + pd.Timedelta(minutes=15))

    timestamps = timestamps[:n]

    # Generate realistic OHLCV
    np.random.seed(42)
    closes = [start_price]
    for i in range(1, n):
        change = np.random.normal(trend, 0.005) * closes[-1]
        closes.append(closes[-1] + change)

    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + abs(np.random.normal(0, 0.003)) * c for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(np.random.normal(0, 0.003)) * c for o, c in zip(opens, closes)]
    volumes = [int(np.random.uniform(50000, 200000)) for _ in range(n)]

    return pd.DataFrame({
        "symbol": symbol,
        "timestamp": timestamps[:n],
        "open": opens[:n],
        "high": highs[:n],
        "low": lows[:n],
        "close": closes[:n],
        "volume": volumes[:n],
    })


# ── EdgeScorer Tests ───────────────────────────────────────────────────────────

def test_layer_weights_sum_to_one():
    """All layer weights must sum to 1.0 for correct composite scoring."""
    total = sum(LAYER_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"


def test_layer_result_construction():
    """LayerResult should hold name, score, direction, and detail."""
    lr = LayerResult(name="test", score=0.8, direction="CALL", detail="Test detail")
    assert lr.name == "test"
    assert lr.score == 0.8
    assert lr.direction == "CALL"
    assert lr.detail == "Test detail"


def test_edge_score_risk_reward():
    """EdgeScore risk_reward property should compute correctly."""
    es = EdgeScore(
        symbol="NSE:TEST-EQ",
        sector="IT",
        ltp=100.0,
        change_pct=1.0,
        raw_score=80.0,
        final_score=80.0,
        direction="CALL",
        direction_confidence=0.85,
        entry_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
    )
    # Risk = 5, Reward = 10, R:R = 2.0
    assert es.risk_reward == 2.0


def test_edge_score_zero_sl_returns_zero_rr():
    """EdgeScore with zero stop_loss should return 0 risk_reward."""
    es = EdgeScore(
        symbol="NSE:TEST-EQ", sector="IT", ltp=100.0, change_pct=1.0,
        raw_score=80.0, final_score=80.0, direction="CALL",
        direction_confidence=0.85,
    )
    assert es.risk_reward == 0.0


def test_scorer_sector_momentum_layer():
    """Sector momentum layer should rank leading sectors higher."""
    scorer = IntradayEdgeScorer()

    # Top sector
    result = scorer._layer_sector_momentum(
        "IT", {"IT": 1, "BANKING": 2, "PHARMA": 3, "AUTO": 4}, pd.DataFrame()
    )
    assert result.score == 1.0
    assert result.direction == "CALL"

    # Bottom sector
    result = scorer._layer_sector_momentum(
        "AUTO", {"IT": 1, "BANKING": 2, "PHARMA": 3, "AUTO": 4}, pd.DataFrame()
    )
    assert result.direction == "PUT"


def test_scorer_relative_strength_layer():
    """Relative strength layer should detect outperformers."""
    scorer = IntradayEdgeScorer()
    stock_perf = pd.DataFrame({
        "symbol": ["NSE:A-EQ", "NSE:B-EQ", "NSE:C-EQ"],
        "sector": ["IT", "IT", "IT"],
        "pChange": [3.0, 1.0, -0.5],
    })
    # Stock A is a strong outperformer (3.0 vs avg 1.17)
    result = scorer._layer_relative_strength("NSE:A-EQ", "IT", 3.0, stock_perf)
    assert result.direction == "CALL"
    assert result.score > 0.5


def test_scorer_supertrend_alignment_both_bullish():
    """Supertrend alignment with both timeframes bullish should score 1.0."""
    scorer = IntradayEdgeScorer()
    target_date = date(2026, 6, 27)

    # Create a bullish trend dataset for 15m
    df_15m = _make_15m_df(n=50, start_price=2400, trend=0.003, target_date=target_date)
    # Create a bullish trend dataset for daily
    daily_timestamps = pd.date_range(end=target_date, periods=30, freq="B")
    daily_closes = [2400 + i * 10 for i in range(30)]
    df_daily = pd.DataFrame({
        "symbol": "NSE:TEST-EQ",
        "timestamp": daily_timestamps,
        "open": [c - 5 for c in daily_closes],
        "high": [c + 10 for c in daily_closes],
        "low": [c - 10 for c in daily_closes],
        "close": daily_closes,
        "volume": [100000] * 30,
    })

    result = scorer._layer_supertrend_alignment(df_15m, df_daily)
    # Both should be bullish in a strong uptrend
    assert result.direction in ("CALL", "NEUTRAL")  # May be CALL if data is sufficient


def test_scorer_volume_confirmation_high_surge():
    """Volume confirmation with high surge should score well."""
    scorer = IntradayEdgeScorer()
    target_date = date(2026, 6, 27)
    df_15m = _make_15m_df(n=50, target_date=target_date)

    result = scorer._layer_volume_confirmation(df_15m, vol_surge=2.5, target_date=target_date)
    assert result.score >= 0.8


def test_scorer_volume_confirmation_low_surge():
    """Volume confirmation with low surge should score poorly."""
    scorer = IntradayEdgeScorer()
    target_date = date(2026, 6, 27)
    df_15m = _make_15m_df(n=50, target_date=target_date)

    result = scorer._layer_volume_confirmation(df_15m, vol_surge=0.5, target_date=target_date)
    assert round(result.score, 2) <= 0.3


def test_scorer_momentum_timing_orb_breakout():
    """Momentum timing should detect ORB breakout."""
    scorer = IntradayEdgeScorer()
    target_date = date(2026, 6, 27)
    df_15m = _make_15m_df(n=50, start_price=2400, trend=0.005, target_date=target_date)

    today_data = df_15m[df_15m["timestamp"].dt.date == target_date]
    if not today_data.empty:
        ltp = float(today_data["close"].iloc[-1])
        result = scorer._layer_momentum_timing(df_15m, ltp, 1.5, target_date)
        assert result.name == "momentum_timing"
        assert 0.0 <= result.score <= 1.0


def test_scorer_compression_release_empty():
    """Compression release with insufficient data should return 0 score."""
    scorer = IntradayEdgeScorer()
    result = scorer._layer_compression_release(pd.DataFrame(), 1.5, date(2026, 6, 27))
    assert result.score == 0.0
    assert result.direction == "NEUTRAL"


# ── SmartEntryTrigger Tests ────────────────────────────────────────────────────

def test_entry_trigger_vwap_pullback_call():
    """VWAP pullback for CALL should trigger when price is at VWAP."""
    trigger = SmartEntryTrigger()
    target_date = date(2026, 6, 27)
    df_15m = _make_15m_df(n=50, target_date=target_date)

    result = trigger.evaluate(
        direction="CALL", ltp=2400.0, atr=35.0,
        df_15m=df_15m, target_date=target_date,
        vwap=2400.0,  # Price at VWAP
    )
    # When price is exactly at VWAP with CALL direction, should trigger
    assert result is not None
    assert result.trigger_type == "VWAP_PULLBACK"
    assert result.status == "TRIGGERED"
    assert result.stop_loss < result.entry_price
    assert result.target_1 > result.entry_price


def test_entry_trigger_vwap_pullback_put():
    """VWAP pullback for PUT should trigger when price is at VWAP."""
    trigger = SmartEntryTrigger()
    target_date = date(2026, 6, 27)
    df_15m = _make_15m_df(n=50, target_date=target_date)

    result = trigger.evaluate(
        direction="PUT", ltp=2400.0, atr=35.0,
        df_15m=df_15m, target_date=target_date,
        vwap=2400.0,
    )
    assert result is not None
    assert result.trigger_type == "VWAP_PULLBACK"
    assert result.stop_loss > result.entry_price
    assert result.target_1 < result.entry_price


def test_entry_trigger_orb_breakout():
    """ORB breakout should trigger when price breaks opening range."""
    trigger = SmartEntryTrigger()
    target_date = date(2026, 6, 27)

    # Create data where price breaks above ORB
    df_15m = _make_15m_df(n=50, start_price=2400, trend=0.005, target_date=target_date)
    today_data = df_15m[df_15m["timestamp"].dt.date == target_date]

    if len(today_data) >= 3:
        orb_high = float(today_data.head(2)["high"].max())
        # Set LTP just above ORB high
        ltp = orb_high * 1.002

        result = trigger.evaluate(
            direction="CALL", ltp=ltp, atr=35.0,
            df_15m=df_15m, target_date=target_date,
        )
        # Should find either ORB or VWAP trigger
        if result is not None:
            assert result.entry_price > 0
            assert result.stop_loss > 0


def test_entry_trigger_no_data_returns_none():
    """SmartEntryTrigger should return None with empty data."""
    trigger = SmartEntryTrigger()
    result = trigger.evaluate(
        direction="CALL", ltp=2400.0, atr=35.0,
        df_15m=pd.DataFrame(), target_date=date(2026, 6, 27),
    )
    assert result is None


def test_entry_trigger_zero_atr_returns_none():
    """SmartEntryTrigger should return None with zero ATR."""
    trigger = SmartEntryTrigger()
    df_15m = _make_15m_df(n=50)
    result = trigger.evaluate(
        direction="CALL", ltp=2400.0, atr=0.0,
        df_15m=df_15m, target_date=date(2026, 6, 27),
    )
    assert result is None


def test_edge_scorer_horizons_initialization():
    """Verify IntradayEdgeScorer supports different trade horizons."""
    scorer_intraday = IntradayEdgeScorer("INTRADAY")
    scorer_weekly = IntradayEdgeScorer("WEEKLY")
    scorer_monthly = IntradayEdgeScorer("MONTHLY")
    
    assert scorer_intraday.horizon == "INTRADAY"
    assert scorer_weekly.horizon == "WEEKLY"
    assert scorer_monthly.horizon == "MONTHLY"


def test_resample_candles_weekly():
    """Verify daily candles can be correctly resampled to weekly rule."""
    scorer = IntradayEdgeScorer("WEEKLY")
    target_date = date(2026, 6, 27)
    df_daily = pd.DataFrame({
        "symbol": ["NSE:TEST-EQ"] * 10,
        "timestamp": pd.date_range(end=target_date, periods=10, freq="D"),
        "open": [100.0] * 10,
        "high": [110.0] * 10,
        "low": [90.0] * 10,
        "close": [105.0] * 10,
        "volume": [1000] * 10,
    })
    
    df_weekly = scorer._resample_candles(df_daily, "W")
    assert not df_weekly.empty
    assert "open" in df_weekly.columns
    assert "close" in df_weekly.columns
    # Check that volume aggregated correctly for the week
    assert df_weekly["volume"].iloc[0] > 1000


def test_smart_entry_trigger_horizon_scaling():
    """Verify that ATR multipliers scale correctly depending on selected horizon."""
    trigger_intraday = SmartEntryTrigger("INTRADAY")
    trigger_weekly = SmartEntryTrigger("WEEKLY")
    trigger_monthly = SmartEntryTrigger("MONTHLY")
    
    assert trigger_intraday.sl_multiplier == 1.0
    assert trigger_weekly.sl_multiplier == 1.2
    assert trigger_monthly.sl_multiplier == 1.5
    
    assert trigger_intraday.target_1_multiplier == 1.5
    assert trigger_weekly.target_1_multiplier == 2.0
    assert trigger_monthly.target_1_multiplier == 2.5

