import pandas as pd
import numpy as np
from trade_system.application.indicators.support_resistance_channels import (
    SupportResistanceChannelDetector,
    SRChannel,
    SRSnapshot,
    SRBreakEvent,
)


def _make_df(opens, highs, lows, closes, n=100):
    """Helper to build a DataFrame from price arrays."""
    timestamps = pd.date_range("2026-01-01 09:15:00", periods=n, freq="15min")
    volumes = [1000.0] * n
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def test_too_few_bars_returns_empty():
    """Fewer bars than 2*pivot_period+2 should return empty snapshots."""
    detector = SupportResistanceChannelDetector(pivot_period=10)
    df = _make_df([100] * 5, [101] * 5, [99] * 5, [100] * 5, n=5)
    result = detector.calculate(df)
    assert len(result) == 5
    for snap in result:
        assert snap.channels == []
        assert snap.pivot_high is None
        assert snap.pivot_low is None
        assert snap.break_event is None


def test_basic_pivot_detection():
    """Ensure at least one pivot high and one pivot low is found in a V-shape."""
    n = 60
    # Build a V-shape: prices rise to index 20, fall to index 40, rise again
    closes = [0.0] * n
    for i in range(n):
        if i <= 20:
            closes[i] = 100.0 + i * 2
        elif i <= 40:
            closes[i] = 140.0 - (i - 20) * 2
        else:
            closes[i] = 100.0 + (i - 40) * 2

    opens = [c - 0.5 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    df = _make_df(opens, highs, lows, closes, n=n)
    detector = SupportResistanceChannelDetector(pivot_period=5, loopback=100)
    snapshots = detector.calculate(df)

    # There should be at least one pivot high and one pivot low across all bars
    has_ph = any(s.pivot_high is not None for s in snapshots)
    has_pl = any(s.pivot_low is not None for s in snapshots)
    assert has_ph, "Expected at least one pivot high in a V-shaped price action"
    assert has_pl, "Expected at least one pivot low in a V-shaped price action"


def test_channels_detected():
    """Ensure channels form around clustered pivots."""
    n = 200
    np.random.seed(42)

    # Create price data that oscillates around 100 with some structure
    base = 100.0
    closes = [0.0] * n
    for i in range(n):
        # Oscillate between 95-105 with periodic spikes to 110 and dips to 90
        if i % 30 < 15:
            closes[i] = base + (i % 15) * 0.5
        else:
            closes[i] = base + 7.5 - (i % 15) * 0.5

    opens = [c - 0.3 for c in closes]
    highs = [c + 1.5 for c in closes]
    lows = [c - 1.5 for c in closes]

    df = _make_df(opens, highs, lows, closes, n=n)
    detector = SupportResistanceChannelDetector(
        pivot_period=4, min_strength=1, max_num_sr=6, loopback=150
    )
    snapshots = detector.calculate(df)

    # The last snapshot should have at least 1 channel
    last = snapshots[-1]
    assert len(last.channels) >= 1, "Expected at least one S/R channel to be detected"


def test_channel_types():
    """Channels should be labeled support, resistance, or inside based on close."""
    n = 200
    closes = [0.0] * n
    for i in range(n):
        if i % 40 < 20:
            closes[i] = 100 + (i % 20) * 0.5
        else:
            closes[i] = 110 - (i % 20) * 0.5

    opens = [c - 0.2 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    df = _make_df(opens, highs, lows, closes, n=n)
    detector = SupportResistanceChannelDetector(pivot_period=4, min_strength=1)
    snapshots = detector.calculate(df)

    last = snapshots[-1]
    for ch in last.channels:
        assert ch.channel_type in ("support", "resistance", "inside")


def test_dataclass_fields():
    """Verify SRChannel and SRBreakEvent dataclass attributes exist."""
    ch = SRChannel(high=105.0, low=100.0, strength=60.0, channel_type="support")
    assert ch.high == 105.0
    assert ch.low == 100.0
    assert ch.strength == 60.0
    assert ch.channel_type == "support"

    be = SRBreakEvent(
        bar_idx=50,
        timestamp=pd.Timestamp("2026-01-01 12:00"),
        break_type="resistance_broken",
        level_high=110.0,
        level_low=108.0,
        close=111.0,
    )
    assert be.break_type == "resistance_broken"
    assert be.close == 111.0
