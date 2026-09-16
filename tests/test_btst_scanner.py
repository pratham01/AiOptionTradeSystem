"""
Unit tests for BTST Institutional Scanner.
"""

from datetime import datetime, date, timedelta
import pandas as pd
import pytest

from trade_system.domains.analysis.application.analysis.btst_scanner import (
    BTSTInstitutionalScanner,
    BTSTCandidate,
)


def _create_mock_candles() -> pd.DataFrame:
    """Generate mock 5-minute candles spanning 09:15 to 15:30 IST."""
    today = date.today()
    timestamps = []
    current = datetime(today.year, today.month, today.day, 9, 15)
    end = datetime(today.year, today.month, today.day, 15, 30)

    while current < end:
        timestamps.append(current)
        current += timedelta(minutes=5)

    n = len(timestamps)
    # Total volume: make earlier candles have volume 1,000 and 15:00-15:30 have volume 10,000 (surge)
    volumes = []
    opens = []
    highs = []
    lows = []
    closes = []
    price = 1000.0

    for ts in timestamps:
        opens.append(price)
        highs.append(price + 2.0)
        lows.append(price - 1.0)
        closes.append(price + 1.5)
        price += 1.0

        if ts.time() >= datetime.strptime("15:00", "%H:%M").time():
            volumes.append(10000)  # Heavy late-session burst
        else:
            volumes.append(1000)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_btst_scanner_long_buildup_scoring():
    scanner = BTSTInstitutionalScanner(broker=None)
    mock_candles = _create_mock_candles()

    quote = {
        "lp": 1075.0,
        "open_price": 1000.0,
        "high_price": 1076.0,  # Within 0.1% of Day High
        "low_price": 998.0,
        "prev_close_price": 1050.0,  # +2.38% gain
        "ch": 25.0,
        "chp": 2.38,
        "volume": mock_candles["volume"].sum(),
    }

    mock_oc = pd.DataFrame([
        {"option_type": "CE", "strike": 1080, "oi": 50000, "oich": -5000},  # Call unwinding
        {"option_type": "PE", "strike": 1060, "oi": 60000, "oich": 15000},  # Heavy Put writing
    ])

    cand = scanner.analyze_single_stock(
        symbol="NSE:TESTSTOCK-EQ",
        quote=quote,
        candles_df=mock_candles,
        option_chain_df=mock_oc,
    )

    assert cand is not None
    assert cand.clean_symbol == "TESTSTOCK"
    assert cand.dist_to_day_high_pct < 0.5
    assert cand.vol_30m_ratio_pct > 20.0  # Last 30m volume is heavy
    assert cand.oi_quadrant == "LONG_BUILDUP"
    assert not cand.is_trap
    assert cand.btst_score >= 75
    assert cand.verdict == "STRONG_BTST_ACCUMULATION"


def test_btst_scanner_short_covering_trap():
    scanner = BTSTInstitutionalScanner(broker=None)
    mock_candles = _create_mock_candles()

    quote = {
        "lp": 1075.0,
        "open_price": 1000.0,
        "high_price": 1076.0,
        "low_price": 998.0,
        "prev_close_price": 1050.0,
        "ch": 25.0,
        "chp": 2.38,
        "volume": mock_candles["volume"].sum(),
    }

    # In a trap: Call OI drops drastically (unwinding), but Put OI is ALSO dropping or negative!
    mock_oc = pd.DataFrame([
        {"option_type": "CE", "strike": 1080, "oi": 30000, "oich": -20000}, # Call unwinding
        {"option_type": "PE", "strike": 1060, "oi": 20000, "oich": -5000},  # Put unwinding (no support!)
    ])

    cand = scanner.analyze_single_stock(
        symbol="NSE:TRAPSTOCK-EQ",
        quote=quote,
        candles_df=mock_candles,
        option_chain_df=mock_oc,
    )

    assert cand is not None
    assert cand.oi_quadrant == "SHORT_COVERING"
    assert cand.is_trap is True
    assert cand.oi_flow_score == 0
    assert cand.verdict == "BTST_TRAP_SHORT_COVERING"
    assert "TRAP" in cand.trap_warning


def test_btst_telegram_alert_format():
    scanner = BTSTInstitutionalScanner(broker=None)
    mock_cand = BTSTCandidate(
        symbol="NSE:RELIANCE-EQ",
        clean_symbol="RELIANCE",
        sector="ENERGY",
        spot_price=3000.0,
        day_open=2950.0,
        day_high=3005.0,
        day_low=2940.0,
        prev_close=2945.0,
        day_change=55.0,
        day_change_pct=1.87,
        dist_to_day_high_pct=0.17,
        total_day_volume=5000000,
        last_30m_volume=1200000,
        vol_30m_ratio_pct=24.0,
        vol_burst_multiplier=3.0,
        vwap=2975.0,
        price_vs_vwap_pct=0.84,
        oi_quadrant="LONG_BUILDUP",
        total_ce_oi=1000000,
        total_pe_oi=1200000,
        net_ce_oich=-50000,
        net_pe_oich=150000,
        pcr_oi=1.2,
        immediate_call_wall=3040.0,
        immediate_put_wall=2960.0,
        price_action_score=25,
        volume_shockwave_score=25,
        oi_flow_score=25,
        trend_score=25,
        btst_score=100,
        verdict="STRONG_BTST_ACCUMULATION",
        verdict_badge_color="#00d084",
        is_trap=False,
        trap_warning="",
        recommended_entry="₹3,000 — ₹3,012",
        stop_loss="₹2,975",
        target_1="₹3,054",
        target_2="₹3,096",
        risk_reward="1 : 2.2",
        rationale="Solid late-day volume burst with long buildup.",
    )

    alert_text = scanner.format_telegram_alert(mock_cand)
    assert "RELIANCE" in alert_text
    assert "100/100" in alert_text
    assert "24.0%" in alert_text
    assert "LONG BUILDUP" in alert_text
