"""
Unit tests for Option Buyer Protocol, Dynamic S/R Zones, Market Regimes, and VIX Dynamics.
"""
from datetime import datetime, time as dtime, date
import numpy as np
import pandas as pd
import pytest

from trade_system.domains.analysis.application.backtesting.modular_backtester import (
    TradeSide,
    EntrySignal,
    ModularBacktester,
    BacktestConfig,
)
from trade_system.domains.analysis.application.backtesting.indicators_library import (
    calculate_adx,
    calculate_bollinger_squeeze,
    calculate_support_resistance_zones,
    calculate_market_regime,
    calculate_vix_dynamics,
    apply_indicator_suite,
)
from trade_system.domains.analysis.application.backtesting.option_buyer_engine import (
    OptionBuyerFilter,
    strategy_option_buyer_momentum_burst,
    strategy_wyckoff_amd_opt,
    strategy_smc_liquidity_sweep_opt,
    strategy_ict_fvg_mitigation_opt,
)


@pytest.fixture
def sample_market_df():
    """Generates 100 bars of realistic intraday candle series with timestamps."""
    np.random.seed(42)
    n = 100
    timestamps = pd.date_range("2026-09-01 09:15", periods=n, freq="15min")
    
    close = 24000.0 + np.cumsum(np.random.randn(n) * 15.0)
    high = close + np.random.uniform(5, 25, size=n)
    low = close - np.random.uniform(5, 25, size=n)
    open_p = close + np.random.uniform(-10, 10, size=n)
    volume = np.random.uniform(10000, 100000, size=n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_option_buyer_time_windows():
    """Tests that OptionBuyerFilter allows prime windows and rejects the midday churn trap."""
    # 09:45 IST -> Morning Impulse (Allowed)
    t_morning = datetime(2026, 9, 1, 9, 45)
    allowed, reason = OptionBuyerFilter.is_prime_time_window(t_morning)
    assert allowed is True
    assert "MORNING" in reason

    # 12:00 IST -> Midday Churn Trap (Vetoed)
    t_midday = datetime(2026, 9, 1, 12, 0)
    allowed, reason = OptionBuyerFilter.is_prime_time_window(t_midday)
    assert allowed is False
    assert "MIDDAY_CHURN_TRAP_VETO" in reason

    # 13:45 IST -> Afternoon Gamma Blast (Allowed)
    t_afternoon = datetime(2026, 9, 1, 13, 45)
    allowed, reason = OptionBuyerFilter.is_prime_time_window(t_afternoon)
    assert allowed is True
    assert "AFTERNOON_GAMMA_WINDOW" in reason

    # 15:15 IST -> Late Day Lockout (Vetoed)
    t_late = datetime(2026, 9, 1, 15, 15)
    allowed, reason = OptionBuyerFilter.is_prime_time_window(t_late)
    assert allowed is False
    assert "LATE_DAY_LOCKOUT_VETO" in reason


def test_sr_clearance_vetoes():
    """Tests that Call buys into Resistance ceilings and Put buys into Support floors are vetoed."""
    # Spot 24500, Resistance 24520 (Distance: 0.08% < 0.35% minimum) -> VETO LONG
    allowed_long, reason_long = OptionBuyerFilter.has_sr_clearance(
        side=TradeSide.LONG,
        price=24500.0,
        nearest_res=24520.0,
        nearest_sup=24300.0,
        min_room_pct=0.35
    )
    assert allowed_long is False
    assert "RESISTANCE_CEILING_VETO" in reason_long

    # Spot 24500, Resistance 24700 (Distance: 0.81% >= 0.35%) -> CLEAR LONG
    allowed_long_ok, _ = OptionBuyerFilter.has_sr_clearance(
        side=TradeSide.LONG,
        price=24500.0,
        nearest_res=24700.0,
        nearest_sup=24300.0,
        min_room_pct=0.35
    )
    assert allowed_long_ok is True

    # Spot 24500, Support 24480 (Distance: 0.08% < 0.35%) -> VETO SHORT
    allowed_short, reason_short = OptionBuyerFilter.has_sr_clearance(
        side=TradeSide.SHORT,
        price=24500.0,
        nearest_res=24700.0,
        nearest_sup=24480.0,
        min_room_pct=0.35
    )
    assert allowed_short is False
    assert "SUPPORT_FLOOR_VETO" in reason_short


def test_vix_dynamics_and_indicators(sample_market_df):
    """Tests calculation of ADX, Bollinger Squeeze, S/R zones, Market Regimes, and VIX."""
    df_ind = apply_indicator_suite(sample_market_df)

    # Validate presence of newly added columns
    expected_cols = [
        "adx", "plus_di", "minus_di",
        "in_squeeze", "squeeze_release", "bandwidth_pctile",
        "swing_high", "swing_low", "nearest_res", "nearest_sup",
        "dist_to_res_pct", "dist_to_sup_pct",
        "market_regime", "vix_level", "vix_change_pct", "vix_regime", "vix_trend"
    ]
    for col in expected_cols:
        assert col in df_ind.columns, f"Missing expected column: {col}"

    # Verify ADX range [0, 100]
    valid_adx = df_ind["adx"].dropna()
    assert (valid_adx >= 0).all() and (valid_adx <= 100).all()

    # Verify Market Regimes classification
    regimes = df_ind["market_regime"].unique()
    valid_regimes = {"TRENDING_BULL", "TRENDING_BEAR", "VOLATILITY_SQUEEZE", "CHOPPY_RANGEBOUND"}
    assert all(r in valid_regimes for r in regimes)

    # Verify S/R distance >= 0
    assert (df_ind["dist_to_res_pct"] >= 0).all()
    assert (df_ind["dist_to_sup_pct"] >= 0).all()


def test_option_buyer_strategies_execution(sample_market_df):
    """Tests backtester execution with option-buyer strategies."""
    df_ind = apply_indicator_suite(sample_market_df)
    cfg = BacktestConfig(symbol="NSE:NIFTY50-INDEX", timeframe="15")
    bt = ModularBacktester(config=cfg)

    # Run Option Buyer Momentum Burst
    res_burst = bt.run(entry_signal_fn=strategy_option_buyer_momentum_burst, custom_candles=df_ind)
    assert res_burst is not None
    assert "net_profit" in res_burst.metrics

    # Run Wyckoff AMD Opt
    res_amd = bt.run(entry_signal_fn=strategy_wyckoff_amd_opt, custom_candles=df_ind)
    assert res_amd is not None
    assert "win_rate_pct" in res_amd.metrics
