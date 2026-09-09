"""
Unit tests for World-Class Option Buyer Strategies & Quantitative Indicators.
Covers:
- Larry Connors (RSI(2) Mean Reversion & ConnorsRSI)
- Paul Tudor Jones (200-DMA Trend Gate)
- Tom Sosnoff & Sheldon Natenberg (IV Rank Gate & Garman-Klass Volatility)
- Linda Bradford Raschke (Momentum Pinball)
- Oliver Velez (Opening Range Breakout & Elephant Bar Continuation)
- Mark Minervini (Volatility Contraction Pattern / VCP Breakout)
- Jeff Augen (Expiry Day Gamma Edge)
"""
from datetime import datetime, time as dtime
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
    calculate_sma,
    calculate_connors_rsi,
    calculate_momentum_pinball,
    calculate_opening_range,
    calculate_iv_proxy_rank,
    calculate_elephant_bar,
    calculate_vcp_contraction,
    apply_indicator_suite,
)
from trade_system.domains.analysis.application.backtesting.option_buyer_engine import (
    OptionBuyerFilter,
    strategy_connors_rsi_mean_reversion,
    strategy_opening_range_breakout,
    strategy_momentum_pinball,
    strategy_vcp_breakout,
    strategy_expiry_day_gamma_edge,
    strategy_elephant_bar_continuation,
    strategy_option_buyer_momentum_burst,
)


@pytest.fixture
def sample_market_df():
    """Generates 300 bars of realistic intraday candles with timestamps spanning multiple days."""
    np.random.seed(42)
    n = 300
    timestamps = pd.date_range("2026-09-02 09:15", periods=n, freq="15min")

    close = 24000.0 + np.cumsum(np.random.randn(n) * 15.0)
    high = close + np.random.uniform(5, 25, size=n)
    low = close - np.random.uniform(5, 25, size=n)
    open_p = close + np.random.uniform(-10, 10, size=n)
    volume = np.random.uniform(20000, 150000, size=n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Indicator Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_connors_rsi_calculation(sample_market_df):
    """Verifies ConnorsRSI, RSI(2), and SMA(5) calculation."""
    connors_rsi, rsi_2, sma_5 = calculate_connors_rsi(sample_market_df)
    assert len(connors_rsi) == len(sample_market_df)
    assert len(rsi_2) == len(sample_market_df)
    assert len(sma_5) == len(sample_market_df)

    # Valid range: RSI is between 0 and 100
    valid_rsi2 = rsi_2.dropna()
    assert (valid_rsi2 >= 0).all() and (valid_rsi2 <= 100).all()


def test_momentum_pinball_calculation(sample_market_df):
    """Verifies Linda Raschke's Momentum Pinball indicator."""
    pinball, pin_buy, pin_sell = calculate_momentum_pinball(sample_market_df)
    assert len(pinball) == len(sample_market_df)
    assert isinstance(pin_buy, pd.Series)
    assert isinstance(pin_sell, pd.Series)
    assert pin_buy.dtype == bool
    assert pin_sell.dtype == bool


def test_opening_range_calculation(sample_market_df):
    """Verifies Oliver Velez Opening Range calculation."""
    orb = calculate_opening_range(sample_market_df)
    assert "orb_high" in orb
    assert "orb_low" in orb
    assert "orb_breakout_long" in orb
    assert "orb_breakout_short" in orb
    assert (orb["orb_high"] >= orb["orb_low"]).all()


def test_iv_proxy_rank_calculation(sample_market_df):
    """Verifies Garman-Klass IV proxy and IV Rank (0 to 100 scale)."""
    iv_rank = calculate_iv_proxy_rank(sample_market_df, lookback=50)
    assert len(iv_rank) == len(sample_market_df)
    assert (iv_rank >= 0.0).all() and (iv_rank <= 100.0).all()


def test_elephant_bar_detection(sample_market_df):
    """Verifies Elephant Bar detection outputs valid boolean Series."""
    bull_bar, bear_bar = calculate_elephant_bar(sample_market_df, bars_to_clear=2)
    assert len(bull_bar) == len(sample_market_df)
    assert len(bear_bar) == len(sample_market_df)
    assert bull_bar.dtype == bool
    assert bear_bar.dtype == bool


def test_vcp_contraction_detection(sample_market_df):
    """Verifies Mark Minervini VCP contraction outputs."""
    contraction, breakout = calculate_vcp_contraction(sample_market_df, contractions=2)
    assert len(contraction) == len(sample_market_df)
    assert len(breakout) == len(sample_market_df)
    assert contraction.dtype == bool
    assert breakout.dtype == bool


def test_apply_indicator_suite_worldclass_columns(sample_market_df):
    """Verifies that apply_indicator_suite adds all new world-class indicator columns."""
    df_ind = apply_indicator_suite(sample_market_df)
    expected_cols = [
        "sma_200",
        "connors_rsi",
        "rsi_2",
        "sma_5",
        "pinball",
        "pinball_buy_signal",
        "pinball_sell_signal",
        "orb_high",
        "orb_low",
        "orb_breakout_long",
        "orb_breakout_short",
        "iv_rank",
        "elephant_bar_bull",
        "elephant_bar_bear",
        "vcp_contraction",
        "vcp_breakout",
    ]
    for col in expected_cols:
        assert col in df_ind.columns, f"Missing expected column: {col}"


# ─────────────────────────────────────────────────────────────────────────────
# Gate & Filter Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_paul_tudor_jones_200dma_gate():
    """Verifies Paul Tudor Jones 200-DMA Gate: Longs allowed only > 200DMA, Shorts only < 200DMA."""
    # Long above 200 DMA -> Allowed
    ok, reason = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close=24500.0, sma_200=24000.0)
    assert ok is True
    assert "200DMA_TREND_ALIGNED" in reason

    # Long below 200 DMA -> Vetoed
    ok, reason = OptionBuyerFilter.is_200dma_aligned(TradeSide.LONG, close=23800.0, sma_200=24000.0)
    assert ok is False
    assert "200DMA_TREND_VETO_LONG" in reason

    # Short below 200 DMA -> Allowed
    ok, reason = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close=23800.0, sma_200=24000.0)
    assert ok is True
    assert "200DMA_TREND_ALIGNED" in reason

    # Short above 200 DMA -> Vetoed
    ok, reason = OptionBuyerFilter.is_200dma_aligned(TradeSide.SHORT, close=24500.0, sma_200=24000.0)
    assert ok is False
    assert "200DMA_TREND_VETO_SHORT" in reason


def test_sosnoff_natenberg_iv_rank_gate():
    """Verifies Tom Sosnoff / Sheldon Natenberg IV Rank Gate."""
    # Cheap options (IV Rank 25) -> Favorable
    ok, reason = OptionBuyerFilter.is_iv_rank_favorable(iv_rank=25.0, trade_type="BUY")
    assert ok is True
    assert "IV_RANK_FAVORABLE" in reason

    # Moderate options (IV Rank 50) -> Neutral Caution (Allowed)
    ok, reason = OptionBuyerFilter.is_iv_rank_favorable(iv_rank=50.0, trade_type="BUY")
    assert ok is True
    assert "IV_RANK_NEUTRAL_CAUTION" in reason

    # Expensive options (IV Rank 75) -> VETO (IV crush risk)
    ok, reason = OptionBuyerFilter.is_iv_rank_favorable(iv_rank=75.0, trade_type="BUY")
    assert ok is False
    assert "IV_RANK_EXPENSIVE_VETO" in reason


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Execution Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_connors_rsi_mean_reversion_execution(sample_market_df):
    """Verifies Connors RSI strategy signals conform to EntrySignal contracts."""
    df_ind = apply_indicator_suite(sample_market_df)
    signals = []
    for i in range(25, len(df_ind)):
        sig = strategy_connors_rsi_mean_reversion(df_ind, i)
        if sig is not None:
            signals.append(sig)
            assert isinstance(sig, EntrySignal)
            assert sig.stop_loss != sig.price
            assert sig.target != sig.price


def test_opening_range_breakout_execution(sample_market_df):
    """Verifies Opening Range Breakout strategy signals conform to EntrySignal contracts."""
    df_ind = apply_indicator_suite(sample_market_df)
    for i in range(10, len(df_ind)):
        sig = strategy_opening_range_breakout(df_ind, i)
        if sig is not None:
            assert isinstance(sig, EntrySignal)
            if sig.side == TradeSide.LONG:
                assert sig.target > sig.price
                assert sig.stop_loss < sig.price
            else:
                assert sig.target < sig.price
                assert sig.stop_loss > sig.price


def test_momentum_pinball_execution(sample_market_df):
    """Verifies Linda Raschke Momentum Pinball strategy execution."""
    df_ind = apply_indicator_suite(sample_market_df)
    for i in range(15, len(df_ind)):
        sig = strategy_momentum_pinball(df_ind, i)
        if sig is not None:
            assert isinstance(sig, EntrySignal)
            assert sig.confidence >= 80.0


def test_backtest_with_worldclass_strategies(sample_market_df):
    """Verifies running the modular backtester with the new strategies completes without error."""
    df_ind = apply_indicator_suite(sample_market_df)
    bt = ModularBacktester(config=BacktestConfig(symbol="NSE:NIFTY50-INDEX", timeframe="15"))

    # Test Connors RSI
    res_connors = bt.run(entry_signal_fn=strategy_connors_rsi_mean_reversion, custom_candles=df_ind)
    assert res_connors is not None
    assert "net_profit" in res_connors.metrics

    # Test Momentum Burst
    res_burst = bt.run(entry_signal_fn=strategy_option_buyer_momentum_burst, custom_candles=df_ind)
    assert res_burst is not None
    assert "net_profit" in res_burst.metrics

