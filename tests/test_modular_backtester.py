"""
Unit and Integration Tests for Modular Multi-Timeframe Backtester.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from trade_system.domains.analysis.application.backtesting.indicators_library import (
    apply_indicator_suite,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_supertrend,
    calculate_vwap,
)
from trade_system.domains.analysis.application.backtesting.modular_backtester import (
    BacktestConfig,
    BacktestStatisticsEngine,
    DurationConfig,
    ExitReason,
    ModularBacktester,
    StopLossConfig,
    StopLossType,
    TargetConfig,
    TargetType,
    TimeframeDataService,
    TradeSide,
)
from trade_system.domains.analysis.application.backtesting.backtest_reporter import (
    BacktestReporter,
)


@pytest.fixture
def synthetic_candles() -> pd.DataFrame:
    """Generates 200 synthetic 15-minute intraday candles."""
    np.random.seed(42)
    n = 200
    base_price = 20000.0
    start_dt = datetime(2026, 8, 1, 9, 15)

    timestamps = []
    curr = start_dt
    for _ in range(n):
        timestamps.append(curr)
        curr += timedelta(minutes=15)
        # Advance to next day 9:15 if past 15:30
        if curr.time() > dtime(15, 30):
            curr = datetime.combine(curr.date() + timedelta(days=1), dtime(9, 15))

    # Random walk
    returns = np.random.normal(0.0005, 0.005, n)
    closes = base_price * np.exp(np.cumsum(returns))
    highs = closes * (1.0 + np.random.uniform(0.001, 0.008, n))
    lows = closes * (1.0 - np.random.uniform(0.001, 0.008, n))
    opens = (closes + np.roll(closes, 1)) / 2.0
    opens[0] = base_price
    volumes = np.random.randint(1000, 50000, n)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    return df


def test_indicator_calculations(synthetic_candles):
    df = synthetic_candles

    # SMA & EMA
    sma = calculate_sma(df["close"], 20)
    ema = calculate_ema(df["close"], 20)
    assert len(sma) == len(df)
    assert not np.isnan(sma.iloc[-1])
    assert not np.isnan(ema.iloc[-1])

    # ATR
    atr = calculate_atr(df, 14)
    assert (atr.dropna() > 0).all()

    # RSI
    rsi = calculate_rsi(df["close"], 14)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    # Bollinger Bands
    upper, mid, lower, width, pct_b = calculate_bollinger_bands(df["close"], 20, 2.0)
    valid_idx = ~upper.isna()
    assert (upper[valid_idx] >= mid[valid_idx]).all()
    assert (mid[valid_idx] >= lower[valid_idx]).all()

    # MACD
    macd, sig, hist = calculate_macd(df["close"])
    assert len(macd) == len(df)
    assert len(sig) == len(df)

    # SuperTrend
    st_val, st_dir = calculate_supertrend(df, 10, 3.0)
    assert len(st_val) == len(df)
    assert set(st_dir.unique()).issubset({-1, 1})

    # Full Suite
    suite_df = apply_indicator_suite(df)
    assert "supertrend" in suite_df.columns
    assert "rsi_14" in suite_df.columns
    assert "ema_200" in suite_df.columns
    assert "vwap" in suite_df.columns


def test_timeframe_data_service_normalization():
    assert TimeframeDataService.normalize_resolution("15m") == "15"
    assert TimeframeDataService.normalize_resolution("5min") == "5"
    assert TimeframeDataService.normalize_resolution("60") == "60"
    assert TimeframeDataService.normalize_resolution("Daily") == "D"
    assert TimeframeDataService.normalize_resolution("day") == "D"


def test_stop_loss_and_target_execution(synthetic_candles):
    # Strategy: enter LONG on bar 10, check that either SL or TP is reached
    cfg = BacktestConfig(
        symbol="SYNTHETIC",
        timeframe="15",
        initial_capital=100000.0,
    )
    sl_cfg = StopLossConfig(sl_type=StopLossType.PERCENT, value=1.0)
    tp_cfg = TargetConfig(target_type=TargetType.PERCENT, value=2.0)
    dur_cfg = DurationConfig(max_bars=100, intraday_cutoff_time=None)

    def simple_entry(df, i):
        if i == 10:
            return TradeSide.LONG
        return None

    backtester = ModularBacktester(
        config=cfg,
        sl_config=sl_cfg,
        target_config=tp_cfg,
        duration_config=dur_cfg,
    )
    result = backtester.run(entry_signal_fn=simple_entry, custom_candles=synthetic_candles)

    assert len(result.trades) >= 1
    t = result.trades[0]
    assert t.side == TradeSide.LONG
    assert t.entry_index == 10
    assert t.exit_reason in [
        ExitReason.STOP_LOSS,
        ExitReason.TARGET,
        ExitReason.DURATION_MAX_BARS,
        ExitReason.END_OF_DATA,
    ]
    assert t.mfe_pct >= 0.0
    assert t.mae_pct >= 0.0


def test_duration_max_bars_timeout(synthetic_candles):
    cfg = BacktestConfig(symbol="SYNTHETIC", timeframe="15")
    # Set impossible SL (90%) and impossible Target (100%) so only max_bars can exit
    sl_cfg = StopLossConfig(sl_type=StopLossType.PERCENT, value=90.0)
    tp_cfg = TargetConfig(target_type=TargetType.PERCENT, value=100.0)
    dur_cfg = DurationConfig(max_bars=15, intraday_cutoff_time=None)

    def force_entry(df, i):
        if i == 5:
            return TradeSide.LONG
        return None

    backtester = ModularBacktester(
        config=cfg,
        sl_config=sl_cfg,
        target_config=tp_cfg,
        duration_config=dur_cfg,
    )
    result = backtester.run(entry_signal_fn=force_entry, custom_candles=synthetic_candles)

    assert len(result.trades) >= 1
    t = result.trades[0]
    assert t.exit_reason == ExitReason.DURATION_MAX_BARS
    assert t.bars_held == 15


def test_intraday_cutoff_execution(synthetic_candles):
    cfg = BacktestConfig(symbol="SYNTHETIC", timeframe="15")
    sl_cfg = StopLossConfig(sl_type=StopLossType.PERCENT, value=90.0)
    tp_cfg = TargetConfig(target_type=TargetType.PERCENT, value=100.0)
    # Cutoff at 15:15
    dur_cfg = DurationConfig(max_bars=100, intraday_cutoff_time=dtime(15, 15))

    # Enter at 14:00 on first day
    def enter_at_14(df, i):
        t = pd.to_datetime(df["timestamp"].iloc[i]).time()
        if t == dtime(14, 0) and i < 20:
            return TradeSide.LONG
        return None

    backtester = ModularBacktester(
        config=cfg,
        sl_config=sl_cfg,
        target_config=tp_cfg,
        duration_config=dur_cfg,
    )
    result = backtester.run(entry_signal_fn=enter_at_14, custom_candles=synthetic_candles)

    assert len(result.trades) >= 1
    t = result.trades[0]
    assert t.exit_reason == ExitReason.INTRADAY_CUTOFF
    assert t.exit_time.time() >= dtime(15, 15)


def test_statistics_engine():
    # Construct mock trades dataframe
    trades_data = [
        {"net_pnl": 500.0, "r_multiple": 2.0, "bars_held": 10, "duration_mins": 150.0, "mfe_pct": 2.5, "mae_pct": 0.5, "exit_reason": "TARGET"},
        {"net_pnl": -250.0, "r_multiple": -1.0, "bars_held": 5, "duration_mins": 75.0, "mfe_pct": 0.5, "mae_pct": 1.2, "exit_reason": "STOP_LOSS"},
        {"net_pnl": 1000.0, "r_multiple": 4.0, "bars_held": 20, "duration_mins": 300.0, "mfe_pct": 4.2, "mae_pct": 0.8, "exit_reason": "TARGET"},
        {"net_pnl": -250.0, "r_multiple": -1.0, "bars_held": 6, "duration_mins": 90.0, "mfe_pct": 0.3, "mae_pct": 1.1, "exit_reason": "STOP_LOSS"},
    ]
    trades_df = pd.DataFrame(trades_data)

    eq_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-01", periods=10, freq="15min"),
            "equity": [100000, 100200, 100500, 100250, 101250, 101000, 101000, 101000, 101000, 101000],
        }
    )

    stats = BacktestStatisticsEngine.calculate_statistics(trades_df, eq_df, initial_capital=100000.0)

    assert stats["total_trades"] == 4
    assert stats["winning_trades"] == 2
    assert stats["losing_trades"] == 2
    assert stats["win_rate_pct"] == 50.0
    assert stats["gross_profit"] == 1500.0
    assert stats["gross_loss"] == 500.0
    assert stats["net_profit"] == 1000.0
    assert stats["profit_factor"] == 3.0  # 1500 / 500
    assert stats["payoff_ratio"] == 3.0   # avg win 750 / avg loss 250
    assert stats["max_consecutive_wins"] == 1
    assert stats["max_consecutive_losses"] == 1


def test_reporter_generation(tmp_path, synthetic_candles):
    cfg = BacktestConfig(symbol="NSE:TEST-EQ", timeframe="15")
    backtester = ModularBacktester(config=cfg)

    def mock_strategy(df, i):
        if i % 30 == 0 and i > 0:
            return TradeSide.LONG
        return None

    result = backtester.run(entry_signal_fn=mock_strategy, custom_candles=synthetic_candles)

    # ASCII
    ascii_rep = BacktestReporter.generate_ascii_summary(result)
    assert "QUANTITATIVE BACKTEST PERFORMANCE REPORT" in ascii_rep
    assert "Profit Factor" in ascii_rep

    # Markdown
    md_rep = BacktestReporter.generate_markdown_report(result)
    assert "Executive KPI Summary" in md_rep
    assert "Trade Duration" in md_rep

    # File exports
    trades_p, eq_p = BacktestReporter.export_csv(result, tmp_path)
    assert trades_p.exists()
    assert eq_p.exists()

    json_p = BacktestReporter.export_json(result, tmp_path / "test.json")
    assert json_p.exists()


def test_advanced_smc_fvg_strategies(synthetic_candles):
    from trade_system.domains.analysis.application.backtesting.advanced_strategies import (
        strategy_smc_liquidity_sweep,
        strategy_ict_fvg_mitigation,
        strategy_order_block_retest,
        strategy_wyckoff_amd,
    )

    cfg = BacktestConfig(symbol="NSE:TEST-EQ", timeframe="15")
    backtester = ModularBacktester(config=cfg)

    # 1. Test SMC Liquidity Sweep strategy
    res_smc = backtester.run(entry_signal_fn=strategy_smc_liquidity_sweep, custom_candles=synthetic_candles)
    assert res_smc.candles_count == len(synthetic_candles)
    assert isinstance(res_smc.metrics, dict)

    # 2. Test ICT FVG Mitigation strategy
    res_fvg = backtester.run(entry_signal_fn=strategy_ict_fvg_mitigation, custom_candles=synthetic_candles)
    assert res_fvg.candles_count == len(synthetic_candles)
    assert isinstance(res_fvg.metrics, dict)

    # 3. Test Order Block Retest strategy
    res_ob = backtester.run(entry_signal_fn=strategy_order_block_retest, custom_candles=synthetic_candles)
    assert res_ob.candles_count == len(synthetic_candles)
    assert isinstance(res_ob.metrics, dict)

    # 4. Test Wyckoff AMD strategy
    res_amd = backtester.run(entry_signal_fn=strategy_wyckoff_amd, custom_candles=synthetic_candles)
    assert res_amd.candles_count == len(synthetic_candles)
    assert isinstance(res_amd.metrics, dict)

