#!/usr/bin/env python3
"""
CLI Runner for Modular Multi-Timeframe Backtester.

Usage:
  python3 scripts/run_modular_backtest.py --symbol NSE:NIFTY50-INDEX --timeframe 15 --strategy supertrend_ema
  python3 scripts/run_modular_backtest.py --symbol NSE:RELIANCE-EQ --timeframe D --strategy rsi_bb_reversion --sl-val 2.0 --tp-val 2.5
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root and src to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import pandas as pd
from trade_system.domains.analysis.application.backtesting.modular_backtester import (
    BacktestConfig,
    DurationConfig,
    ModularBacktester,
    StopLossConfig,
    StopLossType,
    TargetConfig,
    TargetType,
    TradeSide,
)
from trade_system.domains.analysis.application.backtesting.backtest_reporter import (
    BacktestReporter,
)


def strategy_supertrend_ema(df: pd.DataFrame, i: int):
    """Buy when Close > 200 EMA and SuperTrend flips Bullish."""
    if i < 2:
        return None
    curr_st = df["supertrend_dir"].iloc[i]
    prev_st = df["supertrend_dir"].iloc[i - 1]
    close = df["close"].iloc[i]
    ema_200 = df["ema_200"].iloc[i]

    # Bullish Flip above 200 EMA
    if curr_st == 1 and prev_st == -1 and close > ema_200:
        return TradeSide.LONG
    # Bearish Flip below 200 EMA
    elif curr_st == -1 and prev_st == 1 and close < ema_200:
        return TradeSide.SHORT
    return None


def strategy_rsi_bb_reversion(df: pd.DataFrame, i: int):
    """Mean Reversion: Buy when RSI < 30 and price touches lower BB."""
    if i < 2:
        return None
    rsi = df["rsi_14"].iloc[i]
    low = df["low"].iloc[i]
    bb_lower = df["bb_lower"].iloc[i]
    high = df["high"].iloc[i]
    bb_upper = df["bb_upper"].iloc[i]

    if rsi < 32 and low <= bb_lower:
        return TradeSide.LONG
    elif rsi > 68 and high >= bb_upper:
        return TradeSide.SHORT
    return None


def strategy_dual_ema_crossover(df: pd.DataFrame, i: int):
    """Fast 9 EMA crosses Slow 21 EMA."""
    if i < 2:
        return None
    fast_curr = df["ema_9"].iloc[i]
    fast_prev = df["ema_9"].iloc[i - 1]
    slow_curr = df["ema_21"].iloc[i]
    slow_prev = df["ema_21"].iloc[i - 1]

    if fast_prev <= slow_prev and fast_curr > slow_curr:
        return TradeSide.LONG
    elif fast_prev >= slow_prev and fast_curr < slow_curr:
        return TradeSide.SHORT
    return None


def strategy_macd_trend(df: pd.DataFrame, i: int):
    """MACD Line crosses Signal Line."""
    if i < 2:
        return None
    macd_curr = df["macd"].iloc[i]
    macd_prev = df["macd"].iloc[i - 1]
    sig_curr = df["macd_signal"].iloc[i]
    sig_prev = df["macd_signal"].iloc[i - 1]

    if macd_prev <= sig_prev and macd_curr > sig_curr:
        return TradeSide.LONG
    elif macd_prev >= sig_prev and macd_curr < sig_curr:
        return TradeSide.SHORT
    return None


def strategy_smc_liquidity_sweep(df: pd.DataFrame, i: int):
    """
    SMC Liquidity Sweep & Reclaim:
    - Long: Price sweeps below 8-period low (Sell-side Liquidity SSL), reclaims with bullish candle, and trades above 50 EMA.
    - Short: Price sweeps above 8-period high (Buy-side Liquidity BSL), rejects with bearish candle, and trades below 50 EMA.
    """
    if i < 8:
        return None
    bull_sweep = df["bull_sweep"].iloc[i]
    bear_sweep = df["bear_sweep"].iloc[i]
    close = df["close"].iloc[i]
    ema_50 = df["ema_50"].iloc[i]

    if bull_sweep and close > ema_50:
        return TradeSide.LONG
    elif bear_sweep and close < ema_50:
        return TradeSide.SHORT
    return None


def strategy_ict_fvg_mitigation(df: pd.DataFrame, i: int):
    """
    ICT Fair Value Gap (FVG) Mitigation Entry:
    - Long: Bullish FVG formed within last 5 bars. Current bar dips to mitigate FVG (low <= fvg_top) and closes bullish (close > open).
    - Short: Bearish FVG formed within last 5 bars. Current bar rallies to mitigate FVG (high >= fvg_bottom) and closes bearish (close < open).
    """
    if i < 6:
        return None
    close = df["close"].iloc[i]
    open_p = df["open"].iloc[i]
    low = df["low"].iloc[i]
    high = df["high"].iloc[i]
    ema_50 = df["ema_50"].iloc[i]

    recent_bull_fvg = df["bull_fvg"].iloc[max(0, i - 5) : i]
    if recent_bull_fvg.any() and close > ema_50:
        fvg_idx = recent_bull_fvg[recent_bull_fvg].index[-1]
        fvg_top = df.loc[fvg_idx, "fvg_top"]
        fvg_mid = df.loc[fvg_idx, "fvg_mid"]
        if low <= fvg_top and close >= fvg_mid and close > open_p:
            return TradeSide.LONG

    recent_bear_fvg = df["bear_fvg"].iloc[max(0, i - 5) : i]
    if recent_bear_fvg.any() and close < ema_50:
        fvg_idx = recent_bear_fvg[recent_bear_fvg].index[-1]
        fvg_bottom = df.loc[fvg_idx, "fvg_bottom"]
        fvg_mid = df.loc[fvg_idx, "fvg_mid"]
        if high >= fvg_bottom and close <= fvg_mid and close < open_p:
            return TradeSide.SHORT

    return None


def strategy_order_block_retest(df: pd.DataFrame, i: int):
    """
    Institutional Order Block (OB) Re-test:
    - Long: Bullish OB created within last 8 bars. Price tests OB zone and prints green rejection bar.
    - Short: Bearish OB created within last 8 bars. Price tests OB zone and prints red rejection bar.
    """
    if i < 8:
        return None
    close = df["close"].iloc[i]
    open_p = df["open"].iloc[i]
    ema_50 = df["ema_50"].iloc[i]

    recent_bull_ob = df["bull_ob"].iloc[max(0, i - 8) : i]
    if recent_bull_ob.any() and close > ema_50 and close > open_p:
        return TradeSide.LONG

    recent_bear_ob = df["bear_ob"].iloc[max(0, i - 8) : i]
    if recent_bear_ob.any() and close < ema_50 and close < open_p:
        return TradeSide.SHORT

    return None


def strategy_wyckoff_amd(df: pd.DataFrame, i: int):
    """
    Wyckoff / Power of 3 (AMD) Cycle:
    - Long: Manipulation Spring below accumulation range, followed by strong reclaim above range low.
    - Short: Manipulation UTAD above accumulation range, followed by strong rejection below range high.
    """
    if i < 16:
        return None
    range_high = df["high"].iloc[i - 16 : i - 2].max()
    range_low = df["low"].iloc[i - 16 : i - 2].min()
    range_span = range_high - range_low
    atr = df["atr"].iloc[i]

    if range_span < (2.0 * atr):
        close = df["close"].iloc[i]
        low = df["low"].iloc[i]
        high = df["high"].iloc[i]
        open_p = df["open"].iloc[i]

        if low < range_low and close > range_low and close > open_p:
            return TradeSide.LONG
        elif high > range_high and close < range_high and close < open_p:
            return TradeSide.SHORT

    return None


from trade_system.domains.analysis.application.backtesting.option_buyer_engine import (
    strategy_option_buyer_momentum_burst,
    strategy_wyckoff_amd_opt,
    strategy_smc_liquidity_sweep_opt,
    strategy_ict_fvg_mitigation_opt,
    strategy_supertrend_sr_opt,
    strategy_connors_rsi_mean_reversion,
    strategy_opening_range_breakout,
    strategy_momentum_pinball,
    strategy_vcp_breakout,
    strategy_expiry_day_gamma_edge,
    strategy_elephant_bar_continuation,
)

STRATEGIES = {
    # ── World-Class Option Buyer Protocol Strategies ──
    "option_buyer_momentum_burst": strategy_option_buyer_momentum_burst,
    "connors_rsi_mean_reversion": strategy_connors_rsi_mean_reversion,
    "opening_range_breakout": strategy_opening_range_breakout,
    "elephant_bar_continuation": strategy_elephant_bar_continuation,
    "momentum_pinball": strategy_momentum_pinball,
    "vcp_breakout": strategy_vcp_breakout,
    "expiry_day_gamma_edge": strategy_expiry_day_gamma_edge,
    "wyckoff_amd_opt": strategy_wyckoff_amd_opt,
    "smc_liquidity_sweep_opt": strategy_smc_liquidity_sweep_opt,
    "ict_fvg_mitigation_opt": strategy_ict_fvg_mitigation_opt,
    "supertrend_sr_opt": strategy_supertrend_sr_opt,
    # ── Baseline Strategies ──
    "wyckoff_amd": strategy_wyckoff_amd,
    "supertrend_ema": strategy_supertrend_ema,
    "smc_liquidity_sweep": strategy_smc_liquidity_sweep,
    "ict_fvg_mitigation": strategy_ict_fvg_mitigation,
    "order_block_retest": strategy_order_block_retest,
    "rsi_bb_reversion": strategy_rsi_bb_reversion,
    "dual_ema": strategy_dual_ema_crossover,
    "macd_trend": strategy_macd_trend,
}



def main():
    parser = argparse.ArgumentParser(description="Modular Quantitative Backtesting Runner")
    parser.add_argument("--symbol", type=str, default="NSE:NIFTY50-INDEX", help="Symbol to test")
    parser.add_argument("--timeframe", type=str, default="15", help="Timeframe (1, 3, 5, 15, 30, 60, D)")
    parser.add_argument("--strategy", type=str, default="supertrend_ema", choices=list(STRATEGIES.keys()), help="Strategy logic")
    parser.add_argument("--from-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--to-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial Capital (₹)")
    parser.add_argument("--sl-type", type=str, default="ATR", choices=["ATR", "PERCENT", "SWING", "TRAILING_ATR"], help="SL Type")
    parser.add_argument("--sl-val", type=float, default=1.5, help="SL Value (multiplier or %)")
    parser.add_argument("--tp-type", type=str, default="RR_RATIO", choices=["RR_RATIO", "PERCENT", "ATR"], help="Target Type")
    parser.add_argument("--tp-val", type=float, default=2.0, help="Target Value (R:R ratio or %)")
    parser.add_argument("--be-rr", type=float, default=1.0, help="Move SL to Breakeven at R:R multiple (0 to disable)")
    parser.add_argument("--max-bars", type=int, default=50, help="Max holding bars timeout")
    parser.add_argument("--intraday-cutoff", type=str, default="15:15", help="Intraday cutoff time HH:MM (or 'none')")
    parser.add_argument("--export-dir", type=str, default="reports/backtests", help="Output directory for reports")

    args = parser.parse_args()

    f_date = date.fromisoformat(args.from_date) if args.from_date else None
    t_date = date.fromisoformat(args.to_date) if args.to_date else None

    # Configurations
    cfg = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        from_date=f_date,
        to_date=t_date,
        initial_capital=args.capital,
    )

    sl_cfg = StopLossConfig(
        sl_type=StopLossType(args.sl_type),
        value=args.sl_val,
        breakeven_at_rr=args.be_rr if args.be_rr > 0 else None,
    )

    tp_cfg = TargetConfig(
        target_type=TargetType(args.tp_type),
        value=args.tp_val,
    )

    cutoff_time = None
    if args.intraday_cutoff.lower() != "none":
        h, m = map(int, args.intraday_cutoff.split(":"))
        from datetime import time as dtime
        cutoff_time = dtime(h, m)

    dur_cfg = DurationConfig(
        max_bars=args.max_bars if args.max_bars > 0 else None,
        intraday_cutoff_time=cutoff_time,
    )

    print(f"\n🚀 Starting Modular Backtest for {args.symbol} ({args.timeframe}m)...")
    print(f"   Strategy: {args.strategy} | SL: {args.sl_type} ({args.sl_val}) | TP: {args.tp_type} ({args.tp_val})")
    print(f"   Duration: Max {args.max_bars} bars | Intraday Cutoff: {args.intraday_cutoff}\n")

    backtester = ModularBacktester(
        config=cfg,
        sl_config=sl_cfg,
        target_config=tp_cfg,
        duration_config=dur_cfg,
    )

    strategy_fn = STRATEGIES[args.strategy]
    result = backtester.run(entry_signal_fn=strategy_fn)

    # Print summary to terminal
    print(result.summary_terminal())

    # Export Reports
    out_dir = Path(args.export_dir)
    trades_p, eq_p = BacktestReporter.export_csv(result, out_dir)
    md_content = result.summary_markdown(title=f"{args.symbol} {args.strategy.upper()} Backtest")
    md_p = out_dir / f"report_{args.symbol.replace(':', '_')}_{args.timeframe}.md"
    with open(md_p, "w") as f:
        f.write(md_content)

    print(f"\n💾 Reports Exported Successfully:")
    print(f"   • Markdown Report : {md_p}")
    print(f"   • Trades Ledger   : {trades_p}")
    print(f"   • Equity Curve    : {eq_p}\n")


if __name__ == "__main__":
    main()
