"""
Institutional Advanced Strategies Suite (SMC, FVG, Order Blocks, Wyckoff AMD).

Pluggable strategy rules for Modular Backtester and Backtest Studio.
"""
from __future__ import annotations

import pandas as pd
from trade_system.domains.analysis.application.backtesting.modular_backtester import TradeSide


def strategy_supertrend_ema(df: pd.DataFrame, i: int) -> TradeSide | None:
    """Buy when Close > 200 EMA and SuperTrend flips Bullish."""
    if i < 2:
        return None
    curr_st = df["supertrend_dir"].iloc[i]
    prev_st = df["supertrend_dir"].iloc[i - 1]
    close = df["close"].iloc[i]
    ema_200 = df["ema_200"].iloc[i]

    if curr_st == 1 and prev_st == -1 and close > ema_200:
        return TradeSide.LONG
    elif curr_st == -1 and prev_st == 1 and close < ema_200:
        return TradeSide.SHORT
    return None


def strategy_smc_liquidity_sweep(df: pd.DataFrame, i: int) -> TradeSide | None:
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


def strategy_ict_fvg_mitigation(df: pd.DataFrame, i: int) -> TradeSide | None:
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


def strategy_order_block_retest(df: pd.DataFrame, i: int) -> TradeSide | None:
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


def strategy_wyckoff_amd(df: pd.DataFrame, i: int) -> TradeSide | None:
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


def strategy_rsi_bb_reversion(df: pd.DataFrame, i: int) -> TradeSide | None:
    """Mean Reversion: Buy when RSI < 32 and price touches lower BB."""
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


def strategy_dual_ema_crossover(df: pd.DataFrame, i: int) -> TradeSide | None:
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


def strategy_macd_trend(df: pd.DataFrame, i: int) -> TradeSide | None:
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


from trade_system.domains.analysis.application.backtesting.option_buyer_engine import (
    # Core Option Buyer Strategies
    strategy_option_buyer_momentum_burst,
    strategy_smc_liquidity_sweep_opt,
    strategy_ict_fvg_mitigation_opt,
    strategy_supertrend_sr_opt,
    strategy_wyckoff_amd_opt,
    # World-Class Trader Strategies
    strategy_connors_rsi_mean_reversion,
    strategy_opening_range_breakout,
    strategy_momentum_pinball,
    strategy_vcp_breakout,
    strategy_expiry_day_gamma_edge,
    strategy_elephant_bar_continuation,
)

INSTITUTIONAL_STRATEGIES = {
    # ── World-Class Option Buyer Strategies ──
    "option_buyer_momentum_burst": strategy_option_buyer_momentum_burst,
    "connors_rsi_mean_reversion": strategy_connors_rsi_mean_reversion,
    "opening_range_breakout": strategy_opening_range_breakout,
    "momentum_pinball": strategy_momentum_pinball,
    "vcp_breakout": strategy_vcp_breakout,
    "expiry_day_gamma_edge": strategy_expiry_day_gamma_edge,
    "elephant_bar_continuation": strategy_elephant_bar_continuation,
    "wyckoff_amd_opt": strategy_wyckoff_amd_opt,
    "smc_liquidity_sweep_opt": strategy_smc_liquidity_sweep_opt,
    "ict_fvg_mitigation_opt": strategy_ict_fvg_mitigation_opt,
    "supertrend_sr_opt": strategy_supertrend_sr_opt,
    # ── Core Baseline Strategies ──
    "wyckoff_amd": strategy_wyckoff_amd,
    "supertrend_ema": strategy_supertrend_ema,
    "smc_liquidity_sweep": strategy_smc_liquidity_sweep,
    "ict_fvg_mitigation": strategy_ict_fvg_mitigation,
    "order_block_retest": strategy_order_block_retest,
    "rsi_bb_reversion": strategy_rsi_bb_reversion,
    "dual_ema": strategy_dual_ema_crossover,
    "macd_trend": strategy_macd_trend,
}


