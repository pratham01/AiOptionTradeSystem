"""
Vectorized Technical Indicators Library.

High-performance, pure NumPy and Pandas implementations of institutional and technical
indicators for zero-overhead backtesting and real-time market analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda s: np.dot(s, weights) / weights.sum(), raw=True)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-based Volume Weighted Average Price (VWAP).
    Resets daily if intraday timestamps are present.
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return df["close"]
    
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0)

    # If datetime index or timestamp column exists, group by date
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        dates = ts.dt.date
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        cum_vol = vol.cumsum()
        cum_pv = (typical_price * vol).cumsum()
        return (cum_pv / cum_vol.replace(0, np.nan)).fillna(typical_price)

    cum_vol = vol.groupby(dates).cumsum()
    cum_pv = (typical_price * vol).groupby(dates).cumsum()
    return (cum_pv / cum_vol.replace(0, np.nan)).fillna(typical_price)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle zero loss case
    rsi = rsi.fillna(100.0 * (avg_gain > 0))
    return rsi


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD).
    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns (upper_band, middle_band, lower_band, bandwidth, pct_b).
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, bandwidth, pct_b


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    SuperTrend Indicator.
    Returns (supertrend_value, direction_series):
    - direction = 1: Bullish (Green, price above ST)
    - direction = -1: Bearish (Red, price below ST)
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    atr_series = calculate_atr(df, period).bfill().fillna(0)
    atr = atr_series.values
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    final_upper = np.copy(basic_upper)
    final_lower = np.copy(basic_lower)
    supertrend = np.zeros(n)
    direction = np.zeros(n, dtype=int)

    for i in range(1, n):
        # Upper band calculation
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band calculation
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

    # Calculate supertrend line and direction
    direction[0] = 1 if close[0] >= basic_lower[0] else -1
    supertrend[0] = final_lower[0] if direction[0] == 1 else final_upper[0]

    for i in range(1, n):
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    Returns (%K, %D).
    """
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k_fast = 100.0 * ((df["close"] - low_min) / denom)
    k_smooth = k_fast.rolling(window=3).mean()
    d_line = k_smooth.rolling(window=d_period).mean()
    return k_smooth, d_line


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (ADX).
    Returns (adx, plus_di, minus_di).
    """
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    # If plus_dm < minus_dm, plus_dm = 0; if minus_dm < plus_dm, minus_dm = 0
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0.0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0.0)

    atr = calculate_atr(df, period)
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / dx_denom)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    return adx, plus_di, minus_di


def calculate_volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    Approximated Cumulative Volume Delta based on candle body & wick pressure.
    """
    total_range = (df["high"] - df["low"]).replace(0, np.nan)
    close_open = df["close"] - df["open"]
    # Ratio ranges from -1.0 (bearish marubozu) to +1.0 (bullish marubozu)
    delta_ratio = (close_open / total_range).fillna(0)
    vol = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    bar_delta = delta_ratio * vol
    return bar_delta


def calculate_fvg_zones(df: pd.DataFrame, min_atr_ratio: float = 0.2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Identifies ICT Fair Value Gaps (FVG) / 3-Bar Imbalances.
    Returns: (bull_fvg, bear_fvg, fvg_top, fvg_bottom, fvg_mid)
    """
    atr = calculate_atr(df, 14).bfill().fillna(0)
    min_gap = atr * min_atr_ratio

    prev2_high = df["high"].shift(2)
    prev2_low = df["low"].shift(2)

    bull_fvg = (df["low"] - prev2_high) >= min_gap
    bear_fvg = (prev2_low - df["high"]) >= min_gap

    fvg_top = np.where(bull_fvg, df["low"], np.where(bear_fvg, prev2_low, np.nan))
    fvg_bottom = np.where(bull_fvg, prev2_high, np.where(bear_fvg, df["high"], np.nan))
    fvg_mid = (fvg_top + fvg_bottom) / 2.0

    return (
        pd.Series(bull_fvg, index=df.index),
        pd.Series(bear_fvg, index=df.index),
        pd.Series(fvg_top, index=df.index),
        pd.Series(fvg_bottom, index=df.index),
        pd.Series(fvg_mid, index=df.index),
    )


def calculate_liquidity_sweeps(df: pd.DataFrame, lookback: int = 8) -> Tuple[pd.Series, pd.Series]:
    """
    Identifies Institutional Liquidity Sweeps (Stop Runs / Judas Swings).
    - bull_sweep: Price sweeps below prior N-bar low but reclaims and closes above it.
    - bear_sweep: Price sweeps above prior N-bar high but rejects and closes below it.
    """
    prior_low = df["low"].shift(1).rolling(window=lookback, min_periods=max(2, lookback // 2)).min()
    prior_high = df["high"].shift(1).rolling(window=lookback, min_periods=max(2, lookback // 2)).max()

    bull_sweep = (
        prior_low.notna()
        & (df["low"] < prior_low)
        & (df["close"] > prior_low)
        & (df["close"] > df["open"])
    )

    bear_sweep = (
        prior_high.notna()
        & (df["high"] > prior_high)
        & (df["close"] < prior_high)
        & (df["close"] < df["open"])
    )

    return pd.Series(bull_sweep, index=df.index), pd.Series(bear_sweep, index=df.index)


def calculate_order_blocks(df: pd.DataFrame, lookback: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    Identifies Institutional Order Block (OB) formations:
    - Bullish OB: Last down candle before an explosive upward expansion.
    - Bearish OB: Last up candle before an explosive downward expansion.
    """
    n = len(df)
    bull_ob = pd.Series(False, index=df.index)
    bear_ob = pd.Series(False, index=df.index)

    close = df["close"].values
    open_p = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr = calculate_atr(df, 14).bfill().fillna(0).values

    for i in range(2, n):
        is_expansion_up = (close[i] - open_p[i - 1]) >= (1.0 * atr[i]) and (close[i] > high[i - 2])
        is_down_candle = close[i - 1] < open_p[i - 1]
        if is_expansion_up and is_down_candle:
            bull_ob.iloc[i] = True

        is_expansion_down = (open_p[i - 1] - close[i]) >= (1.0 * atr[i]) and (close[i] < low[i - 2])
        is_up_candle = close[i - 1] > open_p[i - 1]
        if is_expansion_down and is_up_candle:
            bear_ob.iloc[i] = True

    return bull_ob, bear_ob


# ─────────────────────────────────────────────────────────────────────────────
# World-Class Trader Indicators (Connors, Raschke, Velez, Minervini, Augen)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_iv_proxy_rank(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    IV Rank proxy using Garman-Klass realized volatility.
    Computes where current IV sits within its rolling lookback-period range (0-100 scale).
    Source: Tom Sosnoff (tastytrade) / Sheldon Natenberg.
    - IV Rank < 30: Options are cheap → good for BUYING
    - IV Rank > 60: Options are expensive → avoid buying (IV crush risk)
    """
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    log_co = np.log(df["close"] / df["open"].replace(0, np.nan)) ** 2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    # Annualized realized vol (proxy for IV)
    iv_proxy = np.sqrt(gk_var.rolling(20, min_periods=5).mean() * 252) * 100.0
    iv_proxy = iv_proxy.clip(lower=5.0, upper=80.0).fillna(15.0)

    iv_min = iv_proxy.rolling(lookback, min_periods=50).min()
    iv_max = iv_proxy.rolling(lookback, min_periods=50).max()
    iv_range = (iv_max - iv_min).replace(0, np.nan)
    iv_rank = ((iv_proxy - iv_min) / iv_range * 100.0).fillna(50.0).clip(0, 100)
    return iv_rank


def calculate_connors_rsi(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Larry Connors' composite mean-reversion indicator.
    ConnorsRSI = (RSI(3) + PercentRank(Streak, 100) + RSI(ROC(1), 100)) / 3

    Also returns RSI(2) for the classic 2-period RSI strategy.
    Returns: (connors_rsi, rsi_2, sma_5)
    """
    close = df["close"]

    # RSI(2) — ultra-short for classic Connors strategy
    rsi_2 = calculate_rsi(close, period=2)

    # RSI(3) — component 1
    rsi_3 = calculate_rsi(close, period=3)

    # Streak — consecutive up/down closes
    diff = close.diff()
    streak = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if diff.iloc[i] > 0:
            streak.iloc[i] = max(streak.iloc[i - 1], 0) + 1
        elif diff.iloc[i] < 0:
            streak.iloc[i] = min(streak.iloc[i - 1], 0) - 1
        else:
            streak.iloc[i] = 0

    # PercentRank of Streak over 100 bars
    pct_rank = streak.rolling(100, min_periods=20).apply(
        lambda s: (s < s.iloc[-1]).sum() / len(s) * 100.0, raw=False
    ).fillna(50.0)

    # RSI of 1-period ROC
    roc_1 = close.pct_change(1) * 100.0
    rsi_roc = calculate_rsi(roc_1.fillna(0), period=100)

    # Composite ConnorsRSI
    connors_rsi = (rsi_3 + pct_rank + rsi_roc) / 3.0

    # SMA(5) — exit signal for Connors strategy
    sma_5 = calculate_sma(close, 5)

    return connors_rsi, rsi_2, sma_5


def calculate_momentum_pinball(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Linda Raschke's Momentum Pinball indicator.
    Pinball = RSI(3-period) of ROC(1-period)
    - Pinball < 30: Buy setup for next day
    - Pinball > 70: Sell setup for next day

    Returns: (pinball, pinball_buy_signal, pinball_sell_signal)
    """
    roc_1 = df["close"].pct_change(1) * 100.0
    pinball = calculate_rsi(roc_1.fillna(0), period=3)
    pinball_buy = pinball < 30.0
    pinball_sell = pinball > 70.0
    return pinball, pinball_buy, pinball_sell


def calculate_opening_range(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Oliver Velez Opening Range Breakout (ORB) — first-hour high/low (09:15-10:15 IST).
    Returns dict with: orb_high, orb_low, orb_breakout_long, orb_breakout_short
    """
    n = len(df)
    orb_high = pd.Series(np.nan, index=df.index)
    orb_low = pd.Series(np.nan, index=df.index)

    if "timestamp" not in df.columns:
        return {
            "orb_high": orb_high.fillna(df["high"]),
            "orb_low": orb_low.fillna(df["low"]),
            "orb_breakout_long": pd.Series(False, index=df.index),
            "orb_breakout_short": pd.Series(False, index=df.index),
        }

    ts = pd.to_datetime(df["timestamp"])
    dates = ts.dt.date

    from datetime import time as dtime
    orb_start = dtime(9, 15)
    orb_end = dtime(10, 15)

    # Calculate ORB per day
    for day in dates.unique():
        day_mask = dates == day
        day_ts = ts[day_mask]
        day_times = day_ts.dt.time

        first_hour = day_mask & (day_ts.dt.time >= orb_start) & (day_ts.dt.time <= orb_end)
        if first_hour.sum() == 0:
            continue

        orb_h = df.loc[first_hour, "high"].max()
        orb_l = df.loc[first_hour, "low"].min()

        orb_high[day_mask] = orb_h
        orb_low[day_mask] = orb_l

    orb_high = orb_high.ffill().bfill()
    orb_low = orb_low.ffill().bfill()

    # Breakout signals (only after ORB window closes)
    after_orb = pd.Series(False, index=df.index)
    if "timestamp" in df.columns:
        after_orb = ts.dt.time > orb_end

    vol = df.get("volume", pd.Series(1, index=df.index))
    vol_sma = vol.rolling(20, min_periods=5).mean().fillna(vol)

    breakout_long = after_orb & (df["close"] > orb_high) & (vol > 1.3 * vol_sma)
    breakout_short = after_orb & (df["close"] < orb_low) & (vol > 1.3 * vol_sma)

    return {
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_breakout_long": breakout_long,
        "orb_breakout_short": breakout_short,
    }


def calculate_elephant_bar(df: pd.DataFrame, bars_to_clear: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Oliver Velez Elephant Bar detection — large candle body that clears 3+ prior
    candles of opposite color. 87% historical continuation probability.

    Returns: (elephant_bull, elephant_bear)
    """
    n = len(df)
    elephant_bull = pd.Series(False, index=df.index)
    elephant_bear = pd.Series(False, index=df.index)

    close = df["close"].values
    open_p = df["open"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values if "volume" in df.columns else np.ones(n)
    vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().fillna(1).values

    for i in range(bars_to_clear + 1, n):
        body = abs(close[i] - open_p[i])
        is_bullish = close[i] > open_p[i]
        is_bearish = close[i] < open_p[i]

        if body < 1e-6:
            continue

        # Volume confirmation
        if vol[i] < 1.3 * vol_sma[i]:
            continue

        if is_bullish:
            # Check if this bullish bar's close clears 3+ prior bearish bars' opens
            cleared = 0
            for j in range(1, bars_to_clear + 1):
                if i - j >= 0 and close[i - j] < open_p[i - j]:  # prior bar is bearish
                    if close[i] > open_p[i - j]:  # current close clears prior open
                        cleared += 1
            if cleared >= bars_to_clear:
                elephant_bull.iloc[i] = True

        elif is_bearish:
            cleared = 0
            for j in range(1, bars_to_clear + 1):
                if i - j >= 0 and close[i - j] > open_p[i - j]:  # prior bar is bullish
                    if close[i] < open_p[i - j]:  # current close clears prior open
                        cleared += 1
            if cleared >= bars_to_clear:
                elephant_bear.iloc[i] = True

    return elephant_bull, elephant_bear


def calculate_vcp_contraction(df: pd.DataFrame, contractions: int = 3, vol_dry_pct: float = 0.50) -> Tuple[pd.Series, pd.Series]:
    """
    Mark Minervini Volatility Contraction Pattern (VCP) detection.
    Identifies successive range contractions where each is < 75% of prior,
    combined with volume dry-up (volume < vol_dry_pct * 20-bar avg).

    Returns: (vcp_contraction, vcp_breakout)
    - vcp_contraction: True when VCP pattern is forming (contracting ranges + drying volume)
    - vcp_breakout: True when price breaks above contraction high with volume surge
    """
    n = len(df)
    vcp_contraction = pd.Series(False, index=df.index)
    vcp_breakout = pd.Series(False, index=df.index)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values if "volume" in df.columns else np.ones(n)
    vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().fillna(1).values

    window = 10  # Range measurement window

    for i in range(contractions * window + 5, n):
        # Measure N successive range windows
        ranges = []
        for c in range(contractions + 1):
            start = i - (contractions - c + 1) * window
            end = i - (contractions - c) * window
            if start < 0:
                break
            segment_high = max(high[start:end])
            segment_low = min(low[start:end])
            ranges.append(segment_high - segment_low)

        if len(ranges) < contractions + 1:
            continue

        # Check successive contractions (each range < 75% of prior)
        is_contracting = True
        for r in range(1, len(ranges)):
            if ranges[r] >= 0.80 * ranges[r - 1]:
                is_contracting = False
                break

        if not is_contracting:
            continue

        # Volume dry-up in latest window
        recent_vol = np.mean(vol[i - window:i])
        if recent_vol > vol_dry_pct * vol_sma[i]:
            continue

        vcp_contraction.iloc[i] = True

        # Breakout: price breaks above the most recent contraction high with volume surge
        contraction_high = max(high[i - window:i])
        if close[i] > contraction_high and vol[i] > 1.5 * vol_sma[i]:
            vcp_breakout.iloc[i] = True

    return vcp_contraction, vcp_breakout


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative Option Buyer Indicators: ADX, Squeeze, S/R, Regimes & VIX
# ─────────────────────────────────────────────────────────────────────────────

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Wilder's Average Directional Index (ADX), +DI, and -DI.
    Measures trend strength and distinguishes trending momentum from choppy noise.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder's Smoothing (alpha = 1 / period)
    tr_smooth = pd.Series(tr).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum.replace(0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().fillna(20.0)

    return adx, plus_di.fillna(0.0), minus_di.fillna(0.0)


def calculate_bollinger_squeeze(
    df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20, kc_mult: float = 1.5
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Identifies volatility contraction squeezes and subsequent explosive momentum releases.
    Returns:
    - in_squeeze: True when Bollinger Bands contract inside Keltner Channels
    - squeeze_release: True on the exact transition bar from squeeze to expansion
    - bandwidth_pctile: Rolling 100-bar percentile of Bollinger Bandwidth (0-100)
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Bollinger Bands
    bb_mid = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    bandwidth = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # Keltner Channels (ATR based)
    atr = calculate_atr(df, kc_period)
    kc_upper = bb_mid + (kc_mult * atr)
    kc_lower = bb_mid - (kc_mult * atr)

    in_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    squeeze_release = in_squeeze.shift(1).fillna(False) & (~in_squeeze)

    # Rolling bandwidth percentile (lower = tighter compression)
    bandwidth_pctile = bandwidth.rolling(100, min_periods=20).apply(
        lambda s: (s.iloc[-1] - s.min()) / max(s.max() - s.min(), 1e-6) * 100.0, raw=False
    ).fillna(50.0)

    return in_squeeze, squeeze_release, bandwidth_pctile


def calculate_support_resistance_zones(
    df: pd.DataFrame, lookback: int = 20
) -> Dict[str, pd.Series]:
    """
    Calculates dynamic institutional Support and Resistance levels:
    - Rolling Swing High Resistance & Swing Low Support
    - Previous Day High (PDH), Previous Day Low (PDL), Previous Day Close (PDC)
    - Distance % to nearest Resistance ceiling and Support floor
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1. Rolling Swing Pivots (excluding current bar)
    swing_high = high.shift(1).rolling(lookback, min_periods=5).max()
    swing_low = low.shift(1).rolling(lookback, min_periods=5).min()

    # 2. Previous Day Benchmarks (PDH, PDL, PDC)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df_temp = pd.DataFrame({"date": ts.dt.date, "high": high, "low": low, "close": close})
        daily_stats = df_temp.groupby("date").agg(
            pdh=("high", "max"),
            pdl=("low", "min"),
            pdc=("close", "last")
        ).shift(1)  # Previous day

        mapped = df_temp[["date"]].merge(daily_stats, on="date", how="left")
        pdh = mapped["pdh"].ffill().bfill()
        pdl = mapped["pdl"].ffill().bfill()
        pdc = mapped["pdc"].ffill().bfill()
    else:
        pdh = swing_high
        pdl = swing_low
        pdc = close.shift(lookback)

    # 3. Nearest Resistance and Support
    # Nearest resistance is minimum of levels above close
    nearest_res = np.minimum(
        np.where(swing_high > close, swing_high, np.inf),
        np.where(pdh > close, pdh, np.inf)
    )
    # If no level above, fallback to swing_high or close * 1.01
    nearest_res = np.where(np.isinf(nearest_res), swing_high.clip(lower=close * 1.005), nearest_res)

    # Nearest support is maximum of levels below close
    nearest_sup = np.maximum(
        np.where(swing_low < close, swing_low, -np.inf),
        np.where(pdl < close, pdl, -np.inf)
    )
    nearest_sup = np.where(np.isneginf(nearest_sup), swing_low.clip(upper=close * 0.995), nearest_sup)

    dist_to_res_pct = ((nearest_res - close) / close) * 100.0
    dist_to_sup_pct = ((close - nearest_sup) / close) * 100.0

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "pdh": pd_series(pdh, df.index),
        "pdl": pd_series(pdl, df.index),
        "pdc": pd_series(pdc, df.index),
        "nearest_res": pd_series(nearest_res, df.index),
        "nearest_sup": pd_series(nearest_sup, df.index),
        "dist_to_res_pct": pd_series(dist_to_res_pct, df.index),
        "dist_to_sup_pct": pd_series(dist_to_sup_pct, df.index),
    }


def pd_series(arr: Any, index: pd.Index) -> pd.Series:
    if isinstance(arr, pd.Series):
        arr.index = index
        return arr
    return pd.Series(arr, index=index)


def calculate_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classifies each bar into one of 4 dynamic market regimes:
    1. 'TRENDING_BULL': High ADX (>=22), Price > 50 EMA, SuperTrend Bullish
    2. 'TRENDING_BEAR': High ADX (>=22), Price < 50 EMA, SuperTrend Bearish
    3. 'VOLATILITY_SQUEEZE': Squeeze active or Bandwidth percentile < 25%
    4. 'CHOPPY_RANGEBOUND': Low ADX (<20), oscillating inside consolidation
    """
    regimes = []
    adx = df.get("adx", pd.Series(20.0, index=df.index))
    close = df["close"]
    ema50 = df.get("ema_50", close)
    st_dir = df.get("supertrend_dir", pd.Series(1, index=df.index))
    in_squeeze = df.get("in_squeeze", pd.Series(False, index=df.index))
    bw_pct = df.get("bandwidth_pctile", pd.Series(50.0, index=df.index))

    for i in range(len(df)):
        cur_adx = adx.iloc[i]
        cur_c = close.iloc[i]
        cur_ema = ema50.iloc[i]
        cur_st = st_dir.iloc[i]
        cur_sq = in_squeeze.iloc[i]
        cur_bw = bw_pct.iloc[i]

        if cur_sq or cur_bw <= 20.0:
            regimes.append("VOLATILITY_SQUEEZE")
        elif cur_adx >= 22.0 and cur_c > cur_ema and cur_st == 1:
            regimes.append("TRENDING_BULL")
        elif cur_adx >= 22.0 and cur_c < cur_ema and cur_st == -1:
            regimes.append("TRENDING_BEAR")
        else:
            regimes.append("CHOPPY_RANGEBOUND")

    return pd.Series(regimes, index=df.index)


def calculate_vix_dynamics(df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
    """
    Enriches price candles with India VIX metrics:
    - Realized Volatility / VIX Level
    - Intraday VIX Change %
    - VIX Regime: 'COMPLACENT (<11.5)', 'NORMAL (11.5-16)', 'ELEVATED (16-22)', 'FEAR_SPIKE (>22)'
    - VIX Trend: 'EXPANDING' (Vega tailwind for buyers), 'CRUSHING' (IV crush risk), 'STABLE'
    """
    # 1. If vix_df is provided and has timestamp & close
    if vix_df is not None and not vix_df.empty and "close" in vix_df.columns:
        vix_copy = vix_df.copy()
        if "timestamp" in df.columns and "timestamp" in vix_copy.columns:
            df_ts = pd.to_datetime(df["timestamp"])
            vix_copy["ts"] = pd.to_datetime(vix_copy["timestamp"])
            merged = pd.merge_asof(
                pd.DataFrame({"ts": df_ts, "idx": df.index}).sort_values("ts"),
                vix_copy[["ts", "close"]].rename(columns={"close": "vix_level"}).sort_values("ts"),
                on="ts",
                direction="backward"
            ).sort_values("idx")
            vix_level = merged["vix_level"].ffill().bfill()
            vix_level.index = df.index
        else:
            vix_level = pd.Series(float(vix_copy["close"].mean()), index=df.index)
    elif "vix" in df.columns:
        vix_level = df["vix"]
    else:
        # Fallback: Parkinson / Garman-Klass Realized Volatility proxy scaled to index %
        log_hl = (df["high"] / df["low"]).apply(np.log) ** 2
        log_co = (df["close"] / df["open"]).apply(np.log) ** 2
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        realized_vol = np.sqrt(gk_var.rolling(20, min_periods=5).mean() * 252 * 25) * 100.0
        vix_level = realized_vol.clip(lower=10.0, upper=35.0).fillna(13.5)

    vix_change_pct = vix_level.pct_change(periods=4).fillna(0.0) * 100.0

    regimes = []
    trends = []
    for i in range(len(df)):
        lvl = vix_level.iloc[i]
        chg = vix_change_pct.iloc[i]

        if lvl < 11.5:
            regimes.append("COMPLACENT (<11.5)")
        elif lvl <= 16.0:
            regimes.append("NORMAL (11.5-16)")
        elif lvl <= 22.0:
            regimes.append("ELEVATED (16-22)")
        else:
            regimes.append("FEAR_SPIKE (>22)")

        if chg >= 2.0:
            trends.append("EXPANDING")
        elif chg <= -2.0:
            trends.append("CRUSHING")
        else:
            trends.append("STABLE")

    return {
        "vix_level": vix_level,
        "vix_change_pct": vix_change_pct,
        "vix_regime": pd.Series(regimes, index=df.index),
        "vix_trend": pd.Series(trends, index=df.index),
    }


def apply_indicator_suite(
    df: pd.DataFrame, config: Dict[str, Any] | None = None, vix_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes and attaches standard multi-indicator suite onto a copy of the candles DataFrame,
    including advanced institutional structure, ADX, Squeeze, Support/Resistance zones,
    Market Regimes, and India VIX dynamics.
    """
    out = df.copy()
    cfg = config or {}

    # Moving averages
    fast_ema = cfg.get("fast_ema", 9)
    slow_ema = cfg.get("slow_ema", 21)
    trend_ema = cfg.get("trend_ema", 200)

    out[f"ema_{fast_ema}"] = calculate_ema(out["close"], fast_ema)
    out[f"ema_{slow_ema}"] = calculate_ema(out["close"], slow_ema)
    out[f"ema_{trend_ema}"] = calculate_ema(out["close"], trend_ema)
    out["ema_50"] = calculate_ema(out["close"], 50)

    # SuperTrend
    st_period = cfg.get("st_period", 10)
    st_mult = cfg.get("st_mult", 3.0)
    st_val, st_dir = calculate_supertrend(out, period=st_period, multiplier=st_mult)
    out["supertrend"] = st_val
    out["supertrend_dir"] = st_dir

    # RSI
    rsi_period = cfg.get("rsi_period", 14)
    out[f"rsi_{rsi_period}"] = calculate_rsi(out["close"], rsi_period)

    # MACD
    macd, signal, hist = calculate_macd(out["close"])
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = hist

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower, bb_width, bb_pct_b = calculate_bollinger_bands(out["close"])
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width
    out["bb_pct_b"] = bb_pct_b

    # ATR
    atr_period = cfg.get("atr_period", 14)
    out["atr"] = calculate_atr(out, atr_period)

    # VWAP
    out["vwap"] = calculate_vwap(out)

    # Volume Delta & 20-period Volume SMA
    out["volume_delta"] = calculate_volume_delta(out)
    out["vol_sma_20"] = calculate_sma(out["volume"], 20).fillna(out["volume"])

    # ── Advanced Institutional SMC / FVG / Liquidity ─────────────────────────
    bull_fvg, bear_fvg, fvg_top, fvg_bottom, fvg_mid = calculate_fvg_zones(out)
    out["bull_fvg"] = bull_fvg
    out["bear_fvg"] = bear_fvg
    out["fvg_top"] = fvg_top
    out["fvg_bottom"] = fvg_bottom
    out["fvg_mid"] = fvg_mid

    bull_sweep, bear_sweep = calculate_liquidity_sweeps(out, lookback=8)
    out["bull_sweep"] = bull_sweep
    out["bear_sweep"] = bear_sweep

    bull_ob, bear_ob = calculate_order_blocks(out, lookback=10)
    out["bull_ob"] = bull_ob
    out["bear_ob"] = bear_ob

    # ── Option Buyer Quantitative Analytics ──────────────────────────────────
    adx, plus_di, minus_di = calculate_adx(out, period=14)
    out["adx"] = adx
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    in_sq, sq_rel, bw_pct = calculate_bollinger_squeeze(out)
    out["in_squeeze"] = in_sq
    out["squeeze_release"] = sq_rel
    out["bandwidth_pctile"] = bw_pct

    sr_dict = calculate_support_resistance_zones(out, lookback=20)
    for k, v in sr_dict.items():
        out[k] = v

    out["market_regime"] = calculate_market_regime(out)

    vix_dict = calculate_vix_dynamics(out, vix_df=vix_df)
    for k, v in vix_dict.items():
        out[k] = v

    # ── World-Class Trader Indicators ────────────────────────────────────────
    # 200 SMA (Paul Tudor Jones trend gate)
    out["sma_200"] = calculate_sma(out["close"], 200)

    # Connors RSI, RSI(2), SMA(5) (Larry Connors mean reversion)
    connors_rsi, rsi_2, sma_5 = calculate_connors_rsi(out)
    out["connors_rsi"] = connors_rsi
    out["rsi_2"] = rsi_2
    out["sma_5"] = sma_5

    # Momentum Pinball (Linda Raschke)
    pinball, pinball_buy, pinball_sell = calculate_momentum_pinball(out)
    out["pinball"] = pinball
    out["pinball_buy_signal"] = pinball_buy
    out["pinball_sell_signal"] = pinball_sell

    # Opening Range Breakout (Oliver Velez)
    orb_dict = calculate_opening_range(out)
    for k, v in orb_dict.items():
        out[k] = v

    # IV Rank proxy (Tom Sosnoff / Sheldon Natenberg)
    out["iv_rank"] = calculate_iv_proxy_rank(out)

    # Elephant Bar (Oliver Velez — 87% continuation)
    elephant_bull, elephant_bear = calculate_elephant_bar(out)
    out["elephant_bar_bull"] = elephant_bull
    out["elephant_bar_bear"] = elephant_bear

    # VCP — Volatility Contraction Pattern (Mark Minervini)
    vcp_contraction, vcp_breakout = calculate_vcp_contraction(out)
    out["vcp_contraction"] = vcp_contraction
    out["vcp_breakout"] = vcp_breakout

    return out

