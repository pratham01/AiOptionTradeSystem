from __future__ import annotations

from datetime import date, timedelta, time as dt_time
import pandas as pd


def aggregate_ticks_to_bars(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed")
    symbol = frame["symbol"].iloc[0]
    grouped = (
        frame.set_index("timestamp")
        .resample(f"{timeframe_minutes}min")
        .agg(
            open=("ltp", "first"),
            high=("ltp", "max"),
            low=("ltp", "min"),
            close=("ltp", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
        .reset_index()
    )
    grouped["symbol"] = symbol
    return grouped


def create_minute_bar(symbol: str, ticks: list[dict], previous_cumulative_volume: float | None = None) -> pd.Series | None:
    if not ticks:
        return None
    sorted_ticks = sorted(ticks, key=lambda item: item["timestamp"])
    prices = [tick["ltp"] for tick in sorted_ticks]
    volume_values = [float(tick.get("volume", 0.0) or 0.0) for tick in sorted_ticks]
    last_cumulative_volume = volume_values[-1] if volume_values else 0.0
    first_cumulative_volume = volume_values[0] if volume_values else 0.0
    if last_cumulative_volume > 0:
        if previous_cumulative_volume is not None and last_cumulative_volume >= previous_cumulative_volume:
            minute_volume = max(last_cumulative_volume - previous_cumulative_volume, 0.0)
        else:
            minute_volume = max(last_cumulative_volume - first_cumulative_volume, 0.0)
    else:
        minute_volume = 0.0
    timestamp = sorted_ticks[0]["timestamp"].replace(second=0, microsecond=0)
    return pd.Series(
        {
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": minute_volume,
            "symbol": symbol,
        },
        name=timestamp,
    )


def resample_to_timeframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "symbol"])
    resampled = (
        df.resample(f"{timeframe_minutes}min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )
    return resampled


def _is_market_timestamp(ts: date | pd.Timestamp, market_start: dt_time, market_end: dt_time) -> bool:
    return market_start <= pd.Timestamp(ts).time() < market_end


def _sanitize_intraday_minutes(df: pd.DataFrame, market_start: dt_time, market_end: dt_time) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy().sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    valid_mask = [(market_start <= ts.time() < market_end) for ts in frame.index]
    return frame.loc[valid_mask]


def _latest_session_minute(df: pd.DataFrame, current_date: date) -> pd.Timestamp | None:
    if df.empty:
        return None
    today = df[df.index.date == current_date]
    if today.empty:
        return None
    return pd.Timestamp(today.index.max())


def _completed_timeframe_bars(
    timeframe_df: pd.DataFrame,
    timeframe_minutes: int,
    latest_minute: pd.Timestamp | None,
) -> pd.DataFrame:
    if timeframe_df.empty or latest_minute is None:
        return timeframe_df.iloc[0:0]
    complete_cutoff = latest_minute - timedelta(minutes=timeframe_minutes - 1)
    return timeframe_df[timeframe_df.index <= complete_cutoff]


def get_gap_adjusted_data(all_data: pd.DataFrame, supertrend_period: int, current_date: date) -> pd.DataFrame:
    if all_data.empty:
        return all_data
    today_candles = all_data[all_data.index.date == current_date].copy()
    prev_candles = all_data[all_data.index.date < current_date].copy()
    if today_candles.empty:
        return all_data
    if prev_candles.empty:
        return today_candles

    today_open = float(today_candles.iloc[0]["open"])
    warmup_data = prev_candles.tail(max(supertrend_period * 3, 90)).copy()
    last_idx = warmup_data.index[-1]
    warmup_data.at[last_idx, "close"] = today_open
    warmup_data.at[last_idx, "high"] = max(float(warmup_data.at[last_idx, "high"]), today_open)
    warmup_data.at[last_idx, "low"] = min(float(warmup_data.at[last_idx, "low"]), today_open)
    adjusted = pd.concat([warmup_data, today_candles]).sort_index()
    return adjusted[~adjusted.index.duplicated(keep="last")]


def detect_smc_signal(timeframe: pd.DataFrame) -> dict[str, object] | None:
    if timeframe.empty or len(timeframe) < 8:
        return None

    df = timeframe.copy().sort_index()
    df["range"] = df["high"] - df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["rolling_low_6"] = df["low"].shift(1).rolling(6, min_periods=3).min()
    df["rolling_high_6"] = df["high"].shift(1).rolling(6, min_periods=3).max()
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev2_high"] = df["high"].shift(2)
    df["prev2_low"] = df["low"].shift(2)
    df["bull_fvg"] = df["prev2_high"].notna() & ((df["low"] - df["prev2_high"]) >= (df["atr14"] * 0.2))
    df["bear_fvg"] = df["prev2_low"].notna() & ((df["prev2_low"] - df["high"]) >= (df["atr14"] * 0.2))
    df["bull_fvg_recent"] = df["bull_fvg"].shift(1).rolling(3, min_periods=1).max().fillna(0).astype(bool)
    df["bear_fvg_recent"] = df["bear_fvg"].shift(1).rolling(3, min_periods=1).max().fillna(0).astype(bool)
    df["bull_liquidity_sweep"] = (
        df["rolling_low_6"].notna()
        & (df["low"] < df["rolling_low_6"])
        & (df["close"] > df["rolling_low_6"])
        & (df["close"] > df["open"])
    )
    df["bear_liquidity_sweep"] = (
        df["rolling_high_6"].notna()
        & (df["high"] > df["rolling_high_6"])
        & (df["close"] < df["rolling_high_6"])
        & (df["close"] < df["open"])
    )
    df["bull_structure_break"] = df["prev_high"].notna() & (df["close"] > df["prev_high"])
    df["bear_structure_break"] = df["prev_low"].notna() & (df["close"] < df["prev_low"])

    row = df.iloc[-1]
    bar_time = df.index[-1]
    if bool(row["bull_liquidity_sweep"]) and (bool(row["bull_fvg_recent"]) or bool(row["bull_structure_break"])) and float(
        row["close"]
    ) > float(row["ema20"]):
        return {
            "bar_time": bar_time,
            "direction": 1,
            "signal_name": "BULLISH_SWEEP_RECLAIM",
            "close": float(row["close"]),
            "reason": "Sell-side liquidity sweep with bullish reclaim and FVG/structure support.",
        }
    if bool(row["bear_liquidity_sweep"]) and (bool(row["bear_fvg_recent"]) or bool(row["bear_structure_break"])) and float(
        row["close"]
    ) < float(row["ema20"]):
        return {
            "bar_time": bar_time,
            "direction": -1,
            "signal_name": "BEARISH_SWEEP_REJECT",
            "close": float(row["close"]),
            "reason": "Buy-side liquidity sweep with bearish rejection and FVG/structure support.",
        }
    return None


def valid_supertrend_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "supertrend" not in df.columns:
        return df.iloc[0:0]
    return df[df["supertrend"] > 0]
