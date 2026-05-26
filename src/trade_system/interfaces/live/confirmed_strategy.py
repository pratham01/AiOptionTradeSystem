from __future__ import annotations

from datetime import time as dt_time

import pandas as pd


def prepare_confirmed_strategy_frame(st_df: pd.DataFrame) -> pd.DataFrame:
    data = st_df.copy()
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [
            (data["high"] - data["low"]).abs(),
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    data["range"] = data["high"] - data["low"]
    return data


def confirmed_entry_payload(today_df: pd.DataFrame) -> dict[str, object] | None:
    if len(today_df) < 2:
        return None
    prev_row = today_df.iloc[-2]
    row = today_df.iloc[-1]
    direction = int(row["supertrend_direction"])
    if direction == int(prev_row["supertrend_direction"]):
        return None
    if direction == 1 and float(row["close"]) <= float(row["ema20"]):
        return None
    if direction == -1 and float(row["close"]) >= float(row["ema20"]):
        return None
    if float(row["range"]) < float(row["atr14"]) * 0.8:
        return None
    return {
        "bar_time": today_df.index[-1],
        "direction": direction,
        "entry_price": float(row["close"]),
        "supertrend": float(row["supertrend"]),
        "ema20": float(row["ema20"]),
    }


def confirmed_exit_payload(position: dict[str, object], row: pd.Series, bar_time: pd.Timestamp) -> dict[str, object] | None:
    direction = int(position["direction"])
    exit_price: float | None = None
    exit_reason: str | None = None
    if direction == 1 and float(row["low"]) <= float(row["supertrend"]):
        exit_price = float(row["supertrend"])
        exit_reason = "SL_HIT"
    elif direction == -1 and float(row["high"]) >= float(row["supertrend"]):
        exit_price = float(row["supertrend"])
        exit_reason = "SL_HIT"
    elif int(row["supertrend_direction"]) != direction:
        exit_price = float(row["close"])
        exit_reason = "TREND_CHANGE"
    elif bar_time.time() >= dt_time(15, 15):
        exit_price = float(row["close"])
        exit_reason = "EOD_SQUARE_OFF"
    if exit_price is None or exit_reason is None:
        return None
    points = (exit_price - float(position["entry_price"])) * direction
    return {
        "bar_time": bar_time,
        "direction": direction,
        "entry_price": float(position["entry_price"]),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "points": points,
    }


def build_confirmed_entry_message(
    symbol: str,
    payload: dict[str, object],
    short_symbol: str,
) -> str:
    direction = int(payload["direction"])
    action = "BUY CALL" if direction == 1 else "BUY PUT"
    color = "🟩" if direction == 1 else "🟥"
    return (
        f"{color} <b>{short_symbol} {pd.Timestamp(payload['bar_time']).strftime('%H:%M')}</b>\n"
        f"15M st_flip_confirmed ENTRY: <b>{'UP' if direction == 1 else 'DOWN'}</b>\n"
        f"Close: ₹{float(payload['entry_price']):.2f}\n"
        f"Supertrend: ₹{float(payload['supertrend']):.2f}\n"
        f"EMA20: ₹{float(payload['ema20']):.2f}\n"
        f"Action: <b>{action}</b>"
    )


def build_confirmed_exit_message(
    symbol: str,
    payload: dict[str, object],
    short_symbol: str,
) -> str:
    direction = int(payload["direction"])
    action = "EXIT CALL" if direction == 1 else "EXIT PUT"
    points = float(payload["points"])
    color = "🟨" if points >= 0 else "⬜"
    return (
        f"{color} <b>{short_symbol} {pd.Timestamp(payload['bar_time']).strftime('%H:%M')}</b>\n"
        f"15M st_flip_confirmed EXIT: <b>{payload['exit_reason']}</b>\n"
        f"Entry: ₹{float(payload['entry_price']):.2f}\n"
        f"Exit: ₹{float(payload['exit_price']):.2f}\n"
        f"Points: <b>{points:.2f}</b>\n"
        f"Action: <b>{action}</b>"
    )
