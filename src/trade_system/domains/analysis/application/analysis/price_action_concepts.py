from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_price_action_concepts(
    frame: pd.DataFrame,
    *,
    internal_lookback: int = 5,
    swing_lookback: int = 12,
    range_window: int = 20,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    data = frame.copy().sort_index()
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
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["range"] = data["high"] - data["low"]
    data["body"] = (data["close"] - data["open"]).abs()
    data["body_ratio"] = np.where(data["range"] > 0, data["body"] / data["range"], 0.0)

    internal_high = data["high"].shift(1).rolling(internal_lookback, min_periods=max(3, internal_lookback // 2)).max()
    internal_low = data["low"].shift(1).rolling(internal_lookback, min_periods=max(3, internal_lookback // 2)).min()
    swing_high = data["high"].shift(1).rolling(swing_lookback, min_periods=max(5, swing_lookback // 2)).max()
    swing_low = data["low"].shift(1).rolling(swing_lookback, min_periods=max(5, swing_lookback // 2)).min()

    data["internal_break_up"] = internal_high.notna() & (data["close"] > internal_high)
    data["internal_break_down"] = internal_low.notna() & (data["close"] < internal_low)
    data["swing_break_up"] = swing_high.notna() & (data["close"] > swing_high)
    data["swing_break_down"] = swing_low.notna() & (data["close"] < swing_low)

    internal_state = []
    swing_state = []
    last_internal = 0
    last_swing = 0
    internal_bos = []
    internal_choch = []
    swing_bos = []
    swing_choch = []

    for row in data.itertuples():
        current_internal = last_internal
        bos = 0
        choch = 0
        if bool(row.internal_break_up):
            current_internal = 1
            if last_internal == -1:
                choch = 1
            elif last_internal in (0, 1):
                bos = 1
        elif bool(row.internal_break_down):
            current_internal = -1
            if last_internal == 1:
                choch = -1
            elif last_internal in (0, -1):
                bos = -1
        internal_state.append(current_internal)
        internal_bos.append(bos)
        internal_choch.append(choch)
        last_internal = current_internal

        current_swing = last_swing
        bos = 0
        choch = 0
        if bool(row.swing_break_up):
            current_swing = 1
            if last_swing == -1:
                choch = 1
            elif last_swing in (0, 1):
                bos = 1
        elif bool(row.swing_break_down):
            current_swing = -1
            if last_swing == 1:
                choch = -1
            elif last_swing in (0, -1):
                bos = -1
        swing_state.append(current_swing)
        swing_bos.append(bos)
        swing_choch.append(choch)
        last_swing = current_swing

    data["internal_structure"] = internal_state
    data["swing_structure"] = swing_state
    data["internal_bos"] = internal_bos
    data["internal_choch"] = internal_choch
    data["swing_bos"] = swing_bos
    data["swing_choch"] = swing_choch
    data["bull_choch_plus"] = (
        (pd.Series(swing_choch, index=data.index) == 1)
        & (data["low"] > data["low"].shift(1).rolling(3, min_periods=2).min())
        & (data["close"] > data["ema20"])
    )
    data["bear_choch_plus"] = (
        (pd.Series(swing_choch, index=data.index) == -1)
        & (data["high"] < data["high"].shift(1).rolling(3, min_periods=2).max())
        & (data["close"] < data["ema20"])
    )

    prev_high = data["high"].shift(1)
    prev_low = data["low"].shift(1)
    prev2_high = data["high"].shift(2)
    prev2_low = data["low"].shift(2)
    rolling_low_6 = data["low"].shift(1).rolling(6, min_periods=3).min()
    rolling_high_6 = data["high"].shift(1).rolling(6, min_periods=3).max()
    data["bull_liquidity_sweep"] = (
        rolling_low_6.notna()
        & (data["low"] < rolling_low_6)
        & (data["close"] > rolling_low_6)
        & (data["close"] > data["open"])
    )
    data["bear_liquidity_sweep"] = (
        rolling_high_6.notna()
        & (data["high"] > rolling_high_6)
        & (data["close"] < rolling_high_6)
        & (data["close"] < data["open"])
    )

    data["bull_fvg"] = prev2_high.notna() & ((data["low"] - prev2_high) >= (data["atr14"] * 0.2))
    data["bear_fvg"] = prev2_low.notna() & ((prev2_low - data["high"]) >= (data["atr14"] * 0.2))
    data["bull_fvg_recent"] = data["bull_fvg"].shift(1).rolling(3, min_periods=1).max().fillna(0).astype(bool)
    data["bear_fvg_recent"] = data["bear_fvg"].shift(1).rolling(3, min_periods=1).max().fillna(0).astype(bool)
    data["bull_structure_break"] = prev_high.notna() & (data["close"] > prev_high)
    data["bear_structure_break"] = prev_low.notna() & (data["close"] < prev_low)

    data["bull_ob_created"] = (
        (data["close"].shift(1) < data["open"].shift(1))
        & (data["internal_break_up"] | data["swing_break_up"])
    )
    data["bear_ob_created"] = (
        (data["close"].shift(1) > data["open"].shift(1))
        & (data["internal_break_down"] | data["swing_break_down"])
    )

    bull_ob_low: list[float] = []
    bull_ob_high: list[float] = []
    bear_ob_low: list[float] = []
    bear_ob_high: list[float] = []
    bull_ob_active: list[bool] = []
    bear_ob_active: list[bool] = []
    bull_ob_retest: list[bool] = []
    bear_ob_retest: list[bool] = []

    current_bull_low = np.nan
    current_bull_high = np.nan
    current_bear_low = np.nan
    current_bear_high = np.nan
    bull_active = False
    bear_active = False

    for row in data.itertuples():
        if bool(row.bull_ob_created):
            current_bull_low = float(row.low)
            current_bull_high = min(float(row.open), float(row.close))
            bull_active = True
        if bool(row.bear_ob_created):
            current_bear_low = max(float(row.open), float(row.close))
            current_bear_high = float(row.high)
            bear_active = True

        current_bull_retest = False
        current_bear_retest = False

        if bull_active:
            if float(row.close) < current_bull_low:
                bull_active = False
            elif float(row.low) <= current_bull_high and float(row.close) >= current_bull_low:
                current_bull_retest = True

        if bear_active:
            if float(row.close) > current_bear_high:
                bear_active = False
            elif float(row.high) >= current_bear_low and float(row.close) <= current_bear_high:
                current_bear_retest = True

        bull_ob_low.append(current_bull_low)
        bull_ob_high.append(current_bull_high)
        bear_ob_low.append(current_bear_low)
        bear_ob_high.append(current_bear_high)
        bull_ob_active.append(bull_active)
        bear_ob_active.append(bear_active)
        bull_ob_retest.append(current_bull_retest)
        bear_ob_retest.append(current_bear_retest)

    data["bull_ob_low"] = bull_ob_low
    data["bull_ob_high"] = bull_ob_high
    data["bear_ob_low"] = bear_ob_low
    data["bear_ob_high"] = bear_ob_high
    data["bull_ob_active"] = bull_ob_active
    data["bear_ob_active"] = bear_ob_active
    data["bull_ob_retest"] = bull_ob_retest
    data["bear_ob_retest"] = bear_ob_retest

    range_high = data["high"].rolling(range_window, min_periods=max(5, range_window // 2)).max()
    range_low = data["low"].rolling(range_window, min_periods=max(5, range_window // 2)).min()
    range_span = (range_high - range_low).replace(0, np.nan)
    equilibrium = (range_high + range_low) / 2.0
    data["range_high_20"] = range_high
    data["range_low_20"] = range_low
    data["equilibrium"] = equilibrium
    data["premium_pct"] = ((data["close"] - range_low) / range_span).clip(0, 1)
    data["discount_pct"] = (1 - data["premium_pct"]).clip(0, 1)
    data["in_discount"] = data["premium_pct"] <= 0.35
    data["in_premium"] = data["premium_pct"] >= 0.65

    return data


__all__ = ["prepare_price_action_concepts"]
