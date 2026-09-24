"""
StockMojo Smart OI Engine
=========================
Core institutional engine to model StockMojo's Smart OI terminal dynamics:
1. Multi-pane synchronized Price, Volume, and Volume Delta / Smart OI Delta.
2. Option chain snapshot aggregation for Put/Call OI Change, Net PE-CE Difference, and PCRs.
3. Intraday timeframe resampling (1m, 3m, 5m, 15m).
4. Automated Price vs Volume / Smart OI Divergence Forensics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class DivergenceSignal:
    """Represents an identified price vs volume/OI divergence."""
    timestamp: datetime
    divergence_type: str  # "BEARISH_DIVERGENCE" or "BULLISH_DIVERGENCE"
    severity: str         # "HIGH", "MEDIUM"
    price_level: float
    delta_val: float
    summary: str
    description: str


def format_indian_number(val: float, precision: int = 2) -> str:
    """
    Format large numbers into Indian numbering format (Cr, L, K).
    e.g. 38,700,000 -> 3.87 Cr
         -5,564,000 -> -55.64 L
         13,130     -> 13.13 K
    """
    if pd.isna(val) or val is None:
        return "0"
    sign = "+" if val > 0 else ("-" if val < 0 else "")
    abs_val = abs(val)
    if abs_val >= 10_000_000:
        return f"{sign}{abs_val / 10_000_000:.{precision}f} Cr"
    elif abs_val >= 100_000:
        return f"{sign}{abs_val / 100_000:.{precision}f} L"
    elif abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:.{precision}f} K"
    else:
        return f"{sign}{abs_val:.0f}"


def aggregate_option_snapshots(
    snapshots: Union[List[Tuple[datetime, pd.DataFrame]], pd.DataFrame]
) -> pd.DataFrame:
    """
    Aggregate raw option chain snapshots into time-series metrics:
    - ce_oi, pe_oi, net_oi (PE - CE)
    - ce_oi_chg, pe_oi_chg, net_oi_chg
    - ce_volume, pe_volume, total_volume
    - smart_oi_delta (change in net PE-CE OI between consecutive snapshots)
    - oi_pcr (pe_oi / ce_oi)
    - vol_pcr (pe_volume / ce_volume)
    """
    if isinstance(snapshots, pd.DataFrame):
        if snapshots.empty:
            return pd.DataFrame()
        df_raw = snapshots.copy()
        if "timestamp" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        else:
            return pd.DataFrame()
    elif isinstance(snapshots, list):
        if not snapshots:
            return pd.DataFrame()
        dfs = []
        for ts, snap_df in snapshots:
            if snap_df is not None and not snap_df.empty:
                temp = snap_df.copy()
                temp["timestamp"] = pd.to_datetime(ts)
                dfs.append(temp)
        if not dfs:
            return pd.DataFrame()
        df_raw = pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

    # Ensure required columns exist
    for col in ["oi", "oi_change", "volume"]:
        if col not in df_raw.columns:
            df_raw[col] = 0.0
        else:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0.0)

    if "option_type" not in df_raw.columns:
        return pd.DataFrame()

    # Group by timestamp and option_type
    grouped = df_raw.groupby(["timestamp", "option_type"]).agg({
        "oi": "sum",
        "oi_change": "sum",
        "volume": "sum"
    }).unstack(fill_value=0.0)

    records = []
    prev_ce_oi = None
    prev_pe_oi = None
    initial_ce_oi = None
    initial_pe_oi = None
    for ts, row in grouped.iterrows():
        ce_oi = float(row.get(("oi", "CE"), 0.0))
        pe_oi = float(row.get(("oi", "PE"), 0.0))
        ce_chg = float(row.get(("oi_change", "CE"), 0.0))
        pe_chg = float(row.get(("oi_change", "PE"), 0.0))
        ce_vol = float(row.get(("volume", "CE"), 0.0))
        pe_vol = float(row.get(("volume", "PE"), 0.0))

        if initial_ce_oi is None:
            initial_ce_oi = ce_oi
            initial_pe_oi = pe_oi

        # If broker does not provide oi_change, compute cumulative day change from opening snapshot
        if ce_chg == 0.0 and pe_chg == 0.0 and initial_ce_oi is not None and initial_pe_oi is not None:
            ce_chg = ce_oi - initial_ce_oi
            pe_chg = pe_oi - initial_pe_oi

        net_oi = pe_oi - ce_oi
        net_oi_chg = pe_chg - ce_chg

        # Dynamic Smart OI Delta between successive intervals
        if prev_ce_oi is not None and prev_pe_oi is not None:
            smart_oi_delta = (pe_oi - prev_pe_oi) - (ce_oi - prev_ce_oi)
        else:
            smart_oi_delta = net_oi_chg

        prev_ce_oi = ce_oi
        prev_pe_oi = pe_oi

        oi_pcr = pe_oi / ce_oi if ce_oi > 0 else 1.0
        vol_pcr = pe_vol / ce_vol if ce_vol > 0 else 1.0

        records.append({
            "timestamp": ts,
            "ce_oi": ce_oi,
            "pe_oi": pe_oi,
            "net_oi": net_oi,
            "ce_oi_chg": ce_chg,
            "pe_oi_chg": pe_chg,
            "net_oi_chg": net_oi_chg,
            "smart_oi_delta": smart_oi_delta,
            "ce_vol": ce_vol,
            "pe_vol": pe_vol,
            "total_opt_vol": ce_vol + pe_vol,
            "oi_pcr": oi_pcr,
            "vol_pcr": vol_pcr
        })

    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"])
        out_df = out_df.sort_values("timestamp").reset_index(drop=True)
    return out_df


def resample_intraday_data(
    price_df: pd.DataFrame,
    snap_df: pd.DataFrame,
    timeframe: str = "3m"
) -> pd.DataFrame:
    """
    Resample 1m OHLCV price candles and option chain snapshots to a target timeframe.
    Timeframe options: '1m', '3m', '5m', '15m'.
    Returns aligned DataFrame with:
    - open, high, low, close, volume, vwap
    - candle_volume_delta
    - cumulative_volume_delta (CVD)
    - ce_oi_chg, pe_oi_chg, net_oi_chg, smart_oi_delta
    - oi_pcr, vol_pcr
    """
    if price_df.empty:
        return pd.DataFrame()

    p_df = price_df.copy()
    p_df["timestamp"] = pd.to_datetime(p_df["timestamp"])
    p_df = p_df.sort_values("timestamp").set_index("timestamp")

    # Map timeframe string to pandas offset
    tf_map = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min"
    }
    offset = tf_map.get(timeframe.lower(), "3min")

    # Resample price candles
    resampled_p = p_df.resample(offset).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    # Calculate Candle Volume Delta:
    # High-precision directional volume flow: V * (Close - Open) / (High - Low + 1e-6)
    delta_ratio = (resampled_p["close"] - resampled_p["open"]) / (
        (resampled_p["high"] - resampled_p["low"]).replace(0, 1e-6)
    )
    # Clip ratio between -1.0 and 1.0
    delta_ratio = delta_ratio.clip(-1.0, 1.0)
    resampled_p["candle_volume_delta"] = resampled_p["volume"] * delta_ratio

    # Calculate Cumulative Volume Delta (CVD)
    resampled_p["cvd"] = resampled_p["candle_volume_delta"].cumsum()

    # Calculate rolling intraday VWAP
    cum_vol = resampled_p["volume"].cumsum().replace(0, 1e-6)
    typical_price = (resampled_p["high"] + resampled_p["low"] + resampled_p["close"]) / 3.0
    resampled_p["vwap"] = (typical_price * resampled_p["volume"]).cumsum() / cum_vol

    if snap_df is None or snap_df.empty:
        # Fill default zero snapshot metrics if none exist
        resampled_p["ce_oi_chg"] = 0.0
        resampled_p["pe_oi_chg"] = 0.0
        resampled_p["net_oi_chg"] = 0.0
        resampled_p["smart_oi_delta"] = resampled_p["candle_volume_delta"]
        resampled_p["oi_pcr"] = 1.0
        resampled_p["vol_pcr"] = 1.0
        return resampled_p

    # Merge Snapshots by nearest or backward merge_asof
    s_df = snap_df.copy()
    s_df["timestamp"] = pd.to_datetime(s_df["timestamp"])
    resampled_p["timestamp"] = pd.to_datetime(resampled_p["timestamp"])

    # Ensure tz-naive consistency across both dataframes
    if hasattr(resampled_p["timestamp"].dt, "tz") and resampled_p["timestamp"].dt.tz is not None:
        resampled_p["timestamp"] = resampled_p["timestamp"].dt.tz_localize(None)
    if hasattr(s_df["timestamp"].dt, "tz") and s_df["timestamp"].dt.tz is not None:
        s_df["timestamp"] = s_df["timestamp"].dt.tz_localize(None)

    s_df = s_df.sort_values("timestamp").reset_index(drop=True)
    resampled_p = resampled_p.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        resampled_p,
        s_df[[
            "timestamp", "ce_oi_chg", "pe_oi_chg", "net_oi_chg",
            "smart_oi_delta", "oi_pcr", "vol_pcr"
        ]],
        on="timestamp",
        direction="nearest"
    )

    # Fill NaN from snapshot metrics
    merged["ce_oi_chg"] = merged["ce_oi_chg"].ffill().bfill().fillna(0.0)
    merged["pe_oi_chg"] = merged["pe_oi_chg"].ffill().bfill().fillna(0.0)
    merged["net_oi_chg"] = merged["net_oi_chg"].ffill().bfill().fillna(0.0)
    merged["smart_oi_delta"] = merged["smart_oi_delta"].fillna(0.0)
    merged["oi_pcr"] = merged["oi_pcr"].ffill().bfill().fillna(1.0)
    merged["vol_pcr"] = merged["vol_pcr"].ffill().bfill().fillna(1.0)

    return merged


def detect_price_volume_divergences(
    aligned_df: pd.DataFrame,
    delta_col: str = "smart_oi_delta",
    window: int = 5
) -> List[DivergenceSignal]:
    """
    Detect institutional divergences between Price Action and Volume / Smart OI Delta:
    1. BEARISH DIVERGENCE (Distribution / Bull Trap):
       Price makes Higher High, but Delta/OI makes Lower High (or deeply negative).
    2. BULLISH DIVERGENCE (Absorption / Bear Trap):
       Price makes Lower Low, but Delta/OI makes Higher Low (or flips strongly positive).
    """
    if aligned_df.empty or len(aligned_df) < window * 2:
        return []

    signals: List[DivergenceSignal] = []
    closes = aligned_df["close"].values
    deltas = aligned_df[delta_col].values
    timestamps = aligned_df["timestamp"].tolist()

    # Find swing highs and swing lows using rolling window
    n = len(aligned_df)
    for i in range(window, n - window):
        # 1. Bearish Divergence Check: Price Swing High
        is_price_swing_high = closes[i] == max(closes[i - window : i + window + 1])
        if is_price_swing_high:
            # Look back for a previous swing high (up to 30 bars back)
            prev_high_idx = None
            for j in range(i - window - 1, max(-1, i - 30), -1):
                if j >= window and closes[j] == max(closes[j - window : j + window + 1]):
                    prev_high_idx = j
                    break

            if prev_high_idx is not None:
                # If current price is higher than previous swing high
                if closes[i] > closes[prev_high_idx]:
                    # But delta is lower or negative
                    if deltas[i] < deltas[prev_high_idx] or deltas[i] < 0:
                        signals.append(DivergenceSignal(
                            timestamp=timestamps[i],
                            divergence_type="BEARISH_DIVERGENCE",
                            severity="HIGH" if deltas[i] < 0 else "MEDIUM",
                            price_level=closes[i],
                            delta_val=deltas[i],
                            summary="⚠️ Bearish Divergence (Bull Trap)",
                            description=(
                                f"Price formed a higher high (₹{closes[i]:,.1f} > ₹{closes[prev_high_idx]:,.1f}), "
                                f"but Delta showed seller dominance ({format_indian_number(deltas[i])} vs {format_indian_number(deltas[prev_high_idx])})."
                            )
                        ))

        # 2. Bullish Divergence Check: Price Swing Low
        is_price_swing_low = closes[i] == min(closes[i - window : i + window + 1])
        if is_price_swing_low:
            # Look back for a previous swing low (up to 30 bars back)
            prev_low_idx = None
            for j in range(i - window - 1, max(-1, i - 30), -1):
                if j >= window and closes[j] == min(closes[j - window : j + window + 1]):
                    prev_low_idx = j
                    break

            if prev_low_idx is not None:
                # If current price is lower than previous swing low
                if closes[i] < closes[prev_low_idx]:
                    # But delta is higher or positive
                    if deltas[i] > deltas[prev_low_idx] or deltas[i] > 0:
                        signals.append(DivergenceSignal(
                            timestamp=timestamps[i],
                            divergence_type="BULLISH_DIVERGENCE",
                            severity="HIGH" if deltas[i] > 0 else "MEDIUM",
                            price_level=closes[i],
                            delta_val=deltas[i],
                            summary="🔥 Bullish Divergence (Smart Absorption)",
                            description=(
                                f"Price formed a lower low (₹{closes[i]:,.1f} < ₹{closes[prev_low_idx]:,.1f}), "
                                f"but Delta showed buyer accumulation ({format_indian_number(deltas[i])} vs {format_indian_number(deltas[prev_low_idx])})."
                            )
                        ))

    # Keep only the most recent non-duplicate signals
    deduped: List[DivergenceSignal] = []
    seen_ts = set()
    for s in signals:
        if s.timestamp not in seen_ts:
            seen_ts.add(s.timestamp)
            deduped.append(s)

    return deduped
