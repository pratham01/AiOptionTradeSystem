"""
Institutional Intraday Trend Reversal Strategy.

A multi-layered institutional reversal model designed for Index instruments:
1. Layer 1: Context & Liquidity — Intraday price sweeps beyond key institutional reference levels:
   - Previous Day High (PDH) / Previous Day Low (PDL)
   - Initial Balance High (IBH) / Initial Balance Low (IBL - first 30-min range)
2. Layer 2: Momentum & Directional Reversal — Supertrend (7, 3.0) direction flip
   confirming that the counter-attack momentum has overwhelmed the initial sweep.
3. Layer 3: Defined Risk & High Asymmetry — Stop-loss placed at the absolute sweep extreme,
   targeting 1:1.8 to 1:2.0 Risk-to-Reward ratio with intraday square-off.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trade_system.domains.strategy.application.strategies.base import (
    BaseStrategy,
    StrategyContext,
    TradeSignal,
)

LOGGER = logging.getLogger(__name__)


class InstitutionalIntradayReversalStrategy(BaseStrategy):
    """
    Institutional Intraday Trend Reversal Strategy (Liquidity Sweep + Supertrend Flip).
    """
    name: str = "institutional_intraday_reversal"
    version: str = "2.0"
    supported_timeframes: List[str] = ["1m", "3m", "5m"]

    def __init__(
        self,
        st_period: int = 7,
        st_multiplier: float = 3.0,
        target_rr: float = 1.8,
        max_risk_pct: float = 0.007,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.target_rr = target_rr
        self.max_risk_pct = max_risk_pct

    @staticmethod
    def calculate_supertrend(data: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate Supertrend indicator and trend series."""
        if data.empty:
            return data

        df = data.copy()
        hl2 = (df["high"] + df["low"]) / 2.0
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        st = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)

        curr_trend = 1
        for i in range(len(df)):
            if i == 0:
                st.iloc[i] = lower.iloc[i]
                trend.iloc[i] = 1
                continue

            c_close = df["close"].iloc[i]
            p_close = df["close"].iloc[i - 1]
            p_st = st.iloc[i - 1]

            if curr_trend == 1:
                if c_close < p_st:
                    curr_trend = -1
                    st.iloc[i] = upper.iloc[i]
                else:
                    st.iloc[i] = max(lower.iloc[i], p_st) if p_close >= p_st else lower.iloc[i]
            else:
                if c_close > p_st:
                    curr_trend = 1
                    st.iloc[i] = lower.iloc[i]
                else:
                    st.iloc[i] = min(upper.iloc[i], p_st) if p_close <= p_st else upper.iloc[i]
            trend.iloc[i] = curr_trend

        df["supertrend"] = st
        df["trend"] = trend
        df["trend_flip"] = (df["trend"] != df["trend"].shift(1)) & (df["trend"].shift(1).notna())
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch evaluation for backtesting across full OHLCV history.
        """
        if data.empty or len(data) < 30:
            return pd.DataFrame()

        df = data.copy()
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            df = df.set_index("timestamp")

        df = df.sort_index()
        df["date_str"] = df.index.strftime("%Y-%m-%d")

        # Compute Supertrend
        df = self.calculate_supertrend(df, period=self.st_period, multiplier=self.st_multiplier)

        # Compute Daily PDH / PDL
        daily = df.groupby("date_str").agg({"high": "max", "low": "min"}).shift(1)
        daily.rename(columns={"high": "pdh", "low": "pdl"}, inplace=True)
        df = df.merge(daily, left_on="date_str", right_index=True, how="left")

        # Signals initialization
        df["signal"] = 0        # 1=BUY_CALL, -1=BUY_PUT
        df["action"] = ""
        df["stop_loss"] = 0.0
        df["target_1"] = 0.0
        df["target_2"] = 0.0
        df["confluence"] = ""

        for date_val, day_df in df.groupby("date_str"):
            if len(day_df) < 15:
                continue

            pdh = day_df["pdh"].iloc[0]
            pdl = day_df["pdl"].iloc[0]
            if pd.isna(pdh) or pd.isna(pdl):
                continue

            # Initial Balance (first 30 mins)
            ibh = day_df["high"].iloc[:10].max()
            ibl = day_df["low"].iloc[:10].min()

            swept_low = False
            swept_high = False
            low_extreme = 0.0
            high_extreme = 0.0

            for i in range(len(day_df)):
                row = day_df.iloc[i]
                idx = day_df.index[i]

                # Track sweeps
                if row["low"] < min(pdl, ibl):
                    swept_low = True
                    low_extreme = min(low_extreme, row["low"]) if low_extreme > 0 else row["low"]
                if row["high"] > max(pdh, ibh):
                    swept_high = True
                    high_extreme = max(high_extreme, row["high"]) if high_extreme > 0 else row["high"]

                # --- BULLISH REVERSAL (BUY CALL) ---
                if swept_low and row["trend_flip"] and row["trend"] == 1:
                    entry = float(row["close"])
                    sl = float(low_extreme)
                    risk_pts = entry - sl

                    if 0 < risk_pts < entry * self.max_risk_pct:
                        tp1 = float(entry + self.target_rr * risk_pts)
                        tp2 = float(entry + (self.target_rr + 0.5) * risk_pts)

                        df.loc[idx, "signal"] = 1
                        df.loc[idx, "action"] = "BUY_CALL"
                        df.loc[idx, "stop_loss"] = sl
                        df.loc[idx, "target_1"] = tp1
                        df.loc[idx, "target_2"] = tp2
                        df.loc[idx, "confluence"] = f"Liquidity Sweep Low (₹{low_extreme:.1f}) + 3m ST Flip UP | 1:{self.target_rr:.1f} R:R"
                        swept_low = False

                # --- BEARISH REVERSAL (BUY PUT) ---
                if swept_high and row["trend_flip"] and row["trend"] == -1:
                    entry = float(row["close"])
                    sl = float(high_extreme)
                    risk_pts = sl - entry

                    if 0 < risk_pts < entry * self.max_risk_pct:
                        tp1 = float(entry - self.target_rr * risk_pts)
                        tp2 = float(entry - (self.target_rr + 0.5) * risk_pts)

                        df.loc[idx, "signal"] = -1
                        df.loc[idx, "action"] = "BUY_PUT"
                        df.loc[idx, "stop_loss"] = sl
                        df.loc[idx, "target_1"] = tp1
                        df.loc[idx, "target_2"] = tp2
                        df.loc[idx, "confluence"] = f"Liquidity Sweep High (₹{high_extreme:.1f}) + 3m ST Flip DOWN | 1:{self.target_rr:.1f} R:R"
                        swept_high = False

        return df

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        """
        Real-time evaluation hook on every completed 3m bar.
        """
        try:
            df = context.history_df
            if df is None or len(df) < 25:
                return None

            signals_df = self.generate_signals(df)
            if signals_df.empty:
                return None

            last_row = signals_df.iloc[-1]
            signal_val = int(last_row.get("signal", 0))
            if signal_val == 0:
                return None

            direction = "CALL" if signal_val == 1 else "PUT"
            action = str(last_row.get("action", "BUY_CALL" if signal_val == 1 else "BUY_PUT"))
            entry_price = float(last_row["close"])
            stop_loss = float(last_row.get("stop_loss", 0.0))
            target_1 = float(last_row.get("target_1", 0.0))
            target_2 = float(last_row.get("target_2", 0.0))
            confluence_str = str(last_row.get("confluence", "Institutional Reversal"))

            metadata = {
                "supertrend": float(last_row.get("supertrend", 0.0)),
                "confluence": confluence_str,
                "strategy_type": "BOTTOM_FISHING" if signal_val == 1 else "TOP_SNIPE",
            }

            ts = last_row.name if isinstance(last_row.name, datetime) else datetime.now()

            return TradeSignal(
                symbol=context.symbol,
                timestamp=ts,
                direction=direction,
                action=action,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                confidence=0.85,
                strategy_name=self.name,
                timeframe=context.timeframe,
                confluence_factors=[confluence_str],
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.error("InstitutionalIntradayReversalStrategy.evaluate error for %s: %s", context.symbol, exc)
            return None

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        action = row.get("action", "REVERSAL")
        direction = "CALL" if "CALL" in action else "PUT"
        icon = "🟢" if direction == "CALL" else "🔴"
        clean_sym = symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "")
        return (
            f"{icon} <b>INSTITUTIONAL INTRADAY REVERSAL: {clean_sym}</b>\n"
            f"Action: <b>{action}</b> ({direction})\n"
            f"Entry Spot: ₹{row['close']:.2f}\n"
            f"Stop Loss: ₹{row.get('stop_loss', 0.0):.2f}\n"
            f"Target 1: ₹{row.get('target_1', 0.0):.2f}\n"
            f"Target 2: ₹{row.get('target_2', 0.0):.2f}\n"
            f"<i>{row.get('confluence', '')}</i>"
        )
