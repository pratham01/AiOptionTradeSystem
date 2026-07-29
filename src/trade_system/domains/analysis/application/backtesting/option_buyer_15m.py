from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

from trade_system.domains.strategy.application.indicators import calculate_supertrend


ENTRY_CUTOFF = time(14, 45)
SQUARE_OFF_BAR = time(15, 15)
TIMEFRAME_MINUTES = 15


@dataclass(slots=True)
class MultiStrategyArtifacts:
    summary: pd.DataFrame
    summary_path: Path
    report_path: Path


class OptionBuyer15mResearch:
    def __init__(self, period: int = 7, multiplier: int = 3) -> None:
        self.period = period
        self.multiplier = multiplier
        self.strategies = {
            "st_flip": self._signal_st_flip,
            "st_flip_confirmed": self._signal_st_flip_confirmed,
            "st_flip_confirmed_smc": self._signal_st_flip_confirmed_smc,
            "orb_breakout": self._signal_orb_breakout,
            "pullback_reclaim": self._signal_pullback_reclaim,
            "prev_day_breakout": self._signal_prev_day_breakout,
            "compression_breakout": self._signal_compression_breakout,
        }

    def run(self, data_dir: Path, output_dir: Path) -> MultiStrategyArtifacts:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for name in self.strategies:
            strategy_dir = output_dir / name
            strategy_dir.mkdir(parents=True, exist_ok=True)
            all_trade_logs: list[pd.DataFrame] = []
            for file_path in files:
                year = int(file_path.stem.rsplit("_", 1)[-1])
                trades = self.run_year(file_path, name)
                trades["year"] = year
                trades.to_csv(strategy_dir / f"{name}_{year}_trades.csv", index=False)
                all_trade_logs.append(trades)
                rows.append(self._summarize_year(name, year, trades))
            combined = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()
            if not combined.empty:
                combined.to_csv(strategy_dir / f"{name}_all_trades.csv", index=False)

        summary = pd.DataFrame(rows).sort_values(["strategy", "year"]).reset_index(drop=True)
        summary_path = output_dir / "option_buyer_15m_summary.csv"
        summary.to_csv(summary_path, index=False)
        report_path = output_dir / "option_buyer_15m_report.md"
        report_path.write_text(self._build_report(summary))
        return MultiStrategyArtifacts(summary=summary, summary_path=summary_path, report_path=report_path)

    def run_year(self, file_path: Path, strategy_name: str) -> pd.DataFrame:
        source = pd.read_csv(file_path, parse_dates=["timestamp"]).sort_values("timestamp")
        data = self._prepare(source)
        signal_fn = self.strategies[strategy_name]
        trades: list[dict] = []
        position: dict | None = None

        for idx in range(1, len(data)):
            prev_row = data.iloc[idx - 1]
            row = data.iloc[idx]
            ts = pd.Timestamp(row["timestamp"])
            prev_ts = pd.Timestamp(prev_row["timestamp"])

            if position and ts.date() != position["entry_time"].date():
                trades.append(self._close(position, prev_ts, float(prev_row["close"]), "EOD_SQUARE_OFF"))
                position = None

            if position:
                stop_level = float(row["supertrend"])
                if position["side"] == 1 and float(row["low"]) <= stop_level:
                    trades.append(self._close(position, ts, stop_level, "SL_HIT"))
                    position = None
                elif position["side"] == -1 and float(row["high"]) >= stop_level:
                    trades.append(self._close(position, ts, stop_level, "SL_HIT"))
                    position = None
                elif int(row["supertrend_direction"]) != position["side"]:
                    trades.append(self._close(position, ts, float(row["close"]), "TREND_CHANGE"))
                    position = None
                elif ts.time() >= SQUARE_OFF_BAR:
                    trades.append(self._close(position, ts, float(row["close"]), "EOD_SQUARE_OFF"))
                    position = None

            if position is None and ts.time() <= ENTRY_CUTOFF:
                direction, reason = signal_fn(data, idx)
                if direction != 0:
                    position = {
                        "entry_time": ts.to_pydatetime(),
                        "entry_price": float(row["close"]),
                        "entry_supertrend": float(row["supertrend"]),
                        "side": direction,
                        "trade_type": "CE" if direction == 1 else "PE",
                        "entry_reason": reason,
                    }

        if position is not None:
            last = data.iloc[-1]
            trades.append(self._close(position, pd.Timestamp(last["timestamp"]), float(last["close"]), "FORCED_CLOSE"))

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return trades_df
        trades_df["holding_minutes"] = (
            pd.to_datetime(trades_df["exit_time"], format="mixed") - pd.to_datetime(trades_df["entry_time"], format="mixed")
        ).dt.total_seconds() / 60.0
        return trades_df

    def _prepare(self, source: pd.DataFrame) -> pd.DataFrame:
        frame = source.copy().set_index("timestamp")
        data = (
            frame.resample(f"{TIMEFRAME_MINUTES}min")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .dropna()
            .reset_index()
        )
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
        data["day"] = data["timestamp"].dt.date
        data["bar_number"] = data.groupby("day").cumcount()
        daily = data.groupby("day").agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
        daily["prev_day_high"] = daily["day_high"].shift(1)
        daily["prev_day_low"] = daily["day_low"].shift(1)
        data = data.merge(daily[["prev_day_high", "prev_day_low"]], left_on="day", right_index=True, how="left")
        orb = data[data["bar_number"] < 2].groupby("day").agg(orb_high=("high", "max"), orb_low=("low", "min"))
        data = data.merge(orb, left_on="day", right_index=True, how="left")
        data = calculate_supertrend(data, period=self.period, multiplier=self.multiplier)
        data["compression"] = data["range"].rolling(4).mean() < (data["atr14"] * 0.8)
        prev_high = data.groupby("day")["high"].shift(1)
        prev_low = data.groupby("day")["low"].shift(1)
        prev2_high = data.groupby("day")["high"].shift(2)
        prev2_low = data.groupby("day")["low"].shift(2)
        rolling_low_6 = data.groupby("day")["low"].transform(lambda s: s.shift(1).rolling(6, min_periods=3).min())
        rolling_high_6 = data.groupby("day")["high"].transform(lambda s: s.shift(1).rolling(6, min_periods=3).max())
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
        data["bull_fvg_recent"] = data.groupby("day")["bull_fvg"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).max().fillna(0).astype(bool)
        )
        data["bear_fvg_recent"] = data.groupby("day")["bear_fvg"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).max().fillna(0).astype(bool)
        )
        data["bull_structure_break"] = prev_high.notna() & (data["close"] > prev_high)
        data["bear_structure_break"] = prev_low.notna() & (data["close"] < prev_low)
        return data

    def _signal_st_flip(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        prev_row = data.iloc[idx - 1]
        row = data.iloc[idx]
        if int(row["supertrend_direction"]) != int(prev_row["supertrend_direction"]):
            return int(row["supertrend_direction"]), "15M_ST_FLIP"
        return 0, ""

    def _signal_st_flip_confirmed(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        direction, _ = self._signal_st_flip(data, idx)
        if direction == 0:
            return 0, ""
        row = data.iloc[idx]
        if not self._ema_aligned(row):
            return 0, ""
        if not self._range_expansion_confirmed(row):
            return 0, ""
        return direction, "15M_ST_FLIP_CONFIRMED"

    def _signal_st_flip_confirmed_smc(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        direction, _ = self._signal_st_flip_confirmed(data, idx)
        if direction == 0:
            return 0, ""
        row = data.iloc[idx]
        if direction == 1:
            smc_ok = bool(row["bull_liquidity_sweep"]) or bool(row["bull_fvg_recent"]) or bool(row["bull_structure_break"])
        else:
            smc_ok = bool(row["bear_liquidity_sweep"]) or bool(row["bear_fvg_recent"]) or bool(row["bear_structure_break"])
        if not smc_ok:
            return 0, ""
        return direction, "15M_ST_FLIP_CONFIRMED_SMC"

    def _signal_orb_breakout(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        row = data.iloc[idx]
        if int(row["bar_number"]) < 2:
            return 0, ""
        if int(row["supertrend_direction"]) == 1 and float(row["close"]) > float(row["orb_high"]):
            return 1, "15M_ORB_HIGH_BREAK"
        if int(row["supertrend_direction"]) == -1 and float(row["close"]) < float(row["orb_low"]):
            return -1, "15M_ORB_LOW_BREAK"
        return 0, ""

    def _signal_pullback_reclaim(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        if idx < 2:
            return 0, ""
        prev_row = data.iloc[idx - 1]
        row = data.iloc[idx]
        if int(row["supertrend_direction"]) == 1 and float(prev_row["low"]) <= float(prev_row["ema20"]) and float(row["close"]) > float(prev_row["high"]):
            return 1, "15M_PULLBACK_RECLAIM_UP"
        if int(row["supertrend_direction"]) == -1 and float(prev_row["high"]) >= float(prev_row["ema20"]) and float(row["close"]) < float(prev_row["low"]):
            return -1, "15M_PULLBACK_RECLAIM_DOWN"
        return 0, ""

    def _signal_prev_day_breakout(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        row = data.iloc[idx]
        if pd.isna(row["prev_day_high"]) or pd.isna(row["prev_day_low"]):
            return 0, ""
        if int(row["supertrend_direction"]) == 1 and float(row["close"]) > float(row["prev_day_high"]):
            return 1, "15M_PREV_DAY_HIGH_BREAK"
        if int(row["supertrend_direction"]) == -1 and float(row["close"]) < float(row["prev_day_low"]):
            return -1, "15M_PREV_DAY_LOW_BREAK"
        return 0, ""

    def _signal_compression_breakout(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        if idx < 1:
            return 0, ""
        prev_row = data.iloc[idx - 1]
        row = data.iloc[idx]
        expanded = float(row["range"]) > float(row["atr14"])
        if bool(prev_row["compression"]) and expanded and int(row["supertrend_direction"]) == 1:
            return 1, "15M_COMPRESSION_BREAKOUT_UP"
        if bool(prev_row["compression"]) and expanded and int(row["supertrend_direction"]) == -1:
            return -1, "15M_COMPRESSION_BREAKOUT_DOWN"
        return 0, ""

    @staticmethod
    def _ema_aligned(row: pd.Series) -> bool:
        if int(row["supertrend_direction"]) == 1:
            return float(row["close"]) > float(row["ema20"])
        return float(row["close"]) < float(row["ema20"])

    @staticmethod
    def _range_expansion_confirmed(row: pd.Series) -> bool:
        return float(row["range"]) >= float(row["atr14"]) * 0.8

    @staticmethod
    def _close(position: dict, exit_time, exit_price: float, exit_reason: str) -> dict:
        side = int(position["side"])
        points = (exit_price - float(position["entry_price"])) * side
        return {
            "trade_type": position["trade_type"],
            "entry_reason": position["entry_reason"],
            "entry_time": position["entry_time"],
            "entry_price": position["entry_price"],
            "entry_supertrend": position["entry_supertrend"],
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "points_captured": round(points, 2),
            "outcome": "WIN" if points > 0 else "LOSS" if points < 0 else "FLAT",
            "side": "LONG" if side == 1 else "SHORT",
        }

    @staticmethod
    def _summarize_year(strategy: str, year: int, trades: pd.DataFrame) -> dict:
        if trades.empty:
            return {
                "strategy": strategy,
                "year": year,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "gross_points": 0.0,
                "avg_points": 0.0,
                "win_rate": 0.0,
                "sl_hits": 0,
                "trend_exits": 0,
                "eod_exits": 0,
                "max_win": 0.0,
                "max_loss": 0.0,
            }
        return {
            "strategy": strategy,
            "year": year,
            "trades": int(len(trades)),
            "wins": int((trades["points_captured"] > 0).sum()),
            "losses": int((trades["points_captured"] < 0).sum()),
            "gross_points": round(float(trades["points_captured"].sum()), 2),
            "avg_points": round(float(trades["points_captured"].mean()), 2),
            "win_rate": round(float((trades["points_captured"] > 0).mean() * 100), 2),
            "sl_hits": int((trades["exit_reason"] == "SL_HIT").sum()),
            "trend_exits": int((trades["exit_reason"] == "TREND_CHANGE").sum()),
            "eod_exits": int((trades["exit_reason"] == "EOD_SQUARE_OFF").sum()),
            "max_win": round(float(trades["points_captured"].max()), 2),
            "max_loss": round(float(trades["points_captured"].min()), 2),
        }

    @staticmethod
    def _build_report(summary: pd.DataFrame) -> str:
        summary_table = _to_markdown_or_string(summary)
        lines = [
            "# 15-Minute Option Buyer Strategy Research",
            "",
            "Strategies:",
            "- `st_flip`: 15-minute Supertrend color change.",
            "- `st_flip_confirmed`: Supertrend flip with EMA20 alignment and ATR-based range expansion.",
            "- `st_flip_confirmed_smc`: confirmed Supertrend flip plus liquidity-sweep/FVG/structure-break confluence.",
            "- `orb_breakout`: opening range breakout with Supertrend direction.",
            "- `pullback_reclaim`: pullback to EMA20 with trend continuation.",
            "- `prev_day_breakout`: previous day high/low break with trend filter.",
            "- `compression_breakout`: breakout after local range compression.",
            "",
            "## Yearly Summary",
            "",
            summary_table,
            "",
            "## Aggregate By Strategy",
            "",
        ]
        agg = (
            summary.groupby("strategy", as_index=False)
            .agg(
                trades=("trades", "sum"),
                gross_points=("gross_points", "sum"),
                avg_points=("avg_points", "mean"),
                mean_win_rate=("win_rate", "mean"),
            )
            .sort_values("gross_points", ascending=False)
        )
        lines.append(_to_markdown_or_string(agg))
        return "\n".join(lines)


def _to_markdown_or_string(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)
