from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

from trade_system.domains.strategy.application.indicators import calculate_supertrend


ENTRY_CUTOFF = time(15, 24)
SQUARE_OFF_BAR = time(15, 27)


@dataclass(slots=True)
class YearlyBacktestArtifacts:
    yearly_summary: pd.DataFrame
    trade_logs: dict[int, pd.DataFrame]
    report_path: Path
    summary_csv_path: Path


class SupertrendYearlyBacktester:
    def __init__(self, period: int = 7, multiplier: int = 3) -> None:
        self.period = period
        self.multiplier = multiplier

    def run_on_directory(self, data_dir: Path, output_dir: Path) -> YearlyBacktestArtifacts:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries: list[dict] = []
        trade_logs: dict[int, pd.DataFrame] = {}

        for file_path in files:
            year = int(file_path.stem.rsplit("_", 1)[-1])
            trades = self.run_year(file_path)
            trade_logs[year] = trades
            trade_csv = output_dir / f"NSE_NIFTY50-INDEX_supertrend_7_3_{year}_trades.csv"
            trades.to_csv(trade_csv, index=False)
            summaries.append(self._summarize_year(year, trades))

        yearly_summary = pd.DataFrame(summaries).sort_values("year")
        summary_csv_path = output_dir / "NSE_NIFTY50-INDEX_supertrend_7_3_yearly_summary.csv"
        yearly_summary.to_csv(summary_csv_path, index=False)
        report_path = output_dir / "NSE_NIFTY50-INDEX_supertrend_7_3_yearly_report.md"
        report_path.write_text(self._build_markdown_report(yearly_summary, trade_logs))
        return YearlyBacktestArtifacts(
            yearly_summary=yearly_summary,
            trade_logs=trade_logs,
            report_path=report_path,
            summary_csv_path=summary_csv_path,
        )

    def run_year(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = calculate_supertrend(df, period=self.period, multiplier=self.multiplier)
        if df.empty:
            return pd.DataFrame()

        trades: list[dict] = []
        position: dict | None = None

        for idx in range(1, len(df)):
            prev_row = df.iloc[idx - 1]
            row = df.iloc[idx]
            ts = pd.Timestamp(row["timestamp"])
            prev_ts = pd.Timestamp(prev_row["timestamp"])

            if position and ts.date() != position["entry_time"].date():
                prior_close = float(prev_row["close"])
                trades.append(self._close_position(position, prev_ts, prior_close, "EOD_SQUARE_OFF"))
                position = None

            if position:
                stop_level = float(row["supertrend"])
                if position["side"] == 1 and float(row["low"]) <= stop_level:
                    trades.append(self._close_position(position, ts, stop_level, "SL_HIT"))
                    position = None
                elif position["side"] == -1 and float(row["high"]) >= stop_level:
                    trades.append(self._close_position(position, ts, stop_level, "SL_HIT"))
                    position = None
                elif position and int(row["supertrend_direction"]) != position["side"]:
                    trades.append(self._close_position(position, ts, float(row["close"]), "TREND_CHANGE"))
                    position = None
                elif position and ts.time() >= SQUARE_OFF_BAR:
                    trades.append(self._close_position(position, ts, float(row["close"]), "EOD_SQUARE_OFF"))
                    position = None

            signal_flip = int(row["supertrend_direction"]) != int(prev_row["supertrend_direction"])
            if not position and signal_flip and ts.time() <= ENTRY_CUTOFF:
                direction = int(row["supertrend_direction"])
                position = {
                    "entry_time": ts.to_pydatetime(),
                    "entry_price": float(row["close"]),
                    "entry_supertrend": float(row["supertrend"]),
                    "side": direction,
                    "instrument": "CE" if direction == 1 else "PE",
                    "entry_reason": "SUPERTREND_GREEN" if direction == 1 else "SUPERTREND_RED",
                }

        if position:
            last_row = df.iloc[-1]
            last_ts = pd.Timestamp(last_row["timestamp"])
            trades.append(self._close_position(position, last_ts, float(last_row["close"]), "FORCED_CLOSE"))

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return trades_df
        trades_df["holding_minutes"] = (
            pd.to_datetime(trades_df["exit_time"], format="mixed") - pd.to_datetime(trades_df["entry_time"], format="mixed")
        ).dt.total_seconds() / 60.0
        return trades_df

    def _close_position(self, position: dict, exit_time, exit_price: float, exit_reason: str) -> dict:
        side = int(position["side"])
        points = (exit_price - float(position["entry_price"])) * side
        return {
            "trade_type": position["instrument"],
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
    def _summarize_year(year: int, trades: pd.DataFrame) -> dict:
        if trades.empty:
            return {
                "year": year,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "sl_hits": 0,
                "trend_exits": 0,
                "eod_exits": 0,
                "gross_points": 0.0,
                "avg_points": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "win_rate": 0.0,
            }

        wins = trades[trades["points_captured"] > 0]
        losses = trades[trades["points_captured"] < 0]
        return {
            "year": year,
            "trades": int(len(trades)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "sl_hits": int((trades["exit_reason"] == "SL_HIT").sum()),
            "trend_exits": int((trades["exit_reason"] == "TREND_CHANGE").sum()),
            "eod_exits": int((trades["exit_reason"] == "EOD_SQUARE_OFF").sum()),
            "gross_points": round(float(trades["points_captured"].sum()), 2),
            "avg_points": round(float(trades["points_captured"].mean()), 2),
            "avg_win": round(float(wins["points_captured"].mean()), 2) if not wins.empty else 0.0,
            "avg_loss": round(float(losses["points_captured"].mean()), 2) if not losses.empty else 0.0,
            "max_win": round(float(trades["points_captured"].max()), 2),
            "max_loss": round(float(trades["points_captured"].min()), 2),
            "win_rate": round(float((trades["points_captured"] > 0).mean() * 100), 2),
        }

    def _build_markdown_report(self, yearly_summary: pd.DataFrame, trade_logs: dict[int, pd.DataFrame]) -> str:
        lines = [
            "# NIFTY50 Supertrend 7,3 Yearly Backtest",
            "",
            "Rules:",
            "- CE entry when Supertrend flips green.",
            "- PE entry when Supertrend flips red.",
            "- Exit on Supertrend stop-loss hit, opposite trend change, or end-of-day square-off before 15:30.",
            "",
            "## Yearly Summary",
            "",
            yearly_summary.to_markdown(index=False),
            "",
            "## Best And Worst Trades By Year",
            "",
        ]

        for year, trades in sorted(trade_logs.items()):
            lines.append(f"### {year}")
            if trades.empty:
                lines.append("No trades")
                lines.append("")
                continue
            best = trades.sort_values("points_captured", ascending=False).iloc[0]
            worst = trades.sort_values("points_captured", ascending=True).iloc[0]
            lines.append(
                f"Best: {best['trade_type']} {best['entry_time']} -> {best['exit_time']} | "
                f"{best['points_captured']} pts | {best['exit_reason']}"
            )
            lines.append(
                f"Worst: {worst['trade_type']} {worst['entry_time']} -> {worst['exit_time']} | "
                f"{worst['points_captured']} pts | {worst['exit_reason']}"
            )
            lines.append("")

        return "\n".join(lines)
