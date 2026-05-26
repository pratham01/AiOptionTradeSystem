from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_system.application.indicators import calculate_supertrend
from trade_system.interfaces.live.collector import detect_smc_signal


ENTRY_CUTOFF_HOUR = 14
ENTRY_CUTOFF_MINUTE = 45
SQUARE_OFF_HOUR = 15
SQUARE_OFF_MINUTE = 15
TIMEFRAME_MINUTES = 15


@dataclass(slots=True)
class SmcSignalArtifacts:
    summary: pd.DataFrame
    summary_path: Path
    report_path: Path


class SmcSignalResearch:
    def __init__(self, period: int = 7, multiplier: int = 3) -> None:
        self.period = period
        self.multiplier = multiplier

    def run(self, data_dir: Path, output_dir: Path) -> SmcSignalArtifacts:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for file_path in files:
            year = int(file_path.stem.rsplit("_", 1)[-1])
            trades = self.run_year(file_path)
            trades.to_csv(output_dir / f"smc_signal_{year}_trades.csv", index=False)
            rows.append(self._summarize_year(year, trades))

        summary = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        summary_path = output_dir / "smc_signal_summary.csv"
        summary.to_csv(summary_path, index=False)
        report_path = output_dir / "smc_signal_report.md"
        report_path.write_text(self._build_report(summary))
        return SmcSignalArtifacts(summary=summary, summary_path=summary_path, report_path=report_path)

    def run_year(self, file_path: Path) -> pd.DataFrame:
        source = pd.read_csv(file_path, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
        timeframe = (
            source.resample(f"{TIMEFRAME_MINUTES}min")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .dropna()
        )
        if timeframe.empty:
            return pd.DataFrame()

        st_df = calculate_supertrend(timeframe.reset_index(), period=self.period, multiplier=self.multiplier).set_index(
            "timestamp"
        )
        frame = timeframe.join(st_df[["supertrend", "supertrend_direction"]], how="left")

        trades: list[dict] = []
        position: dict | None = None
        last_signal_bar_time = None

        for idx in range(len(frame)):
            history = frame.iloc[: idx + 1][["open", "high", "low", "close", "volume"]]
            row = frame.iloc[idx]
            ts = frame.index[idx]

            if position is not None:
                stop = float(row["supertrend"])
                if position["direction"] == 1 and float(row["low"]) <= stop:
                    trades.append(
                        self._close(position, ts, stop, "SL_HIT")
                    )
                    position = None
                elif position["direction"] == -1 and float(row["high"]) >= stop:
                    trades.append(
                        self._close(position, ts, stop, "SL_HIT")
                    )
                    position = None
                elif ts.time().hour > SQUARE_OFF_HOUR or (
                    ts.time().hour == SQUARE_OFF_HOUR and ts.time().minute >= SQUARE_OFF_MINUTE
                ):
                    trades.append(
                        self._close(position, ts, float(row["close"]), "EOD_SQUARE_OFF")
                    )
                    position = None

            signal = detect_smc_signal(history)
            if signal is None:
                continue
            if last_signal_bar_time == signal["bar_time"]:
                continue
            last_signal_bar_time = signal["bar_time"]

            if position is None and (
                ts.time().hour < ENTRY_CUTOFF_HOUR
                or (ts.time().hour == ENTRY_CUTOFF_HOUR and ts.time().minute <= ENTRY_CUTOFF_MINUTE)
            ):
                position = {
                    "signal_name": str(signal["signal_name"]),
                    "direction": int(signal["direction"]),
                    "entry_time": ts,
                    "entry_price": float(row["close"]),
                }

        if position is not None:
            last_ts = frame.index[-1]
            last_close = float(frame.iloc[-1]["close"])
            trades.append(self._close(position, last_ts, last_close, "FORCED_CLOSE"))

        return pd.DataFrame(trades)

    @staticmethod
    def _close(position: dict, exit_time, exit_price: float, exit_reason: str) -> dict:
        direction = int(position["direction"])
        return {
            "signal_name": position["signal_name"],
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_time": position["entry_time"],
            "entry_price": round(float(position["entry_price"]), 2),
            "exit_time": exit_time,
            "exit_price": round(float(exit_price), 2),
            "exit_reason": exit_reason,
            "points_captured": round((float(exit_price) - float(position["entry_price"])) * direction, 2),
        }

    @staticmethod
    def _summarize_year(year: int, trades: pd.DataFrame) -> dict:
        if trades.empty:
            return {
                "year": year,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "gross_points": 0.0,
                "avg_points": 0.0,
                "win_rate": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
            }
        return {
            "year": year,
            "trades": int(len(trades)),
            "wins": int((trades["points_captured"] > 0).sum()),
            "losses": int((trades["points_captured"] < 0).sum()),
            "gross_points": round(float(trades["points_captured"].sum()), 2),
            "avg_points": round(float(trades["points_captured"].mean()), 2),
            "win_rate": round(float((trades["points_captured"] > 0).mean() * 100), 2),
            "max_win": round(float(trades["points_captured"].max()), 2),
            "max_loss": round(float(trades["points_captured"].min()), 2),
        }

    @staticmethod
    def _build_report(summary: pd.DataFrame) -> str:
        try:
            summary_table = summary.to_markdown(index=False)
        except ImportError:
            summary_table = summary.to_string(index=False)
        return "\n".join(
            [
                "# Separate SMC Signal Research",
                "",
                "This report backtests the standalone SMC live signal with:",
                "- entry at signal bar close",
                "- exit on Supertrend breach",
                "- end-of-day square-off at 15:15",
                "",
                "## Yearly Summary",
                "",
                summary_table,
            ]
        )
