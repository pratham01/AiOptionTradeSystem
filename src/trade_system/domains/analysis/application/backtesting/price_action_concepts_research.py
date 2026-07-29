from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.analysis import prepare_price_action_concepts
from trade_system.domains.strategy.application.indicators import calculate_supertrend


ENTRY_CUTOFF = time(14, 45)
SQUARE_OFF_BAR = time(15, 15)
TIMEFRAME_MINUTES = 15


@dataclass(slots=True)
class PriceActionConceptsArtifacts:
    summary: pd.DataFrame
    summary_path: Path
    report_path: Path


class PriceActionConceptsResearch:
    def __init__(self, period: int = 7, multiplier: int = 3) -> None:
        self.period = period
        self.multiplier = multiplier
        self.strategies = {
            "pac_choch_reversal": self._signal_choch_reversal,
            "pac_choch_plus_ob_reversal": self._signal_choch_plus_ob_reversal,
            "pac_bos_continuation": self._signal_bos_continuation,
            "pac_ob_fvg_continuation": self._signal_ob_fvg_continuation,
            "pac_sweep_fvg_reversal": self._signal_sweep_fvg_reversal,
        }

    def run(self, data_dir: Path, output_dir: Path) -> PriceActionConceptsArtifacts:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for name in self.strategies:
            strategy_dir = output_dir / name
            strategy_dir.mkdir(parents=True, exist_ok=True)
            combined_logs: list[pd.DataFrame] = []
            for file_path in files:
                year = int(file_path.stem.rsplit("_", 1)[-1])
                trades = self.run_year(file_path, name)
                trades["year"] = year
                trades.to_csv(strategy_dir / f"{name}_{year}_trades.csv", index=False)
                combined_logs.append(trades)
                rows.append(self._summarize_year(name, year, trades))
            all_trades = pd.concat(combined_logs, ignore_index=True) if combined_logs else pd.DataFrame()
            if not all_trades.empty:
                all_trades.to_csv(strategy_dir / f"{name}_all_trades.csv", index=False)

        summary = pd.DataFrame(rows).sort_values(["strategy", "year"]).reset_index(drop=True)
        summary_path = output_dir / "price_action_concepts_summary.csv"
        summary.to_csv(summary_path, index=False)
        report_path = output_dir / "price_action_concepts_report.md"
        report_path.write_text(self._build_report(summary))
        return PriceActionConceptsArtifacts(summary=summary, summary_path=summary_path, report_path=report_path)

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
        )
        data = prepare_price_action_concepts(data)
        data = calculate_supertrend(data.reset_index(), period=self.period, multiplier=self.multiplier)
        data["day"] = data["timestamp"].dt.date
        data["bar_number"] = data.groupby("day").cumcount()
        return data

    def _signal_choch_reversal(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        row = data.iloc[idx]
        if int(row["swing_choch"]) == 1 and bool(row["in_discount"]) and float(row["close"]) > float(row["ema20"]):
            return 1, "PAC_SWING_CHOCH_DISCOUNT"
        if int(row["swing_choch"]) == -1 and bool(row["in_premium"]) and float(row["close"]) < float(row["ema20"]):
            return -1, "PAC_SWING_CHOCH_PREMIUM"
        return 0, ""

    def _signal_bos_continuation(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        row = data.iloc[idx]
        if int(row["swing_structure"]) == 1 and int(row["internal_bos"]) == 1 and bool(row["bull_fvg_recent"]):
            if float(row["close"]) > float(row["ema20"]) and float(row["body_ratio"]) >= 0.45:
                return 1, "PAC_BOS_CONTINUATION_UP"
        if int(row["swing_structure"]) == -1 and int(row["internal_bos"]) == -1 and bool(row["bear_fvg_recent"]):
            if float(row["close"]) < float(row["ema20"]) and float(row["body_ratio"]) >= 0.45:
                return -1, "PAC_BOS_CONTINUATION_DOWN"
        return 0, ""

    def _signal_choch_plus_ob_reversal(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        row = data.iloc[idx]
        if bool(row["bull_choch_plus"]) and bool(row["bull_ob_retest"]) and bool(row["in_discount"]):
            if float(row["close"]) > float(row["ema20"]) and float(row["body_ratio"]) >= 0.35:
                return 1, "PAC_CHOCH_PLUS_OB_REVERSAL_UP"
        if bool(row["bear_choch_plus"]) and bool(row["bear_ob_retest"]) and bool(row["in_premium"]):
            if float(row["close"]) < float(row["ema20"]) and float(row["body_ratio"]) >= 0.35:
                return -1, "PAC_CHOCH_PLUS_OB_REVERSAL_DOWN"
        return 0, ""

    def _signal_ob_fvg_continuation(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        row = data.iloc[idx]
        if int(row["swing_structure"]) == 1 and bool(row["bull_ob_retest"]) and bool(row["bull_fvg_recent"]):
            if float(row["close"]) > float(row["ema20"]) and float(row["body_ratio"]) >= 0.35:
                return 1, "PAC_OB_FVG_CONTINUATION_UP"
        if int(row["swing_structure"]) == -1 and bool(row["bear_ob_retest"]) and bool(row["bear_fvg_recent"]):
            if float(row["close"]) < float(row["ema20"]) and float(row["body_ratio"]) >= 0.35:
                return -1, "PAC_OB_FVG_CONTINUATION_DOWN"
        return 0, ""

    def _signal_sweep_fvg_reversal(self, data: pd.DataFrame, idx: int) -> tuple[int, str]:
        row = data.iloc[idx]
        if bool(row["bull_liquidity_sweep"]) and (bool(row["bull_fvg_recent"]) or int(row["internal_choch"]) == 1):
            if bool(row["in_discount"]) and float(row["close"]) > float(row["ema20"]):
                return 1, "PAC_SWEEP_FVG_REVERSAL_UP"
        if bool(row["bear_liquidity_sweep"]) and (bool(row["bear_fvg_recent"]) or int(row["internal_choch"]) == -1):
            if bool(row["in_premium"]) and float(row["close"]) < float(row["ema20"]):
                return -1, "PAC_SWEEP_FVG_REVERSAL_DOWN"
        return 0, ""

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
        return "\n".join(
            [
                "# Price Action Concepts Research",
                "",
                "Open implementation based on public feature descriptions only. This is not a source clone of the TradingView script.",
                "",
                "Strategies:",
                "- `pac_choch_reversal`: swing CHoCH reversal in discount/premium with EMA alignment.",
                "- `pac_choch_plus_ob_reversal`: stricter CHoCH+ reversal with order-block retest confluence.",
                "- `pac_bos_continuation`: swing trend continuation using internal BOS and recent FVG support.",
                "- `pac_ob_fvg_continuation`: trend continuation using active order-block retest plus recent FVG support.",
                "- `pac_sweep_fvg_reversal`: liquidity sweep reversal with FVG or internal CHoCH confirmation.",
                "",
                "## Yearly Summary",
                "",
                summary_table,
                "",
                "## Aggregate By Strategy",
                "",
                _to_markdown_or_string(agg),
            ]
        )


def _to_markdown_or_string(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)
