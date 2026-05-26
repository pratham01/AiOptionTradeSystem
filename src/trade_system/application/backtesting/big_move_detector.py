from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_system.application.indicators import calculate_supertrend


TIMEFRAME_MINUTES = 15
LOOKAHEAD_BARS = 4


@dataclass(slots=True)
class BigMoveArtifacts:
    scored_path: Path
    summary_path: Path
    report_path: Path
    summary: pd.DataFrame


class UpsideBigMoveResearch:
    def __init__(
        self,
        period: int = 7,
        multiplier: int = 3,
        lookahead_bars: int = LOOKAHEAD_BARS,
    ) -> None:
        self.period = period
        self.multiplier = multiplier
        self.lookahead_bars = lookahead_bars
        self.thresholds = [60, 70, 80, 90]

    def run(self, data_dir: Path, output_dir: Path) -> BigMoveArtifacts:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        output_dir.mkdir(parents=True, exist_ok=True)
        scored_frames: list[pd.DataFrame] = []
        summary_rows: list[dict] = []

        for file_path in files:
            year = int(file_path.stem.rsplit("_", 1)[-1])
            scored = self.score_year(file_path)
            scored["year"] = year
            scored.to_csv(output_dir / f"upside_big_move_scored_{year}.csv", index=False)
            scored_frames.append(scored)
            summary_rows.extend(self._summarize_year(year, scored))

        combined = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
        scored_path = output_dir / "upside_big_move_scored_all_years.csv"
        combined.to_csv(scored_path, index=False)

        summary = pd.DataFrame(summary_rows).sort_values(["year", "threshold"]).reset_index(drop=True)
        summary_path = output_dir / "upside_big_move_summary.csv"
        summary.to_csv(summary_path, index=False)

        report_path = output_dir / "upside_big_move_report.md"
        report_path.write_text(self._build_report(summary))

        return BigMoveArtifacts(
            scored_path=scored_path,
            summary_path=summary_path,
            report_path=report_path,
            summary=summary,
        )

    def score_year(self, file_path: Path) -> pd.DataFrame:
        source = pd.read_csv(file_path, parse_dates=["timestamp"]).sort_values("timestamp")
        data = self._prepare(source)
        if data.empty:
            return data

        data["upside_score"] = data.apply(self._score_row, axis=1)
        data["upside_label"] = data["upside_score"].apply(self._score_label)
        labeled = self._label_future_move(data)
        return labeled.reset_index(drop=True)

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
        if data.empty:
            return data

        data["day"] = data["timestamp"].dt.date
        data["bar_number"] = data.groupby("day").cumcount()
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
        data["body"] = (data["close"] - data["open"]).abs()
        data["body_ratio"] = data["body"] / data["range"].replace(0, pd.NA)
        data["volume_avg5"] = data.groupby("day")["volume"].transform(lambda s: s.rolling(5, min_periods=1).mean())
        data["range_avg4"] = data.groupby("day")["range"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        data["rolling_high_6"] = data.groupby("day")["high"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).max())
        data["rolling_low_3"] = data.groupby("day")["low"].transform(lambda s: s.shift(1).rolling(3, min_periods=3).min())
        data["prev_low_1"] = data.groupby("day")["low"].shift(1)
        data["prev_low_2"] = data.groupby("day")["low"].shift(2)
        data["prev_high_1"] = data.groupby("day")["high"].shift(1)
        data["prev_high_2"] = data.groupby("day")["high"].shift(2)
        data["compression"] = data["range_avg4"] < (data["atr14"] * 0.8)
        data["near_breakout"] = (
            (data["rolling_high_6"].notna()) &
            ((data["rolling_high_6"] - data["close"]) <= (data["atr14"] * 0.35))
        )
        data["higher_lows"] = (
            data["prev_low_2"].notna() &
            (data["prev_low_1"] > data["prev_low_2"]) &
            (data["low"] >= data["prev_low_1"])
        )
        data["higher_highs"] = (
            data["prev_high_2"].notna() &
            (data["prev_high_1"] > data["prev_high_2"])
        )

        data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3
        data["tpv"] = data["typical_price"] * data["volume"]
        data["cum_tpv"] = data.groupby("day")["tpv"].cumsum()
        data["cum_vol"] = data.groupby("day")["volume"].cumsum().replace(0, float("nan"))
        data["vwap"] = data["cum_tpv"] / data["cum_vol"]

        data = calculate_supertrend(data, period=self.period, multiplier=self.multiplier)
        data["supertrend_gap"] = data["close"] - data["supertrend"]
        data["volume_expansion"] = data["volume"] > (data["volume_avg5"] * 1.2)
        data["trend_quality"] = (
            (data["close"] > data["ema20"]) &
            (data["vwap"].notna()) &
            (data["close"] > data["vwap"]) &
            (data["supertrend_direction"] == 1)
        )
        return data

    def _score_row(self, row: pd.Series) -> int:
        score = 0
        if bool(row.get("trend_quality", False)):
            score += 25
        if bool(row.get("compression", False)):
            score += 15
        if bool(row.get("near_breakout", False)):
            score += 15
        if bool(row.get("higher_lows", False)):
            score += 15
        if bool(row.get("higher_highs", False)):
            score += 10
        if bool(row.get("volume_expansion", False)):
            score += 10
        body_ratio = row.get("body_ratio")
        if pd.notna(body_ratio) and float(body_ratio) >= 0.55:
            score += 10
        return int(score)

    @staticmethod
    def _score_label(score: int) -> str:
        if score >= 90:
            return "PRIME"
        if score >= 80:
            return "STRONG"
        if score >= 70:
            return "BUILDING"
        if score >= 60:
            return "WATCH"
        return "NEUTRAL"

    def _label_future_move(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        future_max = pd.concat(
            [frame["high"].shift(-offset) for offset in range(1, self.lookahead_bars + 1)],
            axis=1,
        ).max(axis=1)
        future_min = pd.concat(
            [frame["low"].shift(-offset) for offset in range(1, self.lookahead_bars + 1)],
            axis=1,
        ).min(axis=1)
        frame["future_max_high"] = future_max
        frame["future_min_low"] = future_min
        frame["forward_upside_points"] = (frame["future_max_high"] - frame["close"]).round(2)
        frame["forward_drawdown_points"] = (frame["close"] - frame["future_min_low"]).round(2)
        frame["upside_target_points"] = (frame["atr14"] * 1.5).round(2)
        frame["big_up_move"] = (
            frame["forward_upside_points"] >= frame["upside_target_points"]
        ) & (
            frame["forward_drawdown_points"] <= frame["atr14"]
        )
        return frame

    def _summarize_year(self, year: int, scored: pd.DataFrame) -> list[dict]:
        rows: list[dict] = []
        valid = scored.dropna(subset=["future_max_high", "future_min_low", "atr14"])
        for threshold in self.thresholds:
            subset = valid[valid["upside_score"] >= threshold]
            event_count = int(len(subset))
            hit_count = int(subset["big_up_move"].sum()) if event_count else 0
            precision = (hit_count / event_count * 100) if event_count else 0.0
            baseline = float(valid["big_up_move"].mean() * 100) if len(valid) else 0.0
            avg_forward = float(subset["forward_upside_points"].mean()) if event_count else 0.0
            avg_drawdown = float(subset["forward_drawdown_points"].mean()) if event_count else 0.0
            rows.append(
                {
                    "year": year,
                    "threshold": threshold,
                    "events": event_count,
                    "hit_rate": round(precision, 2),
                    "baseline_hit_rate": round(baseline, 2),
                    "lift_vs_baseline": round(precision - baseline, 2),
                    "avg_forward_upside_points": round(avg_forward, 2),
                    "avg_forward_drawdown_points": round(avg_drawdown, 2),
                }
            )
        return rows

    @staticmethod
    def _build_report(summary: pd.DataFrame) -> str:
        summary_table = _to_markdown_or_string(summary)
        agg = (
            summary.groupby("threshold", as_index=False)
            .agg(
                events=("events", "sum"),
                mean_hit_rate=("hit_rate", "mean"),
                mean_baseline_hit_rate=("baseline_hit_rate", "mean"),
                mean_lift=("lift_vs_baseline", "mean"),
                mean_forward_upside=("avg_forward_upside_points", "mean"),
                mean_forward_drawdown=("avg_forward_drawdown_points", "mean"),
            )
            .sort_values("threshold")
        )
        lines = [
            "# Upside Big-Move Detector Research",
            "",
            "The detector is research-only. It is not integrated into the live or main trading flow.",
            "",
            "Scoring inputs:",
            "- Supertrend up, above EMA20, and above VWAP",
            "- local compression before expansion",
            "- close near a recent breakout level",
            "- higher lows and higher highs",
            "- volume expansion",
            "- strong candle body ratio",
            "",
            "A bar is labeled `big_up_move` when the next few bars deliver at least `1.5 * ATR14` upside while keeping drawdown within `1.0 * ATR14`.",
            "",
            "## Yearly Threshold Summary",
            "",
            summary_table,
            "",
            "## Aggregate By Threshold",
            "",
            _to_markdown_or_string(agg),
        ]
        return "\n".join(lines)


def _to_markdown_or_string(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)
