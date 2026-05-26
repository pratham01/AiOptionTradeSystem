from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class HighConfidenceSetupArtifacts:
    setup_stats_path: Path
    filtered_stats_path: Path
    setup_events_path: Path
    report_path: Path


class HighConfidenceSetupAnalyzer:
    def run(self, pattern_dir: Path, pattern_context_dir: Path, output_dir: Path) -> HighConfidenceSetupArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        daily_patterns = pd.read_csv(pattern_dir / "candlestick_probability_daily.csv")
        del daily_patterns

        pattern_events = self._build_daily_pattern_events(Path("data"))
        day_context = pd.read_csv(pattern_context_dir / "nifty50_daily_pattern_features.csv", parse_dates=["trade_date"])
        merged = pattern_events.merge(day_context, on="trade_date", how="left")
        events = self._label_setups(merged)
        setup_events = events[events["setup_name"].notna()].copy()
        setup_stats = self._summarize_setups(setup_events)
        filtered_stats = self._filter_and_score_setups(setup_stats)

        setup_stats_path = output_dir / "high_confidence_setup_stats.csv"
        filtered_stats_path = output_dir / "high_confidence_setup_ranked.csv"
        setup_events_path = output_dir / "high_confidence_setup_events.csv"
        report_path = output_dir / "high_confidence_setup_report.md"

        setup_stats.to_csv(setup_stats_path, index=False)
        filtered_stats.to_csv(filtered_stats_path, index=False)
        setup_events.to_csv(setup_events_path, index=False)
        report_path.write_text(self._build_report(setup_stats, filtered_stats))
        return HighConfidenceSetupArtifacts(
            setup_stats_path=setup_stats_path,
            filtered_stats_path=filtered_stats_path,
            setup_events_path=setup_events_path,
            report_path=report_path,
        )

    def _build_daily_pattern_events(self, data_dir: Path) -> pd.DataFrame:
        from trade_system.application.analysis.candlestick_patterns import CandlestickPatternAnalyzer

        intraday = CandlestickPatternAnalyzer().load_intraday(data_dir).set_index("timestamp")
        daily = CandlestickPatternAnalyzer._resample_ohlc(intraday, "D")
        labeled = CandlestickPatternAnalyzer._label_patterns(daily)
        labeled["trade_date"] = pd.to_datetime(labeled["timestamp"]).dt.normalize()
        labeled["next_return_pct"] = labeled["next_return"] / labeled["close"] * 100.0
        labeled["next_two_return"] = labeled["close"].shift(-2) - labeled["close"]
        labeled["next_two_return_pct"] = labeled["next_two_return"] / labeled["close"] * 100.0
        return labeled

    @staticmethod
    def _label_setups(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["near_daily_support"] = (out["low"] <= out["prev_low"] * 1.006) & (out["low"] >= out["prev_low"] * 0.994)
        out["near_daily_resistance"] = (out["high"] <= out["prev_high"] * 1.006) & (out["high"] >= out["prev_high"] * 0.994)
        out["near_weekly_support"] = (out["low"] <= out["prev_week_low"] * 1.008) & (out["low"] >= out["prev_week_low"] * 0.992)
        out["near_weekly_resistance"] = (out["high"] <= out["prev_week_high"] * 1.008) & (out["high"] >= out["prev_week_high"] * 0.992)
        out["strong_bull_close"] = out["close"] >= out["low"] + (out["range"] * 0.7)
        out["strong_bear_close"] = out["close"] <= out["low"] + (out["range"] * 0.3)
        out["bull_regime"] = (out["trend_3"] > 0) | (out["intraday_bias"] == "bullish")
        out["bear_regime"] = (out["trend_3"] < 0) | (out["intraday_bias"] == "bearish")
        out["compressed_day"] = out["range_vs_prev"] <= 0.85
        out["expanded_day"] = out["range_vs_prev"] >= 1.2

        conditions = [
            (
                out["hammer"] & (out["near_daily_support"] | out["near_weekly_support"]) & (out["trend_3"] < 0),
                "hammer_at_support",
            ),
            (
                out["bullish_engulfing"] & (out["near_daily_support"] | out["near_weekly_support"]) & (out["trend_3"] < 0),
                "bullish_engulfing_at_support",
            ),
            (
                out["hanging_man"] & (out["near_daily_resistance"] | out["near_weekly_resistance"]) & (out["trend_3"] > 0),
                "hanging_man_at_resistance",
            ),
            (
                out["shooting_star"] & (out["near_daily_resistance"] | out["near_weekly_resistance"]) & (out["trend_3"] > 0),
                "shooting_star_at_resistance",
            ),
            (
                out["bullish_engulfing"] & out["compressed_day"] & (out["trend_3"] < 0),
                "bullish_engulfing_after_compression",
            ),
            (
                out["bearish_engulfing"] & out["compressed_day"] & (out["trend_3"] > 0),
                "bearish_engulfing_after_compression",
            ),
            (
                out["bullish_marubozu"] & (out["trend_3"] > 0) & out["compressed_day"],
                "bullish_marubozu_breakout",
            ),
            (
                out["bearish_marubozu"] & (out["trend_3"] < 0) & out["compressed_day"],
                "bearish_marubozu_breakdown",
            ),
            (
                out["hammer"] & out["strong_bull_close"] & out["bear_regime"],
                "hammer_reversal",
            ),
            (
                out["inverted_hammer"] & out["strong_bull_close"] & out["bear_regime"],
                "inverted_hammer_reversal",
            ),
            (
                out["bullish_engulfing"] & out["strong_bull_close"] & out["bear_regime"],
                "bullish_engulfing_reversal",
            ),
            (
                out["hanging_man"] & out["strong_bear_close"] & out["bull_regime"],
                "hanging_man_reversal",
            ),
            (
                out["shooting_star"] & out["strong_bear_close"] & out["bull_regime"],
                "shooting_star_reversal",
            ),
            (
                out["bearish_engulfing"] & out["strong_bear_close"] & out["bull_regime"],
                "bearish_engulfing_reversal",
            ),
            (
                out["bullish_marubozu"] & out["compressed_day"] & out["strong_bull_close"],
                "bullish_breakout_continuation",
            ),
            (
                out["bearish_marubozu"] & out["compressed_day"] & out["strong_bear_close"],
                "bearish_breakdown_continuation",
            ),
        ]

        out["setup_name"] = pd.NA
        for mask, name in conditions:
            out.loc[mask & out["setup_name"].isna(), "setup_name"] = name
        return out

    @staticmethod
    def _summarize_setups(events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame(
                columns=[
                    "setup_name",
                    "occurrences",
                    "next_candle_up_pct",
                    "next_candle_down_pct",
                    "positive_two_candle_pct",
                    "avg_next_return_pct",
                    "avg_two_candle_return_pct",
                ]
            )
        rows = []
        for setup_name, sample in events.groupby("setup_name"):
            rows.append(
                {
                    "setup_name": setup_name,
                    "occurrences": len(sample),
                    "next_candle_up_pct": round(float((sample["next_direction"] == 1).mean() * 100), 2),
                    "next_candle_down_pct": round(float((sample["next_direction"] == -1).mean() * 100), 2),
                    "positive_two_candle_pct": round(float((sample["next_two_return"] > 0).mean() * 100), 2),
                    "avg_next_return_pct": round(float(sample["next_return_pct"].mean()), 3),
                    "avg_two_candle_return_pct": round(float(sample["next_two_return_pct"].mean()), 3),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["positive_two_candle_pct", "avg_two_candle_return_pct"],
            ascending=False,
        )

    @staticmethod
    def _filter_and_score_setups(stats: pd.DataFrame) -> pd.DataFrame:
        if stats.empty:
            return stats.copy()
        ranked = stats.copy()
        ranked = ranked[ranked["occurrences"] >= 10].copy()
        if ranked.empty:
            return ranked
        ranked["quality_score"] = (
            ranked["positive_two_candle_pct"] * 0.45
            + ranked["next_candle_up_pct"].clip(lower=0) * 0.15
            + ranked["next_candle_down_pct"].clip(lower=0) * 0.15
            + ranked["occurrences"].clip(upper=200) * 0.10
            + ranked["avg_two_candle_return_pct"].clip(lower=-2, upper=2) * 7.5
        ).round(2)
        return ranked.sort_values(["quality_score", "occurrences"], ascending=False)

    @staticmethod
    def _build_report(stats: pd.DataFrame, filtered_stats: pd.DataFrame) -> str:
        lines = [
            "# High Confidence Candle Setup Report",
            "",
            "These setups combine candle pattern, trend context, and higher-timeframe location.",
            "",
            "## Ranked Setups",
            "",
        ]
        if filtered_stats.empty:
            lines.extend(["No setups passed the minimum sample filter.", ""])
        else:
            lines.extend([filtered_stats.to_markdown(index=False), ""])
        lines.extend([
            "## All Setup Stats",
            "",
            stats.to_markdown(index=False),
            "",
            "## Guidance",
            "",
            "- Use these setups as context filters, not standalone entry rules.",
            "- Prioritize setups with larger sample size and stable positive two-candle follow-through.",
            "- Reversal setups are strongest when they align with support/resistance and compression-to-expansion transitions.",
        ])
        return "\n".join(lines)
