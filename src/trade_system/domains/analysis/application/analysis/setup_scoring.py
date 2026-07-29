from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class SetupScoreArtifacts:
    scored_days_path: Path
    threshold_stats_path: Path
    report_path: Path


class SetupScoringAnalyzer:
    def run(self, setup_events_path: Path, pattern_features_path: Path, output_dir: Path) -> SetupScoreArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(pattern_features_path, parse_dates=["trade_date"])
        events = pd.read_csv(setup_events_path, parse_dates=["trade_date"])
        scored = self._score_days(features, events)
        threshold_stats = self._evaluate_thresholds(scored)

        scored_days_path = output_dir / "setup_scored_days.csv"
        threshold_stats_path = output_dir / "setup_score_threshold_stats.csv"
        report_path = output_dir / "setup_score_report.md"

        scored.to_csv(scored_days_path, index=False)
        threshold_stats.to_csv(threshold_stats_path, index=False)
        report_path.write_text(self._build_report(scored, threshold_stats))
        return SetupScoreArtifacts(
            scored_days_path=scored_days_path,
            threshold_stats_path=threshold_stats_path,
            report_path=report_path,
        )

    @staticmethod
    def _score_days(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        day_scores = features.copy()
        day_scores["trade_date"] = pd.to_datetime(day_scores["trade_date"], format="mixed")
        event_map = events.groupby("trade_date")["setup_name"].apply(list).to_dict()
        day_scores["setup_names"] = day_scores["trade_date"].map(event_map).apply(lambda x: x if isinstance(x, list) else [])

        bull_weights = {
            "hammer_at_support": 2.0,
            "hammer_reversal": 1.5,
            "bullish_engulfing_at_support": 2.0,
            "bullish_engulfing_reversal": 2.0,
            "bullish_breakout_continuation": 1.8,
            "bullish_marubozu_breakout": 1.5,
        }
        bear_weights = {
            "hanging_man_at_resistance": 1.6,
            "shooting_star_at_resistance": 1.8,
            "hanging_man_reversal": 1.6,
            "shooting_star_reversal": 2.0,
            "bearish_engulfing_reversal": 2.2,
            "bearish_engulfing_after_compression": 1.7,
            "bearish_breakdown_continuation": 1.8,
            "bearish_marubozu_breakdown": 1.5,
        }

        def bull_score(setups: list[str]) -> float:
            return round(sum(bull_weights.get(name, 0.0) for name in setups), 2)

        def bear_score(setups: list[str]) -> float:
            return round(sum(bear_weights.get(name, 0.0) for name in setups), 2)

        day_scores["bull_setup_score"] = day_scores["setup_names"].apply(bull_score)
        day_scores["bear_setup_score"] = day_scores["setup_names"].apply(bear_score)
        day_scores["location_score"] = 0.0
        day_scores.loc[day_scores["closed_above_prev_high"], "location_score"] += 1.0
        day_scores.loc[day_scores["closed_below_prev_low"], "location_score"] -= 1.0
        day_scores.loc[day_scores["gap_pct"] > 0.35, "location_score"] += 0.5
        day_scores.loc[day_scores["gap_pct"] < -0.35, "location_score"] -= 0.5

        day_scores["volatility_score"] = 0.0
        day_scores.loc[day_scores["session_type"] == "continuation", "volatility_score"] += 1.0
        day_scores.loc[day_scores["session_type"] == "volatile", "volatility_score"] += 0.5
        day_scores.loc[day_scores["session_type"] == "reversal", "volatility_score"] += 0.5
        day_scores.loc[day_scores["session_type"] == "sideways", "volatility_score"] -= 1.0

        day_scores["trend_score"] = 0.0
        day_scores.loc[(day_scores["day_direction"] == 1) & (day_scores["intraday_bias"] == "bullish"), "trend_score"] += 1.0
        day_scores.loc[(day_scores["day_direction"] == -1) & (day_scores["intraday_bias"] == "bearish"), "trend_score"] -= 1.0
        day_scores.loc[day_scores["close_location"] >= 0.7, "trend_score"] += 0.5
        day_scores.loc[day_scores["close_location"] <= 0.3, "trend_score"] -= 0.5

        day_scores["bull_score"] = (
            day_scores["bull_setup_score"]
            + day_scores["location_score"].clip(lower=0)
            + day_scores["volatility_score"].clip(lower=0)
            + day_scores["trend_score"].clip(lower=0)
        ).round(2)
        day_scores["bear_score"] = (
            day_scores["bear_setup_score"]
            + (-day_scores["location_score"].clip(upper=0))
            + day_scores["volatility_score"].clip(lower=0)
            + (-day_scores["trend_score"].clip(upper=0))
        ).round(2)
        day_scores["score_bias"] = "neutral"
        day_scores.loc[day_scores["bull_score"] > day_scores["bear_score"], "score_bias"] = "bullish"
        day_scores.loc[day_scores["bear_score"] > day_scores["bull_score"], "score_bias"] = "bearish"
        day_scores["score_edge"] = (day_scores["bull_score"] - day_scores["bear_score"]).round(2)
        return day_scores

    @staticmethod
    def _evaluate_thresholds(scored: pd.DataFrame) -> pd.DataFrame:
        rows = []
        thresholds = [1.5, 2.0, 2.5, 3.0, 3.5]
        for threshold in thresholds:
            bull = scored[scored["bull_score"] >= threshold]
            bear = scored[scored["bear_score"] >= threshold]
            rows.append(
                {
                    "threshold": threshold,
                    "bull_days": len(bull),
                    "bull_continuation_pct": round(float((bull["session_type"] == "continuation").mean() * 100), 2) if not bull.empty else 0.0,
                    "bull_reversal_pct": round(float((bull["session_type"] == "reversal").mean() * 100), 2) if not bull.empty else 0.0,
                    "bull_avg_range": round(float(bull["day_range"].mean()), 2) if not bull.empty else 0.0,
                    "bear_days": len(bear),
                    "bear_continuation_pct": round(float((bear["session_type"] == "continuation").mean() * 100), 2) if not bear.empty else 0.0,
                    "bear_reversal_pct": round(float((bear["session_type"] == "reversal").mean() * 100), 2) if not bear.empty else 0.0,
                    "bear_avg_range": round(float(bear["day_range"].mean()), 2) if not bear.empty else 0.0,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_report(scored: pd.DataFrame, thresholds: pd.DataFrame) -> str:
        ranked_examples = scored.sort_values("score_edge", ascending=False)[
            ["trade_date", "bull_score", "bear_score", "score_bias", "session_type", "setup_names"]
        ].head(20)
        return "\n".join(
            [
                "# Setup Score Report",
                "",
                "This score combines candle setup, location, volatility regime, and trend alignment.",
                "",
                "## Threshold Stats",
                "",
                thresholds.to_markdown(index=False),
                "",
                "## Top Bullish Score Examples",
                "",
                ranked_examples.to_markdown(index=False),
            ]
        )
