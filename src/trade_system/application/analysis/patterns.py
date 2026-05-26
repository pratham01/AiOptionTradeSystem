from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class PatternAnalysisArtifacts:
    daily_features_path: Path
    multi_timeframe_levels_path: Path
    probability_report_path: Path
    daily_summary_path: Path


class MarketPatternAnalyzer:
    def load_intraday(self, data_dir: Path) -> pd.DataFrame:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        frames = [pd.read_csv(path, parse_dates=["timestamp"]) for path in files]
        if not frames:
            raise FileNotFoundError("No NIFTY 3-minute data files found.")
        df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        df["trade_date"] = df["timestamp"].dt.date
        return df

    def run(self, data_dir: Path, output_dir: Path) -> PatternAnalysisArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        intraday = self.load_intraday(data_dir)
        daily_features = self.build_daily_features(intraday)
        levels = self.build_multi_timeframe_levels(intraday, daily_features)
        probability_report = self.build_probability_report(daily_features)
        daily_summary = self.build_daily_summary(daily_features)

        daily_features_path = output_dir / "nifty50_daily_pattern_features.csv"
        levels_path = output_dir / "nifty50_multi_timeframe_levels.csv"
        probability_report_path = output_dir / "nifty50_probability_report.md"
        daily_summary_path = output_dir / "nifty50_daily_summary.csv"

        daily_features.to_csv(daily_features_path, index=False)
        levels.to_csv(levels_path, index=False)
        daily_summary.to_csv(daily_summary_path, index=False)
        probability_report_path.write_text(probability_report)

        return PatternAnalysisArtifacts(
            daily_features_path=daily_features_path,
            multi_timeframe_levels_path=levels_path,
            probability_report_path=probability_report_path,
            daily_summary_path=daily_summary_path,
        )

    def build_daily_features(self, intraday: pd.DataFrame) -> pd.DataFrame:
        bars_15m = (
            intraday.set_index("timestamp")
            .resample("15min")
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
        bars_15m["trade_date"] = bars_15m["timestamp"].dt.date
        bars_15m["bar_number"] = bars_15m.groupby("trade_date").cumcount()

        day = bars_15m.groupby("trade_date").agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
        day["day_range"] = day["day_high"] - day["day_low"]
        day["prev_high"] = day["day_high"].shift(1)
        day["prev_low"] = day["day_low"].shift(1)
        day["prev_close"] = day["day_close"].shift(1)
        day["prev_range"] = day["day_range"].shift(1)

        first_hour = bars_15m[bars_15m["bar_number"] < 4].groupby("trade_date").agg(
            first_hour_high=("high", "max"),
            first_hour_low=("low", "min"),
            first_hour_open=("open", "first"),
            first_hour_close=("close", "last"),
        )
        first_hour["first_hour_range"] = first_hour["first_hour_high"] - first_hour["first_hour_low"]

        bar_1 = bars_15m[bars_15m["bar_number"] == 0][["trade_date", "open", "high", "low", "close"]].rename(
            columns={"open": "bar1_open", "high": "bar1_high", "low": "bar1_low", "close": "bar1_close"}
        )
        bar_2 = bars_15m[bars_15m["bar_number"] == 1][["trade_date", "high", "low", "close"]].rename(
            columns={"high": "bar2_high", "low": "bar2_low", "close": "bar2_close"}
        )

        df = day.join(first_hour, how="left")
        df = df.merge(bar_1, on="trade_date", how="left")
        df = df.merge(bar_2, on="trade_date", how="left")

        weekly = day.copy()
        weekly.index = pd.to_datetime(weekly.index)
        weekly_levels = weekly.resample("W-FRI").agg(week_high=("day_high", "max"), week_low=("day_low", "min"))
        weekly_levels["prev_week_high"] = weekly_levels["week_high"].shift(1)
        weekly_levels["prev_week_low"] = weekly_levels["week_low"].shift(1)
        weekly_levels["week_key"] = weekly_levels.index.to_period("W-FRI").astype(str)
        df["week_key"] = pd.to_datetime(df["trade_date"]).dt.to_period("W-FRI").astype(str)
        df = df.merge(weekly_levels[["week_key", "prev_week_high", "prev_week_low"]], on="week_key", how="left")

        monthly = day.copy()
        monthly.index = pd.to_datetime(monthly.index)
        monthly_levels = monthly.resample("ME").agg(month_high=("day_high", "max"), month_low=("day_low", "min"))
        monthly_levels["prev_month_high"] = monthly_levels["month_high"].shift(1)
        monthly_levels["prev_month_low"] = monthly_levels["month_low"].shift(1)
        monthly_levels["month_key"] = monthly_levels.index.to_period("M").astype(str)
        df["month_key"] = pd.to_datetime(df["trade_date"]).dt.to_period("M").astype(str)
        df = df.merge(monthly_levels[["month_key", "prev_month_high", "prev_month_low"]], on="month_key", how="left")

        df["gap_points"] = df["day_open"] - df["prev_close"]
        df["gap_pct"] = df["gap_points"] / df["prev_close"] * 100.0
        df["range_vs_prev"] = df["day_range"] / df["prev_range"]
        df["close_location"] = (df["day_close"] - df["day_low"]) / (df["day_range"]).replace(0, pd.NA)
        df["first_hour_close_location"] = (
            (df["first_hour_close"] - df["first_hour_low"]) / (df["first_hour_range"]).replace(0, pd.NA)
        )
        df["broke_prev_high"] = df["day_high"] > df["prev_high"]
        df["broke_prev_low"] = df["day_low"] < df["prev_low"]
        df["closed_above_prev_high"] = df["day_close"] > df["prev_high"]
        df["closed_below_prev_low"] = df["day_close"] < df["prev_low"]
        df["day_direction"] = 0
        df.loc[df["day_close"] > df["day_open"], "day_direction"] = 1
        df.loc[df["day_close"] < df["day_open"], "day_direction"] = -1

        df["session_type"] = "sideways"
        continuation_mask = (
            (df["gap_points"] >= 0)
            & (df["day_direction"] == 1)
            & (df["close_location"] >= 0.7)
            & (df["closed_above_prev_high"] | (df["day_close"] > df["first_hour_high"]))
        ) | (
            (df["gap_points"] <= 0)
            & (df["day_direction"] == -1)
            & (df["close_location"] <= 0.3)
            & (df["closed_below_prev_low"] | (df["day_close"] < df["first_hour_low"]))
        )
        reversal_mask = (
            (df["gap_points"] > 0)
            & (df["day_direction"] == -1)
            & (df["close_location"] <= 0.35)
        ) | (
            (df["gap_points"] < 0)
            & (df["day_direction"] == 1)
            & (df["close_location"] >= 0.65)
        )
        volatile_mask = (
            (df["range_vs_prev"] >= 1.35)
            & (df["broke_prev_high"])
            & (df["broke_prev_low"])
        ) | (df["first_hour_range"] / df["prev_range"] >= 0.55)

        df.loc[volatile_mask, "session_type"] = "volatile"
        df.loc[reversal_mask & ~volatile_mask, "session_type"] = "reversal"
        df.loc[continuation_mask & ~volatile_mask, "session_type"] = "continuation"

        df["intraday_bias"] = "neutral"
        df.loc[df["day_close"] > df["first_hour_high"], "intraday_bias"] = "bullish"
        df.loc[df["day_close"] < df["first_hour_low"], "intraday_bias"] = "bearish"

        return df.sort_values("trade_date").reset_index(drop=True)

    def build_multi_timeframe_levels(self, intraday: pd.DataFrame, daily_features: pd.DataFrame) -> pd.DataFrame:
        trade_date = pd.to_datetime(daily_features["trade_date"])
        levels = pd.DataFrame({"trade_date": daily_features["trade_date"]})
        levels["support_15m"] = daily_features["first_hour_low"]
        levels["resistance_15m"] = daily_features["first_hour_high"]
        levels["support_daily"] = daily_features["prev_low"]
        levels["resistance_daily"] = daily_features["prev_high"]
        levels["support_weekly"] = daily_features["prev_week_low"]
        levels["resistance_weekly"] = daily_features["prev_week_high"]
        levels["support_monthly"] = daily_features["prev_month_low"]
        levels["resistance_monthly"] = daily_features["prev_month_high"]
        levels["nearest_support"] = levels[
            ["support_15m", "support_daily", "support_weekly", "support_monthly"]
        ].bfill(axis=1).iloc[:, 0]
        levels["nearest_resistance"] = levels[
            ["resistance_15m", "resistance_daily", "resistance_weekly", "resistance_monthly"]
        ].bfill(axis=1).iloc[:, 0]
        return levels

    def build_daily_summary(self, daily_features: pd.DataFrame) -> pd.DataFrame:
        return (
            daily_features.groupby("session_type", as_index=False)
            .agg(
                sessions=("trade_date", "count"),
                avg_range=("day_range", "mean"),
                avg_gap_pct=("gap_pct", "mean"),
                avg_close_location=("close_location", "mean"),
            )
            .sort_values("sessions", ascending=False)
        )

    def build_probability_report(self, daily_features: pd.DataFrame) -> str:
        lines = [
            "# NIFTY50 Pattern Probability Report",
            "",
            "This report is generated from historical 15-minute session features derived from the 3-minute FYERS data.",
            "",
            "## Session Distribution",
            "",
            daily_features["session_type"].value_counts(normalize=True).mul(100).round(2).to_markdown(),
            "",
            "## Conditional Probabilities",
            "",
        ]

        scenarios = {
            "Gap Up And Hold Above Previous High": (
                (daily_features["gap_points"] > 0)
                & (daily_features["day_open"] > daily_features["prev_high"])
            ),
            "Gap Down And Hold Below Previous Low": (
                (daily_features["gap_points"] < 0)
                & (daily_features["day_open"] < daily_features["prev_low"])
            ),
            "First Hour Breaks Both Sides": (
                (daily_features["first_hour_high"] > daily_features["prev_high"])
                & (daily_features["first_hour_low"] < daily_features["prev_low"])
            ),
            "Close Above First Hour High": (daily_features["day_close"] > daily_features["first_hour_high"]),
            "Close Below First Hour Low": (daily_features["day_close"] < daily_features["first_hour_low"]),
            "Large Opening Range": (daily_features["first_hour_range"] >= daily_features["prev_range"] * 0.45),
        }

        rows = []
        for label, mask in scenarios.items():
            sample = daily_features[mask.fillna(False)]
            if sample.empty:
                continue
            rows.append(
                {
                    "scenario": label,
                    "sample_days": len(sample),
                    "continuation_pct": round((sample["session_type"] == "continuation").mean() * 100, 2),
                    "reversal_pct": round((sample["session_type"] == "reversal").mean() * 100, 2),
                    "volatile_pct": round((sample["session_type"] == "volatile").mean() * 100, 2),
                    "sideways_pct": round((sample["session_type"] == "sideways").mean() * 100, 2),
                }
            )

        scenario_df = pd.DataFrame(rows).sort_values("sample_days", ascending=False)
        lines.append(scenario_df.to_markdown(index=False))
        lines.append("")
        lines.append("## Historical Analog Heuristics")
        lines.append("")

        analog_buckets = daily_features.copy()
        analog_buckets["gap_bucket"] = pd.cut(
            analog_buckets["gap_pct"],
            bins=[-100, -0.5, 0.5, 100],
            labels=["gap_down", "flat_open", "gap_up"],
        )
        analog_buckets["range_bucket"] = pd.cut(
            analog_buckets["range_vs_prev"],
            bins=[-1, 0.85, 1.2, 10],
            labels=["compressed", "normal", "expanded"],
        )
        analog_buckets["close_bucket"] = pd.cut(
            analog_buckets["close_location"],
            bins=[-1, 0.33, 0.67, 2],
            labels=["weak_close", "mid_close", "strong_close"],
        )

        analog_df = (
            analog_buckets.groupby(["gap_bucket", "range_bucket", "close_bucket", "session_type"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        if not analog_df.empty:
            for col in ["continuation", "reversal", "volatile", "sideways"]:
                if col not in analog_df.columns:
                    analog_df[col] = 0
            analog_df["total"] = analog_df[["continuation", "reversal", "volatile", "sideways"]].sum(axis=1)
            analog_df = analog_df[analog_df["total"] >= 10].copy()
            analog_df["top_outcome"] = analog_df[["continuation", "reversal", "volatile", "sideways"]].idxmax(axis=1)
            analog_df["top_outcome_pct"] = (
                analog_df[["continuation", "reversal", "volatile", "sideways"]].max(axis=1) / analog_df["total"] * 100
            ).round(2)
            analog_df = analog_df.sort_values("top_outcome_pct", ascending=False).head(15)
            lines.append(analog_df.to_markdown(index=False))

        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        lines.append("- `continuation` means the day closed in the direction of the opening / early break and held structure.")
        lines.append("- `reversal` means the day rejected the opening direction and closed on the opposite side.")
        lines.append("- `volatile` means range expansion with both-side breaks or very large first-hour expansion.")
        lines.append("- `sideways` means none of the above patterns had enough structure.")
        return "\n".join(lines)
