from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class CandlestickPatternArtifacts:
    daily_path: Path
    weekly_path: Path
    monthly_path: Path
    report_path: Path


class CandlestickPatternAnalyzer:
    def load_intraday(self, data_dir: Path) -> pd.DataFrame:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        frames = [pd.read_csv(path, parse_dates=["timestamp"]) for path in files]
        if not frames:
            raise FileNotFoundError("No NIFTY 3-minute data files found.")
        return pd.concat(frames, ignore_index=True).sort_values("timestamp")

    def run(self, data_dir: Path, output_dir: Path) -> CandlestickPatternArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        intraday = self.load_intraday(data_dir).set_index("timestamp")

        daily = self._resample_ohlc(intraday, "D")
        weekly = self._resample_ohlc(intraday, "W-FRI")
        monthly = self._resample_ohlc(intraday, "ME")

        daily_stats = self._analyze_timeframe(daily, "daily")
        weekly_stats = self._analyze_timeframe(weekly, "weekly")
        monthly_stats = self._analyze_timeframe(monthly, "monthly")

        daily_path = output_dir / "candlestick_probability_daily.csv"
        weekly_path = output_dir / "candlestick_probability_weekly.csv"
        monthly_path = output_dir / "candlestick_probability_monthly.csv"
        report_path = output_dir / "candlestick_probability_report.md"

        daily_stats.to_csv(daily_path, index=False)
        weekly_stats.to_csv(weekly_path, index=False)
        monthly_stats.to_csv(monthly_path, index=False)
        report_path.write_text(self._build_report(daily_stats, weekly_stats, monthly_stats))

        return CandlestickPatternArtifacts(
            daily_path=daily_path,
            weekly_path=weekly_path,
            monthly_path=monthly_path,
            report_path=report_path,
        )

    @staticmethod
    def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        out = (
            df.resample(rule)
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
        out["body"] = (out["close"] - out["open"]).abs()
        out["range"] = out["high"] - out["low"]
        out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
        out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
        out["direction"] = 0
        out.loc[out["close"] > out["open"], "direction"] = 1
        out.loc[out["close"] < out["open"], "direction"] = -1
        out["ema5"] = out["close"].ewm(span=5, adjust=False).mean()
        out["ema10"] = out["close"].ewm(span=10, adjust=False).mean()
        out["trend_3"] = out["close"].diff(3)
        out["next_return"] = out["close"].shift(-1) - out["close"]
        out["next_direction"] = 0
        out.loc[out["next_return"] > 0, "next_direction"] = 1
        out.loc[out["next_return"] < 0, "next_direction"] = -1
        return out

    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        labeled = self._label_patterns(df)
        rows: list[dict] = []
        pattern_columns = [
            "hammer",
            "hanging_man",
            "shooting_star",
            "inverted_hammer",
            "doji",
            "bullish_marubozu",
            "bearish_marubozu",
            "bullish_engulfing",
            "bearish_engulfing",
        ]
        for pattern in pattern_columns:
            sample = labeled[labeled[pattern]].copy()
            if sample.empty:
                continue
            next_up = (sample["next_direction"] == 1).mean() * 100
            next_down = (sample["next_direction"] == -1).mean() * 100
            context_uptrend = (sample["trend_3"] > 0).mean() * 100
            context_downtrend = (sample["trend_3"] < 0).mean() * 100
            rows.append(
                {
                    "timeframe": timeframe,
                    "pattern": pattern,
                    "occurrences": len(sample),
                    "next_candle_up_pct": round(float(next_up), 2),
                    "next_candle_down_pct": round(float(next_down), 2),
                    "avg_next_return": round(float(sample["next_return"].mean()), 2),
                    "median_next_return": round(float(sample["next_return"].median()), 2),
                    "uptrend_context_pct": round(float(context_uptrend), 2),
                    "downtrend_context_pct": round(float(context_downtrend), 2),
                }
            )
        return pd.DataFrame(rows).sort_values(["timeframe", "occurrences"], ascending=[True, False])

    @staticmethod
    def _label_patterns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rng = out["range"].replace(0, pd.NA)
        small_body = out["body"] <= rng * 0.3
        very_small_body = out["body"] <= rng * 0.1
        long_lower = out["lower_wick"] >= out["body"] * 2
        long_upper = out["upper_wick"] >= out["body"] * 2
        tiny_upper = out["upper_wick"] <= out["body"]
        tiny_lower = out["lower_wick"] <= out["body"]

        out["hammer"] = small_body & long_lower & tiny_upper & (out["trend_3"] < 0)
        out["hanging_man"] = small_body & long_lower & tiny_upper & (out["trend_3"] > 0)
        out["shooting_star"] = small_body & long_upper & tiny_lower & (out["trend_3"] > 0)
        out["inverted_hammer"] = small_body & long_upper & tiny_lower & (out["trend_3"] < 0)
        out["doji"] = very_small_body
        out["bullish_marubozu"] = (out["direction"] == 1) & (out["body"] / rng >= 0.9)
        out["bearish_marubozu"] = (out["direction"] == -1) & (out["body"] / rng >= 0.9)

        prev = out.shift(1)
        out["bullish_engulfing"] = (
            (prev["direction"] == -1)
            & (out["direction"] == 1)
            & (out["open"] <= prev["close"])
            & (out["close"] >= prev["open"])
        )
        out["bearish_engulfing"] = (
            (prev["direction"] == 1)
            & (out["direction"] == -1)
            & (out["open"] >= prev["close"])
            & (out["close"] <= prev["open"])
        )
        return out

    @staticmethod
    def _build_report(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame) -> str:
        lines = [
            "# Candlestick Pattern Probability Report",
            "",
            "This report estimates next-candle direction probabilities for common candlestick patterns on daily, weekly, and monthly timeframes.",
            "",
            "## Daily",
            "",
            daily.to_markdown(index=False),
            "",
            "## Weekly",
            "",
            weekly.to_markdown(index=False),
            "",
            "## Monthly",
            "",
            monthly.to_markdown(index=False),
            "",
            "## Notes",
            "",
            "- These are descriptive probabilities, not guaranteed predictive signals.",
            "- Pattern context matters more than the candle name alone.",
            "- Best use is as a filter with location, trend, and volatility regime.",
        ]
        return "\n".join(lines)
