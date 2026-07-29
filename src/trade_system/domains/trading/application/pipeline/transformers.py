"""Data transformers for pipeline stages."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from trade_system.domains.strategy.application.indicators import calculate_supertrend
from trade_system.domains.trading.application.pipeline.processor import PipelineStage


class IndicatorTransform(PipelineStage):
    """Add technical indicators to data."""

    def __init__(
        self,
        indicators: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__("indicators")
        self.indicators = indicators or []

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate and add indicators."""
        result = data.copy()

        for ind in self.indicators:
            name = ind.get("name")
            params = ind.get("params", {})

            if name == "supertrend":
                period = params.get("period", 7)
                multiplier = params.get("multiplier", 3)
                st_df = calculate_supertrend(result, period, multiplier)
                result["supertrend"] = st_df["supertrend"]
                result["supertrend_direction"] = st_df["supertrend_direction"]

            elif name == "sma":
                period = params.get("period", 20)
                result[f"sma_{period}"] = result["close"].rolling(window=period).mean()

            elif name == "ema":
                period = params.get("period", 20)
                result[f"ema_{period}"] = result["close"].ewm(span=period).mean()

            elif name == "rsi":
                period = params.get("period", 14)
                delta = result["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return result


class ResampleTransform(PipelineStage):
    """Resample data to different timeframe."""

    def __init__(self, target_timeframe: str, aggregation: dict[str, str] | None = None) -> None:
        super().__init__(f"resample_{target_timeframe}")
        self.target_timeframe = target_timeframe
        self.aggregation = aggregation or {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample to target timeframe."""
        if "timestamp" not in data.columns:
            return data

        result = data.copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], format="mixed")
        result = result.set_index("timestamp")

        resampled = result.resample(self.target_timeframe).agg(self.aggregation)
        resampled = resampled.reset_index()

        return resampled


class SignalTransform(PipelineStage):
    """Generate signals from indicator data."""

    def __init__(
        self,
        conditions: list[dict[str, Any]],
        signal_column: str = "signal",
    ) -> None:
        super().__init__("signal_generation")
        self.conditions = conditions
        self.signal_column = signal_column

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply signal conditions."""
        result = data.copy()
        result[self.signal_column] = None

        for i, row in result.iterrows():
            signal = self._evaluate_conditions(row)
            result.at[i, self.signal_column] = signal

        return result

    def _evaluate_conditions(self, row: pd.Series) -> str | None:
        """Evaluate signal conditions for a row."""
        for condition in self.conditions:
            if self._check_condition(row, condition):
                return condition.get("signal", "neutral")
        return None

    def _check_condition(self, row: pd.Series, condition: dict[str, Any]) -> bool:
        """Check if a condition is met."""
        indicator = condition.get("indicator")
        operator = condition.get("operator", "eq")
        value = condition.get("value")

        if indicator not in row:
            return False

        actual = row[indicator]

        if operator == "eq":
            return actual == value
        elif operator == "gt":
            return actual > value
        elif operator == "gte":
            return actual >= value
        elif operator == "lt":
            return actual < value
        elif operator == "lte":
            return actual <= value
        elif operator == "crosses_above":
            # Would need previous value for proper check
            return False
        elif operator == "crosses_below":
            return False

        return False


class FilterTransform(PipelineStage):
    """Filter data based on conditions."""

    def __init__(self, condition: Callable[[pd.DataFrame], pd.Series]) -> None:
        super().__init__("filter")
        self.condition = condition

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data."""
        mask = self.condition(data)
        return data[mask].reset_index(drop=True)


class LagTransform(PipelineStage):
    """Create lag features for ML models."""

    def __init__(self, columns: list[str], lags: list[int]) -> None:
        super().__init__("lag_features")
        self.columns = columns
        self.lags = lags

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lag features."""
        result = data.copy()
        for col in self.columns:
            if col in result.columns:
                for lag in self.lags:
                    result[f"{col}_lag_{lag}"] = result[col].shift(lag)
        return result
