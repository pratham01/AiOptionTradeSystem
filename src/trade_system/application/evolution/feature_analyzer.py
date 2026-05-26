"""
FeatureAnalyzer — answers "which setup features predict winning trades?"

Uses logistic regression on {features → outcome} data to compute
feature importances. This drives the WeightEvolver.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

# Feature columns used for analysis
NUMERIC_FEATURES = [
    "rsi_daily",
    "rsi_hourly",
    "adx",
    "volume_surge",
    "atr_pct",
    "vix",
    "pcr",
    "alignment_score",
]
BINARY_FEATURES = [
    "near_support",
    "near_resistance",
    "is_compressed",
    "vol_delta_positive",
    "above_vwap",
    "above_poc",
]


@dataclass
class FeatureImportance:
    """Result of a feature analysis run."""
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    neutral_count: int = 0
    win_rate: float = 0.0

    # Feature importances (0.0 = no importance, 1.0 = very important)
    importances: dict[str, float] = field(default_factory=dict)

    # Top winning conditions (human-readable)
    top_conditions: list[str] = field(default_factory=list)

    # Recommended weight adjustments
    weight_adjustments: dict[str, float] = field(default_factory=dict)

    # Was there enough data?
    sufficient_data: bool = False

    def summary(self) -> str:
        lines = [
            f"Feature Analysis: {self.total_trades} trades analyzed",
            f"  Win Rate: {self.win_rate:.1%} ({self.win_count}W / {self.loss_count}L / {self.neutral_count}N)",
        ]
        if self.top_conditions:
            lines.append("  Top Win Conditions:")
            for c in self.top_conditions[:5]:
                lines.append(f"    • {c}")
        if self.importances:
            top_feats = sorted(self.importances.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("  Top Predictive Features:")
            for feat, imp in top_feats:
                lines.append(f"    • {feat}: {imp:.3f}")
        return "\n".join(lines)


class FeatureAnalyzer:
    """
    Analyzes which features in closed trades predict wins.

    Requires scikit-learn (optional dep). Falls back to correlation
    analysis if sklearn is not available.
    """

    MIN_TRADES = 10     # Minimum trades needed for reliable analysis

    def analyze(
        self,
        trades: list[Any],  # list of SuggestedTrade ORM objects with .features
        min_trades: int | None = None,
    ) -> FeatureImportance:
        """
        Analyze feature importance from closed trades.

        Args:
            trades: List of SuggestedTrade ORM objects (must have .features relationship)
            min_trades: Override minimum trades threshold

        Returns:
            FeatureImportance result
        """
        min_trades = min_trades or self.MIN_TRADES

        result = FeatureImportance()
        result.total_trades = len(trades)

        if not trades:
            LOGGER.warning("No trades provided for feature analysis.")
            return result

        # Filter to win/loss only (exclude neutral for binary classification)
        labeled = []
        for t in trades:
            if t.outcome == "WIN":
                labeled.append((t, 1))
                result.win_count += 1
            elif t.outcome == "LOSS":
                labeled.append((t, 0))
                result.loss_count += 1
            else:
                result.neutral_count += 1

        result.win_rate = result.win_count / result.total_trades if result.total_trades else 0.0

        if len(labeled) < min_trades:
            LOGGER.info("Insufficient labeled trades (%d < %d) for analysis.",
                        len(labeled), min_trades)
            result.sufficient_data = False
            return result

        result.sufficient_data = True
        importances = self._compute_importances(labeled)
        result.importances = importances
        result.top_conditions = self._build_top_conditions(labeled)
        result.weight_adjustments = self._compute_weight_adjustments(importances)

        LOGGER.info("Feature analysis complete: %d trades, %.1f%% win rate",
                    result.total_trades, result.win_rate * 100)
        return result

    def _compute_importances(self, labeled: list[tuple[Any, int]]) -> dict[str, float]:
        """
        Compute feature importances. Uses sklearn if available, else correlation.
        """
        try:
            return self._sklearn_importances(labeled)
        except ImportError:
            LOGGER.info("scikit-learn not available — using correlation analysis.")
            return self._correlation_importances(labeled)

    def _sklearn_importances(self, labeled: list[tuple[Any, int]]) -> dict[str, float]:
        """Use sklearn logistic regression for feature importances."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        features, labels = self._build_matrix(labeled)
        if features is None:
            return {}

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(features)
        y = np.array(labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=500, C=1.0)
        model.fit(X_scaled, y)

        all_feats = NUMERIC_FEATURES + BINARY_FEATURES
        coefs = model.coef_[0]
        # Normalize to 0-1
        abs_coefs = abs(coefs)
        max_c = abs_coefs.max() if abs_coefs.max() > 0 else 1.0
        return {feat: round(float(c / max_c), 4) for feat, c in zip(all_feats, abs_coefs)}

    def _correlation_importances(self, labeled: list[tuple[Any, int]]) -> dict[str, float]:
        """Fallback: point-biserial correlation between each feature and outcome."""
        import math

        all_feats = NUMERIC_FEATURES + BINARY_FEATURES
        importances: dict[str, float] = {}

        for feat in all_feats:
            values = []
            outcomes = []
            for trade, label in labeled:
                if trade.features is None:
                    continue
                val = getattr(trade.features, feat, None)
                if val is not None:
                    try:
                        values.append(float(val))
                        outcomes.append(float(label))
                    except (TypeError, ValueError):
                        pass

            if len(values) < 5:
                importances[feat] = 0.0
                continue

            n = len(values)
            mean_v = sum(values) / n
            mean_o = sum(outcomes) / n
            cov = sum((v - mean_v) * (o - mean_o) for v, o in zip(values, outcomes)) / n
            std_v = math.sqrt(sum((v - mean_v) ** 2 for v in values) / n) or 1e-9
            std_o = math.sqrt(sum((o - mean_o) ** 2 for o in outcomes) / n) or 1e-9
            importances[feat] = round(abs(cov / (std_v * std_o)), 4)

        # Normalize
        max_imp = max(importances.values()) if importances else 1.0
        if max_imp > 0:
            importances = {k: round(v / max_imp, 4) for k, v in importances.items()}
        return importances

    def _build_matrix(self, labeled: list[tuple[Any, int]]) -> tuple[Any, list[int]]:
        """Build feature matrix and label list."""
        import numpy as np

        all_feats = NUMERIC_FEATURES + BINARY_FEATURES
        rows = []
        labels = []

        for trade, label in labeled:
            if trade.features is None:
                continue
            row = [getattr(trade.features, feat, None) for feat in all_feats]
            rows.append(row)
            labels.append(label)

        if not rows:
            return None, []

        return np.array(rows, dtype=float), labels

    def _build_top_conditions(self, labeled: list[tuple[Any, int]]) -> list[str]:
        """Build human-readable conditions that appear most in winning trades."""
        win_trades = [t for t, label in labeled if label == 1]
        conditions = []

        if not win_trades:
            return conditions

        total_wins = len(win_trades)

        def _pct(count: int) -> str:
            return f"{count / total_wins:.0%}"

        # Check each binary feature
        for feat in BINARY_FEATURES:
            count = sum(1 for t in win_trades
                        if t.features and getattr(t.features, feat, 0))
            if count / total_wins >= 0.65:
                conditions.append(f"{feat} = True in {_pct(count)} of wins")

        # Check numeric feature thresholds
        for feat, threshold, label in [
            ("rsi_daily", 50.0, "RSI Daily > 50"),
            ("rsi_daily", 65.0, "RSI Daily > 65"),
            ("adx", 20.0, "ADX > 20 (trending)"),
            ("adx", 30.0, "ADX > 30 (strong trend)"),
            ("volume_surge", 1.5, "Volume Surge > 1.5x avg"),
            ("vix", 15.0, "VIX < 15 (low fear)"),
            ("alignment_score", 0.7, "Multi-TF Alignment > 70%"),
        ]:
            values = [
                getattr(t.features, feat, None)
                for t in win_trades
                if t.features and getattr(t.features, feat, None) is not None
            ]
            if not values:
                continue

            if "VIX" in label:
                count = sum(1 for v in values if v < threshold)
            else:
                count = sum(1 for v in values if v > threshold)

            if count / len(values) >= 0.65:
                conditions.append(f"{label} in {_pct(count)} of wins")

        return conditions[:10]

    def _compute_weight_adjustments(self, importances: dict[str, float]) -> dict[str, float]:
        """
        Map raw feature importances to named scoring weights used by agents.
        Returns suggested weight multipliers (> 1.0 = increase, < 1.0 = decrease).
        """
        adjustments: dict[str, float] = {}

        # Map feature groups to agent weight names
        feature_to_weight = {
            "volume_surge": "volume_delta",
            "vol_delta_positive": "volume_delta",
            "is_compressed": "compression",
            "above_poc": "value_area",
            "above_vwap": "value_area",
            "alignment_score": "trend_alignment",
            "rsi_daily": "momentum",
            "adx": "momentum",
            "vix": "market_context",
            "pcr": "market_context",
            "near_support": "price_action",
            "near_resistance": "price_action",
        }

        weight_scores: dict[str, list[float]] = {}
        for feat, imp in importances.items():
            weight_name = feature_to_weight.get(feat, feat)
            weight_scores.setdefault(weight_name, []).append(imp)

        # Average the importance for each weight group
        for weight, scores in weight_scores.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            # Map avg importance 0→1 to multiplier 0.7→1.3
            multiplier = 0.7 + avg * 0.6
            adjustments[weight] = round(multiplier, 3)

        return adjustments
