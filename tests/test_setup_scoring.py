from pathlib import Path

import pandas as pd

from trade_system.application.analysis.setup_scoring import SetupScoringAnalyzer


def test_setup_scoring_outputs(tmp_path: Path):
    setup_events = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "setup_name": ["hammer_at_support", "bearish_engulfing_reversal"],
        }
    )
    features = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "closed_above_prev_high": [True, False],
            "closed_below_prev_low": [False, True],
            "gap_pct": [0.6, -0.7],
            "session_type": ["continuation", "reversal"],
            "day_direction": [1, -1],
            "intraday_bias": ["bullish", "bearish"],
            "close_location": [0.8, 0.2],
            "day_range": [200, 180],
        }
    )
    setup_path = tmp_path / "events.csv"
    features_path = tmp_path / "features.csv"
    setup_events.to_csv(setup_path, index=False)
    features.to_csv(features_path, index=False)

    artifacts = SetupScoringAnalyzer().run(setup_path, features_path, tmp_path / "out")

    assert artifacts.scored_days_path.exists()
    assert artifacts.threshold_stats_path.exists()
    assert artifacts.report_path.exists()
