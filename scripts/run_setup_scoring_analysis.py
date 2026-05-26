import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.analysis.setup_scoring import SetupScoringAnalyzer

if __name__ == "__main__":
    artifacts = SetupScoringAnalyzer().run(
        setup_events_path=Path("reports") / "high_confidence_setups" / "high_confidence_setup_events.csv",
        pattern_features_path=Path("reports") / "pattern_analysis" / "nifty50_daily_pattern_features.csv",
        output_dir=Path("reports") / "setup_scoring",
    )
    print(f"Scored days: {artifacts.scored_days_path}")
    print(f"Threshold stats: {artifacts.threshold_stats_path}")
    print(f"Report: {artifacts.report_path}")
