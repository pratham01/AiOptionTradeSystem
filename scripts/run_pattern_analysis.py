import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.analysis.application.analysis import MarketPatternAnalyzer

if __name__ == "__main__":
    artifacts = MarketPatternAnalyzer().run(
        data_dir=Path("data"),
        output_dir=Path("reports") / "pattern_analysis",
    )
    print(f"Daily features: {artifacts.daily_features_path}")
    print(f"Levels: {artifacts.multi_timeframe_levels_path}")
    print(f"Daily summary: {artifacts.daily_summary_path}")
    print(f"Probability report: {artifacts.probability_report_path}")
