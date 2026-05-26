import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.analysis.high_confidence_setups import HighConfidenceSetupAnalyzer

if __name__ == "__main__":
    artifacts = HighConfidenceSetupAnalyzer().run(
        pattern_dir=Path("reports") / "candlestick_patterns",
        pattern_context_dir=Path("reports") / "pattern_analysis",
        output_dir=Path("reports") / "high_confidence_setups",
    )
    print(f"Stats: {artifacts.setup_stats_path}")
    print(f"Events: {artifacts.setup_events_path}")
    print(f"Report: {artifacts.report_path}")
