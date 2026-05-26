import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.analysis.candlestick_patterns import CandlestickPatternAnalyzer

if __name__ == "__main__":
    artifacts = CandlestickPatternAnalyzer().run(
        data_dir=Path("data"),
        output_dir=Path("reports") / "candlestick_patterns",
    )
    print(f"Daily: {artifacts.daily_path}")
    print(f"Weekly: {artifacts.weekly_path}")
    print(f"Monthly: {artifacts.monthly_path}")
    print(f"Report: {artifacts.report_path}")
