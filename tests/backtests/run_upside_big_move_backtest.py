import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.backtesting.big_move_detector import UpsideBigMoveResearch

if __name__ == "__main__":
    artifacts = UpsideBigMoveResearch().run(
        Path("data"),
        Path("reports") / "upside_big_move",
    )
    print(artifacts.summary.to_string(index=False))
    print(f"\nScored CSV: {artifacts.scored_path}")
    print(f"Summary CSV: {artifacts.summary_path}")
    print(f"Report MD: {artifacts.report_path}")
