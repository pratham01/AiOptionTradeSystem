import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.backtesting.option_buyer_15m import OptionBuyer15mResearch

if __name__ == "__main__":
    artifacts = OptionBuyer15mResearch(period=7, multiplier=3).run(
        Path("data"),
        Path("reports") / "option_buyer_15m",
    )
    print(artifacts.summary.to_string(index=False))
    print(f"\nSummary CSV: {artifacts.summary_path}")
    print(f"Report MD: {artifacts.report_path}")
