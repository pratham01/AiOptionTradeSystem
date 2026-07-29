import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.analysis.application.backtesting.price_action_concepts_research import PriceActionConceptsResearch

if __name__ == "__main__":
    artifacts = PriceActionConceptsResearch().run(
        Path("data"),
        Path("reports") / "price_action_concepts",
    )
    print(artifacts.summary.to_string(index=False))
    print(f"\nSummary CSV: {artifacts.summary_path}")
    print(f"Report MD: {artifacts.report_path}")
