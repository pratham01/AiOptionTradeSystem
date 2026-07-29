import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.analysis.application.backtesting.supertrend_yearly import SupertrendYearlyBacktester

if __name__ == "__main__":
    data_dir = Path("data")
    output_dir = Path("reports") / "supertrend_yearly"
    artifacts = SupertrendYearlyBacktester(period=7, multiplier=3).run_on_directory(data_dir, output_dir)
    print(artifacts.yearly_summary.to_string(index=False))
    print(f"\nSummary CSV: {artifacts.summary_csv_path}")
    print(f"Report MD: {artifacts.report_path}")
