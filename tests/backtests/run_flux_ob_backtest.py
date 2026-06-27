import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.backtesting.flux_ob_research import FluxOrderBlockResearch

if __name__ == "__main__":
    data_dir = Path("data")
    output_dir = Path("reports") / "flux_ob"
    
    # Initialize the backtester with standard parameters
    backtester = FluxOrderBlockResearch(
        swing_length=10,
        max_atr_mult=3.5,
        stop_buffer_points=3.0,
        reward_risk=2.0,
        min_reward_risk=1.2,
        ob_end_method="Wick"
    )
    
    artifacts = backtester.run(data_dir, output_dir)
    print("\n📊 Flux Charts Volumized Order Blocks Backtest Results Summary:")
    print(artifacts.summary.to_string(index=False))
    print(f"\nSummary CSV saved to: {artifacts.summary_path}")
    print(f"Report MD saved to: {artifacts.report_path}")
