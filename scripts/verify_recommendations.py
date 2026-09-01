#!/usr/bin/env python
"""
CLI script to evaluate and verify daily stock recommendations across historical data.
Usage:
    python scripts/verify_recommendations.py --lookback 90 --min-prob 60
"""
import argparse
import logging
from datetime import date
from trade_system.domains.analysis.application.analysis.recommendation_evaluator import RecommendationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate daily recommendation hit-rate and attribution")
    parser.add_argument("--lookback", type=int, default=90, help="Lookback trading days")
    parser.add_argument("--min-prob", type=int, default=60, help="Minimum probability score filter")
    args = parser.parse_args()

    print(f"\n============================================================")
    print(f"📊 EVALUATING DAILY RECOMMENDATIONS (Lookback: {args.lookback}d, Min Prob: {args.min_prob}%)")
    print(f"============================================================\n")

    evaluator = RecommendationEvaluator(lookback_days=args.lookback)
    results = evaluator.evaluate_reversal_recommendations(min_probability=args.min_prob)

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return

    print(f"📈 Total Generated Recommendations: {results['total_recommendations']}")
    print(f"🎯 Confirmed Triggered Trades:      {results['triggered_trades']} ({results['trigger_rate_pct']}%)")
    print(f"🟢 Wins: {results['wins']}  |  🔴 Losses: {results['losses']}")
    print(f"🏆 Win Rate:                        {results['win_rate_pct']}%")
    print(f"💰 Profit Factor:                   {results['profit_factor']}")
    print(f"📊 Average Return per Trade:        {results['avg_pnl_pct']:+.2f}%")
    print(f"🚀 Avg Max Favorable Excursion (MFE): +{results['avg_mfe_pct']:.2f}%")
    print(f"🛡️ Avg Max Adverse Excursion (MAE):   {results['avg_mae_pct']:.2f}%")

    print(f"\n------------------------------------------------------------")
    print(f"🔬 CONFLUENCE EDGE ATTRIBUTION (Which indicators win most?)")
    print(f"------------------------------------------------------------")
    attr = results.get("confluence_attribution", [])
    if attr:
        import pandas as pd
        print(pd.DataFrame(attr).to_string(index=False))
    else:
        print("No confluences met sample size threshold.")

    print(f"\n============================================================\n")


if __name__ == "__main__":
    main()
