import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import pandas as pd

def _table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)

def main() -> int:
    root = Path(__file__).resolve().parent
    option_summary_path = root / "reports" / "option_buyer_15m" / "option_buyer_15m_summary.csv"
    smc_summary_path = root / "reports" / "smc_signal" / "smc_signal_summary_2024_2026.csv"
    out_dir = root / "reports" / "comparative"
    out_dir.mkdir(parents=True, exist_ok=True)

    option_summary = pd.read_csv(option_summary_path)
    smc_summary = pd.read_csv(smc_summary_path)

    years = [2024, 2025, 2026]
    option_subset = option_summary[option_summary["year"].isin(years)].copy()
    smc_subset = smc_summary[smc_summary["year"].isin(years)].copy()
    smc_subset["strategy"] = "smc_signal"

    comparison = pd.concat(
        [
            option_subset[["strategy", "year", "trades", "win_rate", "gross_points", "avg_points"]],
            smc_subset[["strategy", "year", "trades", "win_rate", "gross_points", "avg_points"]],
        ],
        ignore_index=True,
    ).sort_values(["year", "win_rate", "gross_points"], ascending=[True, False, False])

    focus = comparison[
        comparison["strategy"].isin(["smc_signal", "st_flip", "st_flip_confirmed", "st_flip_confirmed_smc"])
    ].copy()
    focus = focus.sort_values(["year", "strategy"]).reset_index(drop=True)

    aggregate = focus.groupby("strategy", as_index=False).agg(
        trades=("trades", "sum"),
        mean_win_rate=("win_rate", "mean"),
        gross_points=("gross_points", "sum"),
        mean_avg_points=("avg_points", "mean"),
    ).sort_values(["mean_win_rate", "gross_points"], ascending=[False, False])

    best_by_year = comparison.groupby("year", as_index=False).first()

    csv_path = out_dir / "strategy_comparison_2024_2026.csv"
    comparison.to_csv(csv_path, index=False)

    report_path = out_dir / "strategy_comparison_2024_2026.md"
    report_path.write_text(
        "\n".join(
            [
                "# Strategy Comparison 2024-2026",
                "",
                "Methodology",
                "- All results use the same yearly Nifty 3-minute source files under `data/`.",
                "- Option-buyer strategies are the 15-minute research variants from `option_buyer_15m`.",
                "- `smc_signal` is the separate SMC alert replay on 15-minute bars.",
                "- The SMC replay assumes entry at the signal bar close, exit on Supertrend breach, and 15:15 square-off.",
                "",
                "## Focus Comparison",
                "",
                _table(focus),
                "",
                "## Aggregate Comparison",
                "",
                _table(aggregate),
                "",
                "## Best Strategy By Year",
                "",
                _table(best_by_year[["year", "strategy", "trades", "win_rate", "gross_points", "avg_points"]]),
                "",
                "## Notes",
                "",
                "- `st_flip_confirmed` remains the strongest accuracy-focused strategy in this comparison.",
                "- `smc_signal` is useful as an independent alert line, but it does not beat the confirmed Supertrend strategy on total edge.",
                "- `st_flip_confirmed_smc` overfilters the already strong confirmed Supertrend strategy.",
            ]
        )
    )

    print(f"Comparison CSV: {csv_path}")
    print(f"Comparison MD: {report_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
