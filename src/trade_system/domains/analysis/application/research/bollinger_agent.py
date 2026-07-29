from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from trade_system.domains.analysis.application.backtesting.engine import BacktestEngine
from trade_system.domains.strategy.application.strategies.bollinger_options import BollingerOptionStrategy
from trade_system.domains.analysis.application.research.llm import ResearchLlmClient


@dataclass(slots=True)
class ResearchRunArtifacts:
    baseline_summary_path: Path
    upgraded_summary_path: Path
    candidate_config_path: Path
    comparison_path: Path
    report_path: Path


class BollingerResearchOrchestrator:
    def __init__(self, *, root: Path | str, llm_client: ResearchLlmClient | None = None) -> None:
        self.root = Path(root)
        self.llm_client = llm_client or ResearchLlmClient()

    def run_upgrade_cycle(self, *, year: int = 2026, symbol: str = "NSE_NIFTY50-INDEX", mode: str = "intraday") -> ResearchRunArtifacts:
        # Resolve data path based on mode
        res = "15min" if mode == "intraday" else "d"
        data_path = self.root / "data" / "fo_historical" / f"{symbol}_{res}_historical.csv"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Historical data not found: {data_path}")

        lab_dir = self.root / "reports" / "agent_lab" / f"bollinger_upgrade_{mode}_{year}"
        baseline_dir = lab_dir / "baseline"
        upgraded_dir = lab_dir / "upgraded"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        upgraded_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load data
        candles = pd.read_csv(data_path, parse_dates=["timestamp"])
        # Filter for the year
        candles = candles[candles["timestamp"].dt.year == year].copy()
        
        if candles.empty:
            raise ValueError(f"No data found for year {year} in {data_path}")

        # 2. Run Baseline
        baseline_config = {
            "mode": mode,
            "period": 20,
            "std_dev": 2.0,
            "squeeze_lookback": 100,
            "squeeze_percentile": 10.0,
            "min_volatility_score": 1.2,
        }
        baseline_strategy = BollingerOptionStrategy(**baseline_config)
        baseline_engine = BacktestEngine(strategy=baseline_strategy, initial_cash=100000.0)
        baseline_result = baseline_engine.run(candles)
        
        baseline_summary_path = baseline_dir / "summary.json"
        baseline_summary_path.write_text(json.dumps(baseline_result.summary, indent=2))
        
        # Save baseline trades
        baseline_trades_df = pd.DataFrame([__import__("dataclasses").asdict(t) for t in baseline_result.trades])
        baseline_trades_path = baseline_dir / "trades.csv"
        if not baseline_trades_df.empty:
            baseline_trades_df.to_csv(baseline_trades_path, index=False)

        # 3. Agentic Optimization
        candidate_config = self._propose_upgrade(baseline_result.summary, baseline_config)
        
        # 4. Run Upgraded
        upgraded_strategy = BollingerOptionStrategy(**candidate_config)
        upgraded_engine = BacktestEngine(strategy=upgraded_strategy, initial_cash=100000.0)
        upgraded_result = upgraded_engine.run(candles)
        
        upgraded_summary_path = upgraded_dir / "summary.json"
        upgraded_summary_path.write_text(json.dumps(upgraded_result.summary, indent=2))
        
        # Save upgraded trades
        upgraded_trades_df = pd.DataFrame([__import__("dataclasses").asdict(t) for t in upgraded_result.trades])
        if not upgraded_trades_df.empty:
            upgraded_trades_df.to_csv(upgraded_dir / "trades.csv", index=False)

        # 5. Build Reports
        candidate_config_path = lab_dir / "candidate_config.json"
        candidate_config_path.write_text(json.dumps(candidate_config, indent=2))

        comparison = self._build_comparison(baseline_result.summary, upgraded_result.summary)
        comparison_path = lab_dir / "comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        provider, llm_note = self.llm_client.review(self._build_llm_prompt(year, candidate_config, comparison))
        report_path = lab_dir / "report.md"
        report_path.write_text(self._build_report(year, mode, candidate_config, comparison, provider, llm_note))

        return ResearchRunArtifacts(
            baseline_summary_path=baseline_summary_path,
            upgraded_summary_path=upgraded_summary_path,
            candidate_config_path=candidate_config_path,
            comparison_path=comparison_path,
            report_path=report_path,
        )

    @staticmethod
    def _propose_upgrade(baseline_summary: dict, baseline_config: dict) -> dict[str, object]:
        """Simple heuristic to simulate the agent deciding on parameter tweaks based on baseline performance."""
        new_config = baseline_config.copy()
        win_rate = baseline_summary.get("win_rate", 0.0)
        trades = baseline_summary.get("trade_count", 0)
        
        if win_rate < 0.45:
            # Low win rate -> tighten criteria
            new_config["squeeze_percentile"] = max(5.0, new_config["squeeze_percentile"] - 2.5)
            new_config["min_volatility_score"] = min(2.0, new_config["min_volatility_score"] + 0.2)
        elif trades < 20:
            # Too few trades -> loosen criteria
            new_config["squeeze_percentile"] = min(25.0, new_config["squeeze_percentile"] + 5.0)
            new_config["min_volatility_score"] = max(0.8, new_config["min_volatility_score"] - 0.2)
        else:
            # Try capturing a tighter standard deviation band
            new_config["std_dev"] = 1.8
            
        return new_config

    @staticmethod
    def _build_comparison(baseline: dict, upgraded: dict) -> pd.DataFrame:
        rows = []
        metrics = sorted(set(baseline.keys()) | set(upgraded.keys()))
        for metric in metrics:
            b_val = float(baseline.get(metric, 0.0))
            u_val = float(upgraded.get(metric, 0.0))
            rows.append(
                {
                    "metric": metric,
                    "baseline": b_val,
                    "upgraded": u_val,
                    "delta": u_val - b_val,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_llm_prompt(year: int, candidate: dict[str, object], comparison: pd.DataFrame) -> str:
        return (
            f"You are reviewing an autonomous trading-strategy research cycle for Bollinger Breakout in {year}.\n\n"
            "The agent ran a baseline backtest, adjusted standard deviation, squeeze percentile and volatility thresholds, "
            "and reran the backtest.\n\n"
            f"Candidate config:\n{json.dumps(candidate, indent=2)}\n\n"
            f"Metric comparison:\n{comparison.to_csv(index=False)}\n\n"
            "Return a concise research review with:\n"
            "1. whether the upgrade is actually better\n"
            "2. what improved\n"
            "3. what degraded\n"
            "4. what next experiment to run\n"
        )

    @staticmethod
    def _build_report(year: int, mode: str, candidate: dict[str, object], comparison: pd.DataFrame, provider: str, llm_note: str) -> str:
        try:
            table = comparison.to_markdown(index=False)
        except ImportError:
            table = comparison.to_string(index=False)
        return "\n".join(
            [
                f"# Autonomous Bollinger Option {mode.capitalize()} Upgrade Cycle {year}",
                "",
                "This experimental agent cycle tests BB squeeze to expansion dynamics for option buyers.",
                "",
                "## Upgraded Config",
                "",
                "```json",
                json.dumps(candidate, indent=2),
                "```",
                "",
                "## Comparison vs Baseline",
                "",
                table,
                "",
                f"## LLM Review ({provider})",
                "",
                llm_note,
            ]
        )
