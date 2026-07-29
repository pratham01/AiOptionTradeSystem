from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.backtesting.zone_engine_research import ZoneEngineResearch
from trade_system.domains.analysis.application.research.llm import ResearchLlmClient


@dataclass(slots=True)
class ResearchRunArtifacts:
    baseline_summary_path: Path
    upgraded_summary_path: Path
    candidate_config_path: Path
    comparison_path: Path
    report_path: Path


class AutonomousResearchOrchestrator:
    def __init__(self, *, root: Path | str, llm_client: ResearchLlmClient | None = None) -> None:
        self.root = Path(root)
        self.llm_client = llm_client or ResearchLlmClient()

    def run_zone_upgrade_cycle(self, *, year: int = 2026, symbol: str = "NSE_NIFTY50-INDEX") -> ResearchRunArtifacts:
        data_path = self.root / "data" / f"{symbol}_3min_{year}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Historical data not found: {data_path}")

        lab_dir = self.root / "reports" / "agent_lab" / f"zone_upgrade_{year}"
        baseline_dir = lab_dir / "baseline"
        upgraded_dir = lab_dir / "upgraded"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        upgraded_dir.mkdir(parents=True, exist_ok=True)

        def _safe_read_csv(path: Path) -> pd.DataFrame:
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        baseline = ZoneEngineResearch().run_year(data_path, baseline_dir)
        baseline_trades = _safe_read_csv(baseline.trades_path)
        candidate = self._propose_zone_upgrade(baseline_trades)

        upgraded = ZoneEngineResearch(
            allowed_zone_types=set(candidate["enabled_zone_types"]),
            min_volume_ratio=float(candidate["min_volume_ratio"]),
            min_body_ratio=float(candidate["min_body_ratio"]),
        ).run_year(data_path, upgraded_dir)
        upgraded_trades = _safe_read_csv(upgraded.trades_path)

        candidate_config_path = lab_dir / "candidate_config.json"
        candidate_config_path.write_text(json.dumps(candidate, indent=2))

        comparison = self._build_comparison(
            baseline_summary=_safe_read_csv(baseline.summary_path),
            upgraded_summary=_safe_read_csv(upgraded.summary_path),
            baseline_trades=baseline_trades,
            upgraded_trades=upgraded_trades,
        )
        comparison_path = lab_dir / "comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        provider, llm_note = self.llm_client.review(self._build_llm_prompt(year, candidate, comparison))
        report_path = lab_dir / "report.md"
        report_path.write_text(self._build_report(year, candidate, comparison, provider, llm_note))

        return ResearchRunArtifacts(
            baseline_summary_path=baseline.summary_path,
            upgraded_summary_path=upgraded.summary_path,
            candidate_config_path=candidate_config_path,
            comparison_path=comparison_path,
            report_path=report_path,
        )

    @staticmethod
    def _propose_zone_upgrade(trades: pd.DataFrame) -> dict[str, object]:
        if trades.empty:
            return {
                "enabled_zone_types": [],
                "disabled_zone_types": [],
                "min_volume_ratio": 1.2,
                "min_body_ratio": 0.45,
            }
        by_zone = (
            trades.groupby("zone_type", as_index=False)
            .agg(
                trades=("zone_type", "count"),
                net_points=("points_captured", "sum"),
                avg_r=("r_multiple", "mean"),
                win_rate=("points_captured", lambda s: (s > 0).mean() * 100),
            )
            .sort_values(["net_points", "avg_r"], ascending=[False, False])
        )
        enabled = by_zone[(by_zone["net_points"] > 0) & (by_zone["trades"] >= 2)]["zone_type"].astype(str).tolist()
        disabled = by_zone[~by_zone["zone_type"].astype(str).isin(enabled)]["zone_type"].astype(str).tolist()
        return {
            "enabled_zone_types": enabled,
            "disabled_zone_types": disabled,
            "min_volume_ratio": 1.25 if len(enabled) <= 4 else 1.2,
            "min_body_ratio": 0.45,
        }

    @staticmethod
    def _build_comparison(
        *,
        baseline_summary: pd.DataFrame,
        upgraded_summary: pd.DataFrame,
        baseline_trades: pd.DataFrame,
        upgraded_trades: pd.DataFrame,
    ) -> pd.DataFrame:
        b = baseline_summary.iloc[0].to_dict() if not baseline_summary.empty else {}
        u = upgraded_summary.iloc[0].to_dict() if not upgraded_summary.empty else {}
        rows = []
        metrics = sorted(set(b) | set(u))
        for metric in metrics:
            if metric in {"trades", "wins", "losses", "target_hits", "sl_hits", "trend_exits", "eod_exits"}:
                baseline_value = float(b.get(metric, 0))
                upgraded_value = float(u.get(metric, 0))
            else:
                baseline_value = float(b.get(metric, 0.0))
                upgraded_value = float(u.get(metric, 0.0))
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline_value,
                    "upgraded": upgraded_value,
                    "delta": upgraded_value - baseline_value,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_llm_prompt(year: int, candidate: dict[str, object], comparison: pd.DataFrame) -> str:
        return (
            f"You are reviewing an autonomous trading-strategy research cycle for {year}.\n\n"
            "The agent ran a baseline zone-engine backtest, removed weak zones, tightened confirmation, "
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
    def _build_report(year: int, candidate: dict[str, object], comparison: pd.DataFrame, provider: str, llm_note: str) -> str:
        try:
            table = comparison.to_markdown(index=False)
        except ImportError:
            table = comparison.to_string(index=False)
        return "\n".join(
            [
                f"# Autonomous Zone Upgrade Cycle {year}",
                "",
                "This experimental agent cycle does not auto-deploy to live trading.",
                "",
                "## Candidate Config",
                "",
                "```json",
                json.dumps(candidate, indent=2),
                "```",
                "",
                "## Comparison",
                "",
                table,
                "",
                f"## LLM Review ({provider})",
                "",
                llm_note,
            ]
        )
