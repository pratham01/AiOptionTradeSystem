from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.research.llm import ResearchLlmClient
from trade_system.domains.analysis.application.research.orchestrator import AutonomousResearchOrchestrator


ROOT = Path(__file__).resolve().parents[1]


def test_research_llm_client_prefers_deterministic_when_unconfigured(monkeypatch):
    monkeypatch.setattr("os.environ", {})
    client = ResearchLlmClient(
        free_base_url="",
        free_model="",
        paid_base_url="",
        paid_model="",
        paid_api_key="",
    )

    provider, note = client.review("test prompt")

    assert provider == "deterministic"
    assert "No LLM configured" in note


def test_propose_zone_upgrade_keeps_positive_zones():
    trades = pd.DataFrame(
        [
            {"zone_type": "bull_fvg", "points_captured": 50.0, "r_multiple": 1.2},
            {"zone_type": "bull_fvg", "points_captured": 30.0, "r_multiple": 0.8},
            {"zone_type": "prev_day_close", "points_captured": -20.0, "r_multiple": -1.0},
            {"zone_type": "prev_day_close", "points_captured": -10.0, "r_multiple": -0.5},
        ]
    )

    candidate = AutonomousResearchOrchestrator._propose_zone_upgrade(trades)

    assert "bull_fvg" in candidate["enabled_zone_types"]
    assert "prev_day_close" in candidate["disabled_zone_types"]


def test_research_agent_cycle_writes_artifacts(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(ResearchLlmClient, "review", lambda *a, **k: ("mock", "mock_note"))
    orchestrator = AutonomousResearchOrchestrator(root=tmp_path)
    data_path = tmp_path / "data" / "NSE_NIFTY50-INDEX_3min_2026.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Needs at least 12 rows for Supertrend
    pd.DataFrame({
        "timestamp": pd.date_range("2026-03-25 09:15:00", periods=15, freq="3min"),
        "open": [100.0] * 15,
        "high": [105.0] * 15,
        "low": [95.0] * 15,
        "close": [102.0] * 15,
        "volume": [1000] * 15
    }).to_csv(data_path, index=False)
    
    artifacts = orchestrator.run_zone_upgrade_cycle(year=2026)

    assert artifacts.baseline_summary_path.exists()
    assert artifacts.upgraded_summary_path.exists()
    assert artifacts.candidate_config_path.exists()
    assert artifacts.comparison_path.exists()
    assert artifacts.report_path.exists()
