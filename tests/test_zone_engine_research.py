from pathlib import Path

from trade_system.application.backtesting.zone_engine_research import ZoneEngineResearch


ROOT = Path(__file__).resolve().parents[1]


def test_zone_engine_run_2026_smoke(tmp_path: Path):
    artifacts = ZoneEngineResearch().run_year(
        ROOT / "data" / "NSE_NIFTY50-INDEX_3min_2026.csv",
        tmp_path,
    )

    assert artifacts.summary_path.exists()
    assert artifacts.trades_path.exists()
    assert artifacts.report_path.exists()
    assert {"trades", "win_rate", "net_points", "avg_points"}.issubset(artifacts.summary.columns)
