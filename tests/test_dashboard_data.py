from trade_system.shared.config.settings import TelegramConfig, FyersConfig
from pathlib import Path
import pandas as pd
from trade_system.interfaces.dashboard.data import load_strategy_summaries, load_strategy_trades

def test_load_strategy_summaries_returns_normalized_rows(tmp_path):
    # Create dummy data
    p = tmp_path / "reports/ict_fvg_liquidity/ict_fvg_liquidity_summary.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"strategy": "ict_fvg_liquidity", "year": "2026", "trades": 10, "win_rate": 50.0}]).to_csv(p, index=False)

    p2 = tmp_path / "reports/supertrend_yearly/NSE_NIFTY50-INDEX_supertrend_7_3_yearly_summary.csv"
    p2.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"strategy": "supertrend_yearly", "year": "2026", "trades": 5, "win_rate": 40.0}]).to_csv(p2, index=False)

    summary = load_strategy_summaries(tmp_path)

    assert not summary.empty
    assert {"strategy", "family", "year", "trades", "win_rate", "net_points", "avg_points"}.issubset(summary.columns)
    assert "ict_fvg_liquidity" in set(summary["strategy"])
    assert "supertrend_yearly" in set(summary["strategy"])

def test_load_strategy_trades_reads_ict_trade_logs(tmp_path):
    p = tmp_path / "reports/ict_fvg_liquidity/ict_fvg_liquidity_2026_trades.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"entry_time": "2026-03-25 10:15:00", "exit_time": "2026-03-25 10:30:00", "points_captured": 15.0, "r_multiple": 1.5}]).to_csv(p, index=False)

    trades = load_strategy_trades(tmp_path, "ict_fvg_liquidity", [2026])

    assert not trades.empty
    assert {"entry_time", "exit_time", "points_captured", "r_multiple", "holding_minutes"}.issubset(trades.columns)
    assert int(trades["year"].dropna().iloc[0]) == 2026
