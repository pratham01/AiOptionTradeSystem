"""
Unit tests for OptionChainMonitorAgent.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pytest

from trade_system.domains.advisory.application.agent.option_chain_monitor_agent import (
    OptionChainMonitorAgent,
    DynamicForensicResult,
)


@pytest.fixture
def temp_state_file(tmp_path: Path) -> Path:
    return tmp_path / "test_oc_state.json"


@pytest.fixture
def monitor_agent(temp_state_file: Path) -> OptionChainMonitorAgent:
    agent = OptionChainMonitorAgent(
        enable_telegram=False,
        state_file=temp_state_file,
    )
    return agent


def create_mock_oc_dataframe(
    strikes: list[float],
    ce_ois: list[int],
    pe_ois: list[int],
    ce_ltps: list[float] | None = None,
    pe_ltps: list[float] | None = None,
) -> pd.DataFrame:
    records = []
    for i, s in enumerate(strikes):
        records.append({
            "strike": s,
            "option_type": "CE",
            "oi": ce_ois[i],
            "volume": 1000,
            "ltp": ce_ltps[i] if ce_ltps else 100.0,
        })
        records.append({
            "strike": s,
            "option_type": "PE",
            "oi": pe_ois[i],
            "volume": 1000,
            "ltp": pe_ltps[i] if pe_ltps else 100.0,
        })
    return pd.DataFrame(records)


def test_first_snapshot_ingestion(monitor_agent: OptionChainMonitorAgent):
    strikes = [24000.0, 24100.0, 24200.0]
    ce_ois = [10000, 25000, 50000]
    pe_ois = [50000, 25000, 10000]
    df1 = create_mock_oc_dataframe(strikes, ce_ois, pe_ois)

    res = monitor_agent.process_snapshot(
        symbol="NSE:NIFTY50-INDEX",
        oc_df=df1,
        spot_price=24100.0,
        timestamp=datetime(2026, 9, 16, 9, 15),
    )

    assert res is not None
    assert res.spot_price == 24100.0
    assert res.atm_strike == 24100.0
    assert res.ce_wall == 24200.0
    assert res.pe_wall == 24000.0
    assert res.max_pain == 24100.0
    assert res.max_pain_shifted is False
    assert len(res.traps) == 0


def test_max_pain_shift_detection(monitor_agent: OptionChainMonitorAgent):
    strikes = [24000.0, 24100.0, 24200.0, 24300.0]
    t1 = datetime(2026, 9, 16, 9, 15)
    t2 = datetime(2026, 9, 16, 9, 18)

    # Snap 1: Max Pain at 24100
    df1 = create_mock_oc_dataframe(strikes, [5000, 20000, 60000, 80000], [80000, 50000, 20000, 5000])
    res1 = monitor_agent.process_snapshot("NSE:NIFTY50-INDEX", df1, spot_price=24120.0, timestamp=t1)
    assert res1.max_pain == 24100.0

    # Snap 2: Huge Put writing at 24200 shifts Max Pain up to 24200
    df2 = create_mock_oc_dataframe(strikes, [5000, 10000, 30000, 60000], [80000, 80000, 95000, 5000])
    res2 = monitor_agent.process_snapshot("NSE:NIFTY50-INDEX", df2, spot_price=24190.0, timestamp=t2)

    assert res2.max_pain == 24200.0
    assert res2.max_pain_shifted is True
    assert res2.max_pain_shift_pts == 100.0
    assert res2.prev_max_pain == 24100.0
    assert len(res2.signals) >= 1
    assert res2.signals[0]["setup"] == "MAX_PAIN_UPWARD_MIGRATION"
    assert res2.signals[0]["bias"] == "BULLISH"


def test_bull_trap_detection(monitor_agent: OptionChainMonitorAgent):
    strikes = [24000.0, 24100.0, 24200.0]
    t1 = datetime(2026, 9, 16, 10, 0)
    t2 = datetime(2026, 9, 16, 10, 3)

    # Snap 1: CE Wall at 24200 with 40,000 OI
    df1 = create_mock_oc_dataframe(strikes, [10000, 20000, 40000], [40000, 20000, 10000])
    monitor_agent.process_snapshot("NSE:NIFTY50-INDEX", df1, spot_price=24150.0, timestamp=t1)

    # Snap 2: Spot tests 24200, but Call writers ADD +25,000 contracts (defending resistance)
    df2 = create_mock_oc_dataframe(strikes, [10000, 20000, 65000], [40000, 20000, 10000])
    res2 = monitor_agent.process_snapshot("NSE:NIFTY50-INDEX", df2, spot_price=24205.0, timestamp=t2)

    assert len(res2.traps) >= 1
    assert "BULL TRAP" in res2.traps[0]


def test_state_file_persistence(monitor_agent: OptionChainMonitorAgent, temp_state_file: Path):
    strikes = [24000.0, 24100.0]
    df = create_mock_oc_dataframe(strikes, [10000, 30000], [30000, 10000])
    monitor_agent.process_snapshot("NSE:NIFTY50-INDEX", df, spot_price=24050.0, timestamp=datetime(2026, 9, 16, 11, 0))

    assert temp_state_file.exists()
    with open(temp_state_file) as f:
        data = json.load(f)
    assert "NIFTY50" in data
    assert data["NIFTY50"]["spot_price"] == 24050.0
