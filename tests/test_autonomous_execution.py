"""
Automated unit tests for Autonomous Execution Router, Position Manager,
and dynamic trailing stop logic.
"""
import json
import pytest
from datetime import datetime
from pathlib import Path

from trade_system.domains.trading.application.execution.position_manager import PositionManager, OpenPosition
from trade_system.domains.trading.application.execution.execution_engine import ExecutionEngine
from trade_system.domains.trading.application.execution.autonomous_router import AutonomousExecutionRouter
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager
from trade_system.domains.strategy.application.strategies.smc_strategy import SmcTradeSetup
from trade_system.domains.analysis.application.analysis.stockmojo_smart_oi_engine import DivergenceSignal
from trade_system.shared.config import Settings


class MockBroker:
    """Mock broker simulating order execution."""
    def __init__(self):
        self.orders = []

    def place_order(self, symbol, side, quantity, order_type, product_type, price=0.0, stoploss=0.0):
        order_dict = {
            "s": "ok",
            "id": f"mock_order_{len(self.orders)+1}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price
        }
        self.orders.append(order_dict)
        return order_dict

    def get_funds(self):
        return {"fund_limit": [{"equityAmount": 100000.0}]}


@pytest.fixture
def clean_test_environment(tmp_path):
    """Sets up an isolated Settings, PositionManager, and Router."""
    settings = Settings.load()
    test_state_file = tmp_path / "autonomous_trades.json"
    
    risk_manager = RiskManager()
    broker = MockBroker()
    
    pos_mgr = PositionManager(broker=broker, risk_manager=risk_manager, settings=settings)
    pos_mgr.state_file = test_state_file
    pos_mgr.active_positions.clear()
    pos_mgr.trade_history.clear()
    
    engine = ExecutionEngine(broker=broker, risk_manager=risk_manager, settings=settings)
    router = AutonomousExecutionRouter(
        execution_engine=engine,
        position_manager=pos_mgr,
        risk_manager=risk_manager,
        settings=settings,
        max_concurrent_positions=3,
        min_smc_confluence=70.0
    )
    return pos_mgr, router, test_state_file


def test_position_manager_persistence(clean_test_environment):
    pos_mgr, _, state_file = clean_test_environment
    
    pos = OpenPosition(
        symbol="NSE:NIFTY50-INDEX",
        side=1,
        quantity=50,
        entry_price=25000.0,
        entry_time=datetime.now(),
        current_stop_loss=24950.0,
        take_profit=25150.0,
        target_1=25075.0,
        target_2=25150.0,
        strategy_name="SMC_INSTITUTIONAL"
    )
    pos_mgr.add_position(pos)
    
    assert pos_mgr.state_file.exists()
    assert len(pos_mgr.active_positions) == 1
    
    # Reload from disk into a fresh PositionManager
    new_pos_mgr = PositionManager(broker=pos_mgr.broker, risk_manager=pos_mgr.risk_manager)
    new_pos_mgr.state_file = state_file
    new_pos_mgr.load_state()
    
    assert "NSE:NIFTY50-INDEX" in new_pos_mgr.active_positions
    reloaded_pos = new_pos_mgr.active_positions["NSE:NIFTY50-INDEX"]
    assert reloaded_pos.entry_price == 25000.0
    assert reloaded_pos.strategy_name == "SMC_INSTITUTIONAL"


def test_two_stage_trailing_stop_breakeven_lock(clean_test_environment):
    pos_mgr, _, _ = clean_test_environment
    
    pos = OpenPosition(
        symbol="NSE:NIFTY50-INDEX",
        side=1,
        quantity=50,
        entry_price=25000.0,
        entry_time=datetime.now(),
        current_stop_loss=24950.0,
        take_profit=25150.0,
        target_1=25075.0,
        target_2=25150.0,
        strategy_name="SMC_INSTITUTIONAL"
    )
    pos_mgr.add_position(pos)
    
    # Price moves upwards to 25040 (not yet TP1)
    pos_mgr.process_tick("NSE:NIFTY50-INDEX", 25040.0)
    assert not pos_mgr.active_positions["NSE:NIFTY50-INDEX"].is_breakeven_locked
    assert pos_mgr.active_positions["NSE:NIFTY50-INDEX"].current_stop_loss == 24950.0
    
    # Price hits TP1 (25080 >= 25075)
    pos_mgr.process_tick("NSE:NIFTY50-INDEX", 25080.0)
    p = pos_mgr.active_positions["NSE:NIFTY50-INDEX"]
    assert p.is_breakeven_locked
    assert p.status == "TP1_HIT"
    # Stop loss should be moved to breakeven + 0.1% buffer = 25025.0
    assert p.current_stop_loss >= 25000.0
    
    # Price reaches TP2 -> Trade should be closed
    pos_mgr.process_tick("NSE:NIFTY50-INDEX", 25155.0)
    assert "NSE:NIFTY50-INDEX" not in pos_mgr.active_positions
    assert len(pos_mgr.trade_history) == 1
    assert pos_mgr.trade_history[0]["exit_reason"] == "TARGET_2_HIT"
    assert pos_mgr.trade_history[0]["is_win"] is True


def test_autonomous_router_smc_setup(clean_test_environment):
    pos_mgr, router, _ = clean_test_environment
    
    # Setup with low confluence (< 70) should be rejected
    low_conf_setup = SmcTradeSetup(
        symbol="NSE:NIFTY50-INDEX",
        action="BUY CALL",
        direction=1,
        setup_type="DECISIONAL_DEMAND",
        confluence_score=60.0,
        entry_price=25000.0,
        stop_loss=24950.0,
        target_1=25075.0,
        target_2=25150.0,
        risk_reward_ratio=3.0,
        is_internal_realigned=True
    )
    res_low = router.route_smc_setup(low_conf_setup)
    assert res_low is None
    assert len(pos_mgr.active_positions) == 0
    
    # High confluence setup (>= 70) should execute
    high_conf_setup = SmcTradeSetup(
        symbol="NSE:NIFTY50-INDEX",
        action="BUY CALL",
        direction=1,
        setup_type="EXTREME_DEMAND",
        confluence_score=85.0,
        entry_price=25000.0,
        stop_loss=24950.0,
        target_1=25075.0,
        target_2=25150.0,
        risk_reward_ratio=3.0,
        is_internal_realigned=True
    )
    res_high = router.route_smc_setup(high_conf_setup)
    assert res_high is not None
    assert res_high["status"] == "EXECUTED"
    assert "NSE:NIFTY50-INDEX" in pos_mgr.active_positions
    assert pos_mgr.active_positions["NSE:NIFTY50-INDEX"].strategy_name == "SMC_INSTITUTIONAL"


def test_autonomous_router_divergence_and_panic_flatten(clean_test_environment):
    pos_mgr, router, _ = clean_test_environment
    
    div_sig = DivergenceSignal(
        timestamp=datetime.now(),
        divergence_type="BEARISH_DIVERGENCE",
        severity="HIGH",
        price_level=25200.0,
        delta_val=-450000,
        summary="Bearish Divergence",
        description="Bearish Divergence / Bull Trap"
    )
    
    res = router.route_divergence_signal("NSE:BANKNIFTY-INDEX", div_sig, current_price=25200.0)
    assert res is not None
    assert "NSE:BANKNIFTY-INDEX" in pos_mgr.active_positions
    pos = pos_mgr.active_positions["NSE:BANKNIFTY-INDEX"]
    assert pos.side == -1 # Short
    assert pos.strategy_name == "SMART_OI_DIVERGENCE"
    
    # Emergency panic flatten
    flushed = router.flatten_all(reason="TEST_PANIC")
    assert len(flushed) == 1
    assert len(pos_mgr.active_positions) == 0
    assert len(pos_mgr.trade_history) == 1
    assert pos_mgr.trade_history[0]["exit_reason"] == "TEST_PANIC"


def test_simulation_sandbox(clean_test_environment):
    pos_mgr, router, _ = clean_test_environment
    
    sim_res = router.simulate_test_trade(
        symbol="NSE:RELIANCE-EQ",
        strategy_name="MIDDAY_BREAKOUT",
        direction=1,
        entry_price=2950.0,
        stop_loss=2930.0,
        target_1=2980.0,
        target_2=3010.0,
        quantity=50
    )
    assert sim_res["status"] == "SIMULATION_SUCCESS"
    assert "NSE:RELIANCE-EQ" in pos_mgr.active_positions
    assert pos_mgr.active_positions["NSE:RELIANCE-EQ"].target_1 == 2980.0
