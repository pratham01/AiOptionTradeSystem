import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, time as dt_time
import json
import pandas as pd
from pathlib import Path

from trade_system.shared import TradeSuggestion, TradeDirection, TradeHorizon, OptionParams
from trade_system.domains.trading.domain.ports.broker import OrderSide
from trade_system.domains.advisory.application.agent.option_buyer_workflow import OptionBuyerExecutionWorkflow
from trade_system.domains.market_data.infrastructure.database.models import Base, SuggestedTrade
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager


class MockMarketQuote:
    def __init__(self, last_price, close=None, open=None):
        self.last_price = last_price
        self.close = close
        self.open = open


@pytest.fixture(name="db_session")
def fixture_db_session():
    # Setup in-memory sqlite database for the test run
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    
    @contextmanager
    def mock_get_db_context():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            
    with patch("trade_system.domains.advisory.application.agent.option_buyer_workflow.get_db_context", mock_get_db_context):
        yield Session


@pytest.fixture(name="temp_positions_file")
def fixture_temp_positions_file(tmp_path: Path):
    return str(tmp_path / "active_option_positions.json")


@pytest.fixture(name="mock_broker")
def fixture_mock_broker():
    broker = MagicMock()
    broker.name = "MockBroker"
    
    # Mock place_order return dynamically based on side
    def mock_place_order(order_req):
        res = MagicMock()
        res.order_id = "ORDER123"
        if order_req.side == OrderSide.SELL:
            res.average_price = None  # Force EOD / Exit logic to fallback to ltp
        else:
            res.average_price = 100.0
        return res
    broker.place_order.side_effect = mock_place_order
    
    # Mock get_quotes response
    broker.get_quotes.return_value = {
        "NSE:NIFTY2661822000CE": MockMarketQuote(last_price=100.0),
        "NSE:NIFTY2661822000PE": MockMarketQuote(last_price=80.0),
        "NSE:NIFTY2661822100CE": MockMarketQuote(last_price=70.0),
    }
    return broker


@pytest.mark.asyncio
async def test_resolve_option_contract_symbol(mock_broker, temp_positions_file):
    workflow = OptionBuyerExecutionWorkflow(broker=mock_broker, positions_file=temp_positions_file)
    
    with patch("trade_system.domains.advisory.application.agent.option_buyer_workflow.OptionChainAnalyzer") as MockAnalyzer:
        mock_analyzer_instance = MockAnalyzer.return_value
        
        # Exact and fallback matching data
        mock_analyzer_instance.get_option_chain_df.return_value = pd.DataFrame([
            {"strike": 22000, "expiry": "2026-06-18", "option_type": "CE", "symbol": "NSE:NIFTY2661822000CE"},
            {"strike": 22100, "expiry": "2026-06-18", "option_type": "CE", "symbol": "NSE:NIFTY2661822100CE"},
            {"strike": 22000, "expiry": "2026-06-18", "option_type": "PE", "symbol": "NSE:NIFTY2661822000PE"},
        ])
        
        # Test Exact match
        resolved_ce = workflow.resolve_option_contract_symbol("NSE:NIFTY50-INDEX", 22000, "CE")
        assert resolved_ce == "NSE:NIFTY2661822000CE"
        
        # Test Fallback to nearest
        resolved_fallback = workflow.resolve_option_contract_symbol("NSE:NIFTY50-INDEX", 22060, "CE")
        assert resolved_fallback == "NSE:NIFTY2661822100CE"


@pytest.mark.asyncio
async def test_execute_buy_order(mock_broker, temp_positions_file, db_session):
    workflow = OptionBuyerExecutionWorkflow(broker=mock_broker, positions_file=temp_positions_file)
    
    # Setup trade suggestion
    suggestion = TradeSuggestion(
        id="trade_1",
        symbol="NSE:NIFTY50-INDEX",
        timestamp=datetime.now(),
        direction=TradeDirection.CALL,
        horizon=TradeHorizon.INTRADAY,
        entry_zone_low=21950,
        entry_zone_high=22050,
        target=22200,
        stop_loss=21800,
        option_params=OptionParams(direction=TradeDirection.CALL, suggested_strike=22000),
        confidence=0.8,
        narrative="Bullish Supertrend",
        setup_features=None
    )
    
    # Seed SuggestedTrade in mocked database first so update_db_trade works
    session = db_session()
    db_trade = SuggestedTrade(
        id="trade_1",
        date=datetime.now().strftime("%Y-%m-%d"),
        symbol="NSE:NIFTY50-INDEX",
        direction="CALL",
        horizon="INTRADAY",
        entry_zone_low=21950,
        entry_zone_high=22050,
        target=22200,
        stop_loss=21800,
        confidence=0.8,
        outcome="PENDING"
    )
    session.add(db_trade)
    session.commit()
    session.close()

    with patch("trade_system.domains.advisory.application.agent.option_buyer_workflow.OptionChainAnalyzer") as MockAnalyzer:
        mock_analyzer_instance = MockAnalyzer.return_value
        mock_analyzer_instance.get_option_chain_df.return_value = pd.DataFrame([
            {"strike": 22000, "expiry": "2026-06-18", "option_type": "CE", "symbol": "NSE:NIFTY2661822000CE"},
        ])
        
        pos = await workflow.execute_buy_order(suggestion)
        assert pos is not None
        assert pos["option_symbol"] == "NSE:NIFTY2661822000CE"
        assert pos["entry_premium"] == 100.0
        assert pos["sl_premium"] == 70.0    # 30% SL
        assert pos["target_premium"] == 160.0  # 60% TP
        assert pos["status"] == "OPEN"
        
        # Verify JSON file has active position
        active_positions = workflow.load_active_positions()
        assert len(active_positions) == 1
        assert active_positions[0]["trade_id"] == "trade_1"
        
        # Verify Database has status updated to OPEN
        session = db_session()
        db_record = session.query(SuggestedTrade).filter_by(id="trade_1").first()
        assert db_record.outcome == "OPEN"
        assert db_record.actual_entry == 100.0
        session.close()


@pytest.mark.asyncio
async def test_manage_open_positions_stop_loss_hit(mock_broker, temp_positions_file, db_session):
    workflow = OptionBuyerExecutionWorkflow(broker=mock_broker, positions_file=temp_positions_file)
    
    # Save open position to JSON
    open_pos = {
        "trade_id": "trade_sl_hit",
        "suggestion_symbol": "NSE:NIFTY50-INDEX",
        "option_symbol": "NSE:NIFTY2661822000CE",
        "direction": "CALL",
        "strike": 22000,
        "quantity": 50,
        "entry_premium": 100.0,
        "current_premium": 100.0,
        "sl_premium": 70.0,
        "target_premium": 160.0,
        "peak_premium": 100.0,
        "order_id": "ORDER123",
        "status": "OPEN",
        "entered_at": datetime.now().isoformat()
    }
    workflow.save_active_positions([open_pos])
    
    # Seed db
    session = db_session()
    db_trade = SuggestedTrade(
        id="trade_sl_hit",
        date=datetime.now().strftime("%Y-%m-%d"),
        symbol="NSE:NIFTY50-INDEX",
        direction="CALL",
        horizon="INTRADAY",
        entry_zone_low=21950,
        entry_zone_high=22050,
        target=22200,
        stop_loss=21800,
        confidence=0.8,
        outcome="OPEN",
        actual_entry=100.0
    )
    session.add(db_trade)
    session.commit()
    session.close()

    # Mock quotes returning low price (SL hit)
    mock_broker.get_quotes.return_value = {
        "NSE:NIFTY2661822000CE": MockMarketQuote(last_price=65.0)
    }
    
    # Mock datetime to standard morning trading hours
    with patch("trade_system.domains.advisory.application.agent.option_buyer_workflow.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 6, 12, 10, 30)
        
        await workflow.manage_open_positions()
        
        # Check position is removed from active positions
        assert len(workflow.load_active_positions()) == 0
        
        # Check DB updated to LOSS
        session = db_session()
        db_record = session.query(SuggestedTrade).filter_by(id="trade_sl_hit").first()
        assert db_record.outcome == "LOSS"
        assert db_record.actual_exit == 65.0
        session.close()


@pytest.mark.asyncio
async def test_manage_open_positions_target_hit(mock_broker, temp_positions_file, db_session):
    workflow = OptionBuyerExecutionWorkflow(broker=mock_broker, positions_file=temp_positions_file)
    
    # Save open position to JSON
    open_pos = {
        "trade_id": "trade_target_hit",
        "suggestion_symbol": "NSE:NIFTY50-INDEX",
        "option_symbol": "NSE:NIFTY2661822000CE",
        "direction": "CALL",
        "strike": 22000,
        "quantity": 50,
        "entry_premium": 100.0,
        "current_premium": 100.0,
        "sl_premium": 70.0,
        "target_premium": 160.0,
        "peak_premium": 100.0,
        "order_id": "ORDER123",
        "status": "OPEN",
        "entered_at": datetime.now().isoformat()
    }
    workflow.save_active_positions([open_pos])
    
    # Seed db
    session = db_session()
    db_trade = SuggestedTrade(
        id="trade_target_hit",
        date=datetime.now().strftime("%Y-%m-%d"),
        symbol="NSE:NIFTY50-INDEX",
        direction="CALL",
        horizon="INTRADAY",
        entry_zone_low=21950,
        entry_zone_high=22050,
        target=22200,
        stop_loss=21800,
        confidence=0.8,
        outcome="OPEN",
        actual_entry=100.0
    )
    session.add(db_trade)
    session.commit()
    session.close()

    # Mock quotes returning high price (Target hit)
    mock_broker.get_quotes.return_value = {
        "NSE:NIFTY2661822000CE": MockMarketQuote(last_price=175.0)
    }
    
    with patch("trade_system.domains.advisory.application.agent.option_buyer_workflow.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 6, 12, 10, 30)
        
        await workflow.manage_open_positions()
        
        assert len(workflow.load_active_positions()) == 0
        
        session = db_session()
        db_record = session.query(SuggestedTrade).filter_by(id="trade_target_hit").first()
        assert db_record.outcome == "WIN"
        assert db_record.actual_exit == 175.0
        session.close()


@pytest.mark.asyncio
async def test_manage_open_positions_trailing_sl_activation_and_lock(mock_broker, temp_positions_file, db_session):
    workflow = OptionBuyerExecutionWorkflow(
        broker=mock_broker, 
        positions_file=temp_positions_file,
        trailing_activation=0.20,
        trailing_lock_pct=0.50
    )
    
    # Save open position to JSON
    open_pos = {
        "trade_id": "trade_trail",
        "suggestion_symbol": "NSE:NIFTY50-INDEX",
        "option_symbol": "NSE:NIFTY2661822000CE",
        "direction": "CALL",
        "strike": 22000,
        "quantity": 50,
        "entry_premium": 100.0,
        "current_premium": 100.0,
        "sl_premium": 70.0,
        "target_premium": 160.0,
        "peak_premium": 100.0,
        "order_id": "ORDER123",
        "status": "OPEN",
        "entered_at": datetime.now().isoformat()
    }
    workflow.save_active_positions([open_pos])
    
    # 1. Price increases to 125 (+25% gain, triggers trailing SL activation to Breakeven = 100, then trailing SL to 112.5)
    mock_broker.get_quotes.return_value = {
        "NSE:NIFTY2661822000CE": MockMarketQuote(last_price=125.0)
    }
    
    with patch("trade_system.domains.advisory.application.agent.option_buyer_workflow.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 6, 12, 10, 30)
        
        await workflow.manage_open_positions()
        
        # Verify SL has trailed to 112.5 (Breakeven 100 + 25 * 0.50)
        active_positions = workflow.load_active_positions()
        assert len(active_positions) == 1
        assert active_positions[0]["sl_premium"] == 112.5
        assert active_positions[0]["peak_premium"] == 125.0
        
        # 2. Price increases to 150 (peak_gain = 50. Trailing lock-in: Breakeven 100 + 50 * 50% = 125)
        mock_broker.get_quotes.return_value = {
            "NSE:NIFTY2661822000CE": MockMarketQuote(last_price=150.0)
        }
        
        await workflow.manage_open_positions()
        
        active_positions = workflow.load_active_positions()
        assert len(active_positions) == 1
        assert active_positions[0]["sl_premium"] == 125.0
        assert active_positions[0]["peak_premium"] == 150.0


@pytest.mark.asyncio
async def test_manage_open_positions_eod_squareoff(mock_broker, temp_positions_file, db_session):
    workflow = OptionBuyerExecutionWorkflow(broker=mock_broker, positions_file=temp_positions_file)
    
    # Save open position to JSON
    open_pos = {
        "trade_id": "trade_eod",
        "suggestion_symbol": "NSE:NIFTY50-INDEX",
        "option_symbol": "NSE:NIFTY2661822000CE",
        "direction": "CALL",
        "strike": 22000,
        "quantity": 50,
        "entry_premium": 100.0,
        "current_premium": 100.0,
        "sl_premium": 70.0,
        "target_premium": 160.0,
        "peak_premium": 100.0,
        "order_id": "ORDER123",
        "status": "OPEN",
        "entered_at": datetime.now().isoformat()
    }
    workflow.save_active_positions([open_pos])
    
    # Seed db
    session = db_session()
    db_trade = SuggestedTrade(
        id="trade_eod",
        date=datetime.now().strftime("%Y-%m-%d"),
        symbol="NSE:NIFTY50-INDEX",
        direction="CALL",
        horizon="INTRADAY",
        entry_zone_low=21950,
        entry_zone_high=22050,
        target=22200,
        stop_loss=21800,
        confidence=0.8,
        outcome="OPEN",
        actual_entry=100.0
    )
    session.add(db_trade)
    session.commit()
    session.close()

    # Mock quotes returning average price (no SL/target hit)
    mock_broker.get_quotes.return_value = {
        "NSE:NIFTY2661822000CE": MockMarketQuote(last_price=110.0)
    }
    
    # Mock datetime to 3:16 PM (EOD cutoff is 3:15 PM)
    with patch("trade_system.domains.advisory.application.agent.option_buyer_workflow.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 6, 12, 15, 16)
        
        await workflow.manage_open_positions()
        
        # Verify position is closed
        assert len(workflow.load_active_positions()) == 0
        
        # Verify DB updated to NEUTRAL
        session = db_session()
        db_record = session.query(SuggestedTrade).filter_by(id="trade_eod").first()
        assert db_record.outcome == "NEUTRAL"
        assert db_record.actual_exit == 110.0
        session.close()
