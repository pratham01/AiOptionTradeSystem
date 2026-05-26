from trade_system.config import Settings
from trade_system.infrastructure.brokers.factory import get_broker_manager
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.repository import SQLAlchemyLogRepository
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.application.agent.orchestrator import TradeOrchestrator
from trade_system.application.agent.premarket_news_agent import PreMarketNewsAgent
from trade_system.application.agent.market_context_agent import MarketContextAgent
from trade_system.application.agent.option_chain_agent import OptionChainAgent
from trade_system.application.agent.nifty_option_buyer_agent import NiftyOptionBuyerAgent
from trade_system.application.agent.fo_stock_suggester_agent import FoStockSuggesterAgent
from trade_system.application.agent.setup_validator_agent import SetupValidatorAgent
from trade_system.application.agent.skill_registry import SkillRegistry
from trade_system.application.evolution.trade_logger import TradeLogger

from trade_system.core.events.bus import EventBus

def build_orchestrator(settings: Settings | None = None) -> TradeOrchestrator:
    """Factory to assemble and inject dependencies into TradeOrchestrator."""
    settings = settings or Settings.load()
    
    broker_manager = get_broker_manager(settings)
    broker = broker_manager.get_broker()
    
    db_engine = get_engine()
    log_repository = SQLAlchemyLogRepository(db_engine)
    
    event_bus = EventBus()
    
    notifier = None
    if settings.telegram_bot_token and settings.telegram_chat_id:
        notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        
    return TradeOrchestrator(
        broker=broker,
        notifier=notifier,
        log_repository=log_repository,
        trade_logger=TradeLogger(),
        skill_registry=SkillRegistry(),
        event_bus=event_bus,
        premarket_agent=PreMarketNewsAgent(),
        market_context_agent=MarketContextAgent(event_bus=event_bus),
        option_chain_agent=OptionChainAgent(broker=broker, event_bus=event_bus),
        nifty_agent=NiftyOptionBuyerAgent(broker=broker, settings=settings),
        fo_suggester_agent=FoStockSuggesterAgent(broker=broker),
        setup_validator=SetupValidatorAgent(broker=broker),
        max_trades_per_day=2,
    )
