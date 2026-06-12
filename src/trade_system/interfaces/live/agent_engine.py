"""
Agentic Live Engine - The core autonomous loop of the AI trading system.

This engine connects the 'brain' (TradeOrchestrator) with the 'hands' (TradingBroker).
It fetches data, prompts the orchestrator to decide, executes the trades, and
logs them for the self-evolution feedback loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from trade_system.config import Settings
from trade_system.infrastructure.brokers.factory import get_broker_manager
from trade_system.core.ports.broker import OrderRequest, OrderSide, OrderType, BrokerError
from trade_system.core.ports.repository import TradeRepository
from trade_system.application.agent.orchestrator import TradeOrchestrator
from trade_system.application.agent.option_buyer_workflow import OptionBuyerExecutionWorkflow

LOGGER = logging.getLogger(__name__)


class AgenticLiveEngine:
    """
    Continuous execution engine for the AI Agent.
    """

    def __init__(self, orchestrator: TradeOrchestrator, trade_repository: TradeRepository, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.broker_manager = get_broker_manager(self.settings)
        self.broker = self.broker_manager.get_broker()
        self.orchestrator = orchestrator
        self.trade_repository = trade_repository
        self.is_running = False
        self.option_buyer_workflow = OptionBuyerExecutionWorkflow(
            broker=self.broker,
            notifier=getattr(self.orchestrator, "notifier", None)
        )

    async def run_loop(self, interval_seconds: int = 3600):
        """
        Run the autonomous trading loop continuously.
        Default interval is 1 hour between agentic deep-dives.
        """
        LOGGER.info("Starting Agentic Live Engine...")
        self.is_running = True

        if not self.broker.is_authenticated:
            LOGGER.info("Authenticating broker...")
            if not self.broker.authenticate():
                LOGGER.error("Broker authentication failed. Exiting.")
                return

        # Start position monitoring loop in the background
        monitor_task = asyncio.create_task(self._monitor_positions_loop())

        try:
            while self.is_running:
                try:
                    # 1. Fetch live context (simplified for now, Orchestrator fetches internally)
                    LOGGER.info("Fetching market data and invoking AI Agents...")
                    
                    # 2. Agent Decision Making
                    plan = await self.orchestrator.run_session()
                    
                    if plan.is_tradeable_day and plan.all_suggestions():
                        LOGGER.info(f"Agents generated {len(plan.all_suggestions())} trade suggestions.")
                        # 3. Execution
                        await self._execute_plan(plan)
                    else:
                        LOGGER.info("No actionable trades generated or market is non-tradeable.")

                except Exception as e:
                    LOGGER.error(f"Error in agentic loop: {e}", exc_info=True)
                
                if self.is_running:
                    LOGGER.info(f"Sleeping for {interval_seconds} seconds...")
                    await asyncio.sleep(interval_seconds)
        finally:
            self.is_running = False
            LOGGER.info("Cancelling option positions monitoring background task...")
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_positions_loop(self):
        """Background loop to monitor open option positions."""
        LOGGER.info("Starting option positions monitoring background loop...")
        while self.is_running:
            try:
                await self.option_buyer_workflow.manage_open_positions()
            except Exception as e:
                LOGGER.error(f"Error in position monitoring background loop: {e}", exc_info=True)
            await asyncio.sleep(15)
        LOGGER.info("Option positions monitoring background loop stopped.")

    async def _execute_plan(self, plan: Any) -> None:
        """Execute the trade suggestions via the active broker."""
        for suggestion in plan.all_suggestions():
            # Route Nifty or suggestions with option parameters to OptionBuyerExecutionWorkflow
            if suggestion.is_nifty or (suggestion.option_params is not None):
                LOGGER.info(f"Routing Suggestion {suggestion.id} for {suggestion.symbol} to Option Buyer Workflow")
                try:
                    await self.option_buyer_workflow.execute_buy_order(suggestion)
                except Exception as e:
                    LOGGER.error(f"Failed to execute buy order in Option Buyer Workflow: {e}", exc_info=True)
                continue

            LOGGER.info(f"Executing Trade: {suggestion.direction.value} on {suggestion.symbol}")
            
            # Map Agent terminology to Broker API terms
            side = OrderSide.BUY  # Option buyers always buy to enter
            
            req = OrderRequest(
                symbol=suggestion.symbol,
                side=side,
                quantity=suggestion.quantity if hasattr(suggestion, 'quantity') else 1, # Default 1 lot
                order_type=OrderType.MARKET, # Simplified
                price=0.0,
                tag="ai_agent_trade"
            )

            try:
                # Type check if broker supports trading
                if hasattr(self.broker, "place_order"):
                    response = self.broker.place_order(req)
                    LOGGER.info(f"Order placed successfully: {response.order_id}")
                    
                    # Update DB that trade is OPEN
                    self.trade_repository.update_trade_status(suggestion.id, "OPEN", response.average_price)
                else:
                    LOGGER.warning(f"Broker {self.broker.name} does not support live execution. Simulating.")
                    self.trade_repository.update_trade_status(suggestion.id, "OPEN", suggestion.entry_zone_high)
            except BrokerError as e:
                LOGGER.error(f"Failed to place order for {suggestion.symbol}: {e}")

    def stop(self):
        """Stop the autonomous loop."""
        LOGGER.info("Stopping Agentic Live Engine...")
        self.is_running = False


def run_main():
    """CLI entry point for the live agent engine."""
    from trade_system.application.agent.factory import build_orchestrator
    from trade_system.infrastructure.database.connection import get_engine
    from trade_system.infrastructure.database.repository import SQLAlchemyTradeRepository

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Dependency Injection assembly
    orchestrator = build_orchestrator()
    db_engine = get_engine()
    trade_repository = SQLAlchemyTradeRepository(db_engine)

    engine = AgenticLiveEngine(orchestrator=orchestrator, trade_repository=trade_repository)
    
    try:
        asyncio.run(engine.run_loop())
    except KeyboardInterrupt:
        engine.stop()
