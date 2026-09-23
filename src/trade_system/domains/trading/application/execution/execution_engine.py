"""
Execution Engine — Handles live order routing, sizing, and risk enforcement.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from trade_system.domains.trading.infrastructure.brokers.legacy.base import BaseBroker
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager
from trade_system.shared import Signal, TradeDirection
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, broker: BaseBroker, risk_manager: RiskManager, settings: Optional[Settings] = None):
        self.broker = broker
        self.risk_manager = risk_manager
        self.settings = settings or Settings.load()
        self.dry_run = getattr(self.settings, 'dry_run_execution', True) # Default to dry run for safety
        
    def execute_signal(self, signal: Signal) -> Optional[Dict[str, Any]]:
        """
        Takes a verified signal, performs final risk/margin checks, calculates size, and routes to broker.
        """
        LOGGER.info("ExecutionEngine received signal: %s %s at %s", signal.direction.value, signal.symbol, signal.price)
        
        # 1. Fetch available funds
        funds = self._get_available_margin()
        if funds <= 0:
            LOGGER.warning("Execution aborted: No available margin.")
            return None
            
        # 2. Final Risk Assessment
        context = {
            "account_value": funds,
            "volatility": 0.01 # Placeholder, could be dynamic
        }
        assessment = self.risk_manager.assess(signal, context)
        
        if not assessment.approved:
            LOGGER.warning("Execution aborted by RiskManager: %s", assessment.reason)
            return None
            
        quantity = assessment.position_size
        if quantity <= 0:
            LOGGER.warning("Execution aborted: Calculated position size is 0.")
            return None
            
        # 3. Prepare Order Payload
        side = 1 if signal.direction in (TradeDirection.LONG, TradeDirection.CALL) or getattr(signal.direction, "value", "") in ("LONG", "CALL", "BUY") else -1
        order_type = 2 # 2 = Limit Order (1 = Market, but limit is safer. Can parameterize)
        product_type = "INTRADAY" # Fyers specific
        
        price = signal.price
        # In live scenarios we might want to cross the spread slightly for limit orders,
        # but for now we use the signal price directly.
        
        LOGGER.info("Executing Trade: %s %s | Qty: %s | Price: %s | Dry Run: %s",
                    "BUY" if side == 1 else "SELL", signal.symbol, quantity, price, self.dry_run)
                    
        if self.dry_run:
            LOGGER.info("[DRY RUN] Would have placed order to broker.")
            # Record it as open to the risk manager so it thinks we are trading
            self.risk_manager.on_trade_open()
            return {
                "s": "ok",
                "message": "[DRY RUN] Order simulated.",
                "id": "dry_run_id_12345",
                "symbol": signal.symbol,
                "qty": quantity,
                "price": price,
                "side": side
            }
            
        # Live Execution
        response = self.broker.place_order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price,
            stoploss=0.0 # We manage SL manually in PositionManager for trailing
        )
        
        if response.get("s") == "ok":
            self.risk_manager.on_trade_open()
            LOGGER.info("Order successfully placed. ID: %s", response.get("id"))
        else:
            LOGGER.error("Order placement failed: %s", response)
            
        return response

    def _get_available_margin(self) -> float:
        if self.dry_run:
            return 100000.0 # Simulate 1L margin
            
        try:
            funds = self.broker.get_funds()
            return float(funds.get("Available Balance", funds.get("Limit", 0.0)))
        except Exception as e:
            LOGGER.error("Failed to fetch funds: %s", e)
            return 0.0
