"""
Position Manager — Tracks live open trades, trails stops, and executes exits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from trade_system.domains.trading.infrastructure.brokers.legacy.base import BaseBroker
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)

@dataclass
class OpenPosition:
    symbol: str
    side: int # 1 for Long, -1 for Short
    quantity: int
    entry_price: float
    entry_time: datetime
    current_stop_loss: float
    take_profit: float
    highest_price_reached: float # For longs
    lowest_price_reached: float  # For shorts
    order_id: str

class PositionManager:
    def __init__(self, broker: BaseBroker, risk_manager: RiskManager, settings: Optional[Settings] = None):
        self.broker = broker
        self.risk_manager = risk_manager
        self.settings = settings or Settings.load()
        self.dry_run = getattr(self.settings, 'dry_run_execution', True)
        
        # In-memory tracking of positions we initiated
        self.active_positions: Dict[str, OpenPosition] = {}
        
    def add_position(self, pos: OpenPosition) -> None:
        self.active_positions[pos.symbol] = pos
        LOGGER.info("Tracking new position: %s %s @ %s (SL: %s, TP: %s)",
                    "LONG" if pos.side == 1 else "SHORT", pos.symbol, pos.entry_price, pos.current_stop_loss, pos.take_profit)

    def process_tick(self, symbol: str, current_price: float, timestamp: datetime) -> None:
        """
        Called on every tick from the websocket to evaluate exit conditions.
        """
        if symbol not in self.active_positions:
            return
            
        pos = self.active_positions[symbol]
        
        # Update extremes for trailing stop logic
        if pos.side == 1:
            pos.highest_price_reached = max(pos.highest_price_reached, current_price)
        else:
            pos.lowest_price_reached = min(pos.lowest_price_reached, current_price)
            
        # 1. Evaluate Trailing Stop Loss Updates
        self._evaluate_trailing_stop(pos, current_price)
            
        # 2. Evaluate Exits (Stop Loss Hit or Take Profit Hit)
        exit_reason = None
        if pos.side == 1:
            if current_price <= pos.current_stop_loss:
                exit_reason = "STOP_LOSS_HIT"
            elif current_price >= pos.take_profit:
                exit_reason = "TAKE_PROFIT_HIT"
        elif pos.side == -1:
            if current_price >= pos.current_stop_loss:
                exit_reason = "STOP_LOSS_HIT"
            elif current_price <= pos.take_profit:
                exit_reason = "TAKE_PROFIT_HIT"
                
        if exit_reason:
            LOGGER.info("EXIT CONDITION MET: %s for %s at %s. Reason: %s",
                        "LONG" if pos.side == 1 else "SHORT", pos.symbol, current_price, exit_reason)
            self._execute_exit(pos, current_price, exit_reason)
            
    def _evaluate_trailing_stop(self, pos: OpenPosition, current_price: float) -> None:
        """
        Trails stop loss automatically.
        Logic: If price reaches 50% of target distance, move SL to breakeven.
        """
        if pos.side == 1:
            target_distance = pos.take_profit - pos.entry_price
            if target_distance <= 0:
                return
                
            # If we've captured 50% of the target, move SL to breakeven + slight buffer
            if pos.highest_price_reached >= pos.entry_price + (target_distance * 0.5):
                new_sl = pos.entry_price * 1.001 # Breakeven + small fee buffer
                if new_sl > pos.current_stop_loss:
                    pos.current_stop_loss = new_sl
                    LOGGER.info("Trailing SL updated for %s to %s (Breakeven)", pos.symbol, new_sl)
                    
        elif pos.side == -1:
            target_distance = pos.entry_price - pos.take_profit
            if target_distance <= 0:
                return
                
            if pos.lowest_price_reached <= pos.entry_price - (target_distance * 0.5):
                new_sl = pos.entry_price * 0.999 # Breakeven - small fee buffer
                if new_sl < pos.current_stop_loss:
                    pos.current_stop_loss = new_sl
                    LOGGER.info("Trailing SL updated for %s to %s (Breakeven)", pos.symbol, new_sl)

    def _execute_exit(self, pos: OpenPosition, exit_price: float, reason: str) -> None:
        """
        Executes the market order to close the position.
        """
        # Close order side is opposite of position side
        exit_side = -1 if pos.side == 1 else 1
        
        if self.dry_run:
            LOGGER.info("[DRY RUN] Executed Exit for %s at %s. Reason: %s", pos.symbol, exit_price, reason)
        else:
            response = self.broker.place_order(
                symbol=pos.symbol,
                side=exit_side,
                quantity=pos.quantity,
                order_type=1, # 1 = Market Order for fast exit
                product_type="INTRADAY"
            )
            
            if response.get("s") == "ok":
                LOGGER.info("Exit order placed successfully. ID: %s", response.get("id"))
            else:
                LOGGER.error("Failed to place exit order: %s", response)
                # We do not pop the position so it keeps retrying on next tick, or we need a retry mechanism.
                # For now, we'll log the error and remove it to avoid spamming the API if it's a hard rejection.
                
        # Calculate approximate PnL for risk manager tracking
        if pos.side == 1:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
            
        self.risk_manager.update_daily_pnl(pnl)
        self.risk_manager.on_trade_close()
        
        # Remove from tracking
        del self.active_positions[pos.symbol]
        LOGGER.info("Closed position %s. Approx PnL: %.2f", pos.symbol, pnl)
