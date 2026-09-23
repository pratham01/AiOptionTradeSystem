"""
Position Manager — Tracks live open trades, trails stops, executes exits,
and persists active positions and trade history to shared state.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from trade_system.domains.trading.infrastructure.brokers.legacy.base import BaseBroker
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    symbol: str
    side: int                           # 1 for Long, -1 for Short
    quantity: int
    entry_price: float
    entry_time: datetime
    current_stop_loss: float
    take_profit: float                  # Final target (or Target 2)
    highest_price_reached: float = 0.0  # For longs
    lowest_price_reached: float = float("inf")  # For shorts
    order_id: str = "sim_0"
    strategy_name: str = "MANUAL"
    target_1: float = 0.0               # Range-to-range target (TP1)
    target_2: float = 0.0               # Swing target (TP2)
    is_breakeven_locked: bool = False
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "OPEN"                # OPEN, TP1_HIT, CLOSED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "direction": "LONG" if self.side == 1 else "SHORT",
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "entry_time": self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else str(self.entry_time),
            "current_stop_loss": round(self.current_stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "highest_price_reached": round(self.highest_price_reached, 2),
            "lowest_price_reached": round(self.lowest_price_reached, 2) if self.lowest_price_reached != float("inf") else self.entry_price,
            "order_id": self.order_id,
            "strategy_name": self.strategy_name,
            "target_1": round(self.target_1, 2) if self.target_1 else round(self.take_profit, 2),
            "target_2": round(self.target_2, 2) if self.target_2 else round(self.take_profit, 2),
            "is_breakeven_locked": self.is_breakeven_locked,
            "current_price": round(self.current_price, 2) if self.current_price else round(self.entry_price, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenPosition":
        entry_time = data.get("entry_time")
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time)
            except Exception:
                entry_time = datetime.now()
        else:
            entry_time = datetime.now()

        return cls(
            symbol=data["symbol"],
            side=int(data.get("side", 1)),
            quantity=int(data.get("quantity", 1)),
            entry_price=float(data.get("entry_price", 0.0)),
            entry_time=entry_time,
            current_stop_loss=float(data.get("current_stop_loss", 0.0)),
            take_profit=float(data.get("take_profit", 0.0)),
            highest_price_reached=float(data.get("highest_price_reached", data.get("entry_price", 0.0))),
            lowest_price_reached=float(data.get("lowest_price_reached", data.get("entry_price", float("inf")))),
            order_id=str(data.get("order_id", "sim_0")),
            strategy_name=str(data.get("strategy_name", "MANUAL")),
            target_1=float(data.get("target_1", 0.0)),
            target_2=float(data.get("target_2", 0.0)),
            is_breakeven_locked=bool(data.get("is_breakeven_locked", False)),
            current_price=float(data.get("current_price", data.get("entry_price", 0.0))),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            status=str(data.get("status", "OPEN")),
            metadata=data.get("metadata", {}),
        )


class PositionManager:
    def __init__(self, broker: BaseBroker, risk_manager: RiskManager, settings: Optional[Settings] = None):
        self.broker = broker
        self.risk_manager = risk_manager
        self.settings = settings or Settings.load()
        self.dry_run = getattr(self.settings, 'dry_run_execution', True)
        
        # State file path
        self.state_file = self.settings.data_dir / "autonomous_trades.json"
        
        # In-memory tracking
        self.active_positions: Dict[str, OpenPosition] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Load persisted state if exists
        self.load_state()
        
    def load_state(self) -> None:
        """Load open positions and trade history from disk."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                payload = json.load(f)
            
            # Load active positions
            loaded_active = {}
            for p_dict in payload.get("active_positions", []):
                pos = OpenPosition.from_dict(p_dict)
                loaded_active[pos.symbol] = pos
            self.active_positions = loaded_active
            
            # Load trade history
            self.trade_history = payload.get("trade_history", [])
            LOGGER.info(
                "PositionManager restored %d active positions and %d historical trades from %s",
                len(self.active_positions), len(self.trade_history), self.state_file
            )
        except Exception as exc:
            LOGGER.warning("Could not load state from %s: %s", self.state_file, exc)

    def save_state(self) -> None:
        """Atomically persist open positions and trade history to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.state_file.with_suffix(".tmp")
            
            data = {
                "last_updated": datetime.now().isoformat(),
                "dry_run": self.dry_run,
                "active_positions": [pos.to_dict() for pos in self.active_positions.values()],
                "trade_history": self.trade_history[-200:], # keep latest 200 trades
            }
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as exc:
            LOGGER.error("Failed to persist autonomous trade state to %s: %s", self.state_file, exc)

    def add_position(self, pos: OpenPosition) -> None:
        if pos.highest_price_reached <= 0.0 and pos.side == 1:
            pos.highest_price_reached = pos.entry_price
        if pos.lowest_price_reached == float("inf") and pos.side == -1:
            pos.lowest_price_reached = pos.entry_price
        if pos.current_price <= 0.0:
            pos.current_price = pos.entry_price
        if pos.target_2 <= 0.0 and pos.take_profit > 0.0:
            pos.target_2 = pos.take_profit

        self.active_positions[pos.symbol] = pos
        self.save_state()
        LOGGER.info(
            "Tracking new position [%s]: %s %s @ %.2f (SL: %.2f, TP1: %.2f, TP2: %.2f)",
            pos.strategy_name,
            "LONG" if pos.side == 1 else "SHORT",
            pos.symbol,
            pos.entry_price,
            pos.current_stop_loss,
            pos.target_1,
            pos.target_2,
        )

    def process_tick(self, symbol: str, current_price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Called on every tick from the websocket or bar completion to evaluate
        dynamic trailing stops and exit triggers.
        """
        if symbol not in self.active_positions:
            return
            
        pos = self.active_positions[symbol]
        pos.current_price = current_price
        
        # Calculate live unrealized PnL
        if pos.side == 1:
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            pos.highest_price_reached = max(pos.highest_price_reached, current_price)
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
            pos.lowest_price_reached = min(pos.lowest_price_reached, current_price)
            
        # 1. Evaluate Trailing Stop Loss & Breakeven Lock
        self._evaluate_trailing_stop(pos, current_price)
            
        # 2. Evaluate Exits (SL hit or final TP hit)
        exit_reason = None
        if pos.side == 1:
            if current_price <= pos.current_stop_loss:
                exit_reason = "BREAKEVEN_STOPPED" if pos.is_breakeven_locked else "STOP_LOSS_HIT"
            elif pos.target_2 > 0 and current_price >= pos.target_2:
                exit_reason = "TARGET_2_HIT"
            elif pos.take_profit > 0 and current_price >= pos.take_profit:
                exit_reason = "TAKE_PROFIT_HIT"
        elif pos.side == -1:
            if current_price >= pos.current_stop_loss:
                exit_reason = "BREAKEVEN_STOPPED" if pos.is_breakeven_locked else "STOP_LOSS_HIT"
            elif pos.target_2 > 0 and current_price <= pos.target_2:
                exit_reason = "TARGET_2_HIT"
            elif pos.take_profit > 0 and current_price <= pos.take_profit:
                exit_reason = "TAKE_PROFIT_HIT"
                
        if exit_reason:
            LOGGER.info(
                "EXIT CONDITION MET: %s for %s [%s] at %.2f. Reason: %s",
                "LONG" if pos.side == 1 else "SHORT",
                pos.symbol,
                pos.strategy_name,
                current_price,
                exit_reason,
            )
            self._execute_exit(pos, current_price, exit_reason)
        else:
            # Periodically sync state on price movement
            self.save_state()

    def _evaluate_trailing_stop(self, pos: OpenPosition, current_price: float) -> None:
        """
        Two-stage institutional trailing stop:
        1. Range Target (TP1) Reached -> Move Stop Loss to Breakeven (+0.1% fee buffer)
           and mark is_breakeven_locked = True.
        2. If beyond TP1, trail stop aggressively to lock in 50% of peak gains.
        """
        if pos.side == 1:
            # Check TP1 milestone
            if pos.target_1 > pos.entry_price and current_price >= pos.target_1 and not pos.is_breakeven_locked:
                be_sl = round(pos.entry_price * 1.001, 2)
                if be_sl > pos.current_stop_loss:
                    pos.current_stop_loss = be_sl
                    pos.is_breakeven_locked = True
                    pos.status = "TP1_HIT"
                    LOGGER.info("🎯 TP1 HIT for %s: SL moved to Breakeven (%.2f)", pos.symbol, be_sl)

            # Trailing stop logic for longs
            target_ref = pos.target_2 if pos.target_2 > 0 else pos.take_profit
            target_distance = target_ref - pos.entry_price
            if target_distance > 0 and pos.highest_price_reached >= pos.entry_price + (target_distance * 0.5):
                half_gain_sl = round(pos.entry_price + ((pos.highest_price_reached - pos.entry_price) * 0.5), 2)
                if half_gain_sl > pos.current_stop_loss:
                    pos.current_stop_loss = half_gain_sl
                    LOGGER.info("Trailing SL updated for %s to %.2f (Trailing 50%% Gain)", pos.symbol, half_gain_sl)
                    
        elif pos.side == -1:
            # Check TP1 milestone
            if pos.target_1 > 0 and pos.target_1 < pos.entry_price and current_price <= pos.target_1 and not pos.is_breakeven_locked:
                be_sl = round(pos.entry_price * 0.999, 2)
                if be_sl < pos.current_stop_loss:
                    pos.current_stop_loss = be_sl
                    pos.is_breakeven_locked = True
                    pos.status = "TP1_HIT"
                    LOGGER.info("🎯 TP1 HIT for %s: SL moved to Breakeven (%.2f)", pos.symbol, be_sl)

            # Trailing stop logic for shorts
            target_ref = pos.target_2 if pos.target_2 > 0 else pos.take_profit
            target_distance = pos.entry_price - target_ref
            if target_distance > 0 and pos.lowest_price_reached <= pos.entry_price - (target_distance * 0.5):
                half_gain_sl = round(pos.entry_price - ((pos.entry_price - pos.lowest_price_reached) * 0.5), 2)
                if half_gain_sl < pos.current_stop_loss:
                    pos.current_stop_loss = half_gain_sl
                    LOGGER.info("Trailing SL updated for %s to %.2f (Trailing 50%% Gain)", pos.symbol, half_gain_sl)

    def manual_close_position(self, symbol: str, exit_price: Optional[float] = None, reason: str = "MANUAL_EXIT") -> Optional[Dict[str, Any]]:
        """Manual override from dashboard to close an open position."""
        if symbol not in self.active_positions:
            LOGGER.warning("Cannot manually close position: %s not found in active positions.", symbol)
            return None
        pos = self.active_positions[symbol]
        actual_price = exit_price or pos.current_price or pos.entry_price
        return self._execute_exit(pos, actual_price, reason)

    def flatten_all_positions(self, reason: str = "EMERGENCY_PANIC_FLATTEN") -> List[Dict[str, Any]]:
        """Emergency kill switch: Closes all currently open positions."""
        closed_records = []
        symbols = list(self.active_positions.keys())
        for sym in symbols:
            pos = self.active_positions.get(sym)
            if pos:
                res = self._execute_exit(pos, pos.current_price or pos.entry_price, reason)
                if res:
                    closed_records.append(res)
        LOGGER.info("Flattened %d active positions. Reason: %s", len(closed_records), reason)
        return closed_records

    def _execute_exit(self, pos: OpenPosition, exit_price: float, reason: str) -> Dict[str, Any]:
        """
        Executes order to close the position, updates risk metrics, records trade history, and persists state.
        """
        exit_side = -1 if pos.side == 1 else 1
        
        if self.dry_run:
            LOGGER.info("[DRY RUN] Executed Exit for %s at %.2f. Reason: %s", pos.symbol, exit_price, reason)
            order_resp = {"s": "ok", "order_id": f"sim_exit_{datetime.now().strftime('%H%M%S')}"}
        else:
            try:
                order_resp = self.broker.place_order(
                    symbol=pos.symbol,
                    side=exit_side,
                    quantity=pos.quantity,
                    order_type=1, # 1 = Market Order
                    product_type="INTRADAY"
                )
                if order_resp.get("s") == "ok":
                    LOGGER.info("Exit order placed successfully. ID: %s", order_resp.get("id"))
                else:
                    LOGGER.error("Failed to place exit order: %s", order_resp)
            except Exception as exc:
                LOGGER.error("Broker order exception during exit for %s: %s", pos.symbol, exc)
                order_resp = {"s": "error", "error": str(exc)}
                
        # Calculate Realized PnL
        if pos.side == 1:
            pnl = (exit_price - pos.entry_price) * pos.quantity
            pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100.0 if pos.entry_price > 0 else 0.0
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
            pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100.0 if pos.entry_price > 0 else 0.0
            
        self.risk_manager.update_daily_pnl(pnl)
        self.risk_manager.on_trade_close()
        
        # Create trade history record
        trade_record = {
            "symbol": pos.symbol,
            "strategy": pos.strategy_name,
            "side": "LONG" if pos.side == 1 else "SHORT",
            "quantity": pos.quantity,
            "entry_price": round(pos.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "entry_time": pos.entry_time.isoformat() if isinstance(pos.entry_time, datetime) else str(pos.entry_time),
            "exit_time": datetime.now().isoformat(),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "exit_reason": reason,
            "order_id": pos.order_id,
            "is_win": pnl > 0,
        }
        self.trade_history.append(trade_record)
        
        # Remove from active tracking
        if pos.symbol in self.active_positions:
            del self.active_positions[pos.symbol]
            
        self.save_state()
        LOGGER.info("Closed position %s [%s]. Realized PnL: ₹%.2f (%.2f%%). Reason: %s",
                    pos.symbol, pos.strategy_name, pnl, pnl_pct, reason)
        return trade_record
