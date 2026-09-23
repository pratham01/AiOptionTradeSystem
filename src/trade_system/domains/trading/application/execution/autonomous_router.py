"""
Autonomous Execution Router — Wires SMC, Smart OI Divergence, and Midday Breakout
signals into automated order execution, risk validation, and persistent position tracking.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from trade_system.domains.trading.application.execution.execution_engine import ExecutionEngine
from trade_system.domains.trading.application.execution.position_manager import OpenPosition, PositionManager
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager
from trade_system.domains.strategy.application.strategies.smc_strategy import SmcTradeSetup
from trade_system.domains.analysis.application.analysis.stockmojo_smart_oi_engine import DivergenceSignal
from trade_system.shared import Signal, TradeDirection
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)


class AutonomousExecutionRouter:
    """
    Central router that receives high-conviction institutional signals from
    all strategies, evaluates risk rules, routes orders to the execution engine,
    and registers positions for automated trailing-stop lifecycle management.
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        settings: Optional[Settings] = None,
        max_concurrent_positions: int = 5,
        min_smc_confluence: float = 70.0,
    ):
        self.execution_engine = execution_engine
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.settings = settings or Settings.load()
        self.max_concurrent_positions = max_concurrent_positions
        self.min_smc_confluence = min_smc_confluence
        self.is_paused = False

    # -----------------------------------------------------------------------
    # 1. SMC Strategy Wiring (Photon Market Structure & JeaFx S/D)
    # -----------------------------------------------------------------------
    def route_smc_setup(self, setup: SmcTradeSetup, current_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receives an SmcTradeSetup from SmartMoneyConceptStrategy.
        Validates minimum confluence, protected SL presence, and risk limits.
        """
        if self.is_paused:
            LOGGER.info("[Autonomous Router] SMC setup skipped for %s: Router is paused.", setup.symbol)
            return None

        if setup.confluence_score < self.min_smc_confluence:
            LOGGER.debug(
                "[Autonomous Router] SMC setup rejected for %s: Confluence %.1f < %.1f",
                setup.symbol, setup.confluence_score, self.min_smc_confluence
            )
            return None

        # Verify risk / reward
        entry = current_price or setup.entry_price
        sl = setup.stop_loss
        tp1 = setup.target_1 or (entry + (abs(entry - sl) * 1.5) if setup.direction == 1 else entry - (abs(entry - sl) * 1.5))
        tp2 = setup.target_2 or (entry + (abs(entry - sl) * 2.5) if setup.direction == 1 else entry - (abs(entry - sl) * 2.5))

        if not self._check_pre_trade_risk(setup.symbol, entry, sl):
            return None

        # Determine lot / quantity
        quantity = self._calculate_position_size(setup.symbol, entry, sl)
        side = 1 if setup.direction == 1 else -1

        sig = Signal(
            symbol=setup.symbol,
            direction=TradeDirection.LONG if side == 1 else TradeDirection.SHORT,
            confidence=setup.confluence_score / 100.0,
            price=entry,
            metadata={
                "strategy": "SMC_INSTITUTIONAL",
                "setup_type": setup.setup_type,
                "stop_loss": sl,
                "target_1": tp1,
                "target_2": tp2,
                "confluence": setup.confluence_score,
            }
        )

        resp = self.execution_engine.execute_signal(sig)
        if resp and resp.get("s") == "ok":
            order_id = resp.get("id", f"smc_{datetime.now().strftime('%H%M%S')}")
            pos = OpenPosition(
                symbol=setup.symbol,
                side=side,
                quantity=resp.get("qty", quantity),
                entry_price=entry,
                entry_time=datetime.now(),
                current_stop_loss=sl,
                take_profit=tp2,
                target_1=tp1,
                target_2=tp2,
                strategy_name="SMC_INSTITUTIONAL",
                order_id=order_id,
                metadata={
                    "setup_type": setup.setup_type,
                    "confluence": setup.confluence_score,
                    "zone_type": getattr(setup, "zone_classification", "DEMAND" if side == 1 else "SUPPLY"),
                }
            )
            self.position_manager.add_position(pos)
            LOGGER.info(
                "🚀 [Autonomous Execution] SMC %s executed for %s @ %.2f (SL: %.2f, TP1: %.2f, TP2: %.2f)",
                "LONG" if side == 1 else "SHORT", setup.symbol, entry, sl, tp1, tp2
            )
            return {"status": "EXECUTED", "position": pos.to_dict(), "response": resp}
        return None

    # -----------------------------------------------------------------------
    # 2. StockMojo Smart OI Price-Volume Divergence Wiring
    # -----------------------------------------------------------------------
    def route_divergence_signal(
        self,
        symbol: str,
        signal: DivergenceSignal,
        current_price: float,
        sl: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Receives a DivergenceSignal from StockMojoSmartOiEngine.
        - BEARISH_DIVERGENCE (Bull Trap) -> SHORT
        - BULLISH_DIVERGENCE (Institutional Absorption) -> LONG
        """
        if self.is_paused:
            LOGGER.info("[Autonomous Router] Divergence signal skipped for %s: Router is paused.", symbol)
            return None

        div_type_str = getattr(signal, "divergence_type", "") or getattr(signal, "div_type", "")
        is_bullish = "BULLISH" in div_type_str.upper() or "ABSORPTION" in div_type_str.upper()
        side = 1 if is_bullish else -1

        entry = current_price
        # Default SL to 0.35% from entry if not specified
        if sl is None or sl <= 0:
            sl = round(entry * (0.9965 if side == 1 else 1.0035), 2)

        risk = abs(entry - sl)
        tp1 = round(entry + (risk * 1.5) if side == 1 else entry - (risk * 1.5), 2)
        tp2 = round(entry + (risk * 2.5) if side == 1 else entry - (risk * 2.5), 2)

        if not self._check_pre_trade_risk(symbol, entry, sl):
            return None

        quantity = self._calculate_position_size(symbol, entry, sl)

        sig = Signal(
            symbol=symbol,
            direction=TradeDirection.LONG if side == 1 else TradeDirection.SHORT,
            confidence=0.85,
            price=entry,
            metadata={
                "strategy": "SMART_OI_DIVERGENCE",
                "divergence_type": div_type_str,
                "stop_loss": sl,
                "target_1": tp1,
                "target_2": tp2,
                "rationale": getattr(signal, "description", ""),
            }
        )

        resp = self.execution_engine.execute_signal(sig)
        if resp and resp.get("s") == "ok":
            order_id = resp.get("id", f"div_{datetime.now().strftime('%H%M%S')}")
            pos = OpenPosition(
                symbol=symbol,
                side=side,
                quantity=resp.get("qty", quantity),
                entry_price=entry,
                entry_time=datetime.now(),
                current_stop_loss=sl,
                take_profit=tp2,
                target_1=tp1,
                target_2=tp2,
                strategy_name="SMART_OI_DIVERGENCE",
                order_id=order_id,
                metadata={
                    "divergence_type": div_type_str,
                    "delta_val": signal.delta_val,
                    "description": getattr(signal, "description", ""),
                }
            )
            self.position_manager.add_position(pos)
            LOGGER.info(
                "🎯 [Autonomous Execution] Smart OI Divergence %s executed for %s @ %.2f (SL: %.2f, TP1: %.2f, TP2: %.2f)",
                "LONG" if side == 1 else "SHORT", symbol, entry, sl, tp1, tp2
            )
            return {"status": "EXECUTED", "position": pos.to_dict(), "response": resp}
        return None

    # -----------------------------------------------------------------------
    # 3. Midday Breakout Engine Wiring
    # -----------------------------------------------------------------------
    def route_midday_breakout(
        self,
        symbol: str,
        direction: int, # 1 for Long, -1 for Short
        current_price: float,
        coil_range_low: float,
        coil_range_high: float,
        notes: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Receives an intraday breakout setup from MiddayBreakoutEngine.
        """
        if self.is_paused:
            LOGGER.info("[Autonomous Router] Midday breakout skipped for %s: Router is paused.", symbol)
            return None

        entry = current_price
        side = 1 if direction == 1 else -1
        # Stop loss is the opposite side of the coiled consolidation
        sl = round(coil_range_low if side == 1 else coil_range_high, 2)
        risk = abs(entry - sl)
        if risk <= 0:
            risk = entry * 0.005 # 0.5% default buffer

        tp1 = round(entry + (risk * 1.5) if side == 1 else entry - (risk * 1.5), 2)
        tp2 = round(entry + (risk * 2.5) if side == 1 else entry - (risk * 2.5), 2)

        if not self._check_pre_trade_risk(symbol, entry, sl):
            return None

        quantity = self._calculate_position_size(symbol, entry, sl)

        sig = Signal(
            symbol=symbol,
            direction=TradeDirection.LONG if side == 1 else TradeDirection.SHORT,
            confidence=0.80,
            price=entry,
            metadata={
                "strategy": "MIDDAY_BREAKOUT",
                "stop_loss": sl,
                "target_1": tp1,
                "target_2": tp2,
                "notes": notes,
            }
        )

        resp = self.execution_engine.execute_signal(sig)
        if resp and resp.get("s") == "ok":
            order_id = resp.get("id", f"mb_{datetime.now().strftime('%H%M%S')}")
            pos = OpenPosition(
                symbol=symbol,
                side=side,
                quantity=resp.get("qty", quantity),
                entry_price=entry,
                entry_time=datetime.now(),
                current_stop_loss=sl,
                take_profit=tp2,
                target_1=tp1,
                target_2=tp2,
                strategy_name="MIDDAY_BREAKOUT",
                order_id=order_id,
                metadata={
                    "coil_high": coil_range_high,
                    "coil_low": coil_range_low,
                    "notes": notes,
                }
            )
            self.position_manager.add_position(pos)
            LOGGER.info(
                "⚡ [Autonomous Execution] Midday Breakout %s executed for %s @ %.2f (SL: %.2f, TP1: %.2f)",
                "LONG" if side == 1 else "SHORT", symbol, entry, sl, tp1
            )
            return {"status": "EXECUTED", "position": pos.to_dict(), "response": resp}
        return None

    # -----------------------------------------------------------------------
    # 4. Sandbox Test Simulation (1-Click Verification)
    # -----------------------------------------------------------------------
    def simulate_test_trade(
        self,
        symbol: str,
        strategy_name: str,
        direction: int,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        quantity: int = 50,
    ) -> Dict[str, Any]:
        """
        Allows dashboard users and automated tests to instantly simulate a trade
        and verify position tracking, trailing stops, and exits end-to-end.
        """
        # If symbol already open, remove it for testing
        if symbol in self.position_manager.active_positions:
            self.position_manager.manual_close_position(symbol, entry_price, reason="SIMULATION_RESET")

        order_id = f"sim_{strategy_name.lower()[:3]}_{datetime.now().strftime('%H%M%S')}"
        pos = OpenPosition(
            symbol=symbol,
            side=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_stop_loss=stop_loss,
            take_profit=target_2,
            target_1=target_1,
            target_2=target_2,
            strategy_name=strategy_name,
            order_id=order_id,
            metadata={"simulated": True, "created_at": datetime.now().isoformat()}
        )
        self.position_manager.add_position(pos)
        return {"status": "SIMULATION_SUCCESS", "position": pos.to_dict()}

    # -----------------------------------------------------------------------
    # Risk and Safety Checks
    # -----------------------------------------------------------------------
    def _check_pre_trade_risk(self, symbol: str, entry: float, sl: float) -> bool:
        """Enforces max concurrent positions, duplicate avoidance, and valid stop loss."""
        # 1. No duplicate open positions for the same symbol
        if symbol in self.position_manager.active_positions:
            LOGGER.warning("[Autonomous Router] Trade skipped: Position for %s is already active.", symbol)
            return False

        # 2. Max concurrent positions limit
        if len(self.position_manager.active_positions) >= self.max_concurrent_positions:
            LOGGER.warning(
                "[Autonomous Router] Trade skipped: Max concurrent positions (%d) reached.",
                self.max_concurrent_positions
            )
            return False

        # 3. Minimum risk sanity
        if entry <= 0 or sl <= 0 or abs(entry - sl) <= 0.01:
            LOGGER.warning("[Autonomous Router] Trade skipped: Invalid entry/SL prices (Entry: %.2f, SL: %.2f)", entry, sl)
            return False

        return True

    def _calculate_position_size(self, symbol: str, entry: float, sl: float) -> int:
        """Determines appropriate position size based on symbol type and risk limits."""
        # Standard index lot sizes
        if "NIFTY50" in symbol or "NIFTY" in symbol:
            return 50
        elif "BANKNIFTY" in symbol:
            return 15
        elif "SENSEX" in symbol:
            return 10
        else:
            # Equities / F&O stock standard allocation (e.g. ₹2,000 risk cap)
            risk_per_share = max(abs(entry - sl), 1.0)
            max_risk_amount = 2000.0
            return max(int(max_risk_amount / risk_per_share), 1)

    def pause_trading(self) -> None:
        """Pauses autonomous order executions (monitoring and trailing continue)."""
        self.is_paused = True
        LOGGER.warning("⏸️ [Autonomous Router] Trading has been PAUSED.")

    def resume_trading(self) -> None:
        """Resumes autonomous order executions."""
        self.is_paused = False
        LOGGER.info("▶️ [Autonomous Router] Trading has been RESUMED.")

    def flatten_all(self, reason: str = "EMERGENCY_PANIC_FLATTEN") -> List[Dict[str, Any]]:
        """Emergency kill switch: Closes all active positions immediately."""
        return self.position_manager.flatten_all_positions(reason=reason)
