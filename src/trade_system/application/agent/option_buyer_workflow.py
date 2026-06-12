"""
OptionBuyerExecutionWorkflow - Autonomous execution, monitoring, and exit management for option buying.
"""
import os
import json
import logging
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from trade_system.core import TradeSuggestion, TradeDirection
from trade_system.core.ports.broker import OrderRequest, OrderSide, OrderType, BrokerError
from trade_system.infrastructure.database.models import SuggestedTrade
from trade_system.infrastructure.database.connection import get_db_context
from trade_system.application.analysis.option_chain_analyzer import OptionChainAnalyzer

LOGGER = logging.getLogger(__name__)


class OptionBuyerExecutionWorkflow:
    """
    Manages the full lifecycle of option buying trades:
    1. Resolves Nifty index/stock suggestions into exact option contract symbols.
    2. Executes BUY market orders via the active broker.
    3. Persists active positions to a JSON file to survive system restarts.
    4. Continuously monitors option contract LTP (every 15s) to check SL, Target, and Trailing SL.
    5. Enforces time-based intraday exits at 3:15 PM (15:15).
    """

    def __init__(
        self,
        broker: Any,
        notifier: Optional[Any] = None,
        positions_file: str = "data/active_option_positions.json",
        sl_pct: float = 0.30,        # 30% Stop Loss on option premium
        target_pct: float = 0.60,    # 60% Target Profit on option premium
        trailing_activation: float = 0.20,  # Move to BE when price gains 20%
        trailing_lock_pct: float = 0.50     # Lock in 50% of peak gains after activation
    ) -> None:
        self.broker = broker
        self.notifier = notifier
        self.positions_file = Path(positions_file)
        self.sl_pct = sl_pct
        self.target_pct = target_pct
        self.trailing_activation = trailing_activation
        self.trailing_lock_pct = trailing_lock_pct

        # Ensure directory exists
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_premium_price(self, quote: Any) -> float:
        """Helper to safely get last/close/open price from object or dict quote."""
        if isinstance(quote, (int, float)):
            return float(quote)
        if hasattr(quote, "last_price") and getattr(quote, "last_price", 0.0):
            return float(getattr(quote, "last_price", 0.0))
        if hasattr(quote, "close") and getattr(quote, "close", 0.0):
            return float(getattr(quote, "close", 0.0))
        if hasattr(quote, "open") and getattr(quote, "open", 0.0):
            return float(getattr(quote, "open", 0.0))
        if isinstance(quote, dict):
            val = quote.get("last_price") or quote.get("close") or quote.get("open") or quote.get("ltp") or 0.0
            return float(val)
        return 0.0


    def resolve_option_contract_symbol(
        self,
        index_symbol: str,
        strike: float,
        option_type: str
    ) -> Optional[str]:
        """
        Resolves Nifty index/stock and strike parameters to a tradable option contract symbol
        using the live OptionChainAnalyzer.
        """
        try:
            clean_symbol = index_symbol.replace("NSE:", "").replace("-INDEX", "").replace("-EQ", "")
            # Default to NIFTY50 if target is Nifty
            symbol_key = "NIFTY50" if "NIFTY" in clean_symbol.upper() else clean_symbol
            
            LOGGER.info(f"Resolving option contract for {symbol_key} | Strike: {strike} | Type: {option_type}")
            
            analyzer = OptionChainAnalyzer(self.broker.fyers if hasattr(self.broker, "fyers") else self.broker, symbol=symbol_key)
            df = analyzer.get_option_chain_df()
            
            if df is None or df.empty:
                LOGGER.warning("Option chain DataFrame is empty or could not be fetched.")
                return None
                
            # Filter by option type and strike
            filtered = df[
                (df["option_type"].str.upper() == option_type.upper()) &
                (df["strike"].round().astype(int) == int(round(strike)))
            ]
            
            if filtered.empty:
                # Fallback: find nearest strike if exact match fails
                df_type = df[df["option_type"].str.upper() == option_type.upper()]
                if not df_type.empty:
                    nearest_idx = (df_type["strike"] - strike).abs().idxmin()
                    resolved_symbol = df_type.loc[nearest_idx, "symbol"]
                    LOGGER.info(f"Exact strike match failed. Fallback to nearest resolved symbol: {resolved_symbol}")
                    return resolved_symbol
                return None
                
            # Sort by expiry to pick the nearest expiry contract (weekly)
            filtered = filtered.sort_values("expiry")
            resolved_symbol = filtered.iloc[0]["symbol"]
            LOGGER.info(f"Successfully resolved option contract symbol: {resolved_symbol}")
            return resolved_symbol
            
        except Exception as e:
            LOGGER.error(f"Error resolving option contract symbol: {e}", exc_info=True)
            return None

    def load_active_positions(self) -> List[Dict[str, Any]]:
        """Load currently active positions from local file."""
        if not self.positions_file.exists():
            return []
        try:
            with open(self.positions_file, "r") as f:
                data = json.load(f)
                return data.get("positions", [])
        except Exception as e:
            LOGGER.error(f"Failed to read active positions file: {e}")
            return []

    def save_active_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Save active positions to local file."""
        try:
            with open(self.positions_file, "w") as f:
                json.dump({"positions": positions, "last_updated": datetime.now().isoformat()}, f, indent=4)
        except Exception as e:
            LOGGER.error(f"Failed to write active positions file: {e}")

    async def execute_buy_order(self, suggestion: TradeSuggestion) -> Optional[Dict[str, Any]]:
        """
        Executes a BUY market order for the appropriate option contract.
        Saves target and stop-loss levels in option premium terms.
        """
        # 1. Resolve contract symbol
        strike = suggestion.option_params.suggested_strike
        opt_type = "CE" if suggestion.direction == TradeDirection.CALL else "PE"
        option_symbol = self.resolve_option_contract_symbol(suggestion.symbol, strike, opt_type)
        
        if not option_symbol:
            msg = f"❌ [Execution Failed] Option symbol resolution failed for {suggestion.symbol} {suggestion.direction.value} Strike {strike}"
            LOGGER.error(msg)
            self._notify(msg)
            return None

        # 2. Get current premium price for entry level
        try:
            quotes = self.broker.get_quotes([option_symbol])
            if not quotes or option_symbol not in quotes:
                raise BrokerError(f"Could not get quotes for resolved option {option_symbol}")
            
            quote = quotes[option_symbol]
            entry_premium = self._get_premium_price(quote)
            if entry_premium <= 0:
                raise BrokerError(f"Invalid last price for {option_symbol}: {entry_premium}")
        except Exception as e:
            msg = f"❌ [Execution Failed] Could not fetch quotes for resolved option {option_symbol}: {e}"
            LOGGER.error(msg)
            self._notify(msg)
            return None

        # 3. Calculate Stop Loss & Take Profit in premium terms
        sl_premium = round(entry_premium * (1 - self.sl_pct), 2)
        target_premium = round(entry_premium * (1 + self.target_pct), 2)

        LOGGER.info(f"Placing Buy order for {option_symbol} | Premium Entry: {entry_premium} | SL: {sl_premium} | TP: {target_premium}")

        # 4. Place Market Order
        # Quantity defaults to standard lot (50 for Nifty) or suggestion qty
        quantity = 50 if "NIFTY" in suggestion.symbol.upper() else 1
        if hasattr(suggestion, "quantity") and suggestion.quantity > 1:
            quantity = suggestion.quantity

        req = OrderRequest(
            symbol=option_symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=0.0,
            tag="option_buyer_wf"
        )

        try:
            if hasattr(self.broker, "place_order"):
                res = self.broker.place_order(req)
                order_id = res.order_id
                fill_price = res.average_price or entry_premium
            else:
                # Simulating for test brokers
                LOGGER.warning("Broker does not support place_order. Simulating execution.")
                order_id = "SIM_" + os.urandom(4).hex()
                fill_price = entry_premium

            # Create position record
            position = {
                "trade_id": suggestion.id,
                "suggestion_symbol": suggestion.symbol,
                "option_symbol": option_symbol,
                "direction": suggestion.direction.value,
                "strike": strike,
                "quantity": quantity,
                "entry_premium": fill_price,
                "current_premium": fill_price,
                "sl_premium": sl_premium,
                "target_premium": target_premium,
                "peak_premium": fill_price,
                "order_id": order_id,
                "status": "OPEN",
                "entered_at": datetime.now().isoformat()
            }

            # Persist locally
            active_positions = self.load_active_positions()
            active_positions.append(position)
            self.save_active_positions(active_positions)

            # Update SQLite DB
            self._update_db_trade(suggestion.id, "OPEN", fill_price)

            msg = (
                f"🟢 <b>[Option Entry Executed]</b>\n"
                f"Contract: <code>{option_symbol}</code>\n"
                f"Direction: {suggestion.direction.value} (Strike: {strike})\n"
                f"Premium Entry: ₹{fill_price:.2f}\n"
                f"Stop Loss: ₹{sl_premium:.2f} (-{self.sl_pct:.0%})\n"
                f"Target: ₹{target_premium:.2f} (+{self.target_pct:.0%})\n"
                f"Lot Size: {quantity}"
            )
            self._notify(msg)
            return position

        except Exception as e:
            msg = f"❌ [Execution Failed] Broker place_order error on {option_symbol}: {e}"
            LOGGER.error(msg)
            self._notify(msg)
            return None

    async def manage_open_positions(self) -> None:
        """
        Evaluates open options positions against SL/TP levels, trails stop loss,
        and manages intraday time-based exits at 3:15 PM (15:15).
        """
        active_positions = self.load_active_positions()
        if not active_positions:
            return

        symbols = [p["option_symbol"] for p in active_positions]
        LOGGER.info(f"Evaluating {len(active_positions)} open options positions: {symbols}")

        try:
            quotes = self.broker.get_quotes(symbols)
        except Exception as e:
            LOGGER.error(f"Failed to fetch quotes for position monitoring: {e}")
            return

        remaining_positions = []
        now_time = datetime.now().time()
        time_exit_trigger = dt_time(15, 15)

        for pos in active_positions:
            opt_sym = pos["option_symbol"]
            trade_id = pos["trade_id"]
            
            if opt_sym not in quotes:
                remaining_positions.append(pos)
                continue

            quote = quotes[opt_sym]
            ltp = self._get_premium_price(quote)
            pos["current_premium"] = ltp

            # Track peak price for trailing stop loss
            if ltp > pos["peak_premium"]:
                pos["peak_premium"] = ltp

            # Stop loss & target calculations
            sl_premium = pos["sl_premium"]
            target_premium = pos["target_premium"]
            entry_premium = pos["entry_premium"]

            # Dynamic Trailing Stop Loss
            # 1. Activation: Move to breakeven (entry premium) if price rises by trailing_activation (e.g. 20%)
            gain_pct = (ltp - entry_premium) / entry_premium
            if gain_pct >= self.trailing_activation:
                # If current SL is still below entry, move SL to entry price
                if pos["sl_premium"] < entry_premium:
                    pos["sl_premium"] = entry_premium
                    LOGGER.info(f"Trailing SL activated for {opt_sym}: SL moved to breakeven (₹{entry_premium:.2f})")
                    self._notify(f"🛡️ <b>[Trailing SL Activated]</b> {opt_sym} reached +{gain_pct:.0%} gain. SL trailed to entry (₹{entry_premium:.2f}).")

                # 2. Lock-in: Lock in trailing_lock_pct (e.g. 50%) of gains above peak
                peak_gain = pos["peak_premium"] - entry_premium
                lock_sl = round(entry_premium + (peak_gain * self.trailing_lock_pct), 2)
                if lock_sl > pos["sl_premium"]:
                    pos["sl_premium"] = lock_sl
                    LOGGER.info(f"Trailing SL trailed higher for {opt_sym}: SL moved to ₹{lock_sl:.2f}")

            # Check Exit Conditions
            exit_reason = None
            outcome = "PENDING"
            
            if ltp <= pos["sl_premium"]:
                exit_reason = f"Stop Loss Hit (LTP: ₹{ltp:.2f} <= SL: ₹{pos['sl_premium']:.2f})"
                outcome = "LOSS" if ltp < entry_premium else "WIN" # Trailed SL can be a win
            elif ltp >= target_premium:
                exit_reason = f"Target Profit Hit (LTP: ₹{ltp:.2f} >= Target: ₹{target_premium:.2f})"
                outcome = "WIN"
            elif now_time >= time_exit_trigger:
                exit_reason = f"Intraday Time Cutoff Hit (15:15 PM EOD squareoff)"
                outcome = "NEUTRAL"

            if exit_reason:
                await self._execute_exit_order(pos, ltp, exit_reason, outcome)
            else:
                remaining_positions.append(pos)

        self.save_active_positions(remaining_positions)

    async def _execute_exit_order(self, pos: Dict[str, Any], exit_price: float, reason: str, outcome: str) -> None:
        """Executes a sell market order to close out the option position."""
        opt_sym = pos["option_symbol"]
        qty = pos["quantity"]
        
        LOGGER.info(f"Exiting position {opt_sym} | Reason: {reason} | Exit price: {exit_price}")

        req = OrderRequest(
            symbol=opt_sym,
            side=OrderSide.SELL,
            quantity=qty,
            order_type=OrderType.MARKET,
            price=0.0,
            tag="option_buyer_exit"
        )

        try:
            if hasattr(self.broker, "place_order"):
                res = self.broker.place_order(req)
                final_exit_price = res.average_price or exit_price
            else:
                LOGGER.warning("Broker does not support place_order. Simulating exit.")
                final_exit_price = exit_price

            # Update SQLite DB
            self._update_db_trade(pos["trade_id"], outcome, pos["entry_premium"], exit_price=final_exit_price)

            pnl_val = (final_exit_price - pos["entry_premium"]) * qty
            pnl_pct = ((final_exit_price - pos["entry_premium"]) / pos["entry_premium"]) * 100.0

            outcome_emoji = "🎉 WIN" if outcome == "WIN" or pnl_pct > 0 else "🔴 LOSS" if outcome == "LOSS" else "⚖️ EOD CLOSE"
            msg = (
                f"🚨 <b>[Option Position Closed]</b>\n"
                f"Contract: <code>{opt_sym}</code>\n"
                f"Outcome: <b>{outcome_emoji}</b>\n"
                f"Entry Price: ₹{pos['entry_premium']:.2f}\n"
                f"Exit Price: ₹{final_exit_price:.2f}\n"
                f"PnL: <b>₹{pnl_val:+.2f} ({pnl_pct:+.2f}%)</b>\n"
                f"Reason: <i>{reason}</i>"
            )
            self._notify(msg)

        except Exception as e:
            LOGGER.error(f"Failed to place exit order for {opt_sym}: {e}", exc_info=True)
            self._notify(f"⚠️ <b>[Exit Order Failed]</b> Could not execute sell order for {opt_sym}: {e}")

    def _update_db_trade(self, trade_id: str, outcome: str, entry_price: float, exit_price: Optional[float] = None) -> None:
        """Helper to update database SuggestedTrade record."""
        try:
            with get_db_context() as session:
                trade = session.query(SuggestedTrade).filter_by(id=trade_id).first()
                if trade:
                    trade.outcome = outcome
                    trade.actual_entry = entry_price
                    if exit_price is not None:
                        trade.actual_exit = exit_price
                        trade.actual_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    trade.outcome_checked_at = datetime.now()
                    session.add(trade)
                    LOGGER.info(f"Database updated for trade {trade_id} with outcome: {outcome}")
        except Exception as e:
            LOGGER.error(f"Failed to update database trade {trade_id}: {e}")

    def _notify(self, message: str) -> None:
        """Telegram notification sender."""
        if self.notifier:
            try:
                self.notifier.send(message)
            except Exception as e:
                LOGGER.error(f"Telegram notification failed: {e}")
