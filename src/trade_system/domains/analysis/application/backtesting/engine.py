from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_system.domains.trading.domain.models.legacy import Position, Trade
from trade_system.domains.strategy.application.strategies.base import Strategy


@dataclass(slots=True)
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.DataFrame
    summary: dict[str, float]


class BacktestEngine:
    def __init__(self, strategy: Strategy, initial_cash: float = 100000.0, quantity: float = 1.0) -> None:
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.quantity = quantity

    def run(self, candles: pd.DataFrame) -> BacktestResult:
        if candles.empty:
            return BacktestResult(trades=[], equity_curve=pd.DataFrame(), summary={})

        prepared = self.strategy.prepare(candles)
        position = Position()
        trades: list[Trade] = []
        equity_rows: list[dict] = []
        realized_pnl = 0.0
        symbol = prepared["symbol"].iloc[0] if "symbol" in prepared.columns and not prepared.empty else "UNKNOWN"

        for idx in range(len(prepared)):
            window = prepared.iloc[: idx + 1]
            row = window.iloc[-1]
            signal = self.strategy.on_bar(window)
            price = float(row["close"])
            timestamp = pd.to_datetime(row["timestamp"], format="mixed").to_pydatetime()

            if signal:
                if signal.action == "BUY":
                    if position.side == -1:
                        realized_pnl += self._close_trade(position, timestamp, price, trades, symbol, signal.reason)
                    if position.side == 0:
                        position = Position(side=1, quantity=self.quantity, entry_price=price, entry_time=timestamp)
                elif signal.action == "SELL":
                    if position.side == 1:
                        realized_pnl += self._close_trade(position, timestamp, price, trades, symbol, signal.reason)
                    if position.side == 0:
                        position = Position(side=-1, quantity=self.quantity, entry_price=price, entry_time=timestamp)
                elif signal.action == "EXIT" and position.side != 0:
                    realized_pnl += self._close_trade(position, timestamp, price, trades, symbol, signal.reason)
                    position = Position()

            unrealized = 0.0
            if position.side != 0:
                unrealized = (price - position.entry_price) * position.quantity * position.side
            equity_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "equity": self.initial_cash + realized_pnl + unrealized,
                }
            )

        if position.side != 0:
            last = prepared.iloc[-1]
            realized_pnl += self._close_trade(
                position,
                pd.to_datetime(last["timestamp"], format="mixed").to_pydatetime(),
                float(last["close"]),
                trades,
                symbol,
                "forced end of test exit",
            )
            equity_rows[-1]["equity"] = self.initial_cash + realized_pnl

        equity_curve = pd.DataFrame(equity_rows)
        summary = {
            "initial_cash": self.initial_cash,
            "final_equity": float(equity_curve["equity"].iloc[-1]),
            "total_return": float(equity_curve["equity"].iloc[-1] - self.initial_cash),
            "trade_count": float(len(trades)),
            "win_rate": float(sum(1 for trade in trades if trade.pnl > 0) / len(trades)) if trades else 0.0,
        }
        return BacktestResult(trades=trades, equity_curve=equity_curve, summary=summary)

    @staticmethod
    def _close_trade(
        position: Position,
        exit_time,
        exit_price: float,
        trades: list[Trade],
        symbol: str,
        reason: str,
    ) -> float:
        pnl = (exit_price - position.entry_price) * position.quantity * position.side
        trades.append(
            Trade(
                symbol=symbol,
                entry_time=position.entry_time,
                exit_time=exit_time,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=pnl,
                reason=reason,
            )
        )
        position.side = 0
        position.quantity = 0.0
        position.entry_price = 0.0
        position.entry_time = None
        return pnl
