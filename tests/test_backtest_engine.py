import pandas as pd

from trade_system.application.backtesting import BacktestEngine
from trade_system.application.strategies.base import Strategy
from trade_system.core.models.legacy import Signal


class DummyStrategy(Strategy):
    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        return candles.copy()

    def on_bar(self, window: pd.DataFrame):
        if len(window) > 1 and float(window.iloc[-1]["close"]) > float(window.iloc[-2]["close"]):
            return Signal(action="BUY", reason="Price Up")
        if len(window) > 1 and float(window.iloc[-1]["close"]) < float(window.iloc[-2]["close"]):
            return Signal(action="SELL", reason="Price Down")
        return None


def test_backtest_engine_generates_trades_and_summary():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:15:00", periods=40, freq="min"),
            "open": list(range(100, 140)),
            "high": list(range(101, 141)),
            "low": list(range(99, 139)),
            "close": list(range(100, 120)) + list(range(119, 99, -1)),
            "volume": [1000] * 40,
            "symbol": ["NSE:NIFTY50-INDEX"] * 40,
        }
    )
    strategy = DummyStrategy()
    result = BacktestEngine(strategy=strategy, initial_cash=100000, quantity=1).run(candles)

    assert result.summary["trade_count"] >= 1
    assert len(result.equity_curve) == len(candles)
    assert "final_equity" in result.summary
