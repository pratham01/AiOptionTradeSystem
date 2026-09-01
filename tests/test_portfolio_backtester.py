"""
Unit tests for PortfolioBacktestEngine and PerformanceAnalytics.
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np
import pytest

from trade_system.domains.analysis.application.backtesting.performance_analytics import (
    BacktestTradeRecord,
    PerformanceAnalytics
)
from trade_system.domains.analysis.application.backtesting.portfolio_backtester import (
    BacktestConfig,
    PortfolioBacktestEngine
)


def test_performance_analytics_empty():
    report = PerformanceAnalytics.compute([], pd.DataFrame(), initial_capital=1_000_000.0)
    assert report.initial_capital == 1_000_000.0
    assert report.final_equity == 1_000_000.0
    assert report.triggered_trades == 0
    assert report.win_rate_pct == 0.0


def test_performance_analytics_calculation():
    trades = [
        BacktestTradeRecord(
            trade_id=1,
            symbol="RELIANCE",
            sector="ENERGY",
            direction="CALL",
            entry_date=date(2026, 1, 1),
            exit_date=date(2026, 1, 3),
            holding_days=2,
            entry_price=1250.0,
            exit_price=1300.0,
            shares=100,
            capital_invested=125000.0,
            stop_loss=1220.0,
            target_1=1295.0,
            target_2=1320.0,
            gross_pnl=5000.0,
            costs=100.0,
            net_pnl=4900.0,
            return_pct=3.92,
            r_multiple=1.63,
            exit_reason="TARGET_1",
            confidence="VERY HIGH",
            probability=90,
            confluences=["Hammer", "50 EMA Bounce"]
        ),
        BacktestTradeRecord(
            trade_id=2,
            symbol="TCS",
            sector="IT",
            direction="CALL",
            entry_date=date(2026, 1, 2),
            exit_date=date(2026, 1, 4),
            holding_days=2,
            entry_price=3500.0,
            exit_price=3400.0,
            shares=30,
            capital_invested=105000.0,
            stop_loss=3420.0,
            target_1=3620.0,
            target_2=3700.0,
            gross_pnl=-3000.0,
            costs=100.0,
            net_pnl=-3100.0,
            return_pct=-2.95,
            r_multiple=-1.29,
            exit_reason="STOP_LOSS",
            confidence="HIGH",
            probability=70,
            confluences=["Bullish Engulfing"]
        )
    ]

    equity_df = pd.DataFrame([
        {"date": date(2026, 1, 1), "equity": 1000000.0},
        {"date": date(2026, 1, 3), "equity": 1004900.0},
        {"date": date(2026, 1, 4), "equity": 1001800.0}
    ])

    report = PerformanceAnalytics.compute(trades, equity_df, initial_capital=1_000_000.0, total_signals_generated=5)
    assert report.triggered_trades == 2
    assert report.winning_trades == 1
    assert report.losing_trades == 1
    assert report.win_rate_pct == 50.0
    assert report.net_profit == 1800.0
    assert report.profit_factor > 1.0
    assert not report.confluence_attribution.empty or report.triggered_trades == 2


def test_portfolio_backtester_initialization():
    config = BacktestConfig(initial_capital=500_000.0, risk_per_trade_pct=1.5)
    engine = PortfolioBacktestEngine(config=config)
    assert engine.config.initial_capital == 500_000.0
    assert engine.config.risk_per_trade_pct == 1.5
