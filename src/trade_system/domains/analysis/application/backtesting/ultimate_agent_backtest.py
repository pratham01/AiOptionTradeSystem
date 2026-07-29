"""
UltimateAgentBacktest — Day-by-day historical replay of the Ultimate Intraday Agent.

For each trading day:
  1. Run UltimateIntradayAgent.scan() to get qualifying trades
  2. Simulate each trade on 15m candles: entry → SL/Target/EOD exit
  3. Record trade outcome with full context (regime, VIX, PCR, BB phase, etc.)

Output:
  - reports/ultimate_agent/trades.csv
  - reports/ultimate_agent/summary.csv
  - reports/ultimate_agent/report.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.advisory.application.agent.ultimate_intraday_agent import (
    UltimateIntradayAgent,
    UltimateTrade,
)

LOGGER = logging.getLogger(__name__)

ENTRY_CUTOFF = dt_time(14, 0)
SQUARE_OFF_TIME = dt_time(15, 15)


@dataclass
class BacktestTrade:
    """One simulated trade from the ultimate agent backtest."""
    trade_date: date
    symbol: str
    sector: str
    direction: str
    edge_score: float
    regime: str
    vix: float
    pcr_bias: str
    bb_phase: str
    volume_surge: float
    entry_type: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    is_expiry_day: bool = False
    exit_time: pd.Timestamp | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_points: float = 0.0
    pnl_pct: float = 0.0
    holding_minutes: float = 0.0
    key_signals: str = ""

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date.isoformat(),
            "symbol": self.symbol,
            "sector": self.sector,
            "direction": self.direction,
            "edge_score": round(self.edge_score, 1),
            "regime": self.regime,
            "vix": round(self.vix, 1),
            "pcr_bias": self.pcr_bias,
            "bb_phase": self.bb_phase,
            "volume_surge": round(self.volume_surge, 1),
            "entry_type": self.entry_type,
            "entry_time": str(self.entry_time),
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_1": round(self.target_1, 2),
            "target_2": round(self.target_2, 2),
            "risk_reward": round(self.risk_reward, 2),
            "is_expiry_day": self.is_expiry_day,
            "exit_time": str(self.exit_time),
            "exit_price": round(self.exit_price, 2),
            "exit_reason": self.exit_reason,
            "pnl_points": round(self.pnl_points, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "holding_minutes": round(self.holding_minutes, 1),
            "key_signals": self.key_signals,
        }


class UltimateAgentBacktest:
    """Day-by-day historical replay of the Ultimate Intraday Agent."""

    def __init__(
        self,
        max_trades_normal: int = 3,
        max_trades_expiry: int = 1,
        min_risk_reward: float = 1.5,
    ) -> None:
        self.engine = get_engine()
        self._agent = UltimateIntradayAgent(
            max_trades_normal=max_trades_normal,
            max_trades_expiry=max_trades_expiry,
            min_risk_reward=min_risk_reward,
        )
        self._edge_scorer = self._agent._edge_scorer

    def run(
        self,
        output_dir: Path,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> Path:
        """Run the full backtest across available dates."""
        output_dir.mkdir(parents=True, exist_ok=True)

        trading_dates = self._get_trading_dates(start_date, end_date)
        LOGGER.info(f"Ultimate Backtest: Running across {len(trading_dates)} trading days.")

        all_trades: list[dict] = []

        for i, td in enumerate(trading_dates):
            print(f"  [{i+1}/{len(trading_dates)}] {td}...", end=" ", flush=True)
            day_trades = self._simulate_day(td)
            for t in day_trades:
                all_trades.append(t.to_dict())
            print(f"{len(day_trades)} trades")

        # ── Save results ──────────────────────────────────────────────────
        trades_df = pd.DataFrame(all_trades)
        trades_path = output_dir / "ultimate_agent_trades.csv"
        trades_df.to_csv(trades_path, index=False)

        summary = self._build_summary(trades_df)
        summary_path = output_dir / "ultimate_agent_summary.csv"
        summary.to_csv(summary_path, index=False)

        report = self._build_report(trades_df, summary)
        report_path = output_dir / "ultimate_agent_report.md"
        report_path.write_text(report)

        LOGGER.info(f"Ultimate Backtest: Complete. {len(all_trades)} total trades. Results in {output_dir}")
        return output_dir

    def _simulate_day(self, td: date) -> list[BacktestTrade]:
        """Simulate the Ultimate Agent for a single trading day."""
        trades: list[BacktestTrade] = []

        try:
            # Get the agent's trade suggestions for this day
            suggestions = self._agent.scan(td)
            if not suggestions:
                return trades

            # Fetch 15m data for trade simulation
            df_15m = self._edge_scorer._fetch_15m_data(td)
            if df_15m.empty:
                return trades

            symbol_15m = {
                sym: grp.sort_values("timestamp")
                for sym, grp in df_15m.groupby("symbol")
            }

            for suggestion in suggestions:
                df_sym = symbol_15m.get(suggestion.symbol)
                if df_sym is None or df_sym.empty:
                    continue

                trade = self._simulate_trade(td, suggestion, df_sym)
                if trade:
                    trades.append(trade)

        except Exception as e:
            LOGGER.error(f"Error simulating {td}: {e}")

        return trades

    def _simulate_trade(
        self,
        td: date,
        suggestion: UltimateTrade,
        df_sym: pd.DataFrame,
    ) -> BacktestTrade | None:
        """Simulate a single trade on 15m candles."""
        today_candles = df_sym[df_sym["timestamp"].dt.date == td].sort_values("timestamp")
        if today_candles.empty:
            return None

        is_long = suggestion.direction == "CALL"

        # Find the first candle where price reaches entry level
        entry_time = None
        for _, candle in today_candles.iterrows():
            ts = candle["timestamp"]
            if ts.time() > ENTRY_CUTOFF:
                break
            if is_long and candle["high"] >= suggestion.entry_price:
                entry_time = ts
                break
            elif not is_long and candle["low"] <= suggestion.entry_price:
                entry_time = ts
                break

        if entry_time is None:
            return None

        trade = BacktestTrade(
            trade_date=td,
            symbol=suggestion.symbol,
            sector=suggestion.sector,
            direction=suggestion.direction,
            edge_score=suggestion.edge_score,
            regime=suggestion.regime,
            vix=suggestion.vix,
            pcr_bias=suggestion.pcr_bias,
            bb_phase=suggestion.bb_phase,
            volume_surge=suggestion.volume_surge,
            entry_type=suggestion.entry_type,
            entry_time=entry_time,
            entry_price=suggestion.entry_price,
            stop_loss=suggestion.stop_loss,
            target_1=suggestion.target_1,
            target_2=suggestion.target_2,
            risk_reward=suggestion.risk_reward,
            is_expiry_day=suggestion.is_expiry_day,
            key_signals=" | ".join(suggestion.key_signals[:3]),
        )

        # Simulate from entry candle forward
        post_entry = today_candles[today_candles["timestamp"] > entry_time]
        for _, candle in post_entry.iterrows():
            ts = candle["timestamp"]

            if is_long:
                # Check SL hit
                if candle["low"] <= suggestion.stop_loss:
                    trade.exit_time = ts
                    trade.exit_price = suggestion.stop_loss
                    trade.exit_reason = "SL_HIT"
                    break
                # Check target hit
                if candle["high"] >= suggestion.target_1:
                    trade.exit_time = ts
                    trade.exit_price = suggestion.target_1
                    trade.exit_reason = "TARGET_1_HIT"
                    break
            else:
                # Short / PUT direction
                if candle["high"] >= suggestion.stop_loss:
                    trade.exit_time = ts
                    trade.exit_price = suggestion.stop_loss
                    trade.exit_reason = "SL_HIT"
                    break
                if candle["low"] <= suggestion.target_1:
                    trade.exit_time = ts
                    trade.exit_price = suggestion.target_1
                    trade.exit_reason = "TARGET_1_HIT"
                    break

            # EOD square-off
            if ts.time() >= SQUARE_OFF_TIME:
                trade.exit_time = ts
                trade.exit_price = float(candle["close"])
                trade.exit_reason = "EOD_SQUAREOFF"
                break

        # If no exit found (data gap), close at last candle
        if trade.exit_time is None and not post_entry.empty:
            last = post_entry.iloc[-1]
            trade.exit_time = last["timestamp"]
            trade.exit_price = float(last["close"])
            trade.exit_reason = "DATA_END"

        if trade.exit_time is None:
            return None

        # Compute PnL
        if is_long:
            trade.pnl_points = trade.exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - trade.exit_price
        trade.pnl_pct = trade.pnl_points / trade.entry_price if trade.entry_price > 0 else 0.0
        trade.holding_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60.0

        return trade

    # ── Data Fetching ────────────────────────────────────────────────────

    def _get_trading_dates(self, start_date: date | None, end_date: date | None) -> list[date]:
        """Get unique trading dates from ohlcv_15m."""
        with self.engine.connect() as conn:
            query = text("""
                SELECT DISTINCT date(timestamp) as trade_date
                FROM ohlcv_15m
                WHERE symbol NOT LIKE '%INDEX%'
                ORDER BY trade_date
            """)
            df = pd.read_sql(query, conn)

        if df.empty:
            return []

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        dates = df["trade_date"].tolist()

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        return dates

    # ── Report Generation ────────────────────────────────────────────────

    def _build_summary(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Build a summary DataFrame."""
        if trades_df.empty:
            return pd.DataFrame()

        total_trades = len(trades_df)
        winners = trades_df[trades_df["pnl_points"] > 0]
        losers = trades_df[trades_df["pnl_points"] <= 0]

        rows = [{
            "total_trades": total_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / total_trades * 100, 1) if total_trades else 0,
            "total_pnl_points": round(trades_df["pnl_points"].sum(), 2),
            "avg_pnl_points": round(trades_df["pnl_points"].mean(), 2),
            "avg_winner": round(winners["pnl_points"].mean(), 2) if not winners.empty else 0,
            "avg_loser": round(losers["pnl_points"].mean(), 2) if not losers.empty else 0,
            "max_winner": round(winners["pnl_points"].max(), 2) if not winners.empty else 0,
            "max_loser": round(losers["pnl_points"].min(), 2) if not losers.empty else 0,
            "avg_holding_minutes": round(trades_df["holding_minutes"].mean(), 1),
            "trades_per_day": round(total_trades / trades_df["trade_date"].nunique(), 1) if trades_df["trade_date"].nunique() else 0,
        }]

        # Breakdown by regime
        for regime in trades_df["regime"].unique():
            r_df = trades_df[trades_df["regime"] == regime]
            r_winners = r_df[r_df["pnl_points"] > 0]
            rows[0][f"win_rate_{regime.lower()}"] = round(len(r_winners) / len(r_df) * 100, 1) if len(r_df) else 0
            rows[0][f"trades_{regime.lower()}"] = len(r_df)

        return pd.DataFrame(rows)

    def _build_report(self, trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
        """Build a Markdown report."""
        if trades_df.empty:
            return "# Ultimate Agent Backtest Report\n\nNo trades were generated."

        s = summary_df.iloc[0] if not summary_df.empty else {}

        lines = [
            "# Ultimate Agent Backtest Report",
            "",
            "## Summary",
            f"- **Total Trades**: {s.get('total_trades', 0)}",
            f"- **Win Rate**: {s.get('win_rate', 0)}%",
            f"- **Total PnL (Points)**: {s.get('total_pnl_points', 0)}",
            f"- **Avg PnL per Trade**: {s.get('avg_pnl_points', 0)} pts",
            f"- **Avg Winner**: {s.get('avg_winner', 0)} pts",
            f"- **Avg Loser**: {s.get('avg_loser', 0)} pts",
            f"- **Max Winner**: {s.get('max_winner', 0)} pts",
            f"- **Max Loser**: {s.get('max_loser', 0)} pts",
            f"- **Avg Holding**: {s.get('avg_holding_minutes', 0)} min",
            f"- **Trades/Day**: {s.get('trades_per_day', 0)}",
            "",
            "## Regime Breakdown",
        ]

        for regime in trades_df["regime"].unique():
            key_wr = f"win_rate_{regime.lower()}"
            key_tc = f"trades_{regime.lower()}"
            lines.append(f"- **{regime}**: {s.get(key_tc, 0)} trades, {s.get(key_wr, 0)}% win rate")

        lines.extend([
            "",
            "## Exit Reason Distribution",
        ])
        for reason, count in trades_df["exit_reason"].value_counts().items():
            lines.append(f"- {reason}: {count} trades")

        lines.extend([
            "",
            "## Top 10 Winning Trades",
        ])
        top_winners = trades_df.nlargest(10, "pnl_points")
        for _, t in top_winners.iterrows():
            lines.append(
                f"- [{t['trade_date']}] {t['symbol']} {t['direction']} | "
                f"Edge: {t['edge_score']} | PnL: {t['pnl_points']} pts | {t['exit_reason']}"
            )

        lines.extend([
            "",
            "## Bottom 5 Losing Trades",
        ])
        bottom_losers = trades_df.nsmallest(5, "pnl_points")
        for _, t in bottom_losers.iterrows():
            lines.append(
                f"- [{t['trade_date']}] {t['symbol']} {t['direction']} | "
                f"Edge: {t['edge_score']} | PnL: {t['pnl_points']} pts | {t['exit_reason']}"
            )

        return "\n".join(lines)
