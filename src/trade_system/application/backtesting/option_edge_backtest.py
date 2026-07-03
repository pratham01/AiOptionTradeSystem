"""
OptionEdgeBacktest — Historical replay of the Intraday Option Edge pipeline.

For each trading day in the database:
  1. Run FOIntradayShortlist on daily data up to that date
  2. Run RegimeDetector on the 15m data for that day
  3. Run IntradayEdgeScorer on the 15m data
  4. Run SmartEntryTrigger for qualifying stocks
  5. Simulate the trade: entry → SL/target exit on subsequent 15m candles
  6. Record trade outcome

Output:
  - reports/option_edge/option_edge_trades.csv
  - reports/option_edge/option_edge_summary.csv
  - reports/option_edge/option_edge_report.md
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.infrastructure.database.connection import get_engine
from trade_system.application.analysis.fo_intraday_shortlist import FOIntradayShortlist
from trade_system.application.analysis.regime_detector import RegimeDetector, MarketRegime
from trade_system.application.analysis.intraday_edge_scorer import IntradayEdgeScorer
from trade_system.application.analysis.smart_entry_trigger import SmartEntryTrigger
from trade_system.application.analysis.option_strike_selector import OptionStrikeSelector

LOGGER = logging.getLogger(__name__)

ENTRY_CUTOFF = dt_time(14, 0)       # No new entries after 2 PM
SQUARE_OFF_TIME = dt_time(15, 15)   # Force close at 3:15 PM


@dataclass
class BacktestTrade:
    """One simulated trade."""
    trade_date: date
    symbol: str
    sector: str
    direction: str          # "CALL" or "PUT"
    edge_score: float
    regime: str
    entry_type: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    exit_time: pd.Timestamp | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_points: float = 0.0
    pnl_pct: float = 0.0
    holding_minutes: float = 0.0
    shortlist_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date.isoformat(),
            "symbol": self.symbol,
            "sector": self.sector,
            "direction": self.direction,
            "edge_score": self.edge_score,
            "regime": self.regime,
            "entry_type": self.entry_type,
            "entry_time": str(self.entry_time),
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_1": round(self.target_1, 2),
            "target_2": round(self.target_2, 2),
            "exit_time": str(self.exit_time),
            "exit_price": round(self.exit_price, 2),
            "exit_reason": self.exit_reason,
            "pnl_points": round(self.pnl_points, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "holding_minutes": round(self.holding_minutes, 1),
            "shortlist_reasons": "; ".join(self.shortlist_reasons),
        }


class OptionEdgeBacktest:
    """
    Day-by-day historical replay of the IOE pipeline.
    """

    def __init__(
        self,
        max_trades_per_day: int = 3,
        min_risk_reward: float = 1.2,
    ) -> None:
        self.engine = get_engine()
        self._shortlist = FOIntradayShortlist()
        self._regime = RegimeDetector()
        self._edge_scorer = IntradayEdgeScorer(horizon="INTRADAY")
        self._entry_trigger = SmartEntryTrigger(horizon="INTRADAY")
        self._strike_selector = OptionStrikeSelector()

        self.max_trades_per_day = max_trades_per_day
        self.min_risk_reward = min_risk_reward

    def run(self, output_dir: Path, start_date: date | None = None, end_date: date | None = None) -> Path:
        """Run the full backtest across available dates."""
        output_dir.mkdir(parents=True, exist_ok=True)

        trading_dates = self._get_trading_dates(start_date, end_date)
        LOGGER.info(f"Running IOE backtest across {len(trading_dates)} trading days.")

        all_trades: list[dict] = []

        for i, td in enumerate(trading_dates):
            print(f"  [{i+1}/{len(trading_dates)}] {td}...", end=" ", flush=True)
            day_trades = self._simulate_day(td)
            for t in day_trades:
                all_trades.append(t.to_dict())
            print(f"{len(day_trades)} trades")

        # ── Save results ──────────────────────────────────────────────────
        trades_df = pd.DataFrame(all_trades)
        trades_path = output_dir / "option_edge_trades.csv"
        trades_df.to_csv(trades_path, index=False)

        summary = self._build_summary(trades_df)
        summary_path = output_dir / "option_edge_summary.csv"
        summary.to_csv(summary_path, index=False)

        report = self._build_report(trades_df, summary)
        report_path = output_dir / "option_edge_report.md"
        report_path.write_text(report)

        LOGGER.info(f"Backtest complete. {len(all_trades)} total trades. Results in {output_dir}")
        return output_dir

    def _simulate_day(self, td: date) -> list[BacktestTrade]:
        """Simulate the IOE pipeline for a single trading day."""
        trades: list[BacktestTrade] = []

        try:
            # Stage 1: Shortlist
            candidates = self._shortlist.shortlist(td)
            if not candidates:
                return trades

            shortlist_lookup = {c.symbol: c for c in candidates}
            shortlisted_symbols = set(shortlist_lookup.keys())

            # Stage 2: Regime
            regime_result = self._regime.detect(td)

            # Stage 3: Edge scoring
            edge_scores = self._edge_scorer.scan(target_date=td)
            if not edge_scores:
                return trades

            # Filter to shortlisted + above threshold
            qualifying = []
            for edge in edge_scores:
                if edge.symbol not in shortlisted_symbols:
                    continue
                if edge.final_score < regime_result.score_threshold:
                    continue
                candidate = shortlist_lookup[edge.symbol]
                qualifying.append((edge, candidate))

            if not qualifying:
                return trades

            # Sort by score
            qualifying.sort(key=lambda x: x[0].final_score, reverse=True)

            # Stage 4: Get 15m data for entry triggers & trade simulation
            df_15m = self._edge_scorer._fetch_15m_data(td)
            if df_15m.empty:
                return trades

            symbol_15m = {
                sym: grp.sort_values("timestamp")
                for sym, grp in df_15m.groupby("symbol")
            }

            # Process qualifying stocks
            for edge, candidate in qualifying:
                if len(trades) >= self.max_trades_per_day:
                    break

                df_sym = symbol_15m.get(edge.symbol)
                if df_sym is None or df_sym.empty:
                    continue

                # Smart entry trigger
                entry = self._entry_trigger.evaluate(
                    direction=edge.direction,
                    ltp=edge.ltp,
                    atr=edge.atr,
                    df_base=df_sym,
                    target_date=td,
                )

                if entry is None:
                    continue

                # Risk-reward check
                risk = abs(entry.entry_price - entry.stop_loss)
                reward = abs(entry.target_1 - entry.entry_price)
                rr = reward / risk if risk > 0 else 0
                if rr < self.min_risk_reward:
                    continue

                # Simulate the trade on remaining 15m candles
                trade = self._simulate_trade(
                    td=td,
                    symbol=edge.symbol,
                    sector=edge.sector,
                    direction=edge.direction,
                    edge_score=edge.final_score,
                    regime=regime_result.regime.value,
                    entry_type=entry.trigger_type,
                    entry_price=entry.entry_price,
                    stop_loss=entry.stop_loss,
                    target_1=entry.target_1,
                    target_2=entry.target_2,
                    df_sym=df_sym,
                    shortlist_reasons=candidate.reasons,
                )

                if trade:
                    trades.append(trade)

        except Exception as e:
            LOGGER.error(f"Error simulating {td}: {e}")

        return trades

    def _simulate_trade(
        self,
        td: date,
        symbol: str,
        sector: str,
        direction: str,
        edge_score: float,
        regime: str,
        entry_type: str,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        df_sym: pd.DataFrame,
        shortlist_reasons: list[str],
    ) -> BacktestTrade | None:
        """
        Simulate a single trade on 15m candles.
        Entry at entry_price, exit at SL, target_1, or EOD square-off.
        """
        # Filter to today's candles only
        today_candles = df_sym[df_sym["timestamp"].dt.date == td].sort_values("timestamp")
        if today_candles.empty:
            return None

        is_long = direction == "CALL"

        # Find the first candle where price reaches entry level
        entry_time = None
        for _, candle in today_candles.iterrows():
            ts = candle["timestamp"]

            # Don't enter after cutoff
            if ts.time() > ENTRY_CUTOFF:
                break

            # Check if this candle reaches our entry level
            if is_long:
                if candle["high"] >= entry_price:
                    entry_time = ts
                    break
            else:
                if candle["low"] <= entry_price:
                    entry_time = ts
                    break

        if entry_time is None:
            return None

        # Now simulate from entry candle forward
        trade = BacktestTrade(
            trade_date=td,
            symbol=symbol,
            sector=sector,
            direction=direction,
            edge_score=edge_score,
            regime=regime,
            entry_type=entry_type,
            entry_time=entry_time,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            shortlist_reasons=shortlist_reasons,
        )

        # Walk forward through candles after entry
        post_entry = today_candles[today_candles["timestamp"] > entry_time]

        for _, candle in post_entry.iterrows():
            ts = candle["timestamp"]

            if is_long:
                # Check SL hit
                if candle["low"] <= stop_loss:
                    trade.exit_time = ts
                    trade.exit_price = stop_loss
                    trade.exit_reason = "SL_HIT"
                    break

                # Check target 1 hit
                if candle["high"] >= target_1:
                    trade.exit_time = ts
                    trade.exit_price = target_1
                    trade.exit_reason = "TARGET_1"
                    break
            else:
                # PUT direction
                # Check SL hit (price goes up above SL)
                if candle["high"] >= stop_loss:
                    trade.exit_time = ts
                    trade.exit_price = stop_loss
                    trade.exit_reason = "SL_HIT"
                    break

                # Check target 1 hit (price goes down to target)
                if candle["low"] <= target_1:
                    trade.exit_time = ts
                    trade.exit_price = target_1
                    trade.exit_reason = "TARGET_1"
                    break

            # EOD square-off
            if ts.time() >= SQUARE_OFF_TIME:
                trade.exit_time = ts
                trade.exit_price = float(candle["close"])
                trade.exit_reason = "EOD_SQUARE_OFF"
                break

        # If no exit yet, close at the last candle
        if trade.exit_time is None:
            last = today_candles.iloc[-1]
            trade.exit_time = last["timestamp"]
            trade.exit_price = float(last["close"])
            trade.exit_reason = "EOD_SQUARE_OFF"

        # Calculate P&L
        if is_long:
            trade.pnl_points = trade.exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - trade.exit_price

        trade.pnl_pct = (trade.pnl_points / trade.entry_price) * 100 if trade.entry_price > 0 else 0.0
        trade.holding_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60.0

        return trade

    def _get_trading_dates(self, start_date: date | None, end_date: date | None) -> list[date]:
        """Get distinct trading dates from the 15m database."""
        query = text("""
            SELECT DISTINCT date(timestamp) as d
            FROM ohlcv_15m
            WHERE symbol NOT LIKE '%INDEX%'
            ORDER BY d ASC
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchall()

        dates = [date.fromisoformat(row[0]) for row in result if row[0]]

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        # Skip the first 10 days (need lookback for indicators)
        if len(dates) > 10:
            dates = dates[10:]

        return dates

    def _build_summary(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Build per-month summary statistics."""
        if trades_df.empty:
            return pd.DataFrame()

        trades_df = trades_df.copy()
        trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"])
        trades_df["month"] = trades_df["trade_date"].dt.to_period("M").astype(str)

        rows = []
        for month, grp in trades_df.groupby("month"):
            wins = grp[grp["pnl_pct"] > 0]
            losses = grp[grp["pnl_pct"] <= 0]

            rows.append({
                "month": month,
                "total_trades": len(grp),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(grp) * 100 if len(grp) > 0 else 0,
                "avg_pnl_pct": grp["pnl_pct"].mean(),
                "total_pnl_pct": grp["pnl_pct"].sum(),
                "avg_win_pct": wins["pnl_pct"].mean() if not wins.empty else 0,
                "avg_loss_pct": losses["pnl_pct"].mean() if not losses.empty else 0,
                "max_win_pct": grp["pnl_pct"].max(),
                "max_loss_pct": grp["pnl_pct"].min(),
                "avg_holding_min": grp["holding_minutes"].mean(),
                "regime_trending": len(grp[grp["regime"] == "TRENDING"]),
                "regime_neutral": len(grp[grp["regime"] == "NEUTRAL"]),
                "regime_choppy": len(grp[grp["regime"] == "CHOPPY"]),
            })

        return pd.DataFrame(rows)

    def _build_report(self, trades_df: pd.DataFrame, summary: pd.DataFrame) -> str:
        """Build a markdown report."""
        if trades_df.empty:
            return "# Option Edge Backtest Report\n\nNo trades generated."

        total = len(trades_df)
        wins = len(trades_df[trades_df["pnl_pct"] > 0])
        win_rate = wins / total * 100 if total > 0 else 0
        avg_pnl = trades_df["pnl_pct"].mean()
        total_pnl = trades_df["pnl_pct"].sum()
        avg_hold = trades_df["holding_minutes"].mean()

        # By exit reason
        by_exit = trades_df.groupby("exit_reason").agg(
            count=("pnl_pct", "count"),
            avg_pnl=("pnl_pct", "mean"),
            total_pnl=("pnl_pct", "sum"),
        ).round(3)

        # By direction
        by_dir = trades_df.groupby("direction").agg(
            count=("pnl_pct", "count"),
            win_rate=("pnl_pct", lambda x: (x > 0).sum() / len(x) * 100),
            avg_pnl=("pnl_pct", "mean"),
        ).round(3)

        # By regime
        by_regime = trades_df.groupby("regime").agg(
            count=("pnl_pct", "count"),
            win_rate=("pnl_pct", lambda x: (x > 0).sum() / len(x) * 100),
            avg_pnl=("pnl_pct", "mean"),
        ).round(3)

        lines = [
            "# Option Edge Backtest Report",
            "",
            "## Overall Performance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Trades | {total} |",
            f"| Win Rate | {win_rate:.1f}% |",
            f"| Avg P&L / Trade | {avg_pnl:+.3f}% |",
            f"| Cumulative P&L | {total_pnl:+.2f}% |",
            f"| Avg Holding Time | {avg_hold:.0f} min |",
            "",
            "## By Exit Reason",
            "",
            by_exit.to_markdown(),
            "",
            "## By Direction",
            "",
            by_dir.to_markdown(),
            "",
            "## By Regime",
            "",
            by_regime.to_markdown(),
            "",
            "## Monthly Summary",
            "",
            summary.to_markdown(index=False) if not summary.empty else "No data",
        ]

        return "\n".join(lines)


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    output = Path("reports/option_edge")
    bt = OptionEdgeBacktest(max_trades_per_day=3, min_risk_reward=1.0)

    print("=" * 70)
    print("🎯 INTRADAY OPTION EDGE — HISTORICAL BACKTEST")
    print("=" * 70)

    bt.run(output)

    print(f"\n✅ Results saved to {output}")
    print(f"  📄 Trades: {output / 'option_edge_trades.csv'}")
    print(f"  📊 Summary: {output / 'option_edge_summary.csv'}")
    print(f"  📝 Report: {output / 'option_edge_report.md'}")
