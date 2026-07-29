"""
gamma_blast_backtest.py — Replay simulator for 0DTE late-day index option buying.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.analysis.application.analysis.gamma_blast_strategy import GammaBlastDetector

LOGGER = logging.getLogger(__name__)

@dataclass
class GammaTrade:
    trade_date: date
    underlying: str
    direction: str          # "CALL" or "PUT"
    trigger_time: pd.Timestamp
    trigger_price: float
    option_symbol: str
    entry_premium: float
    exit_time: pd.Timestamp
    exit_premium: float
    pnl_points: float
    pnl_pct: float
    exit_reason: str        # "TARGET_300%", "ZERO_EXPIRED", "EOD_CLOSE"

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date.isoformat(),
            "underlying": self.underlying,
            "direction": self.direction,
            "trigger_time": str(self.trigger_time),
            "trigger_price": round(self.trigger_price, 2),
            "option_symbol": self.option_symbol,
            "entry_premium": round(self.entry_premium, 2),
            "exit_time": str(self.exit_time),
            "exit_premium": round(self.exit_premium, 2),
            "pnl_points": round(self.pnl_points, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "exit_reason": self.exit_reason,
        }

class GammaBlastBacktest:
    """
    Simulates the 0DTE Gamma Blast strategy across Nifty and Sensex weekly expiries.
    """

    def __init__(self, target_gain_pct: float = 300.0) -> None:
        self.engine = get_engine()
        self.detector = GammaBlastDetector()
        self.target_gain_pct = target_gain_pct

    def run(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all trading dates in database
        dates = self._get_available_dates()
        print(f"Found {len(dates)} total dates in ohlcv_1m. Running 0DTE analysis...")
        
        trades: list[GammaTrade] = []
        
        for d in dates:
            # Check Nifty
            nifty_trades = self._analyze_day_symbol("NSE:NIFTY50-INDEX", d)
            trades.extend(nifty_trades)
            
            # Check Sensex
            sensex_trades = self._analyze_day_symbol("BSE:SENSEX-INDEX", d)
            trades.extend(sensex_trades)

        # Output results
        trades_df = pd.DataFrame([t.to_dict() for t in trades])
        if not trades_df.empty:
            trades_df.to_csv(output_dir / "gamma_blast_trades.csv", index=False)
            
            summary = self._build_summary(trades_df)
            summary.to_csv(output_dir / "gamma_blast_summary.csv", index=False)
            
            report = self._build_report(trades_df, summary)
            (output_dir / "gamma_blast_report.md").write_text(report)
        else:
            print("No trades generated during the backtest.")
            
        return output_dir

    def _get_available_dates(self) -> list[date]:
        query = text("""
            SELECT DISTINCT date(timestamp) as d
            FROM ohlcv_1m
            ORDER BY d ASC
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return [date.fromisoformat(r[0]) for r in result if r[0]]

    def _analyze_day_symbol(self, symbol: str, d: date) -> list[GammaTrade]:
        trades = []
        
        # 1. Verify if it is an expiry day
        if not self.detector.is_expiry_day(symbol, d):
            return trades
            
        # 2. Fetch 1m candles for target day
        query = text("""
            SELECT timestamp, open, high, low, close
            FROM ohlcv_1m
            WHERE symbol = :sym AND date(timestamp) = :d
            ORDER BY timestamp ASC
        """)
        with self.engine.connect() as conn:
            df_1m = pd.read_sql(query, conn, params={"sym": symbol, "d": d.isoformat()})
            
        if len(df_1m) < 180:  # Must have a full day's data
            return trades
            
        df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], format="mixed")
        
        # 3. Fetch all option chain snapshots for this day
        query_oc = text("""
            SELECT timestamp, symbol, strike, option_type, ltp
            FROM option_chain_data
            WHERE underlying_symbol = :sym AND date(timestamp) = :d
            ORDER BY timestamp, symbol
        """)
        with self.engine.connect() as conn:
            df_oc = pd.read_sql(query_oc, conn, params={"sym": symbol, "d": d.isoformat()})
            
        if df_oc.empty:
            return trades
            
        df_oc["timestamp"] = pd.to_datetime(df_oc["timestamp"], format="mixed")
        
        # 4. Get consolidation range between 2:00 PM and 2:30 PM
        obs_high, obs_low = self.detector.get_consolidation_range(df_1m, d)
        if obs_high <= 0 or obs_low <= 0:
            return trades
            
        # 5. Check Bollinger Squeeze
        # Squeeze indicator on 1m chart (using rolling 20 period on close)
        is_squeezed, _ = self.detector.check_volatility_squeeze(df_1m[df_1m["timestamp"].dt.time < dt_time(14, 30)])
        
        # 6. Walk forward from 2:30 PM to 3:05 PM looking for breakout
        trade_candidates = df_1m[(df_1m["timestamp"].dt.time >= dt_time(14, 30)) & (df_1m["timestamp"].dt.time <= dt_time(15, 5))]
        
        trigger_trade = None
        for _, row in trade_candidates.iterrows():
            ts = row["timestamp"]
            spot = float(row["close"])
            
            direction, trigger_price = self.detector.evaluate_breakout(
                current_time=ts.to_pydatetime(),
                current_spot=spot,
                obs_high=obs_high,
                obs_low=obs_low,
                is_squeezed=is_squeezed
            )
            
            if direction:
                # Triggered breakout!
                # Fetch option chain snapshot closest to this timestamp
                snap_oc = df_oc[df_oc["timestamp"] <= ts].tail(250)
                if snap_oc.empty:
                    snap_oc = df_oc[df_oc["timestamp"] >= ts].head(250)
                    
                opt_res = self.detector.select_0dte_option(symbol, spot, direction, snap_oc)
                if opt_res:
                    opt_sym, opt_premium = opt_res
                    trigger_trade = {
                        "direction": direction,
                        "trigger_time": ts,
                        "trigger_price": trigger_price,
                        "option_symbol": opt_sym,
                        "entry_premium": opt_premium
                    }
                    break
                    
        if trigger_trade is None:
            return trades
            
        # 7. Simulate Trade Outcome
        # We trace the premium of the selected option symbol from trigger_time forward
        post_trigger_oc = df_oc[(df_oc["symbol"] == trigger_trade["option_symbol"]) & (df_oc["timestamp"] > trigger_trade["trigger_time"])].sort_values("timestamp")
        
        if post_trigger_oc.empty:
            return trades
            
        entry_premium = trigger_trade["entry_premium"]
        target_premium = entry_premium * (1.0 + self.target_gain_pct / 100.0)
        
        exit_time = None
        exit_premium = 0.0
        exit_reason = "EOD_CLOSE"
        
        for _, oc_row in post_trigger_oc.iterrows():
            curr_ts = oc_row["timestamp"]
            curr_prem = float(oc_row["ltp"])
            
            # Target Hit!
            if curr_prem >= target_premium:
                exit_time = curr_ts
                exit_premium = target_premium
                exit_reason = f"TARGET_{int(self.target_gain_pct)}%"
                break
                
            # Zero / Expired (if premium drops to ₹0.05 or lower)
            if curr_prem <= 0.05:
                exit_time = curr_ts
                exit_premium = 0.0
                exit_reason = "ZERO_EXPIRED"
                break
                
            # Forced Close cutoff at 3:12 PM
            if curr_ts.time() >= dt_time(15, 12):
                exit_time = curr_ts
                exit_premium = curr_prem
                exit_reason = "EOD_CLOSE"
                break
                
        if exit_time is None:
            last = post_trigger_oc.iloc[-1]
            exit_time = last["timestamp"]
            exit_premium = float(last["ltp"])
            exit_reason = "EOD_CLOSE"
            
        pnl = exit_premium - entry_premium
        pnl_pct = (pnl / entry_premium) * 100 if entry_premium > 0 else 0.0
        
        trades.append(GammaTrade(
            trade_date=d,
            underlying=symbol,
            direction=trigger_trade["direction"],
            trigger_time=trigger_trade["trigger_time"],
            trigger_price=trigger_trade["trigger_price"],
            option_symbol=trigger_trade["option_symbol"],
            entry_premium=entry_premium,
            exit_time=exit_time,
            exit_premium=exit_premium,
            pnl_points=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason
        ))
        
        return trades

    def _build_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        df["month"] = pd.to_datetime(df["trade_date"], format="mixed").dt.to_period("M").astype(str)
        rows = []
        for month, grp in df.groupby("month"):
            wins = grp[grp["pnl_pct"] > 0]
            rows.append({
                "month": month,
                "total_trades": len(grp),
                "wins": len(wins),
                "losses": len(grp) - len(wins),
                "win_rate": len(wins) / len(grp) * 100 if len(grp) > 0 else 0,
                "avg_pnl_pct": grp["pnl_pct"].mean(),
                "total_pnl_pct": grp["pnl_pct"].sum(),
                "avg_winner_pct": wins["pnl_pct"].mean() if not wins.empty else 0,
            })
        return pd.DataFrame(rows)

    def _build_report(self, df: pd.DataFrame, summary: pd.DataFrame) -> str:
        total = len(df)
        wins = len(df[df["pnl_pct"] > 0])
        win_rate = wins / total * 100 if total > 0 else 0
        avg_pnl = df["pnl_pct"].mean()
        total_pnl = df["pnl_pct"].sum()
        
        by_reason = df.groupby("exit_reason").agg(count=("pnl_pct", "count"), avg_pnl=("pnl_pct", "mean")).round(2)
        by_symbol = df.groupby("underlying").agg(count=("pnl_pct", "count"), win_rate=("pnl_pct", lambda x: (x > 0).sum() / len(x) * 100), avg_pnl=("pnl_pct", "mean")).round(2)
        
        lines = [
            "# 0DTE Gamma Blast Strategy Backtest Report",
            "",
            "## Overall Performance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Trades | {total} |",
            f"| Win Rate | {win_rate:.1f}% |",
            f"| Avg P&L / Trade | {avg_pnl:+.1f}% |",
            f"| Cumulative P&L | {total_pnl:+.1f}% |",
            "",
            "## By Exit Reason",
            "",
            by_reason.to_markdown(),
            "",
            "## By Symbol",
            "",
            by_symbol.to_markdown(),
            "",
            "## Monthly Summary",
            "",
            summary.to_markdown(index=False),
        ]
        return "\n".join(lines)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = Path("reports/gamma_blast")
    bt = GammaBlastBacktest()
    print("Executing historical Nifty/Sensex 0DTE Gamma Blast Backtest...")
    bt.run(output)
    print(f"Backtest complete. Reports saved in {output}")
