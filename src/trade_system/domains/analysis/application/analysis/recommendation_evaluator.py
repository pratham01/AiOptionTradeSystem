"""
Recommendation Evaluator & Walk-Forward Performance Verifier.

Automatically evaluates and verifies daily recommended stocks (reversals, breakouts, FVG, SMC)
against actual historical/forward price action. Calculates:
- Trigger Rate (% of recommendations that confirmed entry)
- Win Rate (% hitting Target 1 before Stop Loss)
- Profit Factor & Average PnL %
- Confluence Attribution (identifying which indicators produce the highest win rates)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping
from trade_system.domains.analysis.application.analysis.reversal_scanner import DailyReversalScanner

LOGGER = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    date: date
    symbol: str
    sector: str
    direction: str
    probability: int
    confidence: str
    entry_price: float
    stop_loss: float
    target_1: float
    triggered: bool
    outcome: str  # "WIN", "LOSS", "UNTRIGGERED", "OPEN"
    pnl_pct: float
    mfe_pct: float  # Max Favorable Excursion
    mae_pct: float  # Max Adverse Excursion
    holding_days: int
    confluences: List[str]


class RecommendationEvaluator:
    """
    Evaluates past recommendations by running walk-forward forward-testing on historical daily candles.
    """

    def __init__(self, lookback_days: int = 90, max_holding_days: int = 5) -> None:
        self.lookback_days = lookback_days
        self.max_holding_days = max_holding_days
        self.sector_map = get_sector_mapping()

    def evaluate_reversal_recommendations(
        self,
        min_probability: int = 55,
        target_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate daily reversal recommendations over the lookback period.
        """
        t_date = target_date or date.today()
        engine = get_engine()
        with engine.connect() as conn:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM ohlcv_daily 
                WHERE timestamp >= date(:t_date, :lookback) AND timestamp <= :t_date
                ORDER BY symbol, timestamp ASC
            """)
            df_d = pd.read_sql(query, conn, params={"t_date": t_date.isoformat(), "lookback": f"-{self.lookback_days} days"})

        if df_d.empty:
            return {"error": "No historical daily data available"}

        df_d["timestamp"] = pd.to_datetime(df_d["timestamp"], format="mixed")
        unique_dates = sorted(df_d["timestamp"].dt.date.unique())

        if len(unique_dates) < 30:
            return {"error": "Insufficient historical trading days for evaluation"}

        scanner = DailyReversalScanner(min_probability=min_probability)
        trades: List[TradeOutcome] = []

        # Iterate through past dates
        for i in range(25, len(unique_dates) - 1):
            eval_date = unique_dates[i]
            df_sub = df_d[df_d["timestamp"].dt.date <= eval_date]

            for sym, grp in df_sub.groupby("symbol"):
                if len(grp) < 25:
                    continue

                clean_grp = grp.sort_values("timestamp").reset_index(drop=True)
                setup = scanner.evaluate_series(sym, clean_grp, eval_date)

                if setup and setup.probability >= min_probability:
                    # Look at future candles
                    fwd_candles = df_d[(df_d["symbol"] == sym) & (df_d["timestamp"].dt.date > eval_date)].sort_values("timestamp").head(self.max_holding_days)
                    if fwd_candles.empty:
                        continue

                    last_bar = clean_grp.iloc[-1]
                    first_fwd = fwd_candles.iloc[0]
                    is_call = (setup.direction == "CALL")

                    # Check Confirmation Trigger
                    triggered = False
                    entry_p = 0.0

                    if is_call:
                        trigger_lvl = float(last_bar["high"])
                        if float(first_fwd["high"]) >= trigger_lvl:
                            triggered = True
                            entry_p = max(float(first_fwd["open"]), trigger_lvl)
                    else:  # PUT
                        trigger_lvl = float(last_bar["low"])
                        if float(first_fwd["low"]) <= trigger_lvl:
                            triggered = True
                            entry_p = min(float(first_fwd["open"]), trigger_lvl)

                    if not triggered:
                        trades.append(TradeOutcome(
                            date=eval_date,
                            symbol=sym.replace("NSE:", "").replace("-EQ", ""),
                            sector=self.sector_map.get(sym, "OTHER"),
                            direction=setup.direction,
                            probability=setup.probability,
                            confidence=setup.confidence,
                            entry_price=setup.ltp,
                            stop_loss=setup.stop_loss,
                            target_1=setup.target_1,
                            triggered=False,
                            outcome="UNTRIGGERED",
                            pnl_pct=0.0,
                            mfe_pct=0.0,
                            mae_pct=0.0,
                            holding_days=0,
                            confluences=setup.confluences
                        ))
                        continue

                    # Evaluate Forward Outcome
                    sl = setup.stop_loss
                    t1 = setup.target_1
                    hit_target = False
                    hit_sl = False
                    pnl = 0.0
                    max_favorable = 0.0
                    max_adverse = 0.0
                    h_days = 0

                    for _, f_row in fwd_candles.iterrows():
                        h_days += 1
                        f_h = float(f_row["high"])
                        f_l = float(f_row["low"])

                        if is_call:
                            fav = (f_h - entry_p) / entry_p * 100
                            adv = (f_l - entry_p) / entry_p * 100
                            max_favorable = max(max_favorable, fav)
                            max_adverse = min(max_adverse, adv)

                            if f_l <= sl:
                                hit_sl = True
                                pnl = (sl - entry_p) / entry_p * 100
                                break
                            if f_h >= t1:
                                hit_target = True
                                pnl = (t1 - entry_p) / entry_p * 100
                                break
                        else:  # PUT
                            fav = (entry_p - f_l) / entry_p * 100
                            adv = (entry_p - f_h) / entry_p * 100
                            max_favorable = max(max_favorable, fav)
                            max_adverse = min(max_adverse, adv)

                            if f_h >= sl:
                                hit_sl = True
                                pnl = (entry_p - sl) / entry_p * 100
                                break
                            if f_l <= t1:
                                hit_target = True
                                pnl = (entry_p - t1) / entry_p * 100
                                break

                    if not hit_target and not hit_sl:
                        exit_p = float(fwd_candles.iloc[-1]["close"])
                        pnl = (exit_p - entry_p) / entry_p * 100 if is_call else (entry_p - exit_p) / entry_p * 100
                        outcome = "WIN" if pnl > 0 else "LOSS"
                    else:
                        outcome = "WIN" if hit_target else "LOSS"

                    trades.append(TradeOutcome(
                        date=eval_date,
                        symbol=sym.replace("NSE:", "").replace("-EQ", ""),
                        sector=self.sector_map.get(sym, "OTHER"),
                        direction=setup.direction,
                        probability=setup.probability,
                        confidence=setup.confidence,
                        entry_price=entry_p,
                        stop_loss=sl,
                        target_1=t1,
                        triggered=True,
                        outcome=outcome,
                        pnl_pct=round(pnl, 2),
                        mfe_pct=round(max_favorable, 2),
                        mae_pct=round(max_adverse, 2),
                        holding_days=h_days,
                        confluences=setup.confluences
                    ))

        # Aggregate Metrics
        total_recs = len(trades)
        triggered_trades = [t for t in trades if t.triggered]
        n_trig = len(triggered_trades)
        
        if n_trig == 0:
            return {"total_recommendations": total_recs, "triggered_trades": 0}

        wins = [t for t in triggered_trades if t.outcome == "WIN"]
        losses = [t for t in triggered_trades if t.outcome == "LOSS"]
        
        win_rate = len(wins) / n_trig * 100
        avg_pnl = sum(t.pnl_pct for t in triggered_trades) / n_trig
        total_win_pnl = sum(t.pnl_pct for t in wins)
        total_loss_pnl = abs(sum(t.pnl_pct for t in losses))
        profit_factor = round(total_win_pnl / total_loss_pnl, 2) if total_loss_pnl > 0 else 0.0

        # Confluence Performance Attribution
        confluence_stats: Dict[str, Dict[str, Any]] = {}
        for t in triggered_trades:
            for c in t.confluences:
                if c not in confluence_stats:
                    confluence_stats[c] = {"trades": 0, "wins": 0, "pnl_sum": 0.0}
                confluence_stats[c]["trades"] += 1
                if t.outcome == "WIN":
                    confluence_stats[c]["wins"] += 1
                confluence_stats[c]["pnl_sum"] += t.pnl_pct

        confluence_table = []
        for c, stats in confluence_stats.items():
            if stats["trades"] >= 10:
                c_win_rate = round(stats["wins"] / stats["trades"] * 100, 1)
                c_avg_pnl = round(stats["pnl_sum"] / stats["trades"], 2)
                confluence_table.append({
                    "Confluence": c,
                    "Trades": stats["trades"],
                    "Win Rate %": c_win_rate,
                    "Avg PnL %": c_avg_pnl
                })
        confluence_table.sort(key=lambda x: -x["Win Rate %"])

        return {
            "total_recommendations": total_recs,
            "triggered_trades": n_trig,
            "trigger_rate_pct": round(n_trig / total_recs * 100, 1) if total_recs > 0 else 0.0,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": profit_factor,
            "avg_pnl_pct": round(avg_pnl, 2),
            "avg_mfe_pct": round(sum(t.mfe_pct for t in triggered_trades) / n_trig, 2),
            "avg_mae_pct": round(sum(t.mae_pct for t in triggered_trades) / n_trig, 2),
            "confluence_attribution": confluence_table,
            "trades_df": pd.DataFrame([t.__dict__ for t in triggered_trades])
        }
