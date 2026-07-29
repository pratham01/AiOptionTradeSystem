from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import time

from trade_system.domains.market_data.infrastructure.data.fo_universe import FO_METADATA, get_fo_universe
from trade_system.domains.strategy.application.indicators.volume_delta import VolumeDeltaIndicator
from trade_system.domains.strategy.application.indicators.compression import CompressionIndicator
from trade_system.domains.strategy.application.indicators.volume_profile import VolumeProfileIndicator
from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)

@dataclass
class Recommendation:
    symbol: str
    date: str
    entry_price: float
    sector: str
    scores: dict[str, float]
    total_score: float
    outcome: str | None = None  # WIN, LOSS, PENDING
    pnl_pct: float | None = None

class FoRecommenderAgent:
    """
    Agent that monitors F&O stocks, suggests trades, and uses a feedback loop
     to improve its recommendation accuracy over time.
    """

    def __init__(self, broker: FyersBroker):
        self.broker = broker
        self.storage_path = Path("data/agent_feedback.json")
        self.weights_path = Path("data/agent_weights.json")
        self.default_weights = {
            "sector_strength": 0.3,
            "volume_delta": 0.4,
            "compression": 0.2,
            "value_area": 0.1
        }
        self.weights = self._load_weights()
        self.vd_ind = VolumeDeltaIndicator()
        self.comp_ind = CompressionIndicator()
        self.vp_ind = VolumeProfileIndicator(price_step=5.0)

    def _load_weights(self) -> dict[str, float]:
        if self.weights_path.exists():
            try:
                return json.loads(self.weights_path.read_text())
            except:
                pass
        return self.default_weights.copy()

    def _save_weights(self):
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_path.write_text(json.dumps(self.weights, indent=2))

    def _load_history(self) -> list[dict[str, Any]]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text())
            except:
                pass
        return []

    def _save_history(self, history: list[dict[str, Any]]):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(history, indent=2))

    async def generate_daily_picks(self) -> list[Recommendation]:
        """Generate high-conviction picks for the current session."""
        symbols = get_fo_universe()
        picks = []
        
        # 1. Get Sector Performance (simplified for speed)
        # In real-world, we'd use the daily sector report
        
        print(f"Agent Analyzing {len(symbols)} stocks using weights: {self.weights}")
        
        for sym in symbols[:20]:  # Limit for demo/speed
            try:
                time.sleep(0.5)
                # Last 2 days of 5m data
                df = self.broker.get_historical_data(sym, "5", 
                    (date.today() - timedelta(days=3)).isoformat(), 
                    date.today().isoformat())
                
                if df is None or df.empty or len(df) < 50:
                    continue

                # Calculate Indicators
                df = self.vd_ind.calculate(df)
                df = self.comp_ind.calculate(df)
                profile = self.vp_ind.calculate(df.tail(75))
                
                latest = df.iloc[-1]
                
                # Component Scoring (0.0 to 1.0)
                scores = {}
                
                # A. Volume Delta Score
                # High positive CVD and positive last bar delta
                scores["volume_delta"] = 1.0 if latest['delta'] > 0 and latest['cvd'] > 0 else 0.0
                
                # B. Compression Score
                # Higher if currently in a squeeze
                scores["compression"] = 1.0 if latest['is_compressed'] else (latest['range_compression'] / 100)
                
                # C. Value Area Score
                # 1.0 if breaking out of VAH
                va_score = 0.0
                if profile:
                    if latest['close'] > profile.value_area_high: va_score = 1.0
                    elif latest['close'] > profile.point_of_control: va_score = 0.5
                scores["value_area"] = va_score
                
                # D. Sector Strength (Placeholder - would come from sector engine)
                scores["sector_strength"] = 0.5 

                # Final Weighted Score
                total_score = sum(scores[k] * self.weights.get(k, 0) for k in scores)
                
                if total_score > 0.6:
                    picks.append(Recommendation(
                        symbol=sym,
                        date=date.today().isoformat(),
                        entry_price=float(latest['close']),
                        sector=FO_METADATA.get(sym, "Other"),
                        scores=scores,
                        total_score=float(total_score)
                    ))
            except Exception as e:
                LOGGER.error(f"Error picking {sym}: {e}")

        # Save to history for feedback loop
        history = self._load_history()
        for p in picks:
            history.append({
                "symbol": p.symbol,
                "date": p.date,
                "entry_price": p.entry_price,
                "sector": p.sector,
                "scores": p.scores,
                "total_score": p.total_score,
                "outcome": "PENDING"
            })
        self._save_history(history)
        
        return sorted(picks, key=lambda x: x.total_score, reverse=True)

    def apply_feedback_loop(self):
        """
        Updates weights based on the performance of 'PENDING' recommendations.
        """
        history = self._load_history()
        updated = False
        
        for rec in history:
            if rec["outcome"] != "PENDING":
                continue
            
            # 1. Check performance (did it move > 1% in the next day?)
            # This requires fetching data for the day AFTER the recommendation
            rec_date = datetime.strptime(rec["date"], "%Y-%m-%d").date()
            if rec_date >= date.today():
                continue # Too early to check
                
            try:
                # Fetch data for checking outcome
                df = self.broker.get_historical_data(rec["symbol"], "D", 
                    rec["date"], (rec_date + timedelta(days=2)).isoformat())
                
                if df is None or len(df) < 2:
                    continue
                    
                # Look at the day following the recommendation
                target_day = df.iloc[1]
                entry = rec["entry_price"]
                high_reach = target_day["high"]
                
                pnl = (high_reach - entry) / entry
                
                if pnl > 0.015: # 1.5% target
                    rec["outcome"] = "WIN"
                    # Improve weights that contributed to this win
                    for component, score in rec["scores"].items():
                        if score > 0.7:
                            self.weights[component] = min(0.8, self.weights[component] + 0.02)
                elif (target_day["low"] - entry) / entry < -0.01: # 1% SL
                    rec["outcome"] = "LOSS"
                    # Reduce weights that were high during this loss
                    for component, score in rec["scores"].items():
                        if score > 0.7:
                            self.weights[component] = max(0.05, self.weights[component] - 0.02)
                else:
                    rec["outcome"] = "NEUTRAL"
                
                rec["pnl_pct"] = float(pnl)
                updated = True
                
            except Exception as e:
                LOGGER.error(f"Feedback error for {rec['symbol']}: {e}")

        if updated:
            # Re-normalize weights to sum to 1.0
            total = sum(self.weights.values())
            for k in self.weights:
                self.weights[k] = round(self.weights[k] / total, 3)
            
            self._save_weights()
            self._save_history(history)
            print(f"Feedback Loop Complete. New Weights: {self.weights}")
        else:
            print("No new outcomes to process for feedback.")
