"""
AdvanceDeclineAgent — Monitors market breadth across the F&O and Nifty 500 universe.
Provides an institutional 'Sanity Check' on index moves.
"""
from __future__ import annotations

import logging
import pandas as pd
from typing import Dict, Any, List
from trade_system.infrastructure.database.connection import get_engine
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

class AdvanceDeclineAgent:
    def __init__(self):
        self.engine = get_engine()

    def calculate_breadth(self) -> Dict[str, Any]:
        """
        Calculates the Advance-Decline ratio for the current session.
        Uses the top_gainers_stocks table which stores the latest scan results.
        """
        try:
            query = text("""
                SELECT 
                    SUM(CASE WHEN change_pct > 0 THEN 1 ELSE 0 END) as advances,
                    SUM(CASE WHEN change_pct < 0 THEN 1 ELSE 0 END) as declines,
                    COUNT(*) as total
                FROM top_gainers_stocks
                WHERE snapshot_id = (SELECT MAX(id) FROM top_gainers_snapshots)
            """)
            
            with self.engine.connect() as conn:
                res = conn.execute(query).fetchone()
            
            if not res or res[2] == 0:
                return {"ratio": 1.0, "label": "NEUTRAL", "advances": 0, "declines": 0}
            
            adv, dec, total = res
            ratio = adv / (dec if dec > 0 else 1)
            
            if ratio >= 2.0: label = "STRONGLY_BULLISH"
            elif ratio >= 1.2: label = "BULLISH"
            elif ratio <= 0.5: label = "STRONGLY_BEARISH"
            elif ratio <= 0.8: label = "BEARISH"
            else: label = "NEUTRAL"
            
            return {
                "advances": adv,
                "declines": dec,
                "total": total,
                "ratio": round(ratio, 2),
                "label": label,
                "participation_pct": round((adv + dec) / total * 100, 1)
            }
        except Exception as e:
            LOGGER.error(f"Breadth calculation failed: {e}")
            return {"ratio": 1.0, "label": "UNKNOWN"}

if __name__ == "__main__":
    agent = AdvanceDeclineAgent()
    print(agent.calculate_breadth())
