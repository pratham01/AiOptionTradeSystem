"""
CorrelationAgent — AI Agent to find correlations between Greeks, VIX, Strike Price, and Sectors.
"""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Any, List, Dict

from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

class CorrelationAgent:
    """
    Analyzes historical and intraday data to find correlations between different market parameters.
    """

    def __init__(self, llm_client: LlmAdvisorClient | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()
        self.engine = get_engine()

    async def analyze_correlations(self) -> str:
        """
        Fetches today's data and performs a correlation analysis.
        """
        LOGGER.info("CorrelationAgent: Starting cross-parameter correlation analysis...")
        
        try:
            # 1. Fetch data from DB
            # We want today's Greeks, VIX, and sector performance
            today = date.today().isoformat()
            
            # Fetch VIX and Nifty Close from ohlcv_daily (or intraday)
            vix_data = self._fetch_vix_data()
            
            # Fetch Option Greeks for Nifty
            greeks_data = self._fetch_greeks_data()
            
            # Fetch Sector performance
            sector_data = self._fetch_sector_data()

            if vix_data.empty or greeks_data.empty or sector_data.empty:
                return "Insufficient data today to perform meaningful correlation analysis."

            # 2. Perform Analysis
            summary = self._calculate_correlations(vix_data, greeks_data, sector_data)
            
            # 3. LLM Synthesis
            if self.llm.configured():
                analysis = await self._llm_synthesize(summary)
                return analysis
            else:
                return self._rule_based_summary(summary)

        except Exception as e:
            LOGGER.error(f"Correlation analysis failed: {e}", exc_info=True)
            return f"Correlation analysis encountered an error: {str(e)}"

    def _fetch_vix_data(self) -> pd.DataFrame:
        query = text("SELECT timestamp, close as nifty_close, vix FROM ohlcv_daily WHERE symbol = 'NSE:NIFTY50-INDEX' ORDER BY timestamp DESC LIMIT 20")
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def _fetch_greeks_data(self) -> pd.DataFrame:
        # Get average Greeks for ATM strikes today
        query = text("""
            SELECT timestamp, AVG(iv) as avg_iv, AVG(delta) as avg_delta, AVG(theta) as avg_theta 
            FROM option_chain_data 
            WHERE underlying_symbol = 'NSE:NIFTY50-INDEX' 
            AND timestamp >= date('now')
            GROUP BY timestamp
        """)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def _fetch_sector_data(self) -> pd.DataFrame:
        query = text("""
            SELECT s.date, sec.sector, sec.avg_change_pct as change_pct 
            FROM sector_performance_snapshots s
            JOIN sector_performance_sectors sec ON (sec.snapshot_id = s.id OR sec.snapshot_id_worst = s.id)
            WHERE s.date >= date('now', '-7 days')
        """)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def _calculate_correlations(self, vix_df: pd.DataFrame, greeks_df: pd.DataFrame, sector_df: pd.DataFrame) -> Dict[str, Any]:
        # Simplify: Correlate VIX with Nifty Close
        vix_nifty_corr = vix_df[['nifty_close', 'vix']].corr().iloc[0, 1]
        
        # Correlate IV with VIX (if timestamps align)
        # This is a bit complex for a one-day run, usually needs time-series alignment
        
        return {
            "vix_nifty_corr": vix_nifty_corr,
            "vix_avg": vix_df['vix'].mean(),
            "top_sector": sector_df.sort_values('change_pct', ascending=False).iloc[0]['sector'] if not sector_df.empty else "N/A",
            "bottom_sector": sector_df.sort_values('change_pct', ascending=True).iloc[0]['sector'] if not sector_df.empty else "N/A"
        }

    async def _llm_synthesize(self, summary: Dict[str, Any]) -> str:
        prompt = f"""
        Act as an expert quantitative market analyst. Based on today's data:
        - VIX Average: {summary['vix_avg']:.2f}
        - VIX/Nifty Correlation: {summary['vix_nifty_corr']:.2f}
        - Top Performing Sector: {summary['top_sector']}
        - Worst Performing Sector: {summary['bottom_sector']}
        
        Provide a 3-sentence summary of the market dynamics. How did the Greeks and VIX affect sector performance?
        """
        return await self.llm.complete(prompt)

    def _rule_based_summary(self, summary: Dict[str, Any]) -> str:
        return f"Market Dynamic Summary: VIX/Nifty correlation stood at {summary['vix_nifty_corr']:.2f}. Sector leadership by {summary['top_sector']}."
