"""
SupertrendTouchAgent — Scans the universe for stocks retesting their Supertrend line.
Provides high-probability reversal/support suggestions on 15m and Daily timeframes.
"""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Any, List, Dict

from trade_system.core import MarketContext, TradeSuggestion, TradeDirection, TradeHorizon, OptionParams
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.repository import get_market_data, log_agent_thought, save_market_data_batch
from trade_system.infrastructure.data.fo_universe import FO_METADATA
from sqlalchemy.orm import Session

LOGGER = logging.getLogger(__name__)

class SupertrendTouchAgent:
    """
    Identifies 'Pulls back to Supertrend' setups which represent institutional re-entries.
    """

    def __init__(self, broker: Any = None):
        self.broker = broker
        self.engine = get_engine()
        self.st_indicator = SupertrendIndicator(period=7, multiplier=3)

    def _thought(self, message: str, symbol: str | None = None, action: str | None = None):
        LOGGER.info(f"[SupertrendTouchAgent] {message}")
        try:
            with Session(self.engine) as session:
                log_agent_thought(session, "SupertrendTouchAgent", message, symbol, action)
        except: pass

    async def scan_for_touches(self, symbols: List[str], resolution: str = "15", use_cache: bool = False) -> List[TradeSuggestion]:
        """
        Scans provided symbols for Supertrend 'touches'.
        If use_cache is True, prefers DB data over LIVE API.
        """
        mode_label = "CACHED" if use_cache else "LIVE"
        self._thought(f"Starting {mode_label} ST-Touch scan ({resolution}) for {len(symbols)} symbols...", action="SCANNING")
        
        suggestions = []
        horizon = TradeHorizon.INTRADAY if resolution == "15" else TradeHorizon.SWING
        
        with Session(self.engine) as session:
            for symbol in symbols:
                try:
                    df = pd.DataFrame()
                    
                    # 1. Prefer DB if requested
                    if use_cache:
                        limit = 100 if resolution == "15" else 60
                        data = get_market_data(session, symbol, resolution, limit=limit)
                        if data and len(data) >= 20:
                            df = pd.DataFrame([
                                {"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume}
                                for d in data
                            ]).sort_values("timestamp")
                    
                    # 2. Fallback to API if DB is empty or use_cache=False
                    if df.empty:
                        df = self._fetch_live_with_context(symbol, resolution)
                        # Optional: Sync back to DB if fetched from API
                        if df is not None and not df.empty:
                            save_market_data_batch(session, symbol, resolution, df.to_dict("records"))
                            session.commit()

                    if df is None or df.empty or len(df) < 20: 
                        continue
                    
                    # 3. Calculate Supertrend
                    df = self.st_indicator.calculate(df)
                    
                    # 4. Detect "Touch"
                    latest = df.iloc[-1]
                    st_val = float(latest['supertrend'])
                    direction = int(latest['supertrend_direction'])
                    
                    is_touching = latest['low'] <= st_val <= latest['high']
                    
                    if not is_touching: continue
                    
                    # 5. Confirmation
                    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
                    vol_surge = latest['volume'] / vol_ma if vol_ma > 0 else 1.0
                    
                    confidence = 0.75 if vol_surge > 1.1 else 0.60
                    trade_dir = TradeDirection.CALL if direction == 1 else TradeDirection.PUT
                    
                    import uuid
                    suggestions.append(TradeSuggestion(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        timestamp=datetime.now(),
                        direction=trade_dir,
                        horizon=horizon,
                        entry_zone_low=st_val * 0.998,
                        entry_zone_high=st_val * 1.002,
                        target=latest['close'] * 1.02 if trade_dir == TradeDirection.CALL else latest['close'] * 0.98,
                        stop_loss=st_val * 0.99 if trade_dir == TradeDirection.CALL else st_val * 1.01,
                        option_params=OptionParams(direction=trade_dir),
                        confidence=confidence,
                        narrative=f"{mode_label} ST-Touch ({resolution}): Institutional re-entry detected at ₹{st_val:.2f}.",
                        setup_features={"vol_surge": vol_surge, "timeframe": resolution},
                        sector=FO_METADATA.get(symbol, "Other")
                    ))
                    
                    self._thought(f"{mode_label}: Found {trade_dir.value} ST-Touch for {symbol}", symbol=symbol, action="FOUND")
                    
                    if not use_cache:
                        await asyncio.sleep(0.05) # Rate limit protection

                except Exception as e:
                    LOGGER.debug(f"Error in {mode_label} scan for {symbol}: {e}")

        return suggestions

    def _fetch_live_with_context(self, symbol: str, resolution: str) -> pd.DataFrame | None:
        """Fetch fresh live data with historical context."""
        try:
            # Fetch last 5 days to ensure enough bars for Supertrend (especially for 15m)
            end = date.today()
            start = end - timedelta(days=7 if resolution == "15" else 60)
            
            # Using the broker.fetch_history directly to ensure parameters are correct
            # Parameters: symbol, resolution, range_from, range_to
            df = self.broker.fetch_history(
                symbol=symbol, 
                resolution=str(resolution), 
                range_from=start.isoformat(), 
                range_to=end.isoformat()
            )
            
            # Handle if result is empty or not a DataFrame
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return None
            
            # Polymorphic handling (list of objects to DF)
            if isinstance(df, list):
                df = pd.DataFrame([
                    {"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume}
                    for d in df
                ])
            
            return df
        except Exception as e:
            LOGGER.error(f"Live fetch failed for {symbol}: {e}")
            return None

    def _fetch_fresh(self, symbol: str, resolution: str) -> pd.DataFrame:
        """Fetch fresh data from broker and save to DB."""
        try:
            end = date.today()
            start = end - timedelta(days=10 if resolution == "15" else 60)
            # Use fetch_history directly to avoid recursive loops
            df = self.broker.fetch_history(
                symbol=symbol,
                resolution=resolution,
                range_from=start.isoformat(),
                range_to=end.isoformat()
            )
            if not df.empty:
                # Save to DB for future scans
                with Session(self.engine) as session:
                    save_market_data_batch(session, symbol, resolution, df.to_dict("records"))
            return df
        except:
            return pd.DataFrame()
