"""
FoStockSuggesterAgent — Evaluates the F&O universe for high-probability setups.
"""
from __future__ import annotations

import logging
from typing import List
from datetime import datetime

from trade_system.core import TradeSuggestion, MarketContext, DataBroker
from trade_system.application.agent.candidate_screener_agent import CandidateScreenerAgent
from trade_system.application.analysis.top_gainers import NSETop100GainersFetcher
from trade_system.application.analysis.sector_rotation import SectorRotationAnalyzer
from trade_system.application.agent.setup_validator_agent import SetupValidatorAgent

LOGGER = logging.getLogger(__name__)

class FoStockSuggesterAgent:
    """
    Scans the F&O universe and identifies stocks for Intraday or BTST trades.
    """

    def __init__(self, broker: DataBroker) -> None:
        self.broker = broker
        self.screener = CandidateScreenerAgent(broker=self.broker)
        self.gainers_fetcher = NSETop100GainersFetcher(broker=self.broker)
        self.rotation_analyzer = SectorRotationAnalyzer()
        self.validator = SetupValidatorAgent(broker=self.broker)

    async def suggest_trades(
        self, market_context: MarketContext, 
        priority_watchlist: list | None = None,
    ) -> List[TradeSuggestion]:
        """
        Suggest FO stock trades with Institutional Merge verification.
        
        Args:
            market_context: Current market state
            priority_watchlist: Optional list of WatchlistItem from NextDayPredictorAgent.
                               These symbols get scanned first and receive a confidence boost.
        """
        LOGGER.info("FoStockSuggesterAgent: Starting full-universe dynamic scan with Sector RS...")
        
        # Log watchlist if provided
        watchlist_symbols = set()
        if priority_watchlist:
            watchlist_symbols = {w.symbol for w in priority_watchlist}
            wl_str = ", ".join(w.symbol.split(":")[-1].replace("-EQ","") for w in priority_watchlist)
            LOGGER.info(f"FoStockSuggesterAgent: Priority watchlist active ({len(priority_watchlist)} stocks): {wl_str}")
        
        # 0. Get Sector Leadership
        leadership_df = self.rotation_analyzer.get_sector_leadership(lookback_days=15)
        leading_sectors = set()
        if not leadership_df.empty:
            leading_sectors = set(leadership_df[leadership_df['Status'].str.contains("LEADING")]['Sector'].tolist())
            LOGGER.info(f"Leading Sectors identified for RS filter: {leading_sectors}")

        # 1. Fetch real-time top gainers
        try:
            all_quotes = self.gainers_fetcher.fetch_all_quotes(batch_size=200)
            filtered = self.gainers_fetcher.apply_filters(all_quotes)
            top_gainers = self.gainers_fetcher.get_top_gainers(filtered, top_n=50)
            self.gainers_fetcher.save_results(top_gainers)
        except: pass

        # 2. Call the screener
        candidates = self.screener.screen(
            market_context=market_context, 
            include_nifty=False, 
            include_fo=True
        )

        # 3. Call BreakoutScreener point-in-time and merge/boost candidates
        try:
            from trade_system.application.analysis.breakout_screener import BreakoutScreener
            from trade_system.application.agent.candidate_screener_agent import CandidateScore
            from trade_system.core import TradeHorizon
            import pandas as pd
            from sqlalchemy import text
            from trade_system.infrastructure.database.connection import get_engine
            
            target_date = None
            if hasattr(market_context, "timestamp") and market_context.timestamp:
                if isinstance(market_context.timestamp, str):
                    try:
                        target_date = datetime.fromisoformat(market_context.timestamp).date()
                    except:
                        pass
                elif hasattr(market_context.timestamp, "date"):
                    target_date = market_context.timestamp.date()

            screener_obj = BreakoutScreener()
            breakout_alerts = screener_obj.scan_for_breakouts(
                top_sectors_count=3,
                target_date=target_date,
                current_time=market_context.timestamp
            )
            LOGGER.info(f"FoStockSuggesterAgent: BreakoutScreener returned {len(breakout_alerts)} alerts.")
        except Exception as e:
            LOGGER.error(f"Failed to run BreakoutScreener in suggester: {e}")
            breakout_alerts = []

        # Map symbol -> candidate
        cand_map = {c.symbol: c for c in candidates}
        direction_map = {"LONG": "CALL", "SHORT": "PUT"}

        for alert in breakout_alerts:
            symbol = alert["symbol"]
            if symbol in cand_map:
                cand = cand_map[symbol]
                cand.intraday_score = max(cand.intraday_score, 0.95)
                cand.swing_score = max(cand.swing_score, 0.95)
                cand.breakout_type = alert.get("direction", "LONG").lower() + "_breakout"
                cand.pattern = (cand.pattern or "") + ", INTRADAY_ORB"
                cand.is_breakout_bypass = True
                # Boost component scores to ensure it passes SetupValidator confidence
                cand.component_scores["performance"] = max(cand.component_scores.get("performance", 0.0), 0.9)
                cand.component_scores["momentum"] = max(cand.component_scores.get("momentum", 0.0), 0.9)
                cand.component_scores["volume_delta"] = max(cand.component_scores.get("volume_delta", 0.0), 0.9)
                cand.component_scores["value_area"] = max(cand.component_scores.get("value_area", 0.0), 0.9)
            else:
                cand = CandidateScore(
                    symbol=symbol,
                    intraday_score=0.95,
                    swing_score=0.95,
                    horizon=TradeHorizon.INTRADAY,
                    direction=direction_map.get(alert["direction"], "CALL"),
                    sector=alert["sector"],
                    entry_price=alert["close"],
                    volume_surge=alert["volume"] / alert["vol_sma"] if alert["vol_sma"] > 0 else 2.0,
                    is_compressed=False,
                    vol_delta_positive=True,
                    above_poc=True,
                    alignment_score=0.9,
                    breakout_type=alert["direction"].lower() + "_breakout",
                    pattern="INTRADAY_ORB",
                    spread_pct=0.05,
                    is_liquid=True,
                    component_scores={"performance": 0.9, "momentum": 0.9, "volume_delta": 0.9, "value_area": 0.9}
                )
                cand.is_breakout_bypass = True
                cand_map[symbol] = cand

        # ── Boost watchlist symbols from NextDayPredictor ──
        if watchlist_symbols:
            for sym in watchlist_symbols:
                if sym in cand_map:
                    cand = cand_map[sym]
                    # Boost scores but don't override — the intraday data must still confirm
                    cand.intraday_score = max(cand.intraday_score, 0.80)
                    cand.swing_score = max(cand.swing_score, 0.80)
                    cand.pattern = (cand.pattern or "") + ", NEXT_DAY_WATCHLIST"
                    cand.component_scores["watchlist_bonus"] = 0.85
                    LOGGER.info(f"FoStockSuggesterAgent: Boosted watchlist symbol {sym}")

        # Prioritize: watchlist first, then breakout alerts, then rest
        breakout_symbols = [a["symbol"] for a in breakout_alerts]
        priority_order = list(watchlist_symbols) + breakout_symbols
        seen = set()
        prioritized_candidates = []
        for sym in priority_order:
            if sym in cand_map and sym not in seen:
                prioritized_candidates.append(cand_map[sym])
                seen.add(sym)
        for c in candidates:
            if c.symbol not in seen:
                prioritized_candidates.append(c)
                seen.add(c.symbol)

        # Helper to fetch point-in-time local 15m bars to avoid broker API calls in validation
        def fetch_local_15m_bars(symbol: str, target_time: datetime) -> pd.DataFrame:
            engine = get_engine()
            query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_15m 
                WHERE symbol = :symbol AND timestamp <= :target_time
                ORDER BY timestamp DESC
                LIMIT 30
            """
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params={"symbol": symbol, "target_time": target_time})
            if not df.empty:
                df = df.iloc[::-1].reset_index(drop=True)
            return df

        suggestions = []
        for cand in prioritized_candidates[:20]:
            is_breakout_bypass = getattr(cand, "is_breakout_bypass", False)
            is_watchlist = cand.symbol in watchlist_symbols
            if not is_breakout_bypass and not is_watchlist and leading_sectors and cand.sector not in leading_sectors:
                continue
            
            if len(suggestions) >= 5: break
            
            # Fetch local historical bars for precise point-in-time ATR calculation
            target_time = market_context.timestamp if hasattr(market_context, "timestamp") else datetime.now()
            local_bars = fetch_local_15m_bars(cand.symbol, target_time)
            
            # Use the Validator for final Strategy Merge
            sugg = await self.validator.validate(cand, market_context, current_bars=local_bars)
            if sugg:
                if is_breakout_bypass:
                    if "intraday_orb_breakout" not in sugg.tags:
                        sugg.tags.append("intraday_orb_breakout")
                if is_watchlist:
                    if "next_day_watchlist" not in sugg.tags:
                        sugg.tags.append("next_day_watchlist")
                suggestions.append(sugg)

        return suggestions

