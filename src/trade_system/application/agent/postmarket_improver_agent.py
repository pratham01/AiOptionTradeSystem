"""
PostMarketImproverAgent — Autonomous Post-Market Swarm Orchestrator.
Analyzes the day's trades, identifies missed momentum, calculates correlations,
and evolves the system's memory/skills.
"""
from __future__ import annotations

import logging
import asyncio
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Any, List, Dict

from trade_system.application.evolution.evolution_loop import EvolutionLoop
from trade_system.application.agent.skill_creator_agent import SkillCreatorAgent
from trade_system.application.agent.missed_opportunity_agent import MissedOpportunityAgent
from trade_system.application.agent.correlation_agent import CorrelationAgent
from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.infrastructure.data.fo_universe import get_fo_universe, update_fo_universe_from_nse
from trade_system.infrastructure.data.nse_universe import NSE_UNIVERSE
from trade_system.application.analysis.sectoral_analyzer import SectoralAnalyzer
from trade_system.application.analysis.sector_rotation import SectorRotationAnalyzer
from trade_system.application.analysis.mwpl_analyzer import MwplAnalyzer
from trade_system.application.analysis.vcp_scanner import VcpScannerAgent
from trade_system.application.agent.fo_stock_suggester_agent import FoStockSuggesterAgent
from trade_system.application.agent.market_context_agent import MarketContextAgent
from trade_system.application.agent.next_day_predictor_agent import NextDayPredictorAgent
from trade_system.application.agent.supertrend_touch_agent import SupertrendTouchAgent
from trade_system.application.analysis.fo_historical_service import FOHistoricalService
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.config import Settings
from trade_system.application.analysis.top_gainers import NSETop100GainersFetcher
from trade_system.infrastructure.database import log_agent_thought, get_db_session
from trade_system.infrastructure.database.connection import get_engine
from sqlalchemy.orm import Session

from trade_system.application.analysis.weekly_gainers import NSEWeeklyGainersFetcher
from trade_system.application.agent.market_synthesis_agent import MarketSynthesisAgent
from trade_system.application.advisory.llm import LlmAdvisorClient

LOGGER = logging.getLogger(__name__)

class PostMarketImproverAgent:
    """
    Autonomous Post-Market Swarm Orchestrator.
    """

    def __init__(
        self, 
        broker: Any = None, 
        llm_client: LlmAdvisorClient | None = None,
        settings: Settings | None = None
    ) -> None:
        self.broker = broker
        self.settings = settings or Settings.load()
        self.llm_client = llm_client or LlmAdvisorClient()
        self.evolution_loop = EvolutionLoop()
        self.skill_creator = SkillCreatorAgent(llm_client=self.llm_client)
        self.missed_opportunity_agent = MissedOpportunityAgent(broker=broker, llm_client=self.llm_client)
        self.correlation_agent = CorrelationAgent(llm_client=self.llm_client)
        self.next_day_predictor = NextDayPredictorAgent(broker=broker)
        self.st_touch_agent = SupertrendTouchAgent(broker=broker)
        self.market_synthesis_agent = MarketSynthesisAgent(llm_client=self.llm_client, settings=self.settings)
        self.mwpl_analyzer = MwplAnalyzer()
        self.vcp_scanner = VcpScannerAgent()
        self.sectoral_analyzer = SectoralAnalyzer()
        self.rotation_analyzer = SectorRotationAnalyzer()
        self.fo_suggester = FoStockSuggesterAgent(broker=broker)
        self.market_context_agent = MarketContextAgent(llm_client=llm_client)
        self.fo_historical_service = FOHistoricalService(broker=broker, settings=self.settings)
        
        # Initialize Notifiers
        self.notifier = TelegramNotifier(
            self.settings.telegram.bot_token, 
            self.settings.telegram.chat_id
        )
        self.gainer_notifier = TelegramNotifier(
            self.settings.top_gainer_telegram.bot_token or self.settings.telegram.bot_token,
            self.settings.top_gainer_telegram.chat_id or self.settings.telegram.chat_id
        )
        self.sector_notifier = TelegramNotifier(
            self.settings.top_sectors_telegram.bot_token or self.settings.telegram.bot_token,
            self.settings.top_sectors_telegram.chat_id or self.settings.telegram.chat_id
        )

    def _thought(self, agent: str, message: str, action: str | None = None):
        LOGGER.info(f"[{agent}] {message}")
        try:
            session = get_db_session()
            log_agent_thought(session, agent, message, action=action)
            session.close()
        except: pass

    async def run_post_market_analysis(self) -> dict[str, Any]:
        """
        Execute the complete post-market agentic feedback loop.
        """
        self._thought("PostMarketOrchestrator", "Initializing post-market Swarm ritual...", action="START")
        
        # 0. Sync Daily Data
        self._thought("DataService", "Updating official NSE F&O universe list...", action="DATASYNC")
        update_fo_universe_from_nse()
        self._thought("DataService", "Fetching latest MWPL data from NSE...", action="DATASYNC")
        self.mwpl_analyzer.update_data()
        
        # 1. VCP Coiling Scan
        self._thought("MomentumForecaster", "Scanning for VCP coiling candidates...", action="SCANNING")
        vcp_candidates = self.vcp_scanner.scan_for_coiling()

        # 2. Top 10 Gainers (Nifty 500)
        self._thought("MarketAnalyst", "Scanning Nifty 500 universe for top performers...", action="SCANNING")
        gainer_result = await self._analyze_top_gainers_broad()
        
        # 2.5 Weekly Gainers (Fridays only)
        if datetime.now().weekday() == 4:
            self._thought("MarketAnalyst", "Friday detected. Calculating weekly top gainers for Nifty 500...", action="SCANNING")
            weekly_fetcher = NSEWeeklyGainersFetcher(broker=self.broker)
            weekly_gainers = weekly_fetcher.fetch_weekly_gainers()
            weekly_fetcher.send_telegram_report(weekly_gainers, top_n=10)
        
        # 2. Sectoral Analysis & Rotation
        self._thought("SectorAnalyst", "Calculating quantitative Sector Rotation (RS Slope)...", action="EVALUATING")
        sector_result = self._analyze_sectors()
        rotation_df = self.rotation_analyzer.get_sector_leadership(lookback_days=20)
        
        # 3. Trade Performance Analysis
        self._thought("PerformanceCritic", "Comparing today's suggested trades against actual market outcomes...", action="EVALUATING")
        trade_stats = await self.missed_opportunity_agent.analyze_today_performance()
        
        # 3.5 Filter Diagnostics (True/False Negatives)
        self._thought("PerformanceCritic", "Analyzing ORB breakouts that were filtered out today...", action="EVALUATING")
        true_negatives = await self.missed_opportunity_agent.analyze_true_negatives()
        
        # 4. Missed Momentum Detection
        top_symbols = [s.symbol for s in gainer_result.gainers[:15]] if gainer_result else []
        self._thought("OpportunityAgent", "Scanning top gainers for missed high-momentum setups...", action="SCANNING")
        missed_insights = await self.missed_opportunity_agent.analyze_missed_setups(top_symbols)

        # 5. Cross-Parameter Correlations
        self._thought("CorrelationAgent", "Calculating mathematical correlations (VIX, Greeks, Sectors)...", action="CALCULATING")
        correlation_summary = await self.correlation_agent.analyze_correlations()

        # 6. Weight Evolution
        self._thought("EvolutionLoop", "Updating neural weights based on today's trade feedback...", action="EVOLVING")
        evolution_result = self.evolution_loop.run()

        # 7. Skill Creation
        self._thought("SkillCreator", "Synthesizing all insights into new behavioral skills...", action="LEARNING")
        skills = await self._refine_skills(evolution_result, missed_insights, correlation_summary)
        
        # 8. Next-Day Suggestions (BTST)
        self._thought("BTST_Agent", "Generating suggestions for tomorrow's market based on current momentum...", action="EVALUATING")
        fo_universe = get_fo_universe()
        await self._suggest_next_day_trades(symbols=fo_universe)

        # 9. Supertrend Touch Scan (MTF) - Use CACHED data for Post-Market
        self._thought("MTF_Scanner", "Scanning for Supertrend retests using database history...", action="SCANNING")
        st_touches = {
            "15m": await self.st_touch_agent.scan_for_touches(fo_universe, resolution="15", use_cache=True),
            "Daily": await self.st_touch_agent.scan_for_touches(fo_universe, resolution="D", use_cache=True)
        }

        # 10. Candlestick Pattern Analysis
        self._thought("PostMarketOrchestrator", "Analyzing F&O candlestick patterns for technical setups...", action="ANALYZING")
        candle_patterns = self._analyze_candlestick_patterns(fo_universe)
        
        # 11. AI Synthesis Report Generation
        self._thought("PostMarketOrchestrator", "Triggering Swarm intelligence market synthesis...", action="SYNTHESIS")
        synthesis_report = None
        try:
            synthesis_report = await self.market_synthesis_agent.generate_daily_synthesis(
                stats=trade_stats,
                missed=missed_insights,
                correlation=correlation_summary,
                skills=skills,
                patterns=candle_patterns,
                st_touches=st_touches,
                rotation_df=rotation_df,
                vcp=vcp_candidates
            )
        except Exception as exc:
            LOGGER.error(f"Market synthesis generation failed: {exc}")

        # Final Report
        self._send_comprehensive_report(
            trade_stats, missed_insights, correlation_summary, skills,
            candle_patterns, st_touches, rotation_df, vcp=vcp_candidates, synthesis=synthesis_report,
            true_negatives=true_negatives
        )
        self._thought("PostMarketOrchestrator", "Post-market Swarm ritual finalized. Memory updated.", action="FINISH")
        
        return {"stats": trade_stats, "missed": missed_insights, "skills": skills}

    def _analyze_candlestick_patterns(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Identify significant candlestick patterns in the F&O universe."""
        results = []
        from trade_system.application.analysis.candlestick_patterns import CandlestickPatternAnalyzer
        
        with Session(get_engine()) as session:
            for symbol in symbols:
                try:
                    data = get_market_data(session, symbol, "D", limit=5)
                    if not data or len(data) < 4: continue
                    
                    df = pd.DataFrame([
                        {"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume}
                        for d in data
                    ])
                    
                    # Calculate required metrics
                    df["body"] = (df["close"] - df["open"]).abs()
                    df["range"] = df["high"] - df["low"]
                    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
                    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
                    df["direction"] = 0
                    df.loc[df["close"] > df["open"], "direction"] = 1
                    df.loc[df["close"] < df["open"], "direction"] = -1
                    df["trend_3"] = df["close"].diff(3)
                    
                    labeled = CandlestickPatternAnalyzer._label_patterns(df)
                    latest = labeled.iloc[-1]
                    
                    patterns = []
                    if latest.get("doji"): patterns.append("Doji")
                    if latest.get("hammer"): patterns.append("Hammer")
                    if latest.get("inverted_hammer"): patterns.append("Inverted Hammer")
                    if latest.get("bullish_engulfing"): patterns.append("Bullish Engulfing")
                    if latest.get("bearish_engulfing"): patterns.append("Bearish Engulfing")
                    
                    if patterns:
                        results.append({
                            "symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                            "patterns": patterns,
                            "close": latest["close"]
                        })
                except: continue
        return results

    async def _analyze_top_gainers_broad(self) -> Any:
        """Fetch top 100 gainers from NIFTY 500 and send to Telegram (Once per day)."""
        # 1. Market Hours Check
        now = datetime.now()
        if now.hour < 15 or (now.hour == 15 and now.minute < 30):
            LOGGER.warning("Top 10 Gainer report requested before market close. Skipping Telegram send.")
            return None

        # 2. Daily Frequency Check (Persistent Lock)
        today_str = date.today().strftime('%Y%m%d')
        lock_file = Path(self.settings.data_dir) / "top_gainers" / f"sent_lock_{today_str}.txt"
        if lock_file.exists():
            LOGGER.info("Top 10 Gainer report already sent for today. Skipping duplicate.")
            return None

        try:
            fetcher = NSETop100GainersFetcher(broker=self.broker)
            
            # Combine Nifty 500 and FO universe to ensure all FO stocks are fetched
            combined_universe = list(set(NSE_UNIVERSE) | set(get_fo_universe()))
            all_quotes = fetcher.fetch_all_quotes(symbols=combined_universe, batch_size=50)
            
            filtered = fetcher.apply_filters(all_quotes)
            result = fetcher.get_top_gainers(filtered, top_n=100)
            fetcher.save_results(result)
            
            # Send Top 10 to dedicated TOP_GAINER channel, using all_quotes so negative F&O stocks are included
            fetcher.send_telegram_report(result, self.gainer_notifier, max_rows=10, all_quotes=all_quotes)
            
            # Create Lock
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file.write_text(f"Sent at {now.isoformat()}")
            
            return result
        except Exception as e:
            LOGGER.error(f"Broad gainer analysis failed: {e}")
            return None

    def _analyze_sectors(self) -> dict[str, Any]:
        """Analyze sectoral performance and send to Telegram."""
        try:
            sector_data = self.sectoral_analyzer.get_top_and_worst_sectors(top_n=3)
            top = sector_data.get("top", pd.DataFrame())
            worst = sector_data.get("worst", pd.DataFrame())
            
            if not top.empty:
                today_str = date.today().strftime("%d %b %Y")
                lines = [
                    f"📊 <b>Sectoral Performance</b> ({today_str})",
                    "",
                    "🟢 <b>TOP 3 SECTORS:</b>"
                ]
                for _, row in top.iterrows():
                    lines.append(f"• {row['Sector']}: <code>{row['Change_Pct']:+.2f}%</code>")
                
                lines.append("\n🔴 <b>WORST 3 SECTORS:</b>")
                for _, row in worst.iterrows():
                    lines.append(f"• {row['Sector']}: <code>{row['Change_Pct']:+.2f}%</code>")
                
                lines.append("\n<i>Generated by AI Post-Market Agent</i>")
                # Send to dedicated TOP_SECTOR channel
                self.sector_notifier.send("\n".join(lines))
            return sector_data
        except Exception as e:
            LOGGER.error(f"Sectoral analysis failed: {e}")
            return {}

    async def _suggest_next_day_trades(self, symbols: List[str] = None) -> List[Any]:
        """Generate next-day WATCHLIST using the upgraded daily candle predictor."""
        try:
            market_context = await self.market_context_agent.analyze()
            if not symbols: return []
            
            self._thought("NextDayPredictor", "Scanning F&O universe for next-day watchlist (daily candle patterns)...", action="SCANNING")
            watchlist = await self.next_day_predictor.predict_next_day_setups(symbols[:100], market_context)

            if not watchlist: 
                self._thought("NextDayPredictor", "No dual-confirmed setups found for watchlist.", action="IDLE")
                return []

            today_str = date.today().strftime("%d %b %Y")
            regime = "🐂 Bullish" if hasattr(market_context, 'bias') and 'bull' in str(market_context.bias).lower() else "🐻 Bearish" if hasattr(market_context, 'bias') and 'bear' in str(market_context.bias).lower() else "⚖️ Neutral"
            
            lines = [
                f"🔭 <b>Tomorrow's Priority Watchlist</b> ({today_str})",
                f"<i>Regime: {regime} | Based on Daily Candle Analysis</i>",
                ""
            ]
            
            for i, w in enumerate(watchlist, 1):
                sym_clean = w.symbol.split(':')[-1].replace('-EQ', '')
                dir_emoji = "📈" if w.direction == "CALL" else "📉"
                patterns_str = " + ".join(w.patterns)
                lines.append(
                    f"{i}. {dir_emoji} <b>{sym_clean}</b> ({w.direction})\n"
                    f"   Conf: <code>{w.confidence:.0%}</code> | RSI: {w.rsi:.0f}\n"
                    f"   Patterns: {patterns_str}\n"
                    f"   ATR SL: {w.suggested_sl_pct:.1f}% | TP: {w.suggested_tp_pct:.1f}%\n"
                    f"   Sector: {w.sector} ({w.sector_perf:+.1f}%)"
                )
            
            lines.append("")
            lines.append("<i>⚡ These stocks will be prioritized by the intraday agent for real-time confirmation.</i>")
            
            self.notifier.send("\n".join(lines))
            
            # Store watchlist for the intraday agent to pick up
            self._store_watchlist(watchlist)
            
            return watchlist
        except Exception as e:
            LOGGER.error(f"Next-day watchlist generation failed: {e}")
            return []

    def _store_watchlist(self, watchlist: List[Any]) -> None:
        """Persist the watchlist to the database for the intraday agent to use."""
        try:
            import json
            from pathlib import Path
            from trade_system.config import Settings
            
            settings = Settings.load()
            watchlist_dir = Path(settings.data_dir) / "watchlist"
            watchlist_dir.mkdir(parents=True, exist_ok=True)
            
            next_day = date.today().strftime("%Y%m%d")
            path = watchlist_dir / f"watchlist_{next_day}.json"
            
            data = [w.to_dict() for w in watchlist]
            path.write_text(json.dumps(data, indent=2, default=str))
            LOGGER.info(f"Watchlist persisted to {path} ({len(watchlist)} items)")
        except Exception as e:
            LOGGER.error(f"Failed to persist watchlist: {e}")

    async def _refine_skills(self, evolution: dict, missed: list, correlation: str) -> list[str]:
        all_insights = [correlation] + missed[:2]
        if evolution.get('top_conditions'):
            all_insights.append(f"Winning pattern: {', '.join(evolution['top_conditions'][:2])}")
        
        created = []
        for ins in all_insights:
            skill_msg = await self.skill_creator.create_skill(ins, "Post-market analysis")
            created.append(skill_msg)
        return created

    def _send_comprehensive_report(self, stats, missed, correlation, skills, patterns=None, st_touches=None, rotation_df=None, vcp=None, synthesis=None, true_negatives=None):
        lines = [
            "🏁 <b>Post-Market Swarm Analysis</b>",
            f"📅 Session: {date.today().strftime('%d %b %Y')}",
            "",
            "📊 <b>Today's Performance:</b>",
            f" • Suggested Trades: {stats['total']}",
            f" • Win Rate: {stats['wins']/stats['total']:.1%}" if stats['total'] > 0 else " • No trades taken today.",
            "",
            "📉 <b>Momentum Gaps:</b>"
        ]
        for m in missed[:3]:
            lines.append(f" • {m}")

        if true_negatives:
            lines.append("\n🛡️ <b>Filter Diagnostics (True/False Negatives):</b>")
            avoided_losses = [tn for tn in true_negatives if tn["outcome"] == "SL_HIT"]
            missed_wins = [tn for tn in true_negatives if tn["outcome"] == "TARGET_HIT"]
            lines.append(f" • Avoided Losses (True Negatives): {len(avoided_losses)}")
            lines.append(f" • Missed Winners (False Negatives): {len(missed_wins)}")
            
            if avoided_losses:
                lines.append("\n🟢 <b>Saved Trades (Avoided Losses):</b>")
                for tn in avoided_losses[:5]:
                    sym = tn["symbol"].split(':')[-1].replace('-EQ', '')
                    lines.append(f" • <b>{sym}</b> ({tn['direction']}): Rejected because <i>{tn['reason']}</i>. (Simulated SL was hit)")
                    
            if missed_wins:
                lines.append("\n🔴 <b>Missed Winners (False Negatives):</b>")
                for tn in missed_wins[:5]:
                    sym = tn["symbol"].split(':')[-1].replace('-EQ', '')
                    lines.append(f" • <b>{sym}</b> ({tn['direction']}): Rejected because <i>{tn['reason']}</i>. (Simulated Target was hit)")
            
        if patterns:
            lines.append("\n🕯️ <b>Technical Patterns (Daily):</b>")
            # Group by pattern type
            from collections import defaultdict
            p_map = defaultdict(list)
            for p in patterns[:15]: # Limit to top 15 for brevity
                for pat_name in p['patterns']:
                    p_map[pat_name].append(p['symbol'])
            
            for pat_name, symbols in p_map.items():
                lines.append(f" • {pat_name}: {', '.join(symbols)}")

        if rotation_df is not None and not rotation_df.empty:
            lines.append("\n🎡 <b>Sector Rotation (Relative Strength):</b>")
            # Filter LEADING status
            leaders = rotation_df[rotation_df['Status'].str.contains('LEADING', na=False)]['Sector'].tolist()[:3]
            # Filter LAGGING status
            laggards = rotation_df[rotation_df['Status'].str.contains('LAGGING', na=False)]['Sector'].tolist()[:3]
            if leaders: lines.append(f" • 🔥 Leaders: {', '.join(leaders)}")
            if laggards: lines.append(f" • ❄️ Laggards: {', '.join(laggards)}")

        if st_touches:
            lines.append("\n🧭 <b>Supertrend Retests (MTF):</b>")
            if st_touches['Daily']:
                d_syms = [s.symbol.split(':')[-1] for s in st_touches['Daily'][:5]]
                lines.append(f" • Daily (Swing): {', '.join(d_syms)}")
            if st_touches['15m']:
                i_syms = [s.symbol.split(':')[-1] for s in st_touches['15m'][:5]]
                lines.append(f" • 15m (Intraday): {', '.join(i_syms)}")

        if vcp:
            lines.append("\n🔥 <b>Spring-Loaded (VCP Coiling):</b>")
            vcp_syms = [v['symbol'].split(':')[-1].replace('-EQ', '') for v in vcp[:5]]
            lines.append(f" • Potential Breakout: {', '.join(vcp_syms)}")
            lines.append(" <i>Low-volatility accumulation detected.</i>")

        lines += ["", "🧬 <b>Market Correlations:</b>", f" {correlation}", "", "🧠 <b>New Learned Skills:</b>"]
        for s in skills[:2]:
            lines.append(f" • {s}")
            
        lines.append("\n<i>Swarm Memory Synchronized.</i>")
        self.notifier.send("\n".join(lines))
        
        if synthesis:
            try:
                self.notifier.send(synthesis)
            except Exception as e:
                LOGGER.error(f"Failed to send Telegram synthesis report: {e}")

def run_main():
    import asyncio
    logging.basicConfig(level=logging.INFO)
    agent = PostMarketImproverAgent()
    asyncio.run(agent.run_post_market_analysis())

if __name__ == "__main__":
    run_main()
