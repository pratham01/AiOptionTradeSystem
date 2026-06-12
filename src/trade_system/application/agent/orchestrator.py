"""
TradeOrchestrator — Central coordinator for the multi-agent trading pipeline.

Orchestrates the Swarm:
1. PreMarketNewsAgent (Macro/News)
2. MarketContextAgent (VIX/PCR/Regime)
3. OptionChainAgent (Derivatives Data)
4. IndexDecisionAgent (Nifty/BankNifty Live Trades)
5. FoStockSuggesterAgent (Intraday/BTST FO Stocks)
6. SetupValidatorAgent (Validation & Risk Checks)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any

from trade_system.core import (
    SessionPlan,
    TradeSuggestion,
    MarketContext,
    OptionChainSnapshot,
    GlobalContext,
)
from trade_system.application.agent.premarket_news_agent import PreMarketNewsAgent
from trade_system.application.agent.market_context_agent import MarketContextAgent
from trade_system.application.agent.option_chain_agent import OptionChainAgent
from trade_system.application.agent.nifty_option_buyer_agent import NiftyOptionBuyerAgent
from trade_system.application.agent.fo_stock_suggester_agent import FoStockSuggesterAgent
from trade_system.application.agent.setup_validator_agent import SetupValidatorAgent
from trade_system.application.agent.conviction_fuser_agent import ConvictionFuserAgent
from trade_system.application.agent.skill_registry import SkillRegistry
from trade_system.application.evolution.trade_logger import TradeLogger
from trade_system.config import Settings
from trade_system.core.ports.repository import LogRepository
from trade_system.core.events.bus import EventBus, EventType
from trade_system.infrastructure.notifications.formatters import format_session_plan_telegram

LOGGER = logging.getLogger(__name__)


class TradeOrchestrator:
    """
    The brain of the system. Coordinates all specialized agents.
    """

    def __init__(
        self,
        broker: Any,
        notifier: Any,
        log_repository: LogRepository,
        trade_logger: TradeLogger,
        skill_registry: SkillRegistry,
        premarket_agent: PreMarketNewsAgent,
        market_context_agent: MarketContextAgent,
        option_chain_agent: OptionChainAgent,
        nifty_agent: NiftyOptionBuyerAgent,
        fo_suggester_agent: FoStockSuggesterAgent,
        setup_validator: SetupValidatorAgent,
        fuser_agent: ConvictionFuserAgent | None = None,
        event_bus: EventBus | None = None,
        max_trades_per_day: int = 2,
    ) -> None:
        self.broker = broker
        self.notifier = notifier
        self.log_repository = log_repository
        self.event_bus = event_bus or EventBus()
        self.max_total_trades = max_trades_per_day
        self.settings = Settings.load()

        # Injected Swarm
        self.premarket_agent = premarket_agent
        self.market_context_agent = market_context_agent
        self.option_chain_agent = option_chain_agent
        self.nifty_agent = nifty_agent
        self.fo_suggester_agent = fo_suggester_agent
        self.setup_validator = setup_validator
        self.fuser_agent = fuser_agent or ConvictionFuserAgent(max_trades_per_day=3)
        self.trade_logger = trade_logger
        self.skill_registry = skill_registry

    def _thought(self, agent: str, message: str, symbol: str | None = None, action: str | None = None):
        """Helper to log an agent thought to the DB and LOGGER."""
        LOGGER.info(f"[{agent}] {message}")
        try:
            self.log_repository.log_thought(agent, message, symbol, action)
        except Exception as e:
            LOGGER.debug(f"Failed to log thought: {e}")

    async def run_session(
        self,
        vix: float | None = None,
        pcr: float | None = None,
        option_chain: OptionChainSnapshot | None = None,
        global_ctx: GlobalContext | None = None,
        extra_context: dict[str, Any] | None = None,
        user_directive: str | None = None,
    ) -> SessionPlan:
        today = date.today().isoformat()
        plan = SessionPlan(date=today)

        self._thought("TradeOrchestrator", f"Waking up the Swarm for session {today}", action="START")
        
        # 1. Start parallel async tasks for independent data gathering agents
        self._thought("PreMarketNewsAgent", "Scraping headlines from MoneyControl, Mint, and ET...", action="SCANNING")
        news_task = asyncio.create_task(self.premarket_agent.analyze())
        
        self._thought("OptionChainAgent", f"Fetching option chain data for {self.settings.index_symbols}...", action="SCANNING")
        oc_tasks = {
            symbol: asyncio.create_task(self.option_chain_agent.analyze(symbol))
            for symbol in self.settings.index_symbols
        }
        
        # Load learned skills in parallel
        skills = self.skill_registry.get_all_skills_text()
        if skills:
            self._thought("SkillRegistry", f"Loaded {len(self.skill_registry.list_skills())} learned skills from memory.", action="MEMORY")

        full_context = extra_context or {}
        full_context["learned_skills"] = skills
        if user_directive:
            self._thought("TradeOrchestrator", f"Received user directive: {user_directive}", action="DIRECTIVE")
            full_context["user_directive"] = user_directive

        # Emit SESSION_STARTED so event-driven agents can begin
        await self.event_bus.emit(EventType.SESSION_STARTED, full_context)

        # 2. Wait for Pre-market News and Context
        premarket_brief = await news_task
        self._thought("PreMarketNewsAgent", f"Macro Sentiment: {premarket_brief.overall_sentiment}. {premarket_brief.agent_summary[:100]}...", action="DECISION")

        # Step 2: Market Context (VIX, PCR, Regime)
        self._thought("MarketContextAgent", "Determining market regime and volatility bias...", action="EVALUATING")
        market_context = await self.market_context_agent.analyze(
            vix=vix, pcr=pcr, option_chain=option_chain,
            global_ctx=global_ctx, extra_context=full_context,
        )
        plan.market_context = market_context
        self._thought("MarketContextAgent", f"Regime: {market_context.regime} | Bias: {market_context.bias}", action="DECISION")

        if not market_context.tradeable:
            self._thought("MarketContextAgent", f"Session aborted: {market_context.narrative}", action="ABORT")
            plan.is_tradeable_day = False
            plan.agent_notes = f"Session aborted: {market_context.narrative}"
            self._notify_non_tradeable(plan)
            await self.event_bus.emit(EventType.SESSION_PLAN_READY, plan)
            return plan

        await self.event_bus.emit(EventType.MARKET_CONTEXT_READY, market_context)

        # 3. Process Live Data Streams (Option Chain & Index Decisions)
        for symbol, task in oc_tasks.items():
            oc_analysis = await task
            if oc_analysis:
                self._thought("OptionChainAgent", f"PCR: {oc_analysis.pcr:.2f} | Max Pain: {oc_analysis.max_pain} for {symbol}", action="DECISION")
                
                # Step 3: Index Decisions
                self._thought("NiftyOptionBuyerAgent", f"Analyzing high-conviction setups for {symbol}...", action="EVALUATING")
                current_price = oc_analysis.atm_strike 
                suggestion = await self.nifty_agent.analyze_and_suggest(
                    symbol=symbol,
                    market_context=market_context,
                    oc_analysis=oc_analysis,
                    current_price=current_price
                )
                
                if suggestion:
                    self._thought("NiftyOptionBuyerAgent", f"APPROVED: {suggestion.direction.value} setup found for {symbol} at {current_price}", action="APPROVED", symbol=symbol)
                    plan.add_nifty(suggestion)
                else:
                    self._thought("NiftyOptionBuyerAgent", f"No high-conviction setups found for {symbol}.", action="REJECTED")

        # --- Step 4: F&O Stock Suggestions ---
        total_suggestions = len(plan.all_suggestions())
        if total_suggestions < self.max_total_trades:
            self._thought("FoStockSuggesterAgent", f"Starting aggressive scan of 183 F&O stocks...", action="SCANNING")
            
            # Load priority watchlist from previous night's post-market analysis
            priority_watchlist = self._load_priority_watchlist()
            if priority_watchlist:
                wl_str = ", ".join(w.symbol.split(":")[-1].replace("-EQ","") for w in priority_watchlist)
                self._thought("FoStockSuggesterAgent", f"Priority watchlist loaded: {wl_str}", action="WATCHLIST")
            
            fo_suggestions = await self.fo_suggester_agent.suggest_trades(
                market_context, priority_watchlist=priority_watchlist
            )
            
            for sugg in fo_suggestions:
                if len(plan.all_suggestions()) >= self.max_total_trades:
                    break
                self._thought("FoStockSuggesterAgent", f"Candidate Found: {sugg.symbol} ({sugg.direction.value})", action="EVALUATING", symbol=sugg.symbol)
                
                # Check for Golden Setup Merge
                if "unified_swarm_setup" in sugg.tags:
                    self._thought("SwarmBrain", f"🌟 GOLDEN SETUP: {sugg.symbol} passed all institutional layers.", action="CONFLUENCE", symbol=sugg.symbol)
                
                # Log if from watchlist
                if "next_day_watchlist" in sugg.tags:
                    self._thought("SwarmBrain", f"🔭 WATCHLIST HIT: {sugg.symbol} confirmed by real-time intraday data.", action="CONFLUENCE", symbol=sugg.symbol)
                
                plan.add_fo(sugg)
                self._thought("SetupValidatorAgent", f"APPROVED: {sugg.symbol} added to plan.", action="APPROVED", symbol=sugg.symbol)

        # --- Step 5: Conviction Fuser & Discipline Enforcement ---
        self._thought("ConvictionFuserAgent", "Applying macro-confluence scoring and culling weak setups...", action="EVALUATING")
        plan = self.fuser_agent.fuse_and_filter(plan)

        # --- Step 6: Logging and Notifications ---
        plan.total_screened = 100 
        all_suggestions = plan.all_suggestions()
        if all_suggestions:
            self.trade_logger.log_many(all_suggestions)
            LOGGER.info("Logged %d suggestions to database.", len(all_suggestions))

        plan.agent_notes = self._build_plan_notes(plan)
        self._notify_session_plan(plan)
        
        # Announce final plan to EventBus
        await self.event_bus.emit(EventType.SESSION_PLAN_READY, plan)
        
        self._thought("TradeOrchestrator", "Session Plan Finalized. Swarm standing by.", action="FINISH")

        return plan

    def run_session_sync(
        self,
        vix: float | None = None,
        pcr: float | None = None,
        option_chain: OptionChainSnapshot | None = None,
        global_ctx: GlobalContext | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> SessionPlan:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run_session(
                        vix=vix, pcr=pcr, option_chain=option_chain,
                        global_ctx=global_ctx, extra_context=extra_context,
                    ))
                    return future.result()
            else:
                return loop.run_until_complete(self.run_session(
                    vix=vix, pcr=pcr, option_chain=option_chain,
                    global_ctx=global_ctx, extra_context=extra_context,
                ))
        except Exception:
            return asyncio.run(self.run_session(
                vix=vix, pcr=pcr, option_chain=option_chain,
                global_ctx=global_ctx, extra_context=extra_context,
            ))

    def _build_plan_notes(self, plan: SessionPlan) -> str:
        notes = [f"Generated {len(plan.all_suggestions())} valid suggestions using the Multi-Agent Swarm."]
        if plan.market_context:
            notes.append(plan.market_context.narrative)
        return " ".join(notes)

    def _notify_session_plan(self, plan: SessionPlan) -> None:
        if self.notifier is None:
            return
        try:
            msg = format_session_plan_telegram(plan)
            self.notifier.send(msg)
        except Exception as exc:
            LOGGER.error("Failed to send session plan notification: %s", exc)

    def _notify_non_tradeable(self, plan: SessionPlan) -> None:
        if self.notifier is None:
            return
        try:
            ctx = plan.market_context
            vix_str = f"{ctx.vix:.1f}" if ctx and ctx.vix else "N/A"
            msg = (
                f"⚠️ <b>Non-Tradeable Day — {plan.date}</b>\n\n"
                f"VIX: {vix_str} | Risk: {ctx.risk_level if ctx else 'UNKNOWN'}\n"
                f"{plan.agent_notes}"
            )
            self.notifier.send(msg)
        except Exception as exc:
            LOGGER.error("Failed to send non-tradeable notification: %s", exc)

    def _load_priority_watchlist(self) -> list | None:
        """Load the watchlist persisted by PostMarketImproverAgent."""
        try:
            import json
            from pathlib import Path
            from trade_system.application.agent.next_day_predictor_agent import WatchlistItem
            
            watchlist_dir = Path(self.settings.data_dir) / "watchlist"
            today_str = date.today().strftime("%Y%m%d")
            path = watchlist_dir / f"watchlist_{today_str}.json"
            
            if not path.exists():
                LOGGER.debug(f"No watchlist file found at {path}")
                return None
            
            data = json.loads(path.read_text())
            if not data:
                return None
            
            items = []
            for d in data:
                items.append(WatchlistItem(
                    symbol=d["symbol"],
                    direction=d["direction"],
                    confidence=d["confidence"],
                    patterns=d["patterns"],
                    pattern_count=d["pattern_count"],
                    close=d["close"],
                    atr=d["atr"],
                    atr_pct=d["atr_pct"],
                    suggested_sl_pct=d["suggested_sl_pct"],
                    suggested_tp_pct=d["suggested_tp_pct"],
                    rsi=d["rsi"],
                    daily_trend=d["daily_trend"],
                    sector=d["sector"],
                    sector_perf=d["sector_perf"],
                    narrative=d["narrative"],
                ))
            
            LOGGER.info(f"Loaded {len(items)} watchlist items from {path}")
            return items
        except Exception as e:
            LOGGER.error(f"Failed to load priority watchlist: {e}")
            return None



def run_main():
    import asyncio
    from trade_system.application.agent.factory import build_orchestrator
    
    logging.basicConfig(level=logging.INFO)
    orc = build_orchestrator()
    asyncio.run(orc.run_session())
