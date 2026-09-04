import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import logging
import asyncio
import requests
import io
import pandas as pd

from trade_system.shared.config import Settings
from trade_system.shared.utils.logging_utils import configure_logging
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient as FyersBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator
from trade_system.shared.notifications.telegram import TelegramNotifier
from trade_system.domains.analysis.application.analysis.eod_analyzer import EODAnalyzer
from trade_system.domains.advisory.application.agent.postmarket_improver_agent import PostMarketImproverAgent

async def run_eod_workflow():
    # 1. Setup
    settings = Settings.load()
    configure_logging(settings.log_level)
    logger = logging.getLogger("EODWorkflow")

    logger.info("Starting Complete End-of-Day (EOD) Agentic Workflow...")

    # 2. Initialize Broker
    authenticator = FyersAuthenticator(settings)
    broker = FyersBroker(
        client_id=settings.fyers_client_id,
        access_token=settings.fyers_access_token,
        user_id=settings.fyers_user_id,
        authenticator=authenticator
    )
    
    if not broker.authenticate():
        logger.error("Failed to authenticate with Fyers.")
        return

    # 3. Parallel Execution of Phase 1 and Phase 2
    logger.info("Phases 1 & 2: Running Agentic Analysis and Strategy Scan concurrently...")
    
    # We need a function to wrap the sync call for Phase 2
    def run_strategy_scan():
        logger.info("Phase 2: Starting RVOL-Trend Strategy Scan...")
        url = "https://public.fyers.in/sym_details/NSE_FO.csv"
        try:
            res = requests.get(url)
            df_fo = pd.read_csv(io.StringIO(res.text), header=None)
            underlying_names = df_fo[13].unique().tolist()
            
            notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
            analyzer = EODAnalyzer(broker, notifier, settings)
            analyzer.run_full_scan(underlying_names)
            logger.info("✓ Phase 2: RVOL-Trend Strategy Scan completed.")
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")

    # Phase 3: Daily Reversal Walk-Forward Backtest Verification
    def run_reversal_backtest():
        logger.info("Phase 3: Starting Daily Reversal Radar Walk-Forward Backtest...")
        try:
            from trade_system.domains.analysis.application.analysis.recommendation_evaluator import RecommendationEvaluator
            evaluator = RecommendationEvaluator(lookback_days=60)
            res = evaluator.evaluate_reversal_recommendations(min_probability=60)
            if "error" not in res:
                logger.info(
                    f"✓ Phase 3 Complete: Trigger Rate {res['trigger_rate_pct']}% | "
                    f"Win Rate {res['win_rate_pct']}% ({res['wins']}W / {res['losses']}L) | "
                    f"Profit Factor {res['profit_factor']}"
                )
            else:
                logger.warning(f"Phase 3 notice: {res.get('error')}")
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}", exc_info=True)

    improver = PostMarketImproverAgent(broker=broker, settings=settings)
    
    try:
        await asyncio.gather(
            improver.run_post_market_analysis(),
            asyncio.to_thread(run_strategy_scan),
            asyncio.to_thread(run_reversal_backtest)
        )
        logger.info("✓ All post-market phases completed in parallel.")
    except Exception as e:
        logger.error(f"Parallel workflow failed: {e}", exc_info=True)

    logger.info("🏁 Complete EOD Workflow finished.")

if __name__ == "__main__":
    asyncio.run(run_eod_workflow())
