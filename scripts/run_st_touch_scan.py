"""
Script to scan the F&O universe for Supertrend 'touches' on 15m and Daily timeframes.
"""
import asyncio
import logging
from trade_system.application.agent.supertrend_touch_agent import SupertrendTouchAgent
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.infrastructure.data.fo_universe import get_fo_universe
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.config import Settings
from datetime import date

async def run_st_touch_scan():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("st_touch_scan")
    
    settings = Settings.load()
    auth_service = FyersAuthService(settings)
    token = auth_service.get_valid_token()

    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth_service.authenticator
    )

    
    agent = SupertrendTouchAgent(broker=broker)
    notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
    
    universe = get_fo_universe()
    logger.info(f"🚀 Starting Multi-Timeframe ST-Touch Scan for {len(universe)} stocks...")
    
    # 1. Scan 15-Minute touches
    touches_15m = await agent.scan_for_touches(universe, resolution="15")
    
    # 2. Scan Daily touches
    touches_daily = await agent.scan_for_touches(universe, resolution="D")
    
    # 3. Report Results
    if not touches_15m and not touches_daily:
        logger.info("No Supertrend touches found at this moment.")
        return

    today_str = date.today().strftime("%d %b %Y")
    report = [f"🎯 <b>ST-Touch Watchlist</b> ({today_str})", "<i>Institutional re-entry zones identified.</i>", ""]
    
    if touches_daily:
        report.append("📅 <b>Daily Timeframe (Swing/Positional):</b>")
        for s in touches_daily:
            dir_icon = "🟢" if s.direction.value == "CALL" else "🔴"
            report.append(f" • {dir_icon} <b>{s.symbol.split(':')[-1]}</b> (Conf: {s.confidence:.0%})")
        report.append("")

    if touches_15m:
        report.append("⏱️ <b>15-Minute Timeframe (Intraday):</b>")
        for s in touches_15m:
            dir_icon = "🟢" if s.direction.value == "CALL" else "🔴"
            report.append(f" • {dir_icon} <b>{s.symbol.split(':')[-1]}</b> (Conf: {s.confidence:.0%})")
        report.append("")

    report.append("<i>Strategy: Price touching Supertrend line with volume confirmation.</i>")
    
    notifier.send("\n".join(report))
    logger.info("ST-Touch report sent to Telegram.")

if __name__ == "__main__":
    asyncio.run(run_st_touch_scan())
