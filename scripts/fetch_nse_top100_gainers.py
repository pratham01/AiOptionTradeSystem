#!/usr/bin/env python3
"""
Fetch Top 100 Gainers from Complete NSE Stock Universe
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import argparse
import logging
from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient as FyersBroker
from trade_system.domains.analysis.application.analysis.top_gainers import NSETop100GainersFetcher
from trade_system.shared.notifications.telegram import TelegramNotifier

def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Top 100 Gainers from NSE")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top gainers")
    parser.add_argument("--send-telegram", action="store_true", help="Send top gainers summary to Telegram")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        settings = Settings.load()
        from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator
        authenticator = FyersAuthenticator(settings)
        broker = FyersBroker(
            client_id=settings.fyers_client_id,
            access_token=settings.fyers_access_token,
            user_id=settings.fyers_user_id,
            authenticator=authenticator
        )

        if not broker.authenticate():
            return 1

        fetcher = NSETop100GainersFetcher(broker=broker)
        all_quotes = fetcher.fetch_all_quotes()
        filtered = fetcher.apply_filters(all_quotes)
        result = fetcher.get_top_gainers(filtered, top_n=args.top_n)
        fetcher.save_results(result)

        if args.send_telegram:
            telegram_cfg = settings.top_gainer_telegram if settings.top_gainer_telegram.enabled else settings.telegram
            if telegram_cfg.enabled:
                notifier = TelegramNotifier(telegram_cfg.bot_token, telegram_cfg.chat_id)
                fetcher.send_telegram_report(result, notifier, max_rows=10)

        return 0
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
