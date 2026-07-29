import sys
import os
from pathlib import Path
from datetime import datetime
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

from trade_system.domains.analysis.application.analysis.top_gainers import NSETop100GainersFetcher, TopGainersResult, StockQuote
from trade_system.shared.notifications.telegram import TelegramNotifier
from trade_system.shared.config import Settings

def test_telegram_gainers():
    settings = Settings.load()
    notifier = TelegramNotifier(
        settings.top_gainer_telegram.bot_token or settings.telegram.bot_token,
        settings.top_gainer_telegram.chat_id or settings.telegram.chat_id
    )

    all_quotes = [
        StockQuote("NSE:HDFCBANK-EQ", "HDFCBANK", 1500.0, 1400.0, 1550.0, 1380.0, 1450.0, 1000000, 50.0, 3.45, "now"),
        StockQuote("NSE:TCS-EQ", "TCS", 3500.0, 3400.0, 3550.0, 3380.0, 3450.0, 1000000, 50.0, -1.45, "now"),
        StockQuote("NSE:RELIANCE-EQ", "RELIANCE", 2500.0, 2400.0, 2550.0, 2380.0, 2450.0, 1000000, 50.0, 2.04, "now"),
        StockQuote("NSE:INFY-EQ", "INFY", 1500.0, 1400.0, 1550.0, 1380.0, 1450.0, 1000000, 50.0, 5.45, "now"),
        StockQuote("NSE:RANDOM1-EQ", "RANDOM1", 100.0, 90.0, 110.0, 80.0, 95.0, 10000, 5.0, 5.26, "now"),
        StockQuote("NSE:RANDOM2-EQ", "RANDOM2", 100.0, 90.0, 110.0, 80.0, 95.0, 10000, 5.0, 4.26, "now"),
        StockQuote("NSE:SBIN-EQ", "SBIN", 600.0, 590.0, 610.0, 580.0, 605.0, 1000000, -5.0, -0.83, "now"),
        StockQuote("NSE:ITC-EQ", "ITC", 400.0, 390.0, 410.0, 380.0, 405.0, 1000000, -5.0, -2.83, "now"),
    ]

    result = TopGainersResult(
        date=datetime.now().isoformat(),
        total_stocks_analyzed=500,
        stocks_passing_filters=300,
        top_n=10,
        gainers=sorted(all_quotes, key=lambda x: x.change_pct, reverse=True)
    )

    print("Sending telegram report...")
    NSETop100GainersFetcher.send_telegram_report(result, notifier, max_rows=3, all_quotes=all_quotes)
    print("Done")

if __name__ == "__main__":
    test_telegram_gainers()
