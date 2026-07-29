#!/usr/bin/env python3
"""
TradingView Screenshot Agent

Uses Playwright to take a screenshot of a TradingView chart and sends it to Telegram.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from trade_system.notifications.telegram import TelegramNotifier

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: Playwright not installed. Run 'pip install playwright' and 'playwright install chromium'")
    sys.exit(1)

def capture_tradingview_chart(symbol: str, output_path: str):
    """Navigates to TradingView and captures a screenshot of the chart."""
    print(f"Capturing TradingView chart for {symbol}...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()
        
        # Navigate to TradingView India (symbol format e.g. NSE:NIFTY)
        # Note: TradingView public charts can sometimes show popups. We try to handle them.
        url = f"https://in.tradingview.com/chart/?symbol={symbol}"
        print(f"Navigating to {url}")
        
        try:
            page.goto(url, timeout=60000, wait_until="networkidle")
            
            # Wait for the main chart canvas to appear
            page.wait_for_selector(".chart-gui-wrapper", timeout=15000)
            
            # Let it load fully
            time.sleep(5)
            
            # Attempt to close any "Accept Cookies" or "Sign In" popups if they appear
            try:
                page.click("button[data-name='dismiss']", timeout=2000)
            except:
                pass
            
            try:
                page.click(".tv-dialog__close", timeout=2000)
            except:
                pass
                
            # Take screenshot of the main chart area
            chart_element = page.locator(".chart-gui-wrapper")
            chart_element.screenshot(path=output_path)
            
            print(f"Screenshot saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Failed to capture screenshot: {e}")
            # Fallback full page screenshot for debugging
            page.screenshot(path=output_path)
            return False
        finally:
            browser.close()

def main():
    parser = argparse.ArgumentParser(description="Take TradingView Screenshot and send to Telegram")
    parser.add_argument("--symbol", type=str, default="NSE:NIFTY", help="TradingView Symbol format")
    parser.add_argument("--caption", type=str, default="TradingView Chart Snapshot", help="Caption for the image")
    args = parser.parse_args()

    from trade_system.shared.config import Settings
    settings = Settings.load()
    
    # We use the generic telegram bot token or fallback
    token = os.getenv("TELEGRAM_BOT_TOKEN") or settings.top_gainer_telegram.bot_token
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or settings.top_gainer_telegram.chat_id
    
    if not token or not chat_id:
        print("Error: Telegram credentials not found in .env")
        return

    # Create temporary path for screenshot
    out_dir = Path("data/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{args.symbol.replace(':', '_')}_{int(time.time())}.png")

    if capture_tradingview_chart(args.symbol, out_path):
        notifier = TelegramNotifier(token, chat_id)
        print("Sending to Telegram...")
        if notifier.send_photo(out_path, caption=f"📊 <b>{args.symbol}</b>\n{args.caption}"):
            print("Successfully sent photo to Telegram.")
        else:
            print("Failed to send photo to Telegram.")
    else:
        print("Skipping Telegram send due to screenshot failure.")

if __name__ == "__main__":
    main()
