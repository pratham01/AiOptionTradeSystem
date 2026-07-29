import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

# Add project root and src to path
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.advisory.application.agent.nifty500_top_stocks_agent import Nifty500TopStocksAgent
from trade_system.shared.notifications.telegram import TelegramNotifier

async def run_broadcast():
    print("Loading settings...")
    settings = Settings.load()
    
    # Define all Telegram sub groups - restrict strictly to Top Gainers subgroup only
    groups = {
        "Top Gainers": settings.top_gainer_telegram,
    }
    
    print("\nConfigured Telegram Groups:")
    for name, config in groups.items():
        print(f" - {name}: Enabled={config.enabled}, Token={config.bot_token[:10]}..., ChatID={config.chat_id}")

    # Mock datetime to a Monday (June 15th, 2026) to bypass weekend check
    mock_weekday = datetime(2026, 6, 15, 12, 0, 0)
    
    print("\nInstantiating Nifty500TopStocksAgent and fetching real data...")
    agent = Nifty500TopStocksAgent()
    
    with patch("trade_system.domains.advisory.application.agent.nifty500_top_stocks_agent.datetime") as mock_dt:
        mock_dt.now.return_value = mock_weekday
        analysis = await agent.get_top_stocks_analysis(send_telegram=False)
        
    print("\n--- AGENT ANALYSIS START ---")
    print(analysis)
    print("--- AGENT ANALYSIS END ---\n")
    
    # Broadcast to all enabled groups
    sent_count = 0
    for name, config in groups.items():
        if not config.bot_token or not config.chat_id:
            print(f"Skipping {name}: not fully configured.")
            continue
            
        print(f"Sending analysis to {name}...")
        notifier = TelegramNotifier(token=config.bot_token, chat_id=config.chat_id)
        
        # We can send it formatted in Markdown or HTML
        # If the output starts with tables, we can send it as plain text code block or Markdown
        # Let's wrap the analysis narrative in normal Markdown and send it
        success = notifier.send(analysis, parse_mode="Markdown")
        if success:
            print(f"Successfully sent to {name}!")
            sent_count += 1
        else:
            print(f"Failed to send to {name} (trying plain text fallback...)")
            # If markdown parsing failed, fall back to plain text (no parse mode)
            success_plain = notifier.send(analysis, parse_mode=None)
            if success_plain:
                print(f"Successfully sent to {name} in plain text!")
                sent_count += 1
            else:
                print(f"Failed to send to {name} in plain text too.")
                
    print(f"\nBroadcast complete. Sent to {sent_count}/{len(groups)} groups.")

if __name__ == "__main__":
    asyncio.run(run_broadcast())
