import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.shared.notifications.telegram import TelegramNotifier

def main():
    settings = Settings.load()
    notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
    
    message = (
        "📊 <b>Nifty 500 & F&O Market Performance</b>\n"
        "<i>Pre-Market Session Update</i>\n\n"
        
        "🟢 <b>Top 10 Gainers (Nifty 500)</b>\n"
        "<pre>"
        "#  SYMBOL               PRICE      CHANGE%      VOLUME\n"
        " 1 TV18BRDCST              45.27       6.74%     31095065\n"
        " 2 LIKHITHA               227.20       5.00%        88324\n"
        " 3 SMARTLINK              164.35       3.33%        19747\n"
        " 4 KRBL                   352.05       2.49%            0\n"
        " 5 ISEC                   896.20       1.70%      5383172\n"
        " 6 ATUL                  6700.00       1.63%            0\n"
        " 7 ARCHIDPLY               78.90       1.57%            0\n"
        " 8 TARC                   124.98       1.55%            0\n"
        " 9 BAJAJCON               583.60       1.43%            0\n"
        "10 GARFIBRES              664.80       1.36%            0"
        "</pre>\n\n"
        
        "🔥 <b>Top 5 Gainers (F&O)</b>\n"
        "<pre>"
        "#  SYMBOL               PRICE      CHANGE%      VOLUME\n"
        " 1 MFSL                  1588.90       0.72%            0\n"
        " 2 BHARTIARTL            1820.10       0.38%            0\n"
        " 3 INDIGO                4375.00       0.35%            0\n"
        " 4 GRASIM                3060.30       0.33%            0\n"
        " 5 ALKEM                 5370.50       0.26%            0"
        "</pre>\n\n"
        
        "❄️ <b>Worst 5 Losers (F&O)</b>\n"
        "<pre>"
        "#  SYMBOL               PRICE      CHANGE%      VOLUME\n"
        " 1 NUVAMA                1513.60      -1.70%            0\n"
        " 2 SIEMENS               3576.00      -1.32%            0\n"
        " 3 PHOENIXLTD            1722.10      -1.11%            0\n"
        " 4 JSWENERGY              565.00      -1.04%            0\n"
        " 5 AMBUJACEM              411.35      -0.95%            0"
        "</pre>"
    )
    
    print("Sending message to Telegram...")
    if notifier.send(message):
        print("Success: Telegram message sent.")
    else:
        print("Error: Failed to send Telegram message.")

if __name__ == "__main__":
    main()
