import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker
import logging

logging.basicConfig(level=logging.INFO)

def search_symbol(query):
    settings = Settings.load()
    from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
    auth = FyersAuthService(settings)
    token = auth.read_cached_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id
    )
    
    # Fyers doesn't have a direct "search" but we can try to get quotes for a guess
    symbols = [f"NSE:{query}-EQ", f"NSE:{query}", f"BSE:{query}-EQ"]
    try:
        quotes = broker.get_quotes(symbols)
        print(f"Quotes for {query}:")
        for s, q in quotes.items():
            if q.get('s') == 'ok' or q.get('lp'):
                print(f"✅ {s}: LTP {q.get('lp')} | Change {q.get('chp')}%")
            else:
                print(f"❌ {s}: Not found or error")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "SAMMAANCAP"
    search_symbol(query)
