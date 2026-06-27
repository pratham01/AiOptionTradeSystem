import sys
from pathlib import Path

# Add src to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.interfaces.dashboard.sector_scope_dashboard import fetch_live_quotes

try:
    quotes = fetch_live_quotes(['NSE:SBIN-EQ', 'NSE:TATAMOTORS-EQ'])
    print("Successfully fetched quotes:")
    for sym, q in quotes.items():
        print(f"{sym}: ltp={q.last_price}, change={q.change_percent}%")
except Exception as e:
    print(f"Error fetching quotes: {e}")
