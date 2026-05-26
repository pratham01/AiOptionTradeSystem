import asyncio
import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to path
root_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_path))

from trade_system.application.agent.fo_stock_suggester_agent import FoStockSuggesterAgent
from trade_system.core import MarketContext, MarketRegime

class MockBroker:
    pass

async def test_suggestions():
    print("Testing FoStockSuggesterAgent suggestions for 2026-05-25 14:30:00...")
    
    # 1. Build market context mock for today
    ctx = MarketContext(
        timestamp=datetime(2026, 5, 25, 14, 30, 0),
        regime=MarketRegime.TRENDING_BULL,
        bias="BULLISH",
        tradeable=True,
        metadata={
            "hurst_exponent": 0.65,
            "market_breadth": {"label": "BULLISH"}
        }
    )
    
    # 2. Instantiate agent
    broker = MockBroker()
    agent = FoStockSuggesterAgent(broker=broker)
    
    # 3. Call suggest_trades
    suggestions = await agent.suggest_trades(ctx)
    
    # 4. Log suggestions to the database so they appear in the UI
    try:
        from trade_system.application.evolution.trade_logger import TradeLogger
        trade_logger = TradeLogger()
        trade_logger.log_many(suggestions)
        print("Logged suggestions to database successfully!")
    except Exception as e:
        print(f"Failed to log suggestions to database: {e}")
        
    print(f"\nFoStockSuggesterAgent suggested {len(suggestions)} trades:")
    print("=" * 80)
    for i, s in enumerate(suggestions):
        print(f"Suggestion {i+1}: {s.symbol} | {s.direction.value} | Horizon: {s.horizon.value}")
        print(f"  Entry Zone: {s.entry_zone_low} - {s.entry_zone_high}")
        print(f"  Stop Loss: {s.stop_loss} | Target: {s.target}")
        print(f"  Option Strike: {s.option_params.suggested_strike} ({s.option_params.direction.value})")
        print(f"  Confidence: {s.confidence:.2%}")
        print(f"  Tags: {s.tags}")
        print(f"  Narrative: {s.narrative}")
        print("-" * 80)
        
    # Assert check
    symbols = [s.symbol for s in suggestions]
    eicher_ok = any("EICHERMOT" in sym for sym in symbols)
    adani_ok = any("ADANIENT" in sym for sym in symbols)
    
    print("\nVerification Results:")
    print(f"  - EICHERMOT Suggested: {'YES (PASS)' if eicher_ok else 'NO (FAIL)'}")
    print(f"  - ADANIENT Suggested: {'YES (PASS)' if adani_ok else 'NO (FAIL)'}")

if __name__ == "__main__":
    asyncio.run(test_suggestions())
