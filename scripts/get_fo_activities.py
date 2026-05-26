import asyncio
import logging
import pandas as pd
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.infrastructure.data.fo_universe import get_fo_universe
from trade_system.config import Settings
import json

logging.basicConfig(level=logging.INFO)

async def get_top_fo_activities():
    print("\n🚀 ANALYZING TOP F&O ACTIVITIES (NSE Market)...")
    
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.read_cached_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator
    )
    
    fo_symbols = get_fo_universe()
    print(f"Fetching live data for {len(fo_symbols)} F&O stocks...")
    
    try:
        quotes = broker.get_quotes(fo_symbols)
        data = []
        for sym, q in quotes.items():
            # lp: last price, v: volume, chp: change %, oi: open interest
            data.append({
                "Symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                "Change %": float(q.get("chp", 0)),
                "Volume": int(q.get("v", 0)),
                "OI": int(q.get("oi", 0)),
                "LTP": float(q.get("lp", 0))
            })
        
        df = pd.DataFrame(data)
        
        # 1. Top Price Gainers
        top_gainers = df.sort_values("Change %", ascending=False).head(5)
        
        # 2. Top Price Losers
        top_losers = df.sort_values("Change %", ascending=True).head(5)
        
        # 3. Top Volume Shockers (Active Participation)
        top_volume = df.sort_values("Volume", ascending=False).head(5)
        
        print("\n🔥 TOP F&O PRICE GAINERS:")
        print(top_gainers[['Symbol', 'Change %', 'LTP']].to_string(index=False))
        
        print("\n❄️ TOP F&O PRICE LOSERS:")
        print(top_losers[['Symbol', 'Change %', 'LTP']].to_string(index=False))
        
        print("\n📊 MOST ACTIVE F&O STOCKS (BY VOLUME):")
        print(top_volume[['Symbol', 'Volume', 'LTP']].to_string(index=False))

    except Exception as e:
        print(f"❌ Error fetching F&O activities: {e}")

if __name__ == "__main__":
    asyncio.run(get_top_fo_activities())
