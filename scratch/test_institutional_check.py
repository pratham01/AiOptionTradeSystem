import asyncio
import pandas as pd
from datetime import datetime
from sqlalchemy import text

from trade_system.core import MarketContext, MarketRegime, TradeHorizon
from trade_system.application.agent.candidate_screener_agent import CandidateScore
from trade_system.application.agent.setup_validator_agent import SetupValidatorAgent
from trade_system.infrastructure.database.connection import get_engine

async def verify_institutional_logic():
    engine = get_engine()
    
    # 1. Clean and Prepare Test Data in option_chain_data
    with engine.begin() as conn:
        # Create option_chain_data table if not exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS option_chain_data (
                timestamp TEXT,
                underlying_symbol TEXT,
                symbol TEXT,
                expiry TEXT,
                strike REAL,
                option_type TEXT,
                ltp REAL,
                oi REAL,
                oi_change REAL,
                volume REAL,
                iv REAL
            )
        """))
        
        # Clear any old test data
        conn.execute(text("DELETE FROM option_chain_data WHERE underlying_symbol = 'NSE:TESTMOCK-EQ'"))
        
        # Insert mock option chain snapshot showing PUT writing (Bullish for CALL setup)
        # Spot is at 1000. Strikes: 980, 990, 1000 (ATM), 1010, 1020
        now_str = datetime.now().isoformat()
        records = [
            # Call side
            (now_str, 'NSE:TESTMOCK-EQ', 'TESTMOCK26JUN1000CE', '2026-06-26', 1000.0, 'CE', 15.0, 1000.0, 100.0, 500.0, 0.20),
            (now_str, 'NSE:TESTMOCK-EQ', 'TESTMOCK26JUN1010CE', '2026-06-26', 1010.0, 'CE', 8.0, 800.0, 50.0, 300.0, 0.21),
            # Put side - heavy Put open interest near ATM indicating institutional floor (Short Buildup/Put writing)
            (now_str, 'NSE:TESTMOCK-EQ', 'TESTMOCK26JUN1000PE', '2026-06-26', 1000.0, 'PE', 14.0, 5000.0, 1500.0, 800.0, 0.22),
            (now_str, 'NSE:TESTMOCK-EQ', 'TESTMOCK26JUN990PE', '2026-06-26', 990.0, 'PE', 7.0, 4000.0, 1000.0, 600.0, 0.23),
        ]
        for r in records:
            conn.execute(text("""
                INSERT INTO option_chain_data (timestamp, underlying_symbol, symbol, expiry, strike, option_type, ltp, oi, oi_change, volume, iv)
                VALUES (:ts, :underlying, :symbol, :expiry, :strike, :op_type, :ltp, :oi, :oi_change, :vol, :iv)
            """), {"ts": r[0], "underlying": r[1], "symbol": r[2], "expiry": r[3], "strike": r[4], "op_type": r[5], "ltp": r[6], "oi": r[7], "oi_change": r[8], "vol": r[9], "iv": r[10]})

    # 2. Setup mock candidate
    candidate = CandidateScore(
        symbol="NSE:TESTMOCK-EQ",
        intraday_score=0.7,
        swing_score=0.5,
        horizon=TradeHorizon.INTRADAY,
        direction="CALL",
        sector="TEST",
        entry_price=1000.0,
        rsi_daily=60.0,
        rsi_hourly=55.0,
        adx=25.0,
        volume_surge=2.5,
        atr_pct=2.0,
        is_compressed=False,
        vol_delta_positive=True,
        above_poc=True,
        alignment_score=0.8,
        breakout_type="daily_high",
        pattern=None,
        spread_pct=0.02,
        is_liquid=True,
        component_scores={"volume_delta": 0.8, "value_area": 0.8, "momentum": 0.8}
    )
    
    market_context = MarketContext(
        timestamp=datetime.now(),
        regime=MarketRegime.TRENDING_BULL,
        bias="BULLISH",
        risk_level="MEDIUM",
        vix=15.0,
        pcr=1.2,
        score=0.6,
        tradeable=True,
        option_chain=None,
        global_ctx=None,
        metadata={"user_directive": None}
    )

    validator = SetupValidatorAgent()
    
    # Run first validation test: Expecting dynamic boost due to PE writing alignment
    print("--- Running Test 1: CALL setup with heavy Put writing (PE OI > CE OI near ATM) ---")
    sugg = await validator.validate(candidate, market_context, current_bars=None)
    if sugg:
        print(f"✅ Success! suggestion generated: {sugg.symbol} {sugg.direction} | Conf: {sugg.confidence} | Tags: {sugg.tags}")
        assert "institutional_buildup" in sugg.tags
        assert sugg.confidence > 0.7
    else:
        print("❌ Failed: suggestion not generated!")

    # 3. Modify option chain test data: Simulate heavy Call resistance (CE OI > PE OI near ATM)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM option_chain_data WHERE underlying_symbol = 'NSE:TESTMOCK-EQ'"))
        
        # Heavy Call open interest near ATM indicating institutional resistance
        records = [
            # Call side - heavy CE OI
            (now_str, 'NSE:TESTMOCK-EQ', 'TESTMOCK26JUN1000CE', '2026-06-26', 1000.0, 'CE', 15.0, 8000.0, 2500.0, 1500.0, 0.20),
            (now_str, 'NSE:TESTMOCK-EQ', 'TESTMOCK26JUN1010CE', '2026-06-26', 1010.0, 'CE', 8.0, 6000.0, 1800.0, 1200.0, 0.21),
            # Put side - low PE OI
            (now_str, 'NSE:TESTMOCK-EQ', 'TESTMOCK26JUN1000PE', '2026-06-26', 1000.0, 'PE', 14.0, 1000.0, 200.0, 300.0, 0.22),
        ]
        for r in records:
            conn.execute(text("""
                INSERT INTO option_chain_data (timestamp, underlying_symbol, symbol, expiry, strike, option_type, ltp, oi, oi_change, volume, iv)
                VALUES (:ts, :underlying, :symbol, :expiry, :strike, :op_type, :ltp, :oi, :oi_change, :vol, :iv)
            """), {"ts": r[0], "underlying": r[1], "symbol": r[2], "expiry": r[3], "strike": r[4], "op_type": r[5], "ltp": r[6], "oi": r[7], "oi_change": r[8], "vol": r[9], "iv": r[10]})

    print("\n--- Running Test 2: CALL setup with heavy Call resistance (CE OI > PE OI near ATM) ---")
    sugg_fail = await validator.validate(candidate, market_context, current_bars=None)
    if sugg_fail is None:
        print("✅ Success! Suggestion correctly rejected (failing institutional check).")
    else:
        print(f"❌ Failed: Suggestion was not rejected! suggestion: {sugg_fail.symbol} Conf: {sugg_fail.confidence}")

    # Cleanup test data
    with engine.begin() as conn:
         conn.execute(text("DELETE FROM option_chain_data WHERE underlying_symbol = 'NSE:TESTMOCK-EQ'"))
         print("\nCleanup complete.")

if __name__ == "__main__":
    asyncio.run(verify_institutional_logic())
