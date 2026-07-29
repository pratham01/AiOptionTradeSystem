import asyncio
import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to path
root_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_path))

from trade_system.domains.advisory.application.agent.fo_stock_suggester_agent import FoStockSuggesterAgent
from trade_system.domains.advisory.application.agent.candidate_screener_agent import CandidateScreenerAgent
from trade_system.domains.analysis.application.analysis.breakout_screener import BreakoutScreener
from trade_system.shared import MarketContext, MarketRegime

class MockBroker:
    pass

async def debug():
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
    
    # 1. Check breakout screener alerts
    print("--- 1. BreakoutScreener Alerts ---")
    screener = BreakoutScreener()
    alerts = screener.scan_for_breakouts(top_sectors_count=3, target_date=date(2026, 5, 25), current_time=ctx.timestamp)
    print(f"Total alerts: {len(alerts)}")
    for a in alerts:
        print(f"  {a['symbol']} | Direction: {a['direction']} | Sector: {a['sector']} | Bypass: {a.get('is_bypass')}")
        
    # 2. Check candidate screener
    print("\n--- 2. CandidateScreener --")
    cs = CandidateScreenerAgent()
    candidates = cs.screen(ctx, include_nifty=False, include_fo=True)
    print(f"Total screened candidates: {len(candidates)}")
    for c in candidates:
        if "EICHER" in c.symbol or "ADANI" in c.symbol:
            print(f"  Found: {c.symbol} | Score: {c.best_score()} | Sector: {c.sector}")
            
    # 3. Trace suggester agent candidates
    agent = FoStockSuggesterAgent(broker=MockBroker())
    print("\n--- 3. Prioritized Candidates in Suggester Agent ---")
    
    # Run the logic manually
    # Get sector leadership
    leadership_df = agent.rotation_analyzer.get_sector_leadership(lookback_days=15)
    leading_sectors = set()
    if not leadership_df.empty:
        leading_sectors = set(leadership_df[leadership_df['Status'].str.contains("LEADING")]['Sector'].tolist())
        
    cand_map = {c.symbol: c for c in candidates}
    direction_map = {"LONG": "CALL", "SHORT": "PUT"}

    for alert in alerts:
        symbol = alert["symbol"]
        if symbol in cand_map:
            cand = cand_map[symbol]
            cand.intraday_score = max(cand.intraday_score, 0.95)
            cand.swing_score = max(cand.swing_score, 0.95)
            cand.breakout_type = alert.get("direction", "LONG").lower() + "_breakout"
            cand.pattern = (cand.pattern or "") + ", INTRADAY_ORB"
            cand.is_breakout_bypass = True
            cand.component_scores["performance"] = max(cand.component_scores.get("performance", 0.0), 0.9)
            cand.component_scores["momentum"] = max(cand.component_scores.get("momentum", 0.0), 0.9)
            cand.component_scores["volume_delta"] = max(cand.component_scores.get("volume_delta", 0.0), 0.9)
            cand.component_scores["value_area"] = max(cand.component_scores.get("value_area", 0.0), 0.9)
        else:
            cand = CandidateScore(
                symbol=symbol,
                intraday_score=0.95,
                swing_score=0.95,
                horizon=TradeHorizon.INTRADAY,
                direction=direction_map.get(alert["direction"], "CALL"),
                sector=alert["sector"],
                entry_price=alert["close"],
                volume_surge=alert["volume"] / alert["vol_sma"] if alert["vol_sma"] > 0 else 2.0,
                is_compressed=False,
                vol_delta_positive=True,
                above_poc=True,
                alignment_score=0.9,
                breakout_type=alert["direction"].lower() + "_breakout",
                pattern="INTRADAY_ORB",
                spread_pct=0.05,
                is_liquid=True,
                component_scores={"performance": 0.9, "momentum": 0.9, "volume_delta": 0.9, "value_area": 0.9}
            )
            cand.is_breakout_bypass = True
            cand_map[symbol] = cand

    breakout_symbols = [a["symbol"] for a in alerts]
    prioritized_candidates = []
    for sym in breakout_symbols:
        if sym in cand_map:
            prioritized_candidates.append(cand_map[sym])
    for c in candidates:
        if c.symbol not in breakout_symbols:
            prioritized_candidates.append(c)
            
    print(f"Total prioritized candidates: {len(prioritized_candidates)}")
    for i, cand in enumerate(prioritized_candidates[:20]):
        is_bypass = getattr(cand, "is_breakout_bypass", False)
        print(f"  {i+1}. {cand.symbol} | Score: {cand.best_score()} | Bypass: {is_bypass} | Sector: {cand.sector}")
        
    print("\n--- 4. Validator Tracing ---")
    suggestions = []
    for cand in prioritized_candidates[:20]:
        is_breakout_bypass = getattr(cand, "is_breakout_bypass", False)
        if not is_breakout_bypass and leading_sectors and cand.sector not in leading_sectors:
            print(f"  {cand.symbol} REJECTED: Sector not leading ({cand.sector} not in {leading_sectors})")
            continue
            
        local_bars = agent._suggest_trades_local_bars if hasattr(agent, "_suggest_trades_local_bars") else None
        # fetch local bars
        engine = get_engine()
        query = """
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv_15m 
            WHERE symbol = :symbol AND timestamp <= :target_time
            ORDER BY timestamp DESC
            LIMIT 30
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"symbol": cand.symbol, "target_time": ctx.timestamp})
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
            
        sugg = await agent.validator.validate(cand, ctx, current_bars=df)
        if sugg:
            print(f"  {cand.symbol} APPROVED!")
            suggestions.append(sugg)
        else:
            print(f"  {cand.symbol} REJECTED in validator (ATR={cand.atr_pct}, entry={cand.entry_price})")

if __name__ == "__main__":
    asyncio.run(debug())
