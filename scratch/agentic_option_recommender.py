"""
Agentic Option Recommender — Generates institutional-grade option buyer suggestions.
"""
import sys
import os
import json
import asyncio
from datetime import datetime
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

from trade_system.infrastructure.database.connection import get_engine
from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.application.analysis import pro_oc_analyzer

async def main():
    print("🧠 AGENTIC DERIVATIVES FORENSICS & OPTION RECOMMENDER\n")
    engine = get_engine()
    
    # 1. Fetch latest option chain snapshot
    symbol = "NSE:NIFTY50-INDEX"
    with engine.connect() as conn:
        # Find latest timestamp
        ts_res = conn.execute(text("""
            SELECT MAX(timestamp) 
            FROM option_chain_data 
            WHERE underlying_symbol = :symbol
        """), {"symbol": symbol}).first()
        latest_ts = ts_res[0] if ts_res else None
        
        if not latest_ts:
            print(f"❌ No option chain data found for {symbol}.")
            return
            
        print(f"📦 Loading latest option chain snapshot for {symbol} at {latest_ts}...")
        df = pd.read_sql(text("""
            SELECT * 
            FROM option_chain_data 
            WHERE underlying_symbol = :symbol 
              AND timestamp = :ts
        """), conn, params={"symbol": symbol, "ts": latest_ts})
        
        # Load the first snapshot of that day to calculate daily changes
        day_start = str(pd.to_datetime(latest_ts).date()) + " 00:00:00"
        day_end = str(pd.to_datetime(latest_ts).date()) + " 23:59:59"
        first_ts_res = conn.execute(text("""
            SELECT MIN(timestamp) 
            FROM option_chain_data 
            WHERE underlying_symbol = :symbol 
              AND timestamp >= :start 
              AND timestamp <= :end
        """), {"symbol": symbol, "start": day_start, "end": day_end}).first()
        first_ts = first_ts_res[0] if first_ts_res else None
        
        first_df = None
        if first_ts and first_ts != latest_ts:
            first_df = pd.read_sql(text("""
                SELECT * 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                  AND timestamp = :ts
            """), conn, params={"symbol": symbol, "ts": first_ts})
            
        # Get spot price at that timestamp from price history or estimate from option ATM
        spot_res = conn.execute(text("""
            SELECT close 
            FROM ohlcv_15m 
            WHERE symbol = :symbol AND timestamp <= :ts
            ORDER BY timestamp DESC LIMIT 1
        """), {"symbol": symbol, "ts": latest_ts}).first()
        spot_price = spot_res[0] if spot_res else 23500.0 # fallback

    print(f"🎯 Spot Price: ₹{spot_price:,.2f}")
    
    # 2. Run pro option chain analysis computations
    strike_step = 50
    positioning = pro_oc_analyzer.compute_buyer_seller_positioning(df, first_df, spot_price, strike_step)
    oi_conc = pro_oc_analyzer.compute_oi_concentration(df, spot_price, strike_step)
    vol_conc = pro_oc_analyzer.compute_volume_concentration(df, spot_price, strike_step)
    ce_pe_diff = pro_oc_analyzer.compute_ce_pe_difference(df, spot_price, strike_step)
    atm_analysis = pro_oc_analyzer.compute_atm_premium_analysis(df, spot_price, strike_step)
    iv_skew = pro_oc_analyzer.compute_iv_skew(df, spot_price, strike_step)
    greeks_heatmap = pro_oc_analyzer.compute_greeks_heatmap(df, spot_price, strike_step)
    
    # Render quantitative report
    pcr_val = atm_analysis['ce_pe_ratio']
    print("\n--- Derivative Metrics ---")
    print(f"• Put-Call Ratio (PCR): {pcr_val:.2f}")
    print(f"• Max Pain Strike: ₹{atm_analysis['atm_strike']:.0f}")
    print(f"• ATM Implied Volatility (IV): {atm_analysis['atm_iv']:.2f}%" if atm_analysis['atm_iv'] else "• ATM IV: N/A")
    print(f"• Expected Move (Straddle expected move): ±{atm_analysis['expected_move_pct']:.2f}% (±{atm_analysis['expected_move_pts']:.1f} pts)")
    print(f"• Key Support Wall (Put Wall): ₹{oi_conc['highest_pe_wall']['strike']:.0f} (OI: {oi_conc['highest_pe_wall']['oi']:,})")
    print(f"• Key Resistance Wall (Call Wall): ₹{oi_conc['highest_ce_wall']['strike']:.0f} (OI: {oi_conc['highest_ce_wall']['oi']:,})")
    print(f"• Net Market Dominance: {positioning['dominant']}")
    print(f"• IV Skew Pattern: {iv_skew['skew_type']}")
    
    # 3. Call Agent to make recommendation
    llm = LlmAdvisorClient()
    
    prompt = (
        f"You are an expert options trader specializing in high-probability option buying strategies for retail traders.\n"
        f"Analyze the following institutional option chain metrics for {symbol} and generate an actionable trade recommendation:\n\n"
        f"### OPTION CHAIN METRICS:\n"
        f"- Spot Price: ₹{spot_price:,.2f}\n"
        f"- ATM Strike: ₹{atm_analysis['atm_strike']:.0f}\n"
        f"- PCR (Put-Call Ratio): {pcr_val:.2f}\n"
        f"- Expected Move (based on ATM Straddle): ±{atm_analysis['expected_move_pts']:.1f} points (±{atm_analysis['expected_move_pct']:.2f}%)\n"
        f"- Call Wall (Major Resistance): ₹{oi_conc['highest_ce_wall']['strike']:.0f} (OI: {oi_conc['highest_ce_wall']['oi']:,} contracts)\n"
        f"- Put Wall (Major Support): ₹{oi_conc['highest_pe_wall']['strike']:.0f} (OI: {oi_conc['highest_pe_wall']['oi']:,} contracts)\n"
        f"- Institutional Positioning Dominance: {positioning['dominant']} (Call Buyers: {positioning['ce_buyer_oi']:,} | Call Sellers: {positioning['ce_seller_oi']:,} | Put Buyers: {positioning['pe_buyer_oi']:,} | Put Sellers: {positioning['pe_seller_oi']:,})\n"
        f"- Volatility Skew: {iv_skew['skew_type']} (ATM IV: {atm_analysis['atm_iv'] if atm_analysis['atm_iv'] else 'N/A'}%)\n\n"
        f"### PROMPT:\n"
        f"Based on the metrics, draft a trade recommendation for an Option Buyer. Provide:\n"
        f"1. **Market Context & Bias**: Is the market bullish, bearish, or rangebound? Explain who is trapped (e.g. Call writers or Put writers).\n"
        f"2. **Specific Strategy**: Recommends either: Buy Call, Buy Put, or No Trade (due to rangebound theta decay risk).\n"
        f"3. **Contract Parameters**: Recommended Strike Price (ITM/OTM), Expiry Type (Weekly/Monthly), Entry Zone, Target, and Stop Loss.\n"
        f"4. **Execution Rules**: Triggers for entry (e.g., breakout above Call Wall or breakdown below Put Wall) and risk management guidelines.\n\n"
        f"Format your response in professional Markdown."
    )
    
    print("\n🤖 Querying LLM Options Specialist for trade suggestion...")
    recommendation = await llm.complete(prompt)
    
    print("\n" + "="*80)
    print("📜 AGENTIC INSTITUTIONAL OPTION TRADE SUGGESTION")
    print("="*80)
    print(recommendation)
    print("="*80 + "\n")
    
    # Save the recommendation to reports
    report_path = "reports/agentic_option_recommendation.md"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# Agentic Option Chain Analysis & Trade Suggestion\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Data snapshot: {latest_ts})\n\n")
        f.write(f"## 1. Quantitative Metrics Summary\n\n")
        f.write(f"- **Underlying Symbol**: {symbol}\n")
        f.write(f"- **Spot Price**: ₹{spot_price:,.2f}\n")
        f.write(f"- **PCR (Ratio)**: {pcr_val:.2f}\n")
        f.write(f"- **Max Pain Strike**: ₹{atm_analysis['atm_strike']:.0f}\n")
        f.write(f"- **Call Wall**: ₹{oi_conc['highest_ce_wall']['strike']:.0f}\n")
        f.write(f"- **Put Wall**: ₹{oi_conc['highest_pe_wall']['strike']:.0f}\n")
        f.write(f"- **IV Skew**: {iv_skew['skew_type']}\n")
        f.write(f"- **Net Dominance**: {positioning['dominant']}\n\n")
        f.write(f"## 2. Professional Actionable Analysis & Recommendation\n\n")
        f.write(recommendation)
        
    print(f"📁 Report saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
