import streamlit as st
import pandas as pd
import sqlite3
import numpy as np

from trade_system.domains.strategy.application.indicators.bollinger_bands import BollingerBandsDetector

def fetch_latest_data() -> pd.DataFrame:
    """Fetches the latest 15m data for all F&O stocks in the database."""
    conn = sqlite3.connect("data/trade_system.db")
    
    # Get the latest 100 periods for each stock so we have enough data for Squeeze Percentile (100 lookback)
    query = """
    WITH RankedData AS (
        SELECT symbol, timestamp, close, high, low, volume,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
        FROM ohlcv_15m
    )
    SELECT * FROM RankedData WHERE rn <= 120 ORDER BY symbol, timestamp ASC
    """
    
    df = pd.read_sql(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
    return df

def run_main():
    st.header("⚡ Bollinger Breakout Lab")
    st.markdown("Live scan of F&O stocks detecting **Squeezes, Expansions, and Volatility Explosions** for prime option setups.")
    
    with st.spinner("Analyzing Market Volatility..."):
        try:
            df = fetch_latest_data()
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return
            
        if df.empty:
            st.warning("No recent intraday data found in database. Please run your data fetching scripts.")
            return

        detector = BollingerBandsDetector(period=20, std_dev=2.0, squeeze_percentile=10.0)
        
        results = []
        for symbol, group in df.groupby('symbol'):
            if len(group) < 100:
                continue # Not enough data for a squeeze lookback
                
            group = group.set_index('timestamp')
            
            # Handle index zero volume issue for cleaner display
            if group['volume'].sum() == 0:
                group['volume'] = 1000
                
            bb_df = detector.compute(group)
            latest = bb_df.iloc[-1]
            
            results.append({
                "Symbol": symbol.replace('NSE:', '').replace('-EQ', ''),
                "Phase": latest['bb_phase'],
                "Explosion Score": latest['volatility_explosion_score'],
                "Trend": latest['trend_state'],
                "Walking Upper": "Yes" if latest['is_walking_upper_band'] else "-",
                "Walking Lower": "Yes" if latest['is_walking_lower_band'] else "-",
                "%b": round(latest['bb_percent_b'], 2),
                "BBW": round(latest['bbw'], 4)
            })

    if not results:
        st.warning("Not enough data to calculate Bollinger Bands (needs at least 100 historical periods per stock).")
        return
        
    results_df = pd.DataFrame(results)
    
    # Sort: Prioritize explosive setups and active squeezes
    def sort_logic(row):
        score = 0
        if "EXPANSION" in row['Phase']: score += 100
        if row['Phase'] == 'SQUEEZE': score += 50
        score += float(row['Explosion Score']) * 10
        return score
        
    results_df['SortScore'] = results_df.apply(sort_logic, axis=1)
    results_df = results_df.sort_values(by=['SortScore', 'Explosion Score'], ascending=[False, False]).drop(columns=['SortScore'])
    
    # Styling
    def highlight_phase(val):
        color = ''
        if 'BULLISH' in str(val): color = 'color: #00FF00; font-weight: bold;'
        elif 'BEARISH' in str(val): color = 'color: #FF0000; font-weight: bold;'
        elif val == 'SQUEEZE': color = 'color: #FFFF00; font-weight: bold;'
        elif val == 'ACCUMULATION': color = 'color: #ADD8E6;'
        elif val == 'DISTRIBUTION': color = 'color: #FFA07A;'
        return color

    def highlight_explosion(val):
        if val > 1.5:
            return 'background-color: rgba(255, 0, 0, 0.3); font-weight: bold;'
        elif val > 1.0:
            return 'background-color: rgba(255, 165, 0, 0.3);'
        return ''
        
    def highlight_walking(val):
        if val == "Yes":
            return 'background-color: rgba(0, 255, 0, 0.2); font-weight: bold;'
        return ''

    styled_df = results_df.style.map(highlight_phase, subset=['Phase'])\
                                .map(highlight_explosion, subset=['Explosion Score'])\
                                .map(highlight_walking, subset=['Walking Upper', 'Walking Lower'])\
                                .format({'Explosion Score': '{:.2f}'})
                                
    st.dataframe(styled_df, width='stretch', height=600)
    
    # Add a legend/help section
    with st.expander("📖 How to read this Dashboard"):
        st.markdown("""
        - **SQUEEZE**: Volatility is dead. Prepare for a massive breakout. Good time to build straddles/strangles.
        - **EXPANSION_BULLISH/BEARISH**: The squeeze has fired. Look for a high Explosion Score to confirm it's real.
        - **Explosion Score**: Combines Band expansion rate with Volume surges. `> 1.0` is good, `> 1.5` is explosive (perfect for Gamma trades).
        - **Walking Upper/Lower**: Price has hugged the outer band for 3+ consecutive periods. Strong momentum trend. Do not fade this.
        """)

if __name__ == "__main__":
    run_main()
