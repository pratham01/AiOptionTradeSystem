import streamlit as st
import pandas as pd
from sqlalchemy import text
from datetime import datetime

from trade_system.domains.market_data.infrastructure.database.connection import get_engine, SessionLocal
from trade_system.domains.market_data.infrastructure.database.models import TopGainersSnapshot, TopGainersStock, TopMoverAnalysis

def render_trade_book():
    st.header("Detailed Trade Book")
    st.markdown("Deep AI analysis of the top movers to understand *why* they moved and how to predict them in the future.")
    
    with SessionLocal() as session:
        # Get distinct dates from TopGainersSnapshot
        dates = session.execute(text("SELECT date FROM top_gainers_snapshots ORDER BY date DESC")).scalars().all()
        
        if not dates:
            st.warning("No trade book data available. Run the top gainers scanner and trade book generator.")
            return
            
        selected_date = st.selectbox("Select Date", dates)
        
        # Fetch the snapshot for the selected date
        snapshot = session.query(TopGainersSnapshot).filter_by(date=selected_date).first()
        if not snapshot:
            st.error("Snapshot not found.")
            return
            
        # Fetch TopMoverAnalysis joined with TopGainersStock
        results = session.query(TopMoverAnalysis, TopGainersStock).join(TopGainersStock).filter(
            TopGainersStock.snapshot_id == snapshot.id
        ).order_by(TopGainersStock.change_pct.desc()).all()
        
        if not results:
            st.info(f"No AI analysis available for {selected_date}. Please run the trade book generation script.")
            return
            
        for analysis, stock in results:
            with st.expander(f"{stock.symbol} ({stock.name}) - Up {stock.change_pct:.2f}% | Vol: {getattr(stock, 'volume_ratio', 'N/A')}x"):
                st.subheader("What drove this move? (Catalyst)")
                st.write(analysis.catalyst_analysis)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Volume Behavior")
                    st.write(analysis.volume_behavior)
                    
                with col2:
                    st.subheader("How to Predict Next Time")
                    st.write(analysis.predictive_factors)
                    
                st.markdown("---")
                st.caption(f"Technical Setup details: {analysis.technical_setup}")
