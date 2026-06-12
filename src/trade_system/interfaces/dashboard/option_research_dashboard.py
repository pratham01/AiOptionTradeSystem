import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import asyncio

from trade_system.application.agent.option_research_agent import OptionResearchAgent
from trade_system.infrastructure.database.connection import get_engine

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .research-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .stat-box {
        background-color: #161b22;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #00b4d8;
        text-align: center;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00b4d8;
        margin-bottom: 5px;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Agent
agent = OptionResearchAgent()

st.title("🔬 Option Research & Predictability Lab")
st.caption("Autonomous quantitative engine studying option pricing physics, market anomalies, and trader wisdom.")
st.markdown("---")

tab_scholar, tab_predict, tab_journal = st.tabs([
    "📚 Scholar & Pricing Physics",
    "🧪 Next-Day Predictability Test",
    "💾 Auto-Journal & Missed Breaks"
])

# --- TAB 1: SCHOLAR & PRICING PHYSICS ---
with tab_scholar:
    st.subheader("Option Trading Wisdom & Market Mechanics")
    st.caption("Auto-generated scholar analysis on the strategies of options giants and canonical literature.")
    
    if st.button("📖 Fetch Scholar Research Report", type="primary", use_container_width=True):
        with st.spinner("Synthesizing option trading literature & Greeks physics..."):
            try:
                report = asyncio.run(agent.research_option_wisdom())
                st.session_state.scholar_report = report
            except Exception as e:
                st.error(f"Failed to generate report: {e}")
                
    if "scholar_report" in st.session_state:
        st.markdown(f'<div class="research-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.scholar_report)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Click the button above to load the Options Masterclass report.")

# --- TAB 2: NEXT-DAY PREDICTABILITY LAB ---
with tab_predict:
    st.subheader("Quantitative Return Predictability Backtest")
    st.caption("Test if technical indicator conditions on daily candles statistically predict next-day return direction.")
    
    # Load all symbols from database ohlcv_daily
    @st.cache_data(ttl=600)
    def load_ohlcv_symbols():
        try:
            engine = get_engine()
            with engine.connect() as conn:
                from sqlalchemy import text
                query = text("SELECT DISTINCT symbol FROM ohlcv_daily ORDER BY symbol ASC")
                df = pd.read_sql(query, conn)
                return df["symbol"].tolist()
        except Exception:
            return ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
            
    symbols = load_ohlcv_symbols()
    
    # Config parameters
    col_sym, col_ind, col_thresh = st.columns([1.5, 1.5, 1.0])
    with col_sym:
        sel_symbol = st.selectbox("Underlying Symbol", symbols, index=symbols.index("NSE:NIFTY50-INDEX") if "NSE:NIFTY50-INDEX" in symbols else 0)
    with col_ind:
        sel_indicator = st.selectbox(
            "Condition Indicator",
            options=["Momentum Surge", "Volume Ratio", "RSI Oversold", "RSI Overbought", "High-Volume Breakout"]
        )
    with col_thresh:
        # Default thresholds based on indicator
        if sel_indicator == "Momentum Surge":
            thresh = st.number_input("Threshold (Close %)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        elif sel_indicator == "Volume Ratio":
            thresh = st.number_input("Threshold (Volume Ratio)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        elif sel_indicator == "RSI Oversold":
            thresh = st.slider("Threshold (RSI)", 10, 50, 30)
        elif sel_indicator == "RSI Overbought":
            thresh = st.slider("Threshold (RSI)", 50, 90, 70)
        elif sel_indicator == "High-Volume Breakout":
            thresh = st.number_input("Threshold (Vol Ratio)", min_value=0.5, max_value=10.0, value=1.8, step=0.1)

    if st.button("⚡ Execute Predictability Test", type="primary", use_container_width=True):
        with st.spinner("Analyzing historical candle returns and calculating significance..."):
            try:
                res = asyncio.run(agent.run_predictability_backtest(sel_symbol, sel_indicator, float(thresh)))
                st.session_state.backtest_result = res
            except Exception as e:
                st.error(f"Failed to run backtest: {e}")

    if "backtest_result" in st.session_state:
        res = st.session_state.backtest_result
        if not res.get("success", False):
            st.error(res.get("error", "Backtest failed."))
        else:
            # Display stats cards
            st.markdown("### 📊 Backtest Statistics")
            sc1, sc2, sc3, sc4 = st.columns(4)
            
            with sc1:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #00d084;">
                    <div class="stat-val">{res['condition_met_count']}/{res['total_days_analyzed']}</div>
                    <div class="stat-label">Days Condition Met</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #ffb703;">
                    <div class="stat-val">{res['win_rate_when_signal']:.1%}</div>
                    <div class="stat-label">Win Rate on Signal</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #9b5de5;">
                    <div class="stat-val">{res['avg_return_when_signal']:+.3f}%</div>
                    <div class="stat-label">Avg Return on Signal</div>
                </div>
                """, unsafe_allow_html=True)
            with sc4:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #ff4d6d;">
                    <div class="stat-val">{res['p_value']:.4f}</div>
                    <div class="stat-label">T-Test P-Value</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Significance Banner
            st.write("")
            color_badge = "#00d084" if res["is_significant"] else "#ff4d6d"
            st.markdown(f"""
            <div style="background:#1e2130; padding:15px; border-radius:8px; border-left:6px solid {color_badge};">
                <h4 style="margin:0 0 5px 0; color:{color_badge};">{res['verdict']}</h4>
                <p style="margin:0; color:#c9d1d9; font-size:0.95rem;">
                    The condition (<b>{res['description']}</b>) yielded an average next-day return of <b>{res['avg_return_when_signal']:+.3f}%</b> 
                    compared to <b>{res['avg_return_baseline']:+.3f}%</b> for normal days. 
                    T-Statistic: <b>{res['t_stat']:.3f}</b>. 
                    {"The difference is statistically significant, meaning this signal has actual predictive value!" if res['is_significant'] else "The statistical test indicates this difference could be random noise. Tread carefully."}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Monthly performance chart
            st.write("")
            monthly_data = res.get("monthly_performance", [])
            if monthly_data:
                m_df = pd.DataFrame(monthly_data)
                st.subheader("📅 Monthly Performance When Condition Met")
                fig_monthly = px.bar(
                    m_df,
                    x="year_month",
                    y="next_day_return",
                    title="Average Next Day Returns by Month",
                    labels={"year_month": "Month", "next_day_return": "Average Return (%)"},
                    template="plotly_dark",
                    color="next_day_return",
                    color_continuous_scale="RdYlGn"
                )
                fig_monthly.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_monthly, use_container_width=True)

# --- TAB 3: AUTO-JOURNAL & MISSED BREAKS ---
with tab_journal:
    st.subheader("Autonomous Trading Journal & Missed Opportunities")
    st.caption("AI Review of suggested trades, scans of ohlcv_daily for missed breakouts, and technique recommendations.")
    
    if st.button("🔄 Generate EOD Auto-Journal Audit", type="primary", use_container_width=True):
        with st.spinner("Generating autonomous journal audit and scanning missed breakout setups..."):
            try:
                res_journal = asyncio.run(agent.generate_autonomous_journal())
                st.session_state.journal_result = res_journal
            except Exception as e:
                st.error(f"Failed to generate journal: {e}")
                
    if "journal_result" in st.session_state:
        res_j = st.session_state.journal_result
        if not res_j.get("success", False):
            st.error(res_j.get("error", "Failed to load journal."))
        else:
            # Display Missed Breakouts table
            st.subheader("🔭 Detected Missed Breakout Opportunities")
            st.caption("Breakout stocks that surged >= 1.5% with high volume on the last 5 trading days but were not suggested by the system.")
            
            missed_ops = res_j.get("missed_opportunities", [])
            if missed_ops:
                m_df = pd.DataFrame(missed_ops)
                # Format
                m_df = m_df.rename(columns={
                    "date": "Date",
                    "symbol": "Stock Symbol",
                    "close_change": "Close Change %",
                    "volume_ratio": "Volume Ratio",
                    "next_day_return": "Next Day Return %"
                })
                st.dataframe(
                    m_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Close Change %": st.column_config.NumberColumn("Close Change", format="%.2f%%"),
                        "Volume Ratio": st.column_config.NumberColumn("Volume Surge", format="%.2fx"),
                        "Next Day Return %": st.column_config.NumberColumn("Next Day Return", format="%.2f%%")
                    }
                )
            else:
                st.info("No significant missed breakout opportunities detected.")
                
            # Display LLM Audit text
            st.write("")
            st.subheader("📝 Autonomous Trading Journal & Enhancement Audit")
            st.markdown(f'<div class="research-card">', unsafe_allow_html=True)
            st.markdown(res_j["journal_text"])
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Click the button above to generate the journal audit.")
