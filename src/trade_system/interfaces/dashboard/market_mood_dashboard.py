import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from pathlib import Path
import json
import logging

from trade_system.infrastructure.data.fo_universe import FO_METADATA, get_fo_universe, get_stocks_by_sector, get_sector_mapping
from trade_system.application.indicators.volume_delta import VolumeDeltaIndicator
from trade_system.application.indicators.volume_profile import VolumeProfileIndicator
from trade_system.config import Settings
from trade_system.interfaces.dashboard.shared_broker import get_cached_broker

LOGGER = logging.getLogger(__name__)

# Constants
PALETTE = ["#00b4d8", "#ff4d6d", "#2d6a4f", "#f77f00"]


def get_broker():
    """Return the shared cached broker instance."""
    return get_cached_broker()

@st.cache_data(ttl=60)
def fetch_sector_data(_broker, sector):
    """Safely fetch sector stocks and quotes."""
    try:
        symbols = get_stocks_by_sector(sector)
        if not symbols:
            return pd.DataFrame()
        
        quotes = _broker.get_quotes(symbols)
        data = []
        for sym, q in quotes.items():
            data.append({
                "Symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                "LTP": q.last_price,
                "Change %": q.change_percent,
                "Volume": q.volume,
                "FullSymbol": sym
            })
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not load quotes for {sector}. API busy.")
        LOGGER.error(f"Sector data fetch failed for {sector}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_market_rankings(_broker):
    """Safely aggregate performance of all F&O stocks with DB fallback."""
    mapping = get_sector_mapping()
    all_symbols = list(mapping.keys())
    
    try:
        quotes = _broker.get_quotes(all_symbols)
        data = []
        
        # If API returns nothing, try DB fallback
        if not quotes:
            LOGGER.warning("Broker returned zero quotes for rankings. Falling back to DB.")
            from trade_system.infrastructure.database.connection import get_engine
            from sqlalchemy import text
            query = text("SELECT symbol, close FROM ohlcv_15m WHERE timestamp > datetime('now', '-5 days') ORDER BY timestamp ASC")
            with get_engine().connect() as conn:
                db_data = pd.read_sql(query, conn)
            
            if not db_data.empty:
                # Group by symbol to get change
                db_data['date'] = pd.to_datetime(db_data['timestamp']).dt.date
                grouped = db_data.groupby('symbol')
                for sym, group in grouped:
                    if len(group) > 0:
                        sector = mapping.get(sym, "Unknown")
                        dates = group['date'].unique()
                        latest_close = group.iloc[-1]['close']
                        if len(dates) > 1:
                            prev_day_data = group[group['date'] == dates[-2]]
                            first_close = prev_day_data.iloc[-1]['close']
                        else:
                            first_close = group.iloc[0]['close']
                            
                        pct_change = ((latest_close - first_close) / first_close) * 100 if first_close > 0 else 0.0
                        data.append({
                            "Symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                            "Sector": sector,
                            "Change %": pct_change,
                            "LTP": latest_close
                        })
        else:
            for sym, q in quotes.items():
                sector = mapping.get(sym, "Unknown")
                data.append({
                    "Symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                    "Sector": sector,
                    "Change %": q.change_percent,
                    "LTP": q.last_price
                })
        
        df = pd.DataFrame(data)
        if df.empty: return pd.DataFrame(), pd.DataFrame()

        sector_summary = df.groupby("Sector")["Change %"].mean().reset_index()
        sector_summary = sector_summary.sort_values("Change %", ascending=False)
        stock_summary = df.sort_values("Change %", ascending=False)
        
        return sector_summary, stock_summary
    except Exception as e:
        LOGGER.error(f"Market rankings fetch failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

from trade_system.application.indicators.compression import CompressionIndicator
from trade_system.application.indicators.vwap import VWAPIndicator
from trade_system.application.indicators.supertrend import SupertrendIndicator

def fetch_stock_details(broker, symbol):
    end_dt = date.today()
    start_dt = (datetime.now() - pd.Timedelta(days=5)).date()
    
    try:
        data = broker.get_historical_data(symbol, start_dt, end_dt, "5")
        if not data:
            return None, None, None
        
        df = pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "volume": d.volume
            } for d in data
        ])
        
        vwap_ind = VWAPIndicator()
        df = vwap_ind.calculate(df)
        vd_ind = VolumeDeltaIndicator()
        df = vd_ind.calculate(df)
        vp_ind = VolumeProfileIndicator(price_step=5.0)
        profile = vp_ind.calculate(df)
        comp_ind = CompressionIndicator()
        df = comp_ind.calculate(df)
        st_ind = SupertrendIndicator()
        df = st_ind.calculate(df)
        
        expected_cols = {
            'vwap': 0.0,
            'is_compressed': False,
            'range_compression': 0.0,
            'cvd': 0.0,
            'delta': 0.0,
            'inside_bar': False,
            'supertrend': 0.0,
            'supertrend_direction': 0
        }
        for col, default in expected_cols.items():
            if col not in df.columns:
                df[col] = default

        return df, profile, df
    except Exception as e:
        LOGGER.error(f"Failed to fetch stock details for {symbol}: {e}")
        return None, None, None

@st.cache_data(ttl=600)
def get_rsi_label(symbol):
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from trade_system.infrastructure.database.repository import get_market_data
        from sqlalchemy.orm import Session
        engine = get_engine()
        with Session(engine) as session:
            daily_data = get_market_data(session, symbol, "D", limit=15)
            if not daily_data: return ""
            df = pd.DataFrame([{"close": d.close} for d in daily_data])
            if len(df) < 14: return ""
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            val = rsi.iloc[-1]
            if val > 70: return " ⚠️ [OB]"
            if val < 30: return " 🛡️ [OS]"
    except: pass
    return ""

@st.fragment
def render_stock_deep_dive(_broker, top_symbol):
    try:
        df_5m, profile, _ = fetch_stock_details(_broker, top_symbol)
        if df_5m is not None and not df_5m.empty:
            latest = df_5m.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"₹{latest['close']:.2f}")
            c2.metric("CVD", f"{latest.get('cvd', 0.0):,.0f}")
            is_comp = latest.get('is_compressed', False)
            comp_score = latest.get('range_compression', 0.0)
            comp_color = "normal" if is_comp else "off"
            c3.metric("Compression", f"{comp_score:.1f}%", delta_color=comp_color)

            vwap_val = latest.get('vwap', latest['close'])
            dist_vwap = ((latest['close'] - vwap_val) / vwap_val) * 100
            c4.metric("VWAP Dist", f"{dist_vwap:+.2f}%")

            # --- MWPL & FVG Context ---
            st.write("")
            m1, m2, m3 = st.columns(3)

            # MWPL
            try:
                from trade_system.application.analysis.mwpl_analyzer import MwplAnalyzer
                mw_analyzer = MwplAnalyzer()
                mw_data = mw_analyzer.get_mwpl_data()
                clean_sym = top_symbol.replace("NSE:", "").replace("-EQ", "")
                symbol_mwpl = mw_data[mw_data['SYMBOL'] == clean_sym]['MWPL_PCT'].iloc[0] if not mw_data.empty else 0.0

                mw_color = "red" if symbol_mwpl > 90 else ("orange" if symbol_mwpl > 80 else "green")
                m1.markdown(f"**MWPL Heat:** <span style='color:{mw_color}'>{symbol_mwpl:.1f}%</span>", unsafe_allow_html=True)
            except:
                m1.write("**MWPL Heat:** --")

            # FVG
            try:
                from trade_system.application.indicators.retracement import RetracementIndicator
                ret_ind = RetracementIndicator()
                df_5m_calc = ret_ind.calculate(df_5m)
                active_zones = ret_ind.get_active_zones(df_5m_calc)
                fvg_count = sum(1 for z in active_zones if z.type == "FVG")
                m2.markdown(f"**Active FVGs:** {fvg_count}")
            except:
                m2.write("**Active FVGs:** --")

            # Supertrend
            st_dir = latest.get('supertrend_direction', 0)
            trend_text = "🟢 BULLISH" if st_dir == 1 else ("🔴 BEARISH" if st_dir == -1 else "⚪ NEUTRAL")
            m3.markdown(f"**Trend:** {trend_text}")

            # --- Liquidity Status ---
            liq_col1, liq_col2 = st.columns(2)
            try:
                q_map = _broker.get_quotes([top_symbol])
                q = q_map.get(top_symbol)
                if q:
                    bid = q.bid or latest['close'] * 0.999
                    ask = q.ask or latest['close'] * 1.001
                    spread = ((ask - bid) / latest['close']) * 100
                    status = "✅ LIQUID" if spread < 0.5 else "⚠️ ILLIQUID"
                    color = "green" if spread < 0.5 else "red"
                    liq_col1.markdown(f"**Liquidity Status:** <span class='{color}'>{status}</span>", unsafe_allow_html=True)
                    liq_col2.markdown(f"**Bid-Ask Spread:** `{spread:.2f}%` (Bid: ₹{bid:.2f} | Ask: ₹{ask:.2f})")
            except: pass

            tab1, tab2, tab3, tab4 = st.tabs(["Volume Delta & CVD", "VWAP & Price", "Compression Theory", "🧠 AI Deep Research"])
            
            with tab1:
                fig_cvd = go.Figure()
                fig_cvd.add_trace(go.Scatter(x=df_5m['timestamp'], y=df_5m['cvd'], name="CVD", line=dict(color="#00b4d8")))
                if 'delta' in df_5m.columns:
                    fig_cvd.add_trace(go.Bar(x=df_5m['timestamp'], y=df_5m['delta'], name="Delta", marker_color=np.where(df_5m['delta'] > 0, "#2d6a4f", "#c1121f")))
                fig_cvd.update_layout(title=f"{top_symbol} Order Flow", height=400)
                st.plotly_chart(fig_cvd, use_container_width=True, key=f"cvd_chart_{top_symbol}")

            with tab2:
                fig_price = go.Figure()
                fig_price.add_trace(go.Candlestick(x=df_5m['timestamp'], open=df_5m['open'], high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], name="Price"))
                fig_price.add_trace(go.Scatter(x=df_5m['timestamp'], y=df_5m['vwap'], name="VWAP", line=dict(color="orange", width=2)))
                fig_price.add_trace(go.Scatter(x=df_5m['timestamp'], y=df_5m['supertrend'], name="Supertrend", line=dict(color="magenta", width=1, dash="dot")))
                if profile:
                    fig_price.add_hline(y=profile.point_of_control, line_dash="dash", line_color="white", annotation_text="POC")
                fig_price.update_layout(title=f"{top_symbol} Price vs VWAP & ST", height=400)
                st.plotly_chart(fig_price, use_container_width=True, key=f"price_vwap_chart_{top_symbol}")

            with tab3:
                st.subheader("⚡ Sudden Move Predictor")
                if 'range_compression' in df_5m.columns:
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(x=df_5m['timestamp'], y=df_5m['range_compression'], name="Compression %", fill='tozeroy'))
                    fig_comp.update_layout(title="Volatility Coiling Chart", height=400)
                    st.plotly_chart(fig_comp, use_container_width=True, key=f"comp_chart_{top_symbol}")

            with tab4:
                st.subheader("🧠 AI Swarm Synthesis Deep-Dive")
                if st.button(f"🔍 Perform Multi-Agent Research on {top_symbol}", key="multi_agent_research_btn"):
                    with st.spinner("Executing Swarm Deep Research..."):
                        try:
                            from trade_system.application.agent.option_chain_agent import OptionChainAgent
                            oc_agent = OptionChainAgent(_broker)
                            import asyncio
                            oc_analysis = asyncio.run(oc_agent.analyze(top_symbol))
                            if oc_analysis:
                                st.metric("PCR", f"{oc_analysis.pcr:.2f}")
                        except Exception as e:
                            st.error(f"Research failed: {e}")
        else:
            st.warning(f"Insufficient intraday data for {top_symbol}.")
    except Exception as e:
        st.error(f"Deep dive failed: {e}")


# --- THEME & COMPACTNESS ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    div[data-testid="stVerticalBlock"] > div { margin-top: -0.5rem !important; }
    .stMetric { background: #1e2130; padding: 10px; border-radius: 8px; border: 1px solid #30363d; }
    .leaderboard-val { font-family: monospace; font-weight: bold; }
    .green { color: #00d084; font-weight: bold; }
    .red { color: #ff4d6d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if "selected_sector" not in st.session_state:
    st.session_state["selected_sector"] = None

broker = get_broker()

if broker:
    # --- MARKET HEALTH ---
    st.subheader("🏥 Market Health")
    h1, h2, h3 = st.columns(3)
    for i, symbol in enumerate(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]):
        try:
            rsi_status = get_rsi_label(symbol)
            q_dict = broker.get_quotes([symbol])
            q = q_dict.get(symbol)
            if q:
                col = [h1, h2, h3][i]
                change = q.change_percent
                cls = "green" if change >= 0 else "red"
                col.markdown(f"**{symbol.split(':')[-1]}**: <span class='{cls}'>{change:+.2f}%</span> {rsi_status}", unsafe_allow_html=True)
        except Exception as e:
            col = [h1, h2, h3][i]
            col.error(f"Quote error: {symbol.split(':')[-1]}")

    st.markdown("---")
    
    # --- TAB NAVIGATION ---
    tab_main, tab_sideways = st.tabs(["🚀 Market Activity", "🛡️ Sideways Watchlist"])
    
    with tab_main:
        st.subheader("🏁 Market Leaderboards")
        sector_summary, stock_summary = fetch_market_rankings(broker)
        
        if not sector_summary.empty:
            c1, c2, c3 = st.columns([1, 1, 2])
            
            with c1:
                st.caption("🎡 Sector Leadership")
                top_5_sectors = sector_summary.head(5)
                for _, row in top_5_sectors.iterrows():
                    val = row['Change %']
                    cls = "green" if val >= 0 else "red"
                    if st.button(f"{row['Sector']}: {val:+.2f}%", key=f"top_sec_{row['Sector']}", use_container_width=True):
                        st.session_state["selected_sector"] = row['Sector']
            
            with c2:
                st.caption("❄️ Sector Laggards")
                bot_5_sectors = sector_summary.tail(5).iloc[::-1]
                for _, row in bot_5_sectors.iterrows():
                    val = row['Change %']
                    cls = "green" if val >= 0 else "red"
                    if st.button(f"{row['Sector']}: {val:+.2f}%", key=f"bot_sec_{row['Sector']}", use_container_width=True):
                        st.session_state["selected_sector"] = row['Sector']
                    
            with c3:
                fig_sec = px.bar(
                    sector_summary, x="Sector", y="Change %",
                    color="Change %", color_continuous_scale="RdYlGn",
                    height=250, text_auto='.2f'
                )
                fig_sec.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_sec, use_container_width=True, key="sector_leadership_bar")

            s1, s2, s3 = st.columns([1, 1, 2])
            top_5_stocks = stock_summary.head(5)
            bot_5_stocks = stock_summary.tail(5).iloc[::-1]

            with s1:
                st.caption("🚀 Top 5 F&O Stocks")
                for _, row in top_5_stocks.iterrows():
                    rsi_badge = get_rsi_label(f"NSE:{row['Symbol']}-EQ")
                    change = row['Change %']
                    cls = "green" if change >= 0 else "red"
                    st.markdown(f"**{row['Symbol']}**: <span class='{cls}'>{change:+.2f}%</span> (₹{row['LTP']:.2f}){rsi_badge}", unsafe_allow_html=True)
            
            with s2:
                st.caption("📉 Bot 5 F&O Stocks")
                for _, row in bot_5_stocks.iterrows():
                    rsi_badge = get_rsi_label(f"NSE:{row['Symbol']}-EQ")
                    change = row['Change %']
                    cls = "green" if change >= 0 else "red"
                    st.markdown(f"**{row['Symbol']}**: <span class='{cls}'>{change:+.2f}%</span> (₹{row['LTP']:.2f}){rsi_badge}", unsafe_allow_html=True)
                    
            with s3:
                fig_stock = px.scatter(
                    stock_summary.head(20), x="Symbol", y="Change %",
                    size=np.abs(stock_summary.head(20)["Change %"]),
                    color="Change %", color_continuous_scale="RdYlGn",
                    height=220
                )
                fig_stock.update_layout(margin=dict(l=0, r=0, t=30, b=0), yaxis_tickformat=".2f")
                st.plotly_chart(fig_stock, use_container_width=True, key="fo_movers_scatter")

        st.sidebar.divider()
        all_sectors = sorted(list(set(FO_METADATA.values())))
        default_idx = 0
        if st.session_state["selected_sector"] in all_sectors:
            default_idx = all_sectors.index(st.session_state["selected_sector"])
            
        selected_sector = st.sidebar.selectbox("Drill-Down Sector", all_sectors, index=default_idx)
        st.session_state["selected_sector"] = selected_sector
        
        st.markdown("---")
        st.subheader(f"🚀 {selected_sector} Drill-Down")
        try:
            sector_df = fetch_sector_data(broker, selected_sector)
            
            if not sector_df.empty:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(
                        sector_df.sort_values("Change %", ascending=False),
                        column_order=("Symbol", "Change %", "LTP"),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                            "LTP": st.column_config.NumberColumn("LTP", format="%.2f"),
                        }
                    )
                with col2:
                    fig = px.bar(
                        sector_df, x="Symbol", y="Change %", 
                        color="Change %", 
                        color_continuous_scale="RdYlGn",
                        title="Intraday Change %"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="sector_stocks_bar")
                    
                st.divider()
                top_symbol = st.selectbox("Deep Dive Stock", sector_df["FullSymbol"].tolist())
                
                if top_symbol:
                    render_stock_deep_dive(broker, top_symbol)
            else:
                st.info(f"No active data for {selected_sector}.")
        except Exception as e:
            st.error(f"Sector Drill-Down encountered an error: {e}")

    with tab_sideways:
        st.subheader("🛡️ Sideways Graveyard")
        st.info("These stocks were rejected by the **Sideways Shield** (Low ADX) or **Buyer Viability** (Tight Range). Monitor them for a late-session breakout.")

        try:
            from trade_system.infrastructure.database.connection import get_engine
            from sqlalchemy import text
            query = text("""
                SELECT timestamp, symbol, action, message 
                FROM agent_thought_stream 
                WHERE action IN ('SIDEWAYS', 'DORMANT')
                AND timestamp > datetime('now', '-4 hours')
                ORDER BY timestamp DESC
            """)
            with get_engine().connect() as conn:
                sideways_df = pd.read_sql(query, conn)

            if not sideways_df.empty:
                sideways_df = sideways_df.drop_duplicates(subset=['symbol'], keep='first')
                for _, row in sideways_df.iterrows():
                    sym_clean = row['symbol'].split(':')[-1].replace('-EQ', '')
                    label_color = "#9b5de5" if row['action'] == 'SIDEWAYS' else "#f15bb5"
                    st.markdown(f"""
                    <div style="background:#1e2130; padding:10px; border-radius:8px; border-left:5px solid {label_color}; margin-bottom:10px;">
                        <span style="font-weight:bold; color:white; font-size:1.1rem;">{sym_clean}</span>
                        <span style="background:{label_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; margin-left:10px;">{row['action']}</span>
                        <p style="margin:5px 0 0 0; color:#8b949e; font-size:0.9rem;">{row['message']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No sideways/dormant stocks detected.")
        except Exception as e:
            st.error(f"Failed to load sideways watchlist: {e}")
else:
    st.info("Waiting for Fyers authentication...")
