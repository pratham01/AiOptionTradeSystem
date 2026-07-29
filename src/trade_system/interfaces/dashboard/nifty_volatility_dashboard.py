import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from pathlib import Path
import importlib

# Force reload agent module to ensure Streamlit picks up changes without server restart
import trade_system.domains.advisory.application.agent.nifty_volatility_analyzer_agent
importlib.reload(trade_system.domains.advisory.application.agent.nifty_volatility_analyzer_agent)
from trade_system.domains.advisory.application.agent.nifty_volatility_analyzer_agent import NiftyVolatilityAnalyzerAgent

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

# Layout
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e2230;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-title {
        font-size: 14px;
        color: #88c0d0;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #eceff4;
    }
    .signal-card {
        background: linear-gradient(135deg, #1e2230 0%, #2e3440 100%);
        border: 1px solid #4c566a;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin-bottom: 10px;
    }
    .signal-card .metric-value {
        font-size: 28px;
        background: linear-gradient(90deg, #bf616a, #d08770);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .signal-card-green .metric-value {
        background: linear-gradient(90deg, #a3be8c, #8fbcbb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .signal-card-blue .metric-value {
        background: linear-gradient(90deg, #81a1c1, #88c0d0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .signal-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    .badge-5 { background: #bf616a; color: white; }
    .badge-4 { background: #d08770; color: white; }
    .badge-3 { background: #ebcb8b; color: #2e3440; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧭 Nifty 15m Volatility Lab")
st.markdown("Analyze 15-minute Nifty 50 Index candles, discover VIX correlations, and detect breakdown pre-condition signals.")

# Initialize agent
analyzer = NiftyVolatilityAnalyzerAgent()

# Sidebar filters
st.sidebar.header("🔧 Parameters & Filters")

# Date range
default_start = date(2020, 1, 1)
default_end = date(2026, 6, 24)

col_s, col_e = st.sidebar.columns(2)
start_d = col_s.date_input("Start Date", value=default_start)
end_d = col_e.date_input("End Date", value=default_end)

filter_mode = st.sidebar.selectbox(
    "Filter Method",
    options=["Fixed Thresholds", "Statistical Z-Score"],
    index=0,
    help="Fixed Thresholds use specific % moves. Z-Scores evaluate standard deviations above the mean."
)

if filter_mode == "Fixed Thresholds":
    body_threshold = st.sidebar.slider("Min Absolute Body Move (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.05)
    range_threshold = st.sidebar.slider("Min High-Low Range (%)", min_value=0.1, max_value=3.0, value=0.8, step=0.05)
    zscore_threshold = 2.0
else:
    body_threshold = 0.5
    range_threshold = 0.8
    zscore_threshold = st.sidebar.slider("Min Z-Score (Std Devs)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

use_zscore = (filter_mode == "Statistical Z-Score")

# Run Analysis Button
if st.sidebar.button("⚡ Run Volatility Analysis", type="primary", use_container_width=True):
    with st.spinner("Executing analysis agent (fetching VIX & loading database)..."):
        try:
            analyzer.run_analysis(
                start_date=start_d,
                end_date=end_d,
                body_threshold=body_threshold,
                range_threshold=range_threshold,
                use_zscore=use_zscore,
                zscore_threshold=zscore_threshold,
                csv_output_path="data/nifty_big_movements.csv",
                report_output_path="reports/nifty_volatility_analysis_report.md"
            )
            st.sidebar.success("Analysis executed successfully!")
            # Clear caches to force reload
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Error executing analyzer: {e}")

# Cache loading from DB
@st.cache_data
def load_db_candles(start_date, end_date):
    return analyzer.load_nifty_15m_candles(start_date, end_date)

@st.cache_data
def load_vix_cached(start_date, end_date):
    cache_path = Path("data/india_vix_daily_cache.csv")
    if cache_path.exists():
        vix_df = pd.read_csv(cache_path)
        vix_df["date"] = pd.to_datetime(vix_df["date"], format="mixed").dt.date
        return vix_df[(vix_df["date"] >= start_date) & (vix_df["date"] <= end_date)].copy()
    
    # Fallback to empty if not fetched yet
    return pd.DataFrame(columns=["date", "vix_close"])

# Load data
df_nifty = load_db_candles(start_d, end_d)
df_vix = load_vix_cached(start_d, end_d)

if df_nifty.empty:
    st.warning("No Nifty 50 15m candle records found in the database. Please run the sync script or check settings.")
else:
    # Run analytical processing
    df_all, df_filtered = analyzer.analyze_movements(
        df_nifty,
        df_vix,
        body_threshold=body_threshold,
        range_threshold=range_threshold,
        use_zscore=use_zscore,
        zscore_threshold=zscore_threshold
    )

    if df_all.empty:
        st.warning("Data processing returned empty dataframe.")
    else:
        # Run breakdown signal detection on full dataset
        df_all = analyzer.detect_breakdown_signals(df_all)

        # Calculate summary metrics
        total_candles = len(df_all)
        filtered_count = len(df_filtered)
        filtered_pct = (filtered_count / total_candles) * 100 if total_candles > 0 else 0
        
        mean_body = df_all["abs_body_change_pct"].mean()
        mean_range = df_all["range_pct"].mean()
        
        overall_avg_vix = df_all["vix_close"].mean()
        filtered_avg_vix = df_filtered["vix_close"].mean() if not df_filtered.empty else 0.0
        
        # Breakdown signal metrics
        breakdown_count = int(df_all["breakdown_signal"].sum()) if "breakdown_signal" in df_all.columns else 0
        
        # Calculate VIX correlation
        vix_corr_range = 0.0
        valid_vix = df_all.dropna(subset=["vix_close"])
        if len(valid_vix) > 10:
            vix_corr_range = valid_vix["range_pct"].corr(valid_vix["vix_close"])

        # Display Metrics Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Candles Analyzed</div>
                    <div class="metric-value">{total_candles:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Big Movements Flagged</div>
                    <div class="metric-value">{filtered_count:,} ({filtered_pct:.2f}%)</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="signal-card">
                    <div class="metric-title">🚨 Breakdown Alerts</div>
                    <div class="metric-value">{breakdown_count:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Avg VIX on Flagged Days</div>
                    <div class="metric-value">{filtered_avg_vix:.2f} <span style="font-size:12px;color:#88c0d0;">(Overall: {overall_avg_vix:.2f})</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">VIX vs Range Correlation</div>
                    <div class="metric-value">{vix_corr_range:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Setup Tabs
        tab_dist, tab_vix, tab_extremes, tab_breakdown, tab_data = st.tabs([
            "📊 Volatility & Distributions",
            "⚡ VIX Correlations",
            "👑 Extreme Movements",
            "🚨 Breakdown Signals",
            "📋 Filtered Data Grid"
        ])
        
        # TAB 1: Volatility & Distributions
        with tab_dist:
            st.subheader("Distribution Analysis of Big Movements")
            
            # Prepare data
            df_all["year"] = pd.to_datetime(df_all["timestamp"], format="mixed").dt.year
            df_filtered["year"] = pd.to_datetime(df_filtered["timestamp"], format="mixed").dt.year
            
            # Yearly chart
            yearly_all = df_all.groupby("year").size().reset_index(name="total")
            yearly_filt = df_filtered.groupby("year").size().reset_index(name="flagged")
            yearly_merged = pd.merge(yearly_all, yearly_filt, on="year", how="left").fillna(0)
            yearly_merged["percent"] = (yearly_merged["flagged"] / yearly_merged["total"]) * 100
            
            fig_year = px.bar(
                yearly_merged,
                x="year",
                y="flagged",
                text="flagged",
                labels={"flagged": "Count", "year": "Year"},
                title="Number of Big Movements by Year",
                color_discrete_sequence=["#81a1c1"],
                template="plotly_dark"
            )
            fig_year.update_traces(textposition='outside')
            
            # Day of week chart
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            day_filt = df_filtered.groupby("day_of_week").size().reindex(days_order).fillna(0).reset_index(name="count")
            
            fig_day = px.bar(
                day_filt,
                x="day_of_week",
                y="count",
                text="count",
                labels={"count": "Count", "day_of_week": "Day"},
                title="Big Movements by Day of the Week",
                color_discrete_sequence=["#a3be8c"],
                template="plotly_dark"
            )
            fig_day.update_traces(textposition='outside')
            
            # Show side-by-side
            col_y, col_d = st.columns(2)
            col_y.plotly_chart(fig_year, use_container_width=True)
            col_d.plotly_chart(fig_day, use_container_width=True)
            
            # Hourly Time of Day Distribution
            time_filt = df_filtered.groupby("time_of_day").size().reset_index(name="count").sort_values(by="time_of_day")
            
            fig_time = px.bar(
                time_filt,
                x="time_of_day",
                y="count",
                text="count",
                labels={"count": "Count", "time_of_day": "Candle Start Time (IST)"},
                title="Volatility Spike Frequency by Time of Day",
                color_discrete_sequence=["#ebcb8b"],
                template="plotly_dark"
            )
            fig_time.update_traces(textposition='outside')
            st.plotly_chart(fig_time, use_container_width=True)
            st.info("💡 Insight: The 09:15 open candle represents the highest volatility period (~20-25% of all big movements) due to overnight news pricing, followed by closing hours (15:00 onwards).")
            
        # TAB 2: VIX Correlations
        with tab_vix:
            st.subheader("India VIX vs Intraday 15m Volatility")
            
            if not df_vix.empty:
                # Scatter Plot
                df_scatter = df_all.dropna(subset=["vix_close", "range_pct"]).sample(min(len(df_all), 5000), random_state=42)
                fig_scatter = px.scatter(
                    df_scatter,
                    x="vix_close",
                    y="range_pct",
                    opacity=0.4,
                    trendline="ols",
                    trendline_color_override="#bf616a",
                    labels={"vix_close": "India VIX Close", "range_pct": "15m Candle Range (%)"},
                    title="Scatter: India VIX Level vs Nifty 15m Candle High-Low Range (%)",
                    color_discrete_sequence=["#88c0d0"],
                    template="plotly_dark"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Average VIX Bar comparison
                vix_compare = pd.DataFrame({
                    "Category": ["Overall Avg VIX", "Avg VIX on Big Move Days"],
                    "VIX Value": [overall_avg_vix, filtered_avg_vix]
                })
                
                fig_vix_bar = px.bar(
                    vix_compare,
                    x="Category",
                    y="VIX Value",
                    text="VIX Value",
                    color="Category",
                    color_discrete_map={"Overall Avg VIX": "#4c566a", "Avg VIX on Big Move Days": "#bf616a"},
                    title="VIX Environment Comparison",
                    template="plotly_dark"
                )
                fig_vix_bar.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                st.plotly_chart(fig_vix_bar, use_container_width=True)
                
            else:
                st.warning("No VIX cached data loaded to calculate correlations. Please run the analysis first to download VIX.")
                
        # TAB 3: Extreme Movements
        with tab_extremes:
            st.subheader("👑 Historical Extreme 15-Minute Movements")
            
            # Fetch top 10 absolute returns
            top_body = df_all.sort_values(by="abs_body_change_pct", ascending=False).head(10).copy()
            top_body["timestamp"] = top_body["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            top_body["body_change_pct"] = top_body["body_change_pct"].apply(lambda x: f"{x:+.2f}%")
            top_body["range_pct"] = top_body["range_pct"].apply(lambda x: f"{x:.2f}%")
            
            # Fetch top 10 ranges
            top_range = df_all.sort_values(by="range_pct", ascending=False).head(10).copy()
            top_range["timestamp"] = top_range["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            top_range["range_pct"] = top_range["range_pct"].apply(lambda x: f"{x:.2f}%")
            top_range["body_change_pct"] = top_range["body_change_pct"].apply(lambda x: f"{x:+.2f}%")
            
            col_b, col_r = st.columns(2)
            
            with col_b:
                st.markdown("### Top 10 Largest Body Returns (Close vs Open)")
                st.dataframe(
                    top_body[["timestamp", "day_of_week", "open", "close", "body_change_pct", "range_pct", "vix_close"]],
                    use_container_width=True,
                    hide_index=True
                )
                
            with col_r:
                st.markdown("### Top 10 Largest Candle High-Low Ranges")
                st.dataframe(
                    top_range[["timestamp", "day_of_week", "high", "low", "range_pct", "body_change_pct", "vix_close"]],
                    use_container_width=True,
                    hide_index=True
                )

        # TAB 4: Breakdown Signals
        with tab_breakdown:
            st.subheader("🚨 Breakdown Pre-Condition Signal Detection")
            st.markdown(
                "This tab identifies candles where **3 or more** of the 5 pre-conditions fire simultaneously, "
                "signaling a high probability of an imminent large directional move."
            )

            if "breakdown_signal" not in df_all.columns:
                st.warning("Breakdown signals not computed. Please run the analysis first.")
            else:
                df_bd = df_all[df_all["breakdown_signal"]].copy()

                if df_bd.empty:
                    st.info("No breakdown signals detected in the selected date range.")
                else:
                    # --- Metrics Row ---
                    bd_total = len(df_bd)
                    bd_5 = int((df_bd["breakdown_signal_count"] == 5).sum())
                    bd_4 = int((df_bd["breakdown_signal_count"] == 4).sum())
                    bd_3 = int((df_bd["breakdown_signal_count"] == 3).sum())

                    # Count unique days with breakdown signals
                    if "date" in df_bd.columns:
                        bd_days = df_bd["date"].nunique()
                    else:
                        df_bd["_date"] = pd.to_datetime(df_bd["timestamp"], format="mixed").dt.date
                        bd_days = df_bd["_date"].nunique()

                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    with mc1:
                        st.markdown(
                            f'<div class="signal-card"><div class="metric-title">Total Alerts</div>'
                            f'<div class="metric-value">{bd_total:,}</div></div>',
                            unsafe_allow_html=True
                        )
                    with mc2:
                        st.markdown(
                            f'<div class="signal-card signal-card-blue"><div class="metric-title">Trading Days Affected</div>'
                            f'<div class="metric-value">{bd_days:,}</div></div>',
                            unsafe_allow_html=True
                        )
                    with mc3:
                        st.markdown(
                            f'<div class="signal-card"><div class="metric-title">'
                            f'<span class="signal-badge badge-5">5/5</span> Max Conviction</div>'
                            f'<div class="metric-value">{bd_5}</div></div>',
                            unsafe_allow_html=True
                        )
                    with mc4:
                        st.markdown(
                            f'<div class="signal-card"><div class="metric-title">'
                            f'<span class="signal-badge badge-4">4/5</span> High Conviction</div>'
                            f'<div class="metric-value">{bd_4}</div></div>',
                            unsafe_allow_html=True
                        )
                    with mc5:
                        st.markdown(
                            f'<div class="signal-card signal-card-green"><div class="metric-title">'
                            f'<span class="signal-badge badge-3">3/5</span> Standard</div>'
                            f'<div class="metric-value">{bd_3}</div></div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # --- Signal Type Distribution (Radar + Bar) ---
                    sig_cols = {
                        "sig_resistance_rejection": "Resistance\nRejection",
                        "sig_bearish_streak": "Bearish\nStreak 4+",
                        "sig_volatility_squeeze": "Volatility\nSqueeze",
                        "sig_volume_divergence": "Volume\nDivergence",
                        "sig_lower_highs": "Lower\nHighs 3+",
                    }

                    col_radar, col_bar = st.columns(2)

                    with col_radar:
                        # Radar chart of signal type frequency
                        radar_values = []
                        radar_labels = []
                        for col, label in sig_cols.items():
                            if col in df_bd.columns:
                                pct = (df_bd[col].sum() / bd_total) * 100
                                radar_values.append(pct)
                                radar_labels.append(label)

                        if radar_values:
                            # Close the polygon
                            radar_values_closed = radar_values + [radar_values[0]]
                            radar_labels_closed = radar_labels + [radar_labels[0]]

                            fig_radar = go.Figure()
                            fig_radar.add_trace(go.Scatterpolar(
                                r=radar_values_closed,
                                theta=radar_labels_closed,
                                fill='toself',
                                fillcolor='rgba(136, 192, 208, 0.2)',
                                line=dict(color='#88c0d0', width=2),
                                name="Signal Frequency"
                            ))
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100],
                                        ticksuffix="%",
                                        gridcolor="#3b4252",
                                    ),
                                    angularaxis=dict(gridcolor="#3b4252"),
                                    bgcolor="#2e3440",
                                ),
                                template="plotly_dark",
                                title="Signal Type Frequency (% of All Alerts)",
                                height=400,
                                margin=dict(t=60, b=40),
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)

                    with col_bar:
                        # Signal count bar chart
                        count_dist = df_bd["breakdown_signal_count"].value_counts().sort_index().reset_index()
                        count_dist.columns = ["Signal Count", "Candles"]

                        color_map = {3: "#ebcb8b", 4: "#d08770", 5: "#bf616a"}
                        count_dist["Color"] = count_dist["Signal Count"].map(
                            lambda x: color_map.get(x, "#81a1c1")
                        )

                        fig_count = go.Figure()
                        for _, row in count_dist.iterrows():
                            fig_count.add_trace(go.Bar(
                                x=[f"{int(row['Signal Count'])}/5 Signals"],
                                y=[row["Candles"]],
                                marker_color=color_map.get(int(row["Signal Count"]), "#81a1c1"),
                                text=[f"{int(row['Candles']):,}"],
                                textposition="outside",
                                name=f"{int(row['Signal Count'])}/5",
                                showlegend=False,
                            ))
                        fig_count.update_layout(
                            title="Alert Distribution by Conviction Level",
                            xaxis_title="Conviction Level",
                            yaxis_title="Number of Candles",
                            template="plotly_dark",
                            height=400,
                            margin=dict(t=60, b=40),
                        )
                        st.plotly_chart(fig_count, use_container_width=True)

                    # --- Timeline: Monthly Breakdown Alert Frequency ---
                    st.markdown("---")
                    st.markdown("### 📅 Breakdown Signal Timeline")

                    df_bd_ts = df_bd.copy()
                    df_bd_ts["month"] = pd.to_datetime(df_bd_ts["timestamp"], format="mixed").dt.to_period("M").astype(str)

                    monthly_signals = df_bd_ts.groupby(["month", "breakdown_signal_count"]).size().reset_index(name="count")
                    monthly_signals["conviction"] = monthly_signals["breakdown_signal_count"].map(
                        {3: "3/5 Standard", 4: "4/5 High", 5: "5/5 Maximum"}
                    )

                    fig_timeline = px.bar(
                        monthly_signals,
                        x="month",
                        y="count",
                        color="conviction",
                        color_discrete_map={
                            "3/5 Standard": "#ebcb8b",
                            "4/5 High": "#d08770",
                            "5/5 Maximum": "#bf616a",
                        },
                        labels={"count": "Alerts", "month": "Month"},
                        title="Monthly Breakdown Alert Frequency by Conviction",
                        template="plotly_dark",
                        barmode="stack",
                    )
                    fig_timeline.update_layout(
                        xaxis=dict(tickangle=-45, dtick=3),
                        height=350,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # --- Day Drill-Down ---
                    st.markdown("---")
                    st.markdown("### 🔍 Day Drill-Down")
                    st.caption("Select a specific date to see the intraday signal progression and identify the pre-conditions building up before a move.")

                    # Get unique dates with breakdown signals
                    if "date" not in df_bd.columns:
                        df_bd["date"] = pd.to_datetime(df_bd["timestamp"], format="mixed").dt.date
                    bd_dates = sorted(df_bd["date"].unique(), reverse=True)
                    bd_date_strings = [str(d) for d in bd_dates]

                    if bd_date_strings:
                        selected_date_str = st.selectbox(
                            "Select date to inspect:",
                            options=bd_date_strings,
                            index=0,
                            help="Dates are sorted most recent first. Only dates with breakdown alerts are shown."
                        )
                        selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d").date()

                        # Get ALL candles for that day (not just flagged ones)
                        day_all = df_all[df_all["timestamp"].dt.date == selected_date].copy()

                        if not day_all.empty:
                            # Intraday signal progression chart
                            day_all["time_str"] = day_all["timestamp"].dt.strftime("%H:%M")
                            day_all["signal_count"] = day_all.get("breakdown_signal_count", 0).fillna(0).astype(int)

                            # Create a dual-axis chart: candlestick + signal bars
                            fig_drill = go.Figure()

                            # Price line
                            fig_drill.add_trace(go.Scatter(
                                x=day_all["time_str"],
                                y=day_all["close"],
                                mode="lines+markers",
                                name="Close Price",
                                line=dict(color="#88c0d0", width=2),
                                marker=dict(size=5),
                                yaxis="y",
                            ))

                            # Signal count as bars
                            bar_colors = day_all["signal_count"].apply(
                                lambda x: "#bf616a" if x >= 5 else (
                                    "#d08770" if x >= 4 else (
                                        "#ebcb8b" if x >= 3 else (
                                            "#4c566a" if x > 0 else "rgba(0,0,0,0)"
                                        )
                                    )
                                )
                            )

                            fig_drill.add_trace(go.Bar(
                                x=day_all["time_str"],
                                y=day_all["signal_count"],
                                name="Signal Count",
                                marker_color=bar_colors,
                                opacity=0.7,
                                yaxis="y2",
                            ))

                            fig_drill.update_layout(
                                title=f"Intraday Signal Progression — {selected_date_str} ({day_all.iloc[0].get('day_of_week', '')})",
                                template="plotly_dark",
                                height=400,
                                xaxis=dict(title="Time (IST)", tickangle=-45),
                                yaxis=dict(title="Close Price", side="left", showgrid=False),
                                yaxis2=dict(
                                    title="Signal Count (/5)",
                                    side="right",
                                    overlaying="y",
                                    range=[0, 6],
                                    showgrid=False,
                                ),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                bargap=0.3,
                            )
                            st.plotly_chart(fig_drill, use_container_width=True)

                            # Detail table for that day
                            day_bd = day_all[day_all["breakdown_signal"] == True].copy()
                            if not day_bd.empty:
                                st.markdown(f"**🚨 {len(day_bd)} breakdown alerts on {selected_date_str}:**")
                                day_disp = day_bd.copy()
                                day_disp["timestamp"] = day_disp["timestamp"].dt.strftime("%H:%M")

                                disp_cols = ["timestamp", "close", "body_change_pct", "range_pct",
                                             "bearish_streak", "bandwidth", "lower_high_streak",
                                             "breakdown_signal_count", "breakdown_reasons"]
                                disp_cols = [c for c in disp_cols if c in day_disp.columns]

                                fmt = {}
                                if "body_change_pct" in disp_cols:
                                    fmt["body_change_pct"] = "{:+.3f}%"
                                if "range_pct" in disp_cols:
                                    fmt["range_pct"] = "{:.3f}%"
                                if "close" in disp_cols:
                                    fmt["close"] = "{:.2f}"
                                if "bandwidth" in disp_cols:
                                    fmt["bandwidth"] = "{:.3f}"

                                st.dataframe(
                                    day_disp[disp_cols].style.format(
                                        {k: v for k, v in fmt.items() if k in disp_cols}
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                # Day summary
                                day_open = day_all.iloc[0]["open"]
                                day_close = day_all.iloc[-1]["close"]
                                day_high = day_all["high"].max()
                                day_low = day_all["low"].min()
                                day_move = day_close - day_open
                                day_move_pct = (day_move / day_open) * 100

                                col_ds1, col_ds2, col_ds3 = st.columns(3)
                                with col_ds1:
                                    move_color = "#bf616a" if day_move < 0 else "#a3be8c"
                                    st.markdown(
                                        f'<div class="metric-card"><div class="metric-title">Day Move</div>'
                                        f'<div class="metric-value" style="color:{move_color}">'
                                        f'{day_move:+.2f} pts ({day_move_pct:+.2f}%)</div></div>',
                                        unsafe_allow_html=True
                                    )
                                with col_ds2:
                                    day_range = day_high - day_low
                                    st.markdown(
                                        f'<div class="metric-card"><div class="metric-title">Day Range</div>'
                                        f'<div class="metric-value">{day_range:.2f} pts</div></div>',
                                        unsafe_allow_html=True
                                    )
                                with col_ds3:
                                    first_alert_time = day_bd.iloc[0]["timestamp"]
                                    st.markdown(
                                        f'<div class="signal-card"><div class="metric-title">First Alert At</div>'
                                        f'<div class="metric-value">{first_alert_time}</div></div>',
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("All candles for this day are loaded but no breakdown alerts found in the filtered view.")

                    # --- Full Breakdown Alerts Data Grid ---
                    st.markdown("---")
                    st.markdown("### 📋 All Breakdown Alerts")

                    # Filter controls
                    min_signals = st.select_slider(
                        "Minimum signal count:",
                        options=[3, 4, 5],
                        value=3,
                        help="Filter alerts by minimum number of active pre-conditions"
                    )

                    df_bd_display = df_bd[df_bd["breakdown_signal_count"] >= min_signals].copy()
                    df_bd_display["timestamp"] = pd.to_datetime(df_bd_display["timestamp"], format="mixed").dt.strftime("%Y-%m-%d %H:%M")
                    df_bd_display.sort_values(by="timestamp", ascending=False, inplace=True)

                    grid_cols = [
                        "timestamp", "day_of_week", "close", "body_change_pct", "range_pct",
                        "bearish_streak", "bandwidth", "lower_high_streak",
                        "breakdown_signal_count", "breakdown_reasons", "vix_close"
                    ]
                    grid_cols = [c for c in grid_cols if c in df_bd_display.columns]

                    grid_fmt = {}
                    if "body_change_pct" in grid_cols:
                        grid_fmt["body_change_pct"] = "{:+.3f}%"
                    if "range_pct" in grid_cols:
                        grid_fmt["range_pct"] = "{:.3f}%"
                    if "close" in grid_cols:
                        grid_fmt["close"] = "{:.2f}"
                    if "vix_close" in grid_cols:
                        grid_fmt["vix_close"] = "{:.2f}"
                    if "bandwidth" in grid_cols:
                        grid_fmt["bandwidth"] = "{:.3f}"

                    st.dataframe(
                        df_bd_display[grid_cols].style.format(
                            {k: v for k, v in grid_fmt.items() if k in grid_cols}
                        ),
                        use_container_width=True,
                        hide_index=True,
                        height=500,
                    )

                    st.caption(f"Showing {len(df_bd_display):,} alerts with {min_signals}+ signals")

                    # Download button
                    csv_bd = df_bd.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download All Breakdown Alerts as CSV",
                        data=csv_bd,
                        file_name="nifty_breakdown_alerts.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        # TAB 5: Filtered Data Grid
        with tab_data:
            st.subheader("Detailed Filtered Volatility Records")
            
            if not df_filtered.empty:
                # Format datetime column for display
                df_disp = df_filtered.copy()
                df_disp["timestamp"] = df_disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                
                # Sort descending by timestamp
                df_disp.sort_values(by="timestamp", ascending=False, inplace=True)
                
                # Export clean columns
                show_cols = [
                    "timestamp", "day_of_week", "open", "high", "low", "close", 
                    "body_change_pct", "range_pct", "trigger_reasons", "vix_close"
                ]
                show_cols = [col for col in show_cols if col in df_disp.columns]
                
                format_dict = {
                    "open": "{:.2f}",
                    "high": "{:.2f}",
                    "low": "{:.2f}",
                    "close": "{:.2f}",
                    "body_change_pct": "{:+.2f}%",
                    "range_pct": "{:.2f}%",
                    "vix_close": "{:.2f}"
                }
                format_dict = {k: v for k, v in format_dict.items() if k in show_cols}
                
                st.dataframe(
                    df_disp[show_cols].style.format(format_dict),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download CSV button
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv_data,
                    file_name="nifty_filtered_volatility.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No records matched the current parameters.")
