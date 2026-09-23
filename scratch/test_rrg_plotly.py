import plotly.graph_objects as go
import pandas as pd
from trade_system.domains.analysis.application.analysis.sector_rotation_rrg import SectorRRGEngine
from trade_system.interfaces.dashboard.sector_scope_dashboard import fetch_target_date_and_data, get_sector_mapping

def test_plot():
    latest_date_str, df, last_candle_ts = fetch_target_date_and_data("2026-09-22")
    fo_metadata = get_sector_mapping()
    df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))
    target_date = pd.to_datetime(latest_date_str).date()
    today_15m = df[df['timestamp'].dt.date == target_date].sort_values('timestamp')

    # Construct mock merged_closes
    grouped = today_15m.groupby(['symbol', 'sector'])
    rows = []
    for (sym, sec), grp in grouped:
        p_chg = (grp['close'].iloc[-1] - grp['open'].iloc[0]) / grp['open'].iloc[0] * 100
        rows.append({"symbol": sym, "sector": sec, "pChange": p_chg, "vol_surge": 1.2})
    merged_closes = pd.DataFrame(rows)

    rrg_data = SectorRRGEngine.compute_rrg(today_15m, merged_closes, tail_bars=4)

    fig = go.Figure()

    # Determine axis ranges centered around 100
    r_vals = [p.rs_ratio for p in rrg_data.values()]
    m_vals = [p.rs_momentum for p in rrg_data.values()]
    for p in rrg_data.values():
        for r_tail, m_tail in p.history_tail:
            r_vals.append(r_tail)
            m_vals.append(m_tail)

    r_min, r_max = min(r_vals + [97.0]), max(r_vals + [103.0])
    m_min, m_max = min(m_vals + [97.0]), max(m_vals + [103.0])
    pad_r = max((r_max - r_min) * 0.15, 1.0)
    pad_m = max((m_max - m_min) * 0.15, 1.0)
    x_range = [r_min - pad_r, r_max + pad_r]
    y_range = [m_min - pad_m, m_max + pad_m]

    # Shaded Quadrant Backgrounds
    # Top-Right: LEADING (Green)
    fig.add_shape(type="rect", x0=100, y0=100, x1=x_range[1], y1=y_range[1],
                  fillcolor="rgba(34, 197, 94, 0.08)", line_width=0, layer="below")
    # Bottom-Right: WEAKENING (Yellow)
    fig.add_shape(type="rect", x0=100, y0=y_range[0], x1=x_range[1], y1=100,
                  fillcolor="rgba(234, 179, 8, 0.08)", line_width=0, layer="below")
    # Bottom-Left: LAGGING (Red)
    fig.add_shape(type="rect", x0=x_range[0], y0=y_range[0], x1=100, y1=100,
                  fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, layer="below")
    # Top-Left: IMPROVING (Blue)
    fig.add_shape(type="rect", x0=x_range[0], y0=100, x1=100, y1=y_range[1],
                  fillcolor="rgba(59, 130, 246, 0.08)", line_width=0, layer="below")

    # Benchmark Center Crosshairs
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", line_width=1.5)
    fig.add_vline(x=100, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", line_width=1.5)

    # Quadrant Watermark Annotations
    fig.add_annotation(x=x_range[1] - pad_r * 0.3, y=y_range[1] - pad_m * 0.3,
                       text="<b>LEADING</b>", showarrow=False, font=dict(color="rgba(34, 197, 94, 0.5)", size=16))
    fig.add_annotation(x=x_range[1] - pad_r * 0.3, y=y_range[0] + pad_m * 0.3,
                       text="<b>WEAKENING</b>", showarrow=False, font=dict(color="rgba(234, 179, 8, 0.5)", size=16))
    fig.add_annotation(x=x_range[0] + pad_r * 0.3, y=y_range[0] + pad_m * 0.3,
                       text="<b>LAGGING</b>", showarrow=False, font=dict(color="rgba(239, 68, 68, 0.5)", size=16))
    fig.add_annotation(x=x_range[0] + pad_r * 0.3, y=y_range[1] - pad_m * 0.3,
                       text="<b>IMPROVING</b>", showarrow=False, font=dict(color="rgba(59, 130, 246, 0.5)", size=16))

    # Add Trajectory Tails and Markers for each sector
    colors = {
        "LEADING": "#22c55e",
        "WEAKENING": "#eab308",
        "LAGGING": "#ef4444",
        "IMPROVING": "#38bdf8"
    }

    for sec, pt in rrg_data.items():
        c = colors.get(pt.quadrant, "#cbd5e1")
        # Tail line
        if len(pt.history_tail) > 1:
            t_xs = [t[0] for t in pt.history_tail]
            t_ys = [t[1] for t in pt.history_tail]
            fig.add_trace(go.Scatter(
                x=t_xs, y=t_ys,
                mode="lines",
                line=dict(color=c, width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False
            ))

        # Current Point Marker
        fig.add_trace(go.Scatter(
            x=[pt.rs_ratio],
            y=[pt.rs_momentum],
            mode="markers+text",
            name=sec,
            text=[f"<b>{sec}</b>"],
            textposition="top center",
            textfont=dict(size=10, color="#f8fafc"),
            marker=dict(size=11, color=c, line=dict(color="#ffffff", width=1.5)),
            hovertemplate=(
                f"<b>{sec}</b> ({pt.quadrant})<br>"
                f"RS-Ratio: %{{x:.2f}}<br>"
                f"RS-Momentum: %{{y:.2f}}<br>"
                f"Return: {pt.sector_return:+.2f}%<br>"
                f"Advances: {pt.advances} / Declines: {pt.declines}<br>"
                f"Avg Vol Surge: {pt.avg_vol_surge:.1f}x<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title="<b>🌌 Sector Rotation RRG (StockMojo Model)</b> — Benchmark: NIFTY 50",
        xaxis=dict(title="<b>RS-Ratio</b> (Relative Strength vs Nifty 50)", range=x_range, zeroline=False, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="<b>RS-Momentum</b> (Relative Momentum)", range=y_range, zeroline=False, gridcolor="rgba(255,255,255,0.06)"),
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    print("Plotly RRG Figure generated successfully!")

if __name__ == "__main__":
    test_plot()
