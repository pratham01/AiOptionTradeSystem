import sqlite3
import pandas as pd
import plotly.graph_objects as go
from trade_system.domains.strategy.application.indicators.bollinger_bands import BollingerBandsDetector, BBPhase

def run_backtest():
    print("Fetching 15m historical data for Nifty 50 (NSE:NIFTY50-INDEX) from local database...")
    
    conn = sqlite3.connect("data/trade_system.db")
    query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE symbol='NSE:NIFTY50-INDEX' ORDER BY timestamp ASC"
    df = pd.read_sql(query, conn)
    
    if df.empty:
        print("Failed to fetch data.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # We must have valid volume. Nifty index often reports 0 volume.
    if df['volume'].sum() == 0:
        print("Warning: Volume data is zero. Simulating average volume.")
        df['volume'] = 1000
    
    print("Applying Refined Bollinger Bands...")
    detector = BollingerBandsDetector(period=20, std_dev=2.0, squeeze_percentile=10.0)
    df = detector.compute(df)
    
    print("Running Backtest Simulation...\n")
    
    trades = []
    in_position = False
    position_type = None
    entry_price = 0.0
    entry_date = None
    stop_loss = 0.0
    
    for dt, row in df.iterrows():
        if not in_position:
            # ENTRY LOGIC
            is_bullish = row['bb_phase'] == BBPhase.EXPANSION_BULLISH.value
            is_bearish = row['bb_phase'] == BBPhase.EXPANSION_BEARISH.value
            high_volatility = row['volatility_explosion_score'] > 1.5
            
            if is_bullish and high_volatility:
                in_position = True
                position_type = "LONG (CALL)"
                entry_price = row['close']
                entry_date = dt
                stop_loss = entry_price * 0.995 # 0.5% stop loss
            elif is_bearish and high_volatility:
                in_position = True
                position_type = "SHORT (PUT)"
                entry_price = row['close']
                entry_date = dt
                stop_loss = entry_price * 1.005 # 0.5% stop loss
                
        else:
            # EXIT LOGIC
            exit_price = row['close']
            exit_reason = None
            
            if position_type == "LONG (CALL)":
                if exit_price <= stop_loss:
                    exit_reason = "Stop Loss"
                elif row['bb_percent_b'] < 0.7:
                    exit_reason = "Momentum Died (%b < 0.7)"
                    
                if exit_reason:
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    trades.append({
                        "Type": position_type,
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Exit Date": dt,
                        "Exit Price": exit_price,
                        "Reason": exit_reason,
                        "PnL %": pnl_pct
                    })
                    in_position = False
                    
            elif position_type == "SHORT (PUT)":
                if exit_price >= stop_loss:
                    exit_reason = "Stop Loss"
                elif row['bb_percent_b'] > 0.3:
                    exit_reason = "Momentum Died (%b > 0.3)"
                    
                if exit_reason:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    trades.append({
                        "Type": position_type,
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Exit Date": dt,
                        "Exit Price": exit_price,
                        "Reason": exit_reason,
                        "PnL %": pnl_pct
                    })
                    in_position = False
                    
    # Results
    if not trades:
        print("No trades generated based on these strict rules over the last 60 days.")
        return
        
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['PnL %'] > 0])
    win_rate = (winning_trades / total_trades) * 100
    avg_pnl = trades_df['PnL %'].mean()
    cum_pnl = trades_df['PnL %'].sum()
    
    print("=" * 60)
    print("             BOLLINGER BAND BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Avg PnL/Trade:{avg_pnl:.2f}% (Underlying Index %)")
    print(f"Cum. Index PnL:{cum_pnl:.2f}%")
    print("\nNote: Options PnL would be magnified significantly by Delta/Gamma.")
    print("-" * 60)
    print("Detailed Trade Log (Last 15 trades):")
    
    for _, t in trades_df.tail(15).iterrows():
        pnl_str = f"+{t['PnL %']:.2f}%" if t['PnL %'] > 0 else f"{t['PnL %']:.2f}%"
        print(f"[{t['Entry Date'].strftime('%Y-%m-%d %H:%M')}] {t['Type']:<12} | Entry: {t['Entry Price']:.2f} -> Exit: {t['Exit Price']:.2f} ({t['Exit Date'].strftime('%m-%d %H:%M')}) | {pnl_str:>8} | {t['Reason']}")

    print("\nGenerating Interactive Visual Validation Chart...")
    
    # Plotly Chart
    fig = go.Figure()
    
    # Add Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price",
        increasing_line_color='grey', decreasing_line_color='black'
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='rgba(0,0,255,0.3)', width=1), name="Upper Band"))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='rgba(0,0,255,0.3)', width=1), name="Lower Band", fill='tonexty', fillcolor='rgba(0,0,255,0.05)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_basis'], line=dict(color='orange', width=1, dash='dot'), name="Basis (SMA)"))
    
    # Add Entry/Exit Markers
    long_entries_x = trades_df[trades_df['Type'] == 'LONG (CALL)']['Entry Date']
    long_entries_y = trades_df[trades_df['Type'] == 'LONG (CALL)']['Entry Price']
    
    short_entries_x = trades_df[trades_df['Type'] == 'SHORT (PUT)']['Entry Date']
    short_entries_y = trades_df[trades_df['Type'] == 'SHORT (PUT)']['Entry Price']
    
    exits_x = trades_df['Exit Date']
    exits_y = trades_df['Exit Price']
    
    fig.add_trace(go.Scatter(
        x=long_entries_x, y=long_entries_y,
        mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
        name="Long Entry"
    ))
    
    fig.add_trace(go.Scatter(
        x=short_entries_x, y=short_entries_y,
        mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
        name="Short Entry"
    ))
    
    fig.add_trace(go.Scatter(
        x=exits_x, y=exits_y,
        mode='markers', marker=dict(symbol='x', size=8, color='black'),
        name="Exit"
    ))
    
    fig.update_layout(
        title="Bollinger Bands Squeeze & Breakout Validation Chart",
        yaxis_title="Price",
        xaxis_title="Time",
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )
    
    chart_path = "bb_backtest_validation.html"
    fig.write_html(chart_path)
    print(f"Chart successfully saved to: {chart_path}")
    print("Open this file in your web browser to visually inspect every trade!")

if __name__ == "__main__":
    run_backtest()
