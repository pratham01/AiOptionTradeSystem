import os, sys, sqlite3
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

def test_rr_backtest():
    conn = sqlite3.connect('data/trade_system.db')
    symbols = [
        "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:INFY-EQ",
        "NSE:TCS-EQ", "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ",
        "NSE:LT-EQ", "NSE:AXISBANK-EQ", "NSE:ITC-EQ", "NSE:BAJFINANCE-EQ"
    ]
    
    trades = []
    
    for symbol in symbols:
        df = pd.read_sql("SELECT timestamp, open, high, low, close FROM ohlcv_15m WHERE symbol = ? ORDER BY timestamp ASC", conn, params=(symbol,))
        if len(df) < 200:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['date'] = df['timestamp'].dt.date
        
        n = len(df)
        for i in range(50, n - 20):
            curr = df.iloc[i]
            c_open = curr['open']
            c_high = curr['high']
            c_low = curr['low']
            c_close = curr['close']
            c_body = abs(c_close - c_open)
            c_upper_wick = c_high - max(c_close, c_open)
            c_lower_wick = min(c_close, c_open) - c_low
            
            bar_hour = curr['timestamp'].hour
            bar_min = curr['timestamp'].minute
            if bar_hour < 9 or (bar_hour == 9 and bar_min < 30) or bar_hour >= 14:
                continue
                
            past = df.iloc[max(0, i-30) : i]
            prior_low = past['low'].min()
            prior_high = past['high'].max()
            
            sig = None
            direction = None
            sl = None
            tp = None
            
            # Bullish Sweep: took out prior low, closed green, reclaimed above prior low
            if c_low < prior_low and c_close > prior_low and c_close > c_open and c_lower_wick > c_upper_wick:
                sig = "Bullish Sweep"
                direction = 1
                sl = c_low * 0.998 # just below sweep low
                risk = c_close - sl
                tp = c_close + 2.0 * risk # 1:2 RR
                
            # Bearish Sweep: took out prior high, closed red, rejected below prior high
            elif c_high > prior_high and c_close < prior_high and c_close < c_open and c_upper_wick > c_lower_wick:
                sig = "Bearish Sweep"
                direction = -1
                sl = c_high * 1.002 # just above sweep high
                risk = sl - c_close
                tp = c_close - 2.0 * risk # 1:2 RR
                
            if sig and direction and risk > 0:
                # Walk forward candle by candle until TP, SL, or EOD
                outcome = "EOD"
                pnl = 0.0
                curr_date = curr['date']
                for fwd in range(1, 20):
                    if i + fwd >= n:
                        break
                    f_bar = df.iloc[i + fwd]
                    if f_bar['date'] != curr_date:
                        # EOD exit at last candle of the day
                        exit_p = df.iloc[i + fwd - 1]['close']
                        pnl = ((exit_p - c_close) / c_close * 100) if direction == 1 else ((c_close - exit_p) / c_close * 100)
                        outcome = "EOD"
                        break
                    
                    if direction == 1:
                        if f_bar['low'] <= sl:
                            outcome = "SL"
                            pnl = ((sl - c_close) / c_close * 100)
                            break
                        elif f_bar['high'] >= tp:
                            outcome = "TP"
                            pnl = ((tp - c_close) / c_close * 100)
                            break
                    else:
                        if f_bar['high'] >= sl:
                            outcome = "SL"
                            pnl = ((c_close - sl) / c_close * 100)
                            break
                        elif f_bar['low'] <= tp:
                            outcome = "TP"
                            pnl = ((c_close - tp) / c_close * 100)
                            break
                            
                trades.append({
                    'symbol': symbol,
                    'time': curr['timestamp'],
                    'sig': sig,
                    'direction': "LONG" if direction == 1 else "SHORT",
                    'outcome': outcome,
                    'pnl': pnl
                })
                
    tdf = pd.DataFrame(trades)
    print("================================================================")
    print("🎯 SMC INTRADAY SWEEPS WITH 1:2 RISK-TO-REWARD")
    print("================================================================")
    print(f"Total Trades: {len(tdf)}")
    print(tdf['outcome'].value_counts())
    win_rate = (tdf['pnl'] > 0).mean() * 100
    avg_pnl = tdf['pnl'].mean()
    tot_pnl = tdf['pnl'].sum()
    pos = tdf[tdf['pnl'] > 0]['pnl'].sum()
    neg = abs(tdf[tdf['pnl'] <= 0]['pnl'].sum())
    pf = pos / neg if neg > 0 else 999.0
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average PnL per trade: {avg_pnl:+.2f}%")
    print(f"Total Cumulative PnL: {tot_pnl:+.2f}%")
    print(f"Profit Factor: {pf:.2f}")
    print("================================================================\n")

if __name__ == "__main__":
    test_rr_backtest()
