import sys, os
from datetime import datetime, timedelta
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
import numpy as np
from sqlalchemy import text
from trade_system.infrastructure.database.connection import get_engine
import warnings
warnings.filterwarnings('ignore')

engine = get_engine()

print("📦 Loading daily OHLCV from DB...")
with engine.connect() as conn:
    df_all = pd.read_sql(text("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv_daily
        WHERE timestamp >= date('now', '-260 days')
        ORDER BY symbol, timestamp ASC
    """), conn)

if df_all.empty:
    with engine.connect() as conn:
        df_all = pd.read_sql(text("""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM market_data
            WHERE timeframe = 'D' AND timestamp >= date('now', '-260 days')
            ORDER BY symbol, timestamp ASC
        """), conn)

df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], format='mixed')
df_all['date'] = df_all['timestamp'].dt.date
df_all = df_all.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

# Get unique trading dates sorted
all_dates = sorted(df_all['date'].unique())
if len(all_dates) < 10:
    print("Not enough data to backtest.")
    sys.exit(1)

# Pick the last 6 trading dates. We scan on dates[-6:-1] and test on the day after
test_dates = all_dates[-6:-1]
print(f"🗓️ Backtesting over last week's trading dates: {[str(d) for d in test_dates]}")

def calc_supertrend(df, period=10, multiplier=3.0):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    return direction

results_log = []

for target_date in test_dates:
    # Filter data strictly up to target_date for scanning
    df_scan = df_all[df_all['date'] <= target_date]
    # Keep future data for performance evaluation (next 3 trading days)
    df_future = df_all[df_all['date'] > target_date]

    for symbol, grp in df_scan.groupby('symbol'):
        grp = grp.sort_values('timestamp').reset_index(drop=True)
        if len(grp) < 60:
            continue
            
        close = grp['close']
        high  = grp['high']
        low   = grp['low']
        vol   = grp['volume'].replace(0, np.nan)
        last  = grp.iloc[-1]
        prev  = grp.iloc[-2]
        
        # Ensure the last date in grp is actually the target_date
        if last['date'] != target_date:
            continue

        clean = symbol.replace('NSE:', '').replace('-EQ', '').replace('-INDEX', '')

        # Indicators
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bbu   = sma20 + 2 * std20
        bbl   = sma20 - 2 * std20
        bbw   = (bbu - bbl) / sma20 * 100

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)

        prev_c = close.shift(1)
        tr     = pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
        atr14  = tr.rolling(14).mean()

        vsma20 = vol.rolling(20).mean()

        w52_high = high.rolling(252).max()
        w52_low  = low.rolling(252).min()

        latest_rsi  = rsi.iloc[-1]
        latest_bbw  = bbw.iloc[-1]
        bbw_5pct    = bbw.quantile(0.07)
        latest_atr  = atr14.iloc[-1]
        atr_sma50   = atr14.rolling(50).mean().iloc[-1]
        latest_vol  = vol.iloc[-1]
        latest_vsma = vsma20.iloc[-1]
        latest_52wh = w52_high.iloc[-1]

        if pd.isna(latest_rsi) or pd.isna(latest_bbw):
            continue

        signals = []

        # A. Volatility Squeeze
        if (latest_bbw <= bbw_5pct and not pd.isna(atr_sma50) and latest_atr < atr_sma50 and
                (latest_52wh - last['close']) / last['close'] < 0.04):
            signals.append(('LONG', 'Squeeze'))

        # B. RSI Oversold Reversal
        rsi_3ago = rsi.iloc[-3] if len(rsi) >= 3 else np.nan
        if (latest_rsi < 35 and not pd.isna(rsi_3ago) and latest_rsi > rsi.iloc[-2] and
                last['close'] > last['low'] * 1.005 and last['close'] > prev['close']):
            signals.append(('LONG', 'RSI_Oversold'))

        # C. RSI Overbought Short
        if (latest_rsi > 72 and last['close'] < prev['close'] and last['close'] < last['open'] and
                not pd.isna(latest_vol) and not pd.isna(latest_vsma) and latest_vol > latest_vsma * 1.3):
            signals.append(('SHORT', 'RSI_Overbought'))

        # D. Supertrend Flip
        if len(grp) >= 20:
            st_dir = calc_supertrend(grp.tail(60))
            if len(st_dir) >= 2:
                prev_d = st_dir.iloc[-2]
                curr_d = st_dir.iloc[-1]
                if prev_d == -1 and curr_d == 1:
                    signals.append(('LONG', 'ST_Flip_Up'))
                elif prev_d == 1 and curr_d == -1:
                    signals.append(('SHORT', 'ST_Flip_Down'))

        # E. 52-week high breakout
        if (not pd.isna(latest_52wh) and last['close'] >= latest_52wh * 0.995 and
                last['close'] > prev['close'] and not pd.isna(latest_vol) and not pd.isna(latest_vsma) and
                latest_vol > latest_vsma * 1.4):
            signals.append(('LONG', '52W_High'))

        # F. Accumulation
        if len(grp) >= 15:
            last10_c = close.tail(10)
            last10_v = vol.tail(10)
            price_range_pct = (last10_c.max() - last10_c.min()) / last10_c.min() * 100
            early5_vol  = last10_v.head(5).mean()
            late5_vol   = last10_v.tail(5).mean()
            if (price_range_pct < 4.0 and not pd.isna(early5_vol) and not pd.isna(late5_vol) and
                    late5_vol > early5_vol * 1.25 and 40 < latest_rsi < 65):
                signals.append(('LONG', 'Accumulation'))

        # G. Trendline Breakout
        if len(grp) >= 65:
            older_high = float(high.iloc[-65:-20].max())
            recent_high = float(high.tail(20).max())
            today_close = float(last['close'])
            if (older_high > recent_high * 1.03 and today_close > recent_high * 0.99 and
                    last['close'] > prev['close'] and not pd.isna(latest_vol) and not pd.isna(latest_vsma) and
                    latest_vol > latest_vsma):
                signals.append(('LONG', 'TL_Breakout'))

        if not signals:
            continue
            
        # Get next day's price action to evaluate the signal
        fut = df_future[df_future['symbol'] == symbol].sort_values('timestamp')
        if fut.empty:
            continue
            
        next_day = fut.iloc[0]
        entry_price = float(last['close'])
        
        for direction, strat_name in signals:
            if direction == 'LONG':
                max_gain_pct = (next_day['high'] - entry_price) / entry_price * 100
                max_loss_pct = (next_day['low'] - entry_price) / entry_price * 100
                is_win = max_gain_pct >= 1.0  # considered a win if it went up 1% intraday
            else:
                max_gain_pct = (entry_price - next_day['low']) / entry_price * 100
                max_loss_pct = (entry_price - next_day['high']) / entry_price * 100
                is_win = max_gain_pct >= 1.0
                
            results_log.append({
                'ScanDate': str(target_date),
                'TradeDate': str(next_day['date']),
                'Symbol': clean,
                'Strategy': strat_name,
                'Direction': direction,
                'Entry (Close)': round(entry_price, 2),
                'Next Day High': round(next_day['high'], 2),
                'Next Day Low': round(next_day['low'], 2),
                'Max Intraday Gain %': round(max_gain_pct, 2),
                'Max Intraday Drawdown %': round(max_loss_pct, 2),
                'Hit 1% Target?': '✅ YES' if is_win else '❌ NO'
            })

if not results_log:
    print("\nNo signals generated during the test period.")
    sys.exit(0)

# Print Summary
res_df = pd.DataFrame(results_log)
print("\n" + "="*80)
print(f"📊 BACKTEST RESULTS (Last Week): {len(res_df)} total signals triggered")
print("="*80)

# Group by Strategy
strat_summary = res_df.groupby('Strategy').agg(
    Signals=('Symbol', 'count'),
    Win_Rate=('Hit 1% Target?', lambda x: (x == '✅ YES').mean() * 100),
    Avg_Max_Gain=('Max Intraday Gain %', 'mean')
).round(2)

print("\n📈 Strategy Performance (Target = 1% Intraday Move Next Day)")
print("-" * 60)
print(strat_summary.to_string())

print("\n🔎 Detailed Trades Log:")
print("-" * 80)
print(res_df.to_string(index=False))
