"""
Multi-strategy EOD scan → tomorrow's best trade setups.
Strategies:
  A. Volatility Squeeze (VCP/BB squeeze) — near breakout
  B. RSI Oversold Reversal — RSI < 35, price stabilizing
  C. RSI Overbought for Short — RSI > 70 with bearish candle
  D. Supertrend Flip Today — fresh 3m/daily trend change
  E. 52-week high breakout — closing near 52w high
  F. Accumulation (Volume Rising, Price Flat) — smart money entry
  G. Trendline breakout — price breaks above falling trendline
"""
import sys, os
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from sqlalchemy import text
from trade_system.infrastructure.database.connection import get_engine
import warnings
warnings.filterwarnings('ignore')

engine = get_engine()

# ── Load daily data for all FO stocks ─────────────────────────────────────────
print("📦 Loading daily OHLCV from DB...")
with engine.connect() as conn:
    df_all = pd.read_sql(text("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv_daily
        WHERE timestamp >= date('now', '-260 days')
        ORDER BY symbol, timestamp ASC
    """), conn)

if df_all.empty:
    print("❌ No daily data found in ohlcv_daily table. Trying market_data table...")
    with engine.connect() as conn:
        df_all = pd.read_sql(text("""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM market_data
            WHERE timeframe = 'D' AND timestamp >= date('now', '-260 days')
            ORDER BY symbol, timestamp ASC
        """), conn)

print(f"   {len(df_all)} rows | {df_all['symbol'].nunique()} symbols")
if df_all.empty:
    print("❌ Still no data. Exiting.")
    sys.exit(1)

df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], format='mixed')
df_all = df_all.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

results = {
    'squeeze':       [],
    'rsi_oversold':  [],
    'rsi_overbought':[],
    'st_flip_up':    [],
    'st_flip_down':  [],
    '52w_high':      [],
    'accumulation':  [],
    'trendline_bo':  [],
}

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

total = df_all['symbol'].nunique()
processed = 0

for symbol, grp in df_all.groupby('symbol'):
    grp = grp.sort_values('timestamp').copy().reset_index(drop=True)
    if len(grp) < 60:
        continue
    processed += 1
    close = grp['close']
    high  = grp['high']
    low   = grp['low']
    vol   = grp['volume'].replace(0, np.nan)
    last  = grp.iloc[-1]
    prev  = grp.iloc[-2]
    clean = symbol.replace('NSE:', '').replace('-EQ', '').replace('-INDEX', '')

    # ── Indicators ────────────────────────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bbu   = sma20 + 2 * std20
    bbl   = sma20 - 2 * std20
    bbw   = (bbu - bbl) / sma20 * 100

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)

    # ATR-14
    prev_c = close.shift(1)
    tr     = pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
    atr14  = tr.rolling(14).mean()

    # Volume SMA
    vsma20 = vol.rolling(20).mean()

    # 52-week high/low
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
    latest_52wl = w52_low.iloc[-1]

    if pd.isna(latest_rsi) or pd.isna(latest_bbw):
        continue

    # ── A. Volatility Squeeze ─────────────────────────────────────────────────
    if (latest_bbw <= bbw_5pct and
            not pd.isna(atr_sma50) and latest_atr < atr_sma50 and
            (latest_52wh - last['close']) / last['close'] < 0.04):
        res20  = float(high.tail(20).max())
        sup20  = float(low.tail(20).min())
        rng_pct = (res20 - sup20) / sup20 * 100
        results['squeeze'].append({
            'symbol': clean, 'close': last['close'], 'resistance': res20,
            'support': sup20, 'bbw': round(latest_bbw, 2),
            'range_pct': round(rng_pct, 1), 'rsi': round(latest_rsi, 1),
            'note': 'BB squeeze + ATR contracting + near 52w high'
        })

    # ── B. RSI Oversold Reversal ──────────────────────────────────────────────
    rsi_3ago = rsi.iloc[-3] if len(rsi) >= 3 else np.nan
    if (latest_rsi < 35 and not pd.isna(rsi_3ago) and
            latest_rsi > rsi.iloc[-2] and  # RSI turning up
            last['close'] > last['low'] * 1.005 and  # not closing at low
            last['close'] > prev['close']):  # green candle today
        results['rsi_oversold'].append({
            'symbol': clean, 'close': last['close'], 'rsi': round(latest_rsi, 1),
            'note': f"RSI oversold ({latest_rsi:.0f}) + green candle today"
        })

    # ── C. RSI Overbought Short ───────────────────────────────────────────────
    if (latest_rsi > 72 and
            last['close'] < prev['close'] and  # red candle today
            last['close'] < last['open'] and
            not pd.isna(latest_vol) and not pd.isna(latest_vsma) and
            latest_vol > latest_vsma * 1.3):
        results['rsi_overbought'].append({
            'symbol': clean, 'close': last['close'], 'rsi': round(latest_rsi, 1),
            'note': f"RSI overbought ({latest_rsi:.0f}) + bearish candle + volume surge"
        })

    # ── D. Supertrend Flip Today (daily) ─────────────────────────────────────
    if len(grp) >= 20:
        st_dir = calc_supertrend(grp.tail(60))
        if len(st_dir) >= 2:
            prev_d = st_dir.iloc[-2]
            curr_d = st_dir.iloc[-1]
            if prev_d == -1 and curr_d == 1:
                results['st_flip_up'].append({
                    'symbol': clean, 'close': last['close'], 'rsi': round(latest_rsi, 1),
                    'note': 'Daily supertrend flipped BULLISH today'
                })
            elif prev_d == 1 and curr_d == -1:
                results['st_flip_down'].append({
                    'symbol': clean, 'close': last['close'], 'rsi': round(latest_rsi, 1),
                    'note': 'Daily supertrend flipped BEARISH today'
                })

    # ── E. 52-week high breakout ──────────────────────────────────────────────
    if (not pd.isna(latest_52wh) and
            last['close'] >= latest_52wh * 0.995 and
            last['close'] > prev['close'] and
            not pd.isna(latest_vol) and not pd.isna(latest_vsma) and
            latest_vol > latest_vsma * 1.4):
        results['52w_high'].append({
            'symbol': clean, 'close': last['close'],
            '52w_high': round(latest_52wh, 2), 'rsi': round(latest_rsi, 1),
            'vol_surge': round(latest_vol / latest_vsma, 1) if latest_vsma else 'N/A',
            'note': '52-week high breakout with volume'
        })

    # ── F. Accumulation (Smart Money) ─────────────────────────────────────────
    # Price range contracting over 10 days, but volume rising
    if len(grp) >= 15:
        last10_c = close.tail(10)
        last10_v = vol.tail(10)
        price_range_pct = (last10_c.max() - last10_c.min()) / last10_c.min() * 100
        early5_vol  = last10_v.head(5).mean()
        late5_vol   = last10_v.tail(5).mean()
        if (price_range_pct < 4.0 and
                not pd.isna(early5_vol) and not pd.isna(late5_vol) and
                late5_vol > early5_vol * 1.25 and
                latest_rsi > 40 and latest_rsi < 65):
            results['accumulation'].append({
                'symbol': clean, 'close': last['close'],
                'range_pct': round(price_range_pct, 1),
                'vol_ratio': round(late5_vol / early5_vol, 2),
                'rsi': round(latest_rsi, 1),
                'note': f"Price flat ({price_range_pct:.1f}%) + volume rising ({late5_vol/early5_vol:.2f}x)"
            })

    # ── G. Trendline Breakout (falling trendline) ─────────────────────────────
    # Simple: Compare recent 20-day high to 60-20 day high — if price breaks above
    # the declining resistance, it's a trendline breakout
    if len(grp) >= 65:
        older_high = float(high.iloc[-65:-20].max())
        recent_high = float(high.tail(20).max())
        today_close = float(last['close'])
        # Trendline was declining (older high > recent high by > 3%)
        if (older_high > recent_high * 1.03 and
                today_close > recent_high * 0.99 and  # price at/above 20-day high
                last['close'] > prev['close'] and
                not pd.isna(latest_vol) and not pd.isna(latest_vsma) and
                latest_vol > latest_vsma):
            results['trendline_bo'].append({
                'symbol': clean, 'close': today_close,
                'resistance_broke': round(recent_high, 2),
                'prior_peak': round(older_high, 2),
                'rsi': round(latest_rsi, 1),
                'note': f"Broke falling trendline (prior peak ₹{older_high:.0f} → now ₹{today_close:.0f})"
            })

if processed == 0:
    print("❌ No symbols processed (insufficient data). Check DB population.")
    sys.exit(1)

print(f"\n✅ Scanned {processed} stocks\n{'='*65}")

# ── Print Results ─────────────────────────────────────────────────────────────
def print_section(title, emoji, items, key_fields):
    print(f"\n{emoji}  {title} ({len(items)} stocks)")
    print("-" * 60)
    if not items:
        print("   None found")
        return
    for i, s in enumerate(sorted(items, key=lambda x: x['rsi'] if 'rsi' in x else 0), 1):
        fields = "  |  ".join(f"{k}: {s[k]}" for k in key_fields if k in s)
        print(f"  {i:2}. {s['symbol']:<18} {fields}")
        print(f"       💬 {s['note']}")

print_section("VOLATILITY SQUEEZE (pre-breakout)",       "🔥", results['squeeze'],
              ['close', 'resistance', 'bbw', 'range_pct', 'rsi'])
print_section("RSI OVERSOLD REVERSAL (long opportunity)","💚", results['rsi_oversold'],
              ['close', 'rsi'])
print_section("RSI OVERBOUGHT SHORT (short setup)",      "🔴", results['rsi_overbought'],
              ['close', 'rsi'])
print_section("DAILY ST FLIP → BULLISH (fresh uptrend)", "🟢", results['st_flip_up'],
              ['close', 'rsi'])
print_section("DAILY ST FLIP → BEARISH (fresh downtrend)","🔻", results['st_flip_down'],
              ['close', 'rsi'])
print_section("52-WEEK HIGH BREAKOUT",                   "🚀", results['52w_high'],
              ['close', '52w_high', 'vol_surge', 'rsi'])
print_section("ACCUMULATION (smart money entry)",        "📦", results['accumulation'],
              ['close', 'range_pct', 'vol_ratio', 'rsi'])
print_section("TRENDLINE BREAKOUT (falling TL broken)",  "📈", results['trendline_bo'],
              ['close', 'resistance_broke', 'prior_peak', 'rsi'])

# ── Top Conviction Picks ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("🏆  TOP CONVICTION PICKS FOR TOMORROW")
print("    (stocks appearing in 2+ strategies = higher confidence)")
print("-" * 65)

from collections import Counter, defaultdict
symbol_count = Counter()
symbol_tags  = defaultdict(list)

tag_map = {
    'squeeze':       '🔥 Squeeze',
    'rsi_oversold':  '💚 RSI-Oversold',
    'rsi_overbought':'🔴 RSI-OB Short',
    'st_flip_up':    '🟢 ST-Flip-Up',
    'st_flip_down':  '🔻 ST-Flip-Down',
    '52w_high':      '🚀 52W-High',
    'accumulation':  '📦 Accumulation',
    'trendline_bo':  '📈 TL-Breakout',
}

all_symbols = {}
for strategy, items in results.items():
    for item in items:
        sym = item['symbol']
        symbol_count[sym] += 1
        symbol_tags[sym].append(tag_map[strategy])
        all_symbols[sym] = item

top_picks = [(sym, cnt) for sym, cnt in symbol_count.items() if cnt >= 2]
top_picks.sort(key=lambda x: -x[1])

if top_picks:
    for rank, (sym, cnt) in enumerate(top_picks[:15], 1):
        item = all_symbols[sym]
        tags = " + ".join(symbol_tags[sym])
        print(f"  #{rank:2}  {sym:<18}  ₹{item.get('close', '?'):<10}  [{cnt} signals]  {tags}")
else:
    print("  No stocks appeared in 2+ strategies today.")
    print("  Single-signal picks (top 5 per category above) are still valid setups.")

print(f"\n{'='*65}")
print("⚠️  DISCLAIMER: This is a quantitative scan, not financial advice.")
print("    Always check chart structure, news, and sector momentum before trading.")
print(f"{'='*65}\n")
