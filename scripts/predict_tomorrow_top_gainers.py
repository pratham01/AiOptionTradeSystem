#!/usr/bin/env python3
"""
T-1 Top Gainer Predictive Screener.

Identifies tomorrow's potential +5% to +15% top gainers by detecting institutional
pre-breakout footprints on Day T-1:
1. Range Coiling / Volatility Contraction (NR7 / Inside Bar / Daily Range < 0.8x ATR)
2. Volume Dry-Up (Volume < 0.80x 20-DMA showing seller exhaustion)
3. Bollinger Bandwidth Compression (Squeeze)
4. Primary Trend Baseline (Price above rising 20 EMA and 50 EMA)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

import sqlite3
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def run_t1_screener(db_path: str = "data/trade_system.db", min_score: float = 65.0, top_n: int = 15):
    con = sqlite3.connect(db_path)
    
    # Get all distinct symbols from daily table
    symbols_query = "SELECT DISTINCT symbol FROM ohlcv_daily WHERE symbol LIKE 'NSE:%'"
    symbols = [row[0] for row in con.cursor().execute(symbols_query).fetchall()]
    
    candidates = []
    
    for sym in symbols:
        q = f"""
        SELECT timestamp, open, high, low, close, volume 
        FROM ohlcv_daily 
        WHERE symbol = '{sym}'
        ORDER BY timestamp DESC
        LIMIT 60
        """
        df = pd.read_sql_query(q, con)
        if len(df) < 30:
            continue
            
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Latest bar (Day T-1 relative to tomorrow)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        close = float(latest['close'])
        open_p = float(latest['open'])
        high = float(latest['high'])
        low = float(latest['low'])
        vol = float(latest['volume'])
        
        # Volume 20 SMA
        vol_sma = df['volume'].iloc[-21:-1].mean()
        if vol_sma <= 0:
            continue
        vol_ratio = vol / vol_sma
        
        # ATR 14
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            )
        )
        atr_14 = tr.iloc[-15:-1].mean()
        if atr_14 <= 0:
            continue
            
        cur_range = high - low
        range_atr_ratio = cur_range / atr_14
        
        # Moving averages
        ema_20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        # Bollinger Bands & Squeeze
        bb_mid = df['close'].iloc[-21:-1].mean()
        bb_std = df['close'].iloc[-21:-1].std()
        bb_width = (4.0 * bb_std) / bb_mid * 100.0 if bb_mid > 0 else 50.0
        
        # Rolling BB Width 50-bar percentile
        rolling_bbw = []
        for j in range(max(0, len(df) - 30), len(df) - 1):
            s_mid = df['close'].iloc[j-20:j].mean()
            s_std = df['close'].iloc[j-20:j].std()
            if s_mid > 0:
                rolling_bbw.append((4.0 * s_std) / s_mid * 100.0)
        
        bb_pctile = (
            sum(w < bb_width for w in rolling_bbw) / len(rolling_bbw) * 100.0
            if rolling_bbw else 50.0
        )
        
        # Pattern detection
        recent_7_ranges = (df['high'] - df['low']).iloc[-8:-1]
        is_nr7 = cur_range <= recent_7_ranges.min() if len(recent_7_ranges) >= 6 else False
        is_inside_bar = (high < prev['high']) and (low > prev['low'])
        day_gain = ((close - float(prev['close'])) / float(prev['close'])) * 100.0
        
        # ── Scoring Algorithm (0 - 100) ──
        score = 0.0
        reasons = []
        
        # 1. Trend Baseline (Max 25 pts)
        if close > ema_20:
            score += 15.0
            reasons.append("Above 20 EMA")
        if close > ema_50:
            score += 10.0
            reasons.append("Above 50 EMA")
            
        # 2. Volatility Contraction / Squeeze (Max 35 pts)
        if is_nr7:
            score += 20.0
            reasons.append("NR7 Coiling")
        elif is_inside_bar:
            score += 15.0
            reasons.append("Inside Bar")
        elif range_atr_ratio < 0.75:
            score += 12.0
            reasons.append(f"Tight Range ({range_atr_ratio:.2f} ATR)")
            
        if bb_pctile <= 25.0:
            score += 15.0
            reasons.append(f"BB Squeeze ({bb_pctile:.0f}th pctile)")
        elif bb_pctile <= 40.0:
            score += 8.0
            
        # 3. Supply Exhaustion / Volume Dry-Up (Max 25 pts)
        if vol_ratio < 0.55:
            score += 25.0
            reasons.append(f"Extreme Vol Dry-up ({vol_ratio:.2f}x)")
        elif vol_ratio < 0.80:
            score += 18.0
            reasons.append(f"Vol Dry-up ({vol_ratio:.2f}x)")
        elif vol_ratio > 2.0:
            score += 10.0
            reasons.append(f"Pre-ignition Vol ({vol_ratio:.2f}x)")
            
        # 4. Stealth Consolidation / Price Near Flat (Max 15 pts)
        if abs(day_gain) <= 1.0:
            score += 15.0
            reasons.append(f"Stealth Flat ({day_gain:+.2f}%)")
        elif abs(day_gain) <= 2.0:
            score += 8.0
            
        if score >= min_score:
            candidates.append({
                "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                "full_symbol": sym,
                "close": close,
                "day_gain_pct": day_gain,
                "vol_ratio": vol_ratio,
                "range_atr": range_atr_ratio,
                "is_nr7": is_nr7,
                "is_inside": is_inside_bar,
                "score": score,
                "reasons": ", ".join(reasons)
            })
            
    con.close()
    
    cand_df = pd.DataFrame(candidates)
    if cand_df.empty:
        print("No candidates met the minimum score threshold.")
        return
        
    cand_df = cand_df.sort_values("score", ascending=False).reset_index(drop=True)
    top_picks = cand_df.head(top_n)
    
    print("\n" + "=" * 90)
    print(f"🎯 TOMORROW'S PREDICTED TOP GAINERS (T-1 Leading Institutional Footprint Screener)")
    print(f"   Evaluated: {len(symbols)} NSE Stocks | Candidates Found: {len(cand_df)} | Showing Top {len(top_picks)}")
    print("=" * 90)
    
    for idx, row in top_picks.iterrows():
        star = "🌟" if row['score'] >= 80 else "⚡"
        pattern = "NR7" if row['is_nr7'] else ("Inside Bar" if row['is_inside'] else "Coiling")
        print(f"\n{idx+1}. {star} {row['symbol']:<15} | Score: {row['score']:.0f}/100 | LTP: ₹{row['close']:<9.2f} | T-1 Move: {row['day_gain_pct']:+.2f}%")
        print(f"   • Footprint: Vol: {row['vol_ratio']:.2f}x avg | Range: {row['range_atr']:.2f}x ATR | Pattern: {pattern}")
        print(f"   • Catalysts: {row['reasons']}")
        
    print("\n" + "=" * 90)
    print("💡 Execution Playbook for Tomorrow's Session:")
    print("   1. Watch pre-market (09:00 - 09:08 IST) for gap-ups on above symbols.")
    print("   2. Enter on 15m Opening Range Breakout (ORB) above T-1 High with volume > 1.5x avg.")
    print("   3. Invalidation Stop Loss: Below Day T-1 Low (or opposite side of ORB).")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-1 Top Gainer Predictive Screener")
    parser.add_argument("--min-score", type=float, default=65.0, help="Minimum probability score (0-100)")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top candidates to display")
    args = parser.parse_args()
    
    run_t1_screener(min_score=args.min_score, top_n=args.top_n)
