import sqlite3
import pandas as pd
import numpy as np

def run_multi_tier_btst_backtest():
    conn = sqlite3.connect("data/trade_system.db")
    
    query_dates = """
        SELECT DISTINCT substr(timestamp, 1, 10) as dt 
        FROM ohlcv_5m 
        WHERE timestamp >= '2026-07-01' AND timestamp <= '2026-08-04'
        ORDER BY dt ASC;
    """
    dates_df = pd.read_sql_query(query_dates, conn)
    trading_days = dates_df['dt'].tolist()

    symbols_query = """
        SELECT symbol, count(*) as cnt 
        FROM ohlcv_5m 
        WHERE timestamp >= '2026-07-01' 
        GROUP BY symbol 
        HAVING cnt > 1500 
        ORDER BY cnt DESC;
    """
    syms_df = pd.read_sql_query(symbols_query, conn)
    universe = [s for s in syms_df['symbol'].tolist() if "-INDEX" not in s]

    all_candidates = []

    for i in range(len(trading_days) - 1):
        day_t = trading_days[i]
        day_next = trading_days[i+1]

        query = f"""
            SELECT symbol, timestamp, open, high, low, close, volume 
            FROM ohlcv_5m 
            WHERE (timestamp LIKE '{day_t}%' OR timestamp LIKE '{day_next}%')
              AND symbol IN ({','.join([repr(s) for s in universe])})
            ORDER BY symbol, timestamp ASC;
        """
        df_days = pd.read_sql_query(query, conn)
        df_days['timestamp'] = pd.to_datetime(df_days['timestamp'])
        df_days['date_str'] = df_days['timestamp'].dt.strftime('%Y-%m-%d')
        df_days['time_str'] = df_days['timestamp'].dt.strftime('%H:%M')

        for sym, group in df_days.groupby('symbol'):
            day_t_candles = group[group['date_str'] == day_t].copy()
            day_next_candles = group[group['date_str'] == day_next].copy()

            if len(day_t_candles) < 60 or len(day_next_candles) < 10:
                continue

            day_t_up_to_close = day_t_candles[day_t_candles['time_str'] <= '15:25']
            if day_t_up_to_close.empty:
                continue

            total_vol = day_t_up_to_close['volume'].sum()
            if total_vol <= 0:
                continue

            last_30m = day_t_up_to_close[day_t_up_to_close['time_str'] >= '15:00']
            last_30m_vol = last_30m['volume'].sum()
            vol_ratio = last_30m_vol / total_vol
            baseline_burst = (vol_ratio / (len(last_30m) / len(day_t_up_to_close))) if len(day_t_up_to_close) > 0 else 1.0

            day_high = day_t_up_to_close['high'].max()
            day_open = day_t_up_to_close.iloc[0]['open']
            entry_price = day_t_up_to_close.iloc[-1]['close']
            vwap = (day_t_up_to_close['close'] * day_t_up_to_close['volume']).sum() / total_vol

            day_chg_pct = (entry_price - day_open) / day_open * 100.0
            dist_high_pct = (day_high - entry_price) / day_high * 100.0

            next_open = day_next_candles.iloc[0]['open']
            c_0920 = day_next_candles[day_next_candles['time_str'] <= '09:25']
            exit_0920 = c_0920.iloc[-1]['close'] if not c_0920.empty else next_open

            first_30m_next = day_next_candles[day_next_candles['time_str'] <= '09:45']
            next_high = first_30m_next['high'].max() if not first_30m_next.empty else next_open
            next_low = first_30m_next['low'].min() if not first_30m_next.empty else next_open

            gap_pct = (next_open - entry_price) / entry_price * 100.0
            ret_0920 = (exit_0920 - entry_price) / entry_price * 100.0
            mfe_pct = (next_high - entry_price) / entry_price * 100.0
            mae_pct = (next_low - entry_price) / entry_price * 100.0

            all_candidates.append({
                "date_in": day_t,
                "date_out": day_next,
                "symbol": sym,
                "entry_price": entry_price,
                "above_vwap": entry_price > vwap,
                "day_chg": day_chg_pct,
                "dist_high": dist_high_pct,
                "vol_30m_ratio": vol_ratio,
                "burst": baseline_burst,
                "gap_pct": gap_pct,
                "ret_0920": ret_0920,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
            })

    conn.close()
    df = pd.DataFrame(all_candidates)

    print("="*65)
    print("🔬 COMPARATIVE BTST PERFORMANCE BY CONVICTION FILTER")
    print("="*65)

    filters = [
        ("Tier 1: Loose Filter (dist_high <= 0.6%, vol_ratio >= 15%)", 
         (df['above_vwap']) & (df['dist_high'] <= 0.6) & (df['day_chg'] >= 0.5) & (df['vol_30m_ratio'] >= 0.15)),
        
        ("Tier 2: Standard Filter (dist_high <= 0.4%, vol_ratio >= 20%)", 
         (df['above_vwap']) & (df['dist_high'] <= 0.4) & (df['day_chg'] >= 1.5) & (df['vol_30m_ratio'] >= 0.20)),
        
        ("Tier 3: High Institutional Conviction (dist_high <= 0.25%, vol >= 25% or 3x burst, chg >= 2.0%)", 
         (df['above_vwap']) & (df['dist_high'] <= 0.25) & (df['day_chg'] >= 2.0) & ((df['vol_30m_ratio'] >= 0.25) | (df['burst'] >= 3.0))),
    ]

    for label, mask in filters:
        sub = df[mask].copy()
        if sub.empty:
            print(f"\n{label}: No trades")
            continue
        
        # Scenario A: Fixed 09:20 Exit
        win_0920 = len(sub[sub['ret_0920'] > 0])
        win_rate_0920 = win_0920 / len(sub) * 100
        avg_ret_0920 = sub['ret_0920'].mean()
        avg_gap = sub['gap_pct'].mean()
        avg_mfe = sub['mfe_pct'].mean()
        avg_mae = sub['mae_pct'].mean()

        # Scenario B: Target +1.0% / SL -0.5%
        targets = len(sub[(sub['mfe_pct'] >= 1.0) & (sub['mae_pct'] > -0.5)])
        sls = len(sub[sub['mae_pct'] <= -0.5])
        timed = len(sub) - targets - sls
        pnl = targets * 1.0 - sls * 0.5 + sub[(sub['mfe_pct'] < 1.0) & (sub['mae_pct'] > -0.5)]['ret_0920'].sum()
        avg_pnl = pnl / len(sub)
        gp = targets * 1.0 + sub[(sub['mfe_pct'] < 1.0) & (sub['mae_pct'] > -0.5) & (sub['ret_0920'] > 0)]['ret_0920'].sum()
        gl = abs(-sls * 0.5 + sub[(sub['mfe_pct'] < 1.0) & (sub['mae_pct'] > -0.5) & (sub['ret_0920'] < 0)]['ret_0920'].sum())
        pf = (gp / gl) if gl > 0 else 999.0

        print(f"\n📌 {label}")
        print(f"  • Total Trades: {len(sub)}")
        print(f"  • Average Overnight Gap: {avg_gap:+.2f}%")
        print(f"  • Peak Upward Move (MFE): {avg_mfe:+.2f}% | Max Drawdown (MAE): {avg_mae:+.2f}%")
        print(f"  • Rule A (09:20 AM Timed Exit): Win Rate {win_rate_0920:.1f}%, Avg Return {avg_ret_0920:+.2f}%")
        print(f"  • Rule B (Target +1.0% / SL -0.5%): Profit Factor {pf:.2f}, Avg PnL {avg_pnl:+.2f}% | Targets: {targets}, SLs: {sls}, Timed: {timed}")

if __name__ == "__main__":
    run_multi_tier_btst_backtest()
