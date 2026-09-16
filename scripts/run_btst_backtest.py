import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def run_btst_backtest():
    conn = sqlite3.connect("data/trade_system.db")
    
    # Check available dates in July 2026
    query_dates = """
        SELECT DISTINCT substr(timestamp, 1, 10) as dt 
        FROM ohlcv_5m 
        WHERE timestamp >= '2026-07-01' AND timestamp <= '2026-08-04'
        ORDER BY dt ASC;
    """
    dates_df = pd.read_sql_query(query_dates, conn)
    trading_days = dates_df['dt'].tolist()
    print(f"Total trading days available: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")

    # Liquid F&O stock sample
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
    print(f"Total F&O stocks with full data: {len(universe)}")

    btst_trades = []
    stbt_trades = []

    # Iterate over consecutive pairs of trading days (Day T and Day T+1)
    for i in range(len(trading_days) - 1):
        day_t = trading_days[i]
        day_next = trading_days[i+1]

        # Fetch all 5m candles for Day T and Day T+1 for the universe
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

            # Day T data up to 15:25
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
            day_low = day_t_up_to_close['low'].min()
            day_open = day_t_up_to_close.iloc[0]['open']
            entry_candle = day_t_up_to_close.iloc[-1]
            entry_price = entry_candle['close']

            # Session VWAP
            vwap = (day_t_up_to_close['close'] * day_t_up_to_close['volume']).sum() / total_vol

            day_chg_pct = (entry_price - day_open) / day_open * 100.0
            dist_high_pct = (day_high - entry_price) / day_high * 100.0
            dist_low_pct = (entry_price - day_low) / entry_price * 100.0

            # Next Day metrics
            next_open = day_next_candles.iloc[0]['open']
            
            # 09:20 AM exit
            c_0920 = day_next_candles[day_next_candles['time_str'] <= '09:25']
            exit_0920 = c_0920.iloc[-1]['close'] if not c_0920.empty else next_open

            # First 30m extreme (09:15 - 09:45)
            first_30m_next = day_next_candles[day_next_candles['time_str'] <= '09:45']
            next_high = first_30m_next['high'].max() if not first_30m_next.empty else next_open
            next_low = first_30m_next['low'].min() if not first_30m_next.empty else next_open

            # BTST CRITERIA:
            # 1. dist_high_pct <= 0.6% (closing near high)
            # 2. day_chg_pct >= +1.0%
            # 3. entry_price > vwap
            # 4. vol_ratio >= 0.18 OR baseline_burst >= 2.0x
            if dist_high_pct <= 0.6 and day_chg_pct >= 1.0 and entry_price > vwap and (vol_ratio >= 0.18 or baseline_burst >= 2.0):
                gap_pct = (next_open - entry_price) / entry_price * 100.0
                ret_0920 = (exit_0920 - entry_price) / entry_price * 100.0
                mfe_pct = (next_high - entry_price) / entry_price * 100.0
                mae_pct = (next_low - entry_price) / entry_price * 100.0

                # Target (+1.0%) & SL (-0.5%) rule
                if mfe_pct >= 1.0 and mae_pct > -0.5:
                    outcome = "TARGET_HIT"
                    trade_ret = 1.0
                elif mae_pct <= -0.5:
                    outcome = "SL_HIT"
                    trade_ret = -0.5
                else:
                    outcome = "TIMED_EXIT_0920"
                    trade_ret = ret_0920

                btst_trades.append({
                    "date_in": day_t,
                    "date_out": day_next,
                    "symbol": sym,
                    "entry_price": entry_price,
                    "day_chg": day_chg_pct,
                    "dist_high": dist_high_pct,
                    "vol_30m_ratio": vol_ratio,
                    "burst": baseline_burst,
                    "gap_pct": gap_pct,
                    "ret_0920": ret_0920,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "outcome": outcome,
                    "trade_ret": trade_ret
                })

            # STBT CRITERIA:
            # 1. dist_low_pct <= 0.6%
            # 2. day_chg_pct <= -1.0%
            # 3. entry_price < vwap
            # 4. vol_ratio >= 0.18 OR baseline_burst >= 2.0x
            if dist_low_pct <= 0.6 and day_chg_pct <= -1.0 and entry_price < vwap and (vol_ratio >= 0.18 or baseline_burst >= 2.0):
                gap_pct = (entry_price - next_open) / entry_price * 100.0
                ret_0920 = (entry_price - exit_0920) / entry_price * 100.0
                mfe_pct = (entry_price - next_low) / entry_price * 100.0
                mae_pct = (entry_price - next_high) / entry_price * 100.0

                if mfe_pct >= 1.0 and mae_pct > -0.5:
                    outcome = "TARGET_HIT"
                    trade_ret = 1.0
                elif mae_pct <= -0.5:
                    outcome = "SL_HIT"
                    trade_ret = -0.5
                else:
                    outcome = "TIMED_EXIT_0920"
                    trade_ret = ret_0920

                stbt_trades.append({
                    "date_in": day_t,
                    "date_out": day_next,
                    "symbol": sym,
                    "entry_price": entry_price,
                    "day_chg": day_chg_pct,
                    "dist_low": dist_low_pct,
                    "vol_30m_ratio": vol_ratio,
                    "burst": baseline_burst,
                    "gap_pct": gap_pct,
                    "ret_0920": ret_0920,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "outcome": outcome,
                    "trade_ret": trade_ret
                })

    conn.close()

    df_btst = pd.DataFrame(btst_trades)
    df_stbt = pd.DataFrame(stbt_trades)

    print("\n" + "="*50)
    print("🎯 BTST HISTORICAL BACKTEST RESULTS (July - Aug 2026)")
    print("="*50)
    if not df_btst.empty:
        total = len(df_btst)
        wins = len(df_btst[df_btst['trade_ret'] > 0])
        win_rate = wins / total * 100
        avg_ret = df_btst['trade_ret'].mean()
        avg_gap = df_btst['gap_pct'].mean()
        avg_mfe = df_btst['mfe_pct'].mean()
        gross_profit = df_btst[df_btst['trade_ret'] > 0]['trade_ret'].sum()
        gross_loss = abs(df_btst[df_btst['trade_ret'] < 0]['trade_ret'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        print(f"Total BTST Setups Triggered: {total}")
        print(f"Win Rate: {win_rate:.2f}% ({wins}/{total})")
        print(f"Average Return per Trade: {avg_ret:+.2f}%")
        print(f"Average Overnight Gap: {avg_gap:+.2f}%")
        print(f"Average Max Favorable Excursion (MFE): {avg_mfe:+.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("\nOutcome Breakdown:")
        print(df_btst['outcome'].value_counts())
        print("\nSample Trades:")
        print(df_btst[['date_in', 'symbol', 'day_chg', 'vol_30m_ratio', 'gap_pct', 'trade_ret', 'outcome']].head(10))
    else:
        print("No BTST trades found.")

    print("\n" + "="*50)
    print("🎯 STBT HISTORICAL BACKTEST RESULTS (July - Aug 2026)")
    print("="*50)
    if not df_stbt.empty:
        total_s = len(df_stbt)
        wins_s = len(df_stbt[df_stbt['trade_ret'] > 0])
        win_rate_s = wins_s / total_s * 100
        avg_ret_s = df_stbt['trade_ret'].mean()
        avg_gap_s = df_stbt['gap_pct'].mean()
        gross_p_s = df_stbt[df_stbt['trade_ret'] > 0]['trade_ret'].sum()
        gross_l_s = abs(df_stbt[df_stbt['trade_ret'] < 0]['trade_ret'].sum())
        pf_s = (gross_p_s / gross_l_s) if gross_l_s > 0 else 999.0

        print(f"Total STBT Setups Triggered: {total_s}")
        print(f"Win Rate: {win_rate_s:.2f}% ({wins_s}/{total_s})")
        print(f"Average Return per Trade: {avg_ret_s:+.2f}%")
        print(f"Average Overnight Gap: {avg_gap_s:+.2f}%")
        print(f"Profit Factor: {pf_s:.2f}")

if __name__ == "__main__":
    run_btst_backtest()
