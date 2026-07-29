"""
Backtester for GEX (Gamma Exposure) + EMA Stack Breakout Strategy.
Inspired by: "Why I stopped trading price action & only use gamma exposure ($1k/day strategy)"

Rules:
1. GEX Environment Check:
   - POSITIVE_GEX: dealers long gamma -> volatility suppressed. Size = 100%, Target = 2.0x ATR (1:2 R:R).
   - NEGATIVE_GEX: dealers short gamma -> volatility amplified. Size = 50%, Target = 1.0x ATR (1:1 R:R).
2. Trend Stack:
   - EMA Stack: 9 EMA > 21 EMA > 50 EMA (CALL/Long) or 9 EMA < 21 EMA < 50 EMA (PUT/Short).
3. Breakout Entry:
   - Price breaks consolidation range with Volume > 1.5x of the 20-period average volume.
"""

import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime, date
from typing import List, Any, Dict
from sqlalchemy import text
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


def compute_daily_gex_series(symbol: str, engine) -> pd.DataFrame:
    """
    Query the option_chain_data table and compute daily net GEX values.
    Returns a DataFrame with columns ['trade_date', 'total_gex', 'gex_label'].
    """
    upper_sym = symbol.upper()
    if "NIFTY50" in upper_sym or "NIFTY-INDEX" in upper_sym:
        symbol_key = "NSE:NIFTY50-INDEX"
        lot_size = 50
    elif "BANK" in upper_sym:
        symbol_key = "NSE:NIFTYBANK-INDEX"
        lot_size = 25
    elif "SENSEX" in upper_sym:
        symbol_key = "BSE:SENSEX-INDEX"
        lot_size = 10
    else:
        symbol_key = symbol
        lot_size = 250
        
    query = text("""
        SELECT date(timestamp) as trade_date, strike, option_type, gamma, oi, ltp
        FROM option_chain_data
        WHERE underlying_symbol = :sym
        ORDER BY timestamp ASC
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sym": symbol_key})
        if df.empty:
            return pd.DataFrame()
            
        # GEX calculation formula: spot * 0.01 * gamma * oi * lot_size
        df['gamma'] = df['gamma'].fillna(0)
        df['oi'] = df['oi'].fillna(0)
        df['ltp'] = df['ltp'].fillna(0)
        df['strike_gex'] = df['ltp'] * 0.01 * df['gamma'] * df['oi'] * lot_size
        df.loc[df['option_type'] == 'PE', 'strike_gex'] *= -1
        
        # Aggregate per trade date
        daily = df.groupby('trade_date')['strike_gex'].sum().reset_index(name='total_gex')
        daily['gex_label'] = np.where(daily['total_gex'] > 0, "POSITIVE_GEX", "NEGATIVE_GEX")
        return daily
    except Exception as e:
        LOGGER.warning(f"Could not compute GEX from DB for {symbol_key}: {e}")
        return pd.DataFrame()


def run_gex_backtest(df: pd.DataFrame, symbol: str, engine) -> List[dict]:
    if df.empty or len(df) < 50:
        return []

    # Sort and parse times
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    # Calculate EMA stack
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    # Calculate ATR (14-period)
    df["tr"] = np.maximum(
        df["high"] - df["low"], 
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(), 
            (df["low"] - df["close"].shift(1)).abs()
        )
    )
    df["atr"] = df["tr"].rolling(14).mean()
    
    # Volume MA
    df["vol_ma"] = df["volume"].rolling(20).mean()
    
    # Fetch GEX regimes
    gex_df = compute_daily_gex_series(symbol, engine)
    if not gex_df.empty:
        gex_df['trade_date'] = pd.to_datetime(gex_df['trade_date']).dt.date
        gex_map = dict(zip(gex_df['trade_date'], gex_df['gex_label']))
    else:
        gex_map = {}

    trades = []
    position = None
    entry_price = 0.0
    entry_time = None
    target_price = 0.0
    sl_price = 0.0
    gex_regime = "POSITIVE_GEX"
    size_mult = 1.0

    for i in range(50, len(df)):
        row = df.iloc[i]
        curr_time = row["time"]
        curr_price = row["close"]
        curr_atr = row["atr"] if not pd.isna(row["atr"]) else (row["high"] - row["low"])
        trade_date = row["date"]
        
        # Get today's GEX regime (default to POSITIVE if unknown)
        today_gex = gex_map.get(trade_date, "POSITIVE_GEX")

        # EOD Square-off (15:15)
        is_eod = curr_time >= datetime.strptime("15:15:00", "%H:%M:%S").time()

        if position is not None:
            # Evaluate Exits
            if position == "LONG":
                risk = abs(entry_price - sl_price)
                if curr_price >= target_price:
                    pts = (target_price - entry_price) * size_mult
                    r_mult = pts / risk if risk > 0 else 0.0
                    trades.append({
                        "symbol": symbol, "direction": "LONG", "points_captured": pts,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price, "stop_loss": sl_price, "take_profit": target_price,
                        "exit_time": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price": target_price, "exit_reason": "TARGET_HIT",
                        "risk_points": risk, "r_multiple": r_mult, "gex": gex_regime
                    })
                    position = None
                elif curr_price <= sl_price:
                    pts = (sl_price - entry_price) * size_mult
                    r_mult = pts / risk if risk > 0 else 0.0
                    trades.append({
                        "symbol": symbol, "direction": "LONG", "points_captured": pts,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price, "stop_loss": sl_price, "take_profit": target_price,
                        "exit_time": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price": sl_price, "exit_reason": "SL_HIT",
                        "risk_points": risk, "r_multiple": r_mult, "gex": gex_regime
                    })
                    position = None
                elif is_eod:
                    pts = (curr_price - entry_price) * size_mult
                    r_mult = pts / risk if risk > 0 else 0.0
                    trades.append({
                        "symbol": symbol, "direction": "LONG", "points_captured": pts,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price, "stop_loss": sl_price, "take_profit": target_price,
                        "exit_time": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price": curr_price, "exit_reason": "EOD_EXIT",
                        "risk_points": risk, "r_multiple": r_mult, "gex": gex_regime
                    })
                    position = None
            elif position == "SHORT":
                risk = abs(sl_price - entry_price)
                if curr_price <= target_price:
                    pts = (entry_price - target_price) * size_mult
                    r_mult = pts / risk if risk > 0 else 0.0
                    trades.append({
                        "symbol": symbol, "direction": "SHORT", "points_captured": pts,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price, "stop_loss": sl_price, "take_profit": target_price,
                        "exit_time": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price": target_price, "exit_reason": "TARGET_HIT",
                        "risk_points": risk, "r_multiple": r_mult, "gex": gex_regime
                    })
                    position = None
                elif curr_price >= sl_price:
                    pts = (entry_price - sl_price) * size_mult
                    r_mult = pts / risk if risk > 0 else 0.0
                    trades.append({
                        "symbol": symbol, "direction": "SHORT", "points_captured": pts,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price, "stop_loss": sl_price, "take_profit": target_price,
                        "exit_time": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price": sl_price, "exit_reason": "SL_HIT",
                        "risk_points": risk, "r_multiple": r_mult, "gex": gex_regime
                    })
                    position = None
                elif is_eod:
                    pts = (entry_price - curr_price) * size_mult
                    r_mult = pts / risk if risk > 0 else 0.0
                    trades.append({
                        "symbol": symbol, "direction": "SHORT", "points_captured": pts,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price, "stop_loss": sl_price, "take_profit": target_price,
                        "exit_time": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price": curr_price, "exit_reason": "EOD_EXIT",
                        "risk_points": risk, "r_multiple": r_mult, "gex": gex_regime
                    })
                    position = None
        else:
            # Restrict entries past 15:00
            if curr_time >= datetime.strptime("15:00:00", "%H:%M:%S").time():
                continue

            # EMA stack signals
            ema_long = row["ema_9"] > row["ema_21"] > row["ema_50"] and curr_price > row["ema_9"]
            ema_short = row["ema_9"] < row["ema_21"] < row["ema_50"] and curr_price < row["ema_9"]
            
            # Volume confirmation (breakout bar volume > 1.5x of 20-period avg)
            vol_confirm = row["volume"] > row["vol_ma"] * 1.5 if row["vol_ma"] > 0 else False
            
            # Consolidation check: high-low compression over previous 4 candles
            prev_ranges = df.iloc[i-4:i]
            compression = (prev_ranges["high"].max() - prev_ranges["low"].min()) < curr_atr * 2.0
            
            if vol_confirm and compression:
                # Determine R:R and Sizing from GEX Regime
                gex_regime = today_gex
                if gex_regime == "POSITIVE_GEX":
                    size_mult = 1.0
                    target_mult = 2.0
                    sl_mult = 1.0
                else:  # NEGATIVE_GEX
                    size_mult = 0.5  # Reduce position size by 50%
                    target_mult = 1.0  # Tighter target due to wild swings
                    sl_mult = 1.0
                
                if ema_long:
                    position = "LONG"
                    entry_price = curr_price
                    entry_time = row["timestamp"]
                    target_price = entry_price + target_mult * curr_atr
                    sl_price = entry_price - sl_mult * curr_atr
                elif ema_short:
                    position = "SHORT"
                    entry_price = curr_price
                    entry_time = row["timestamp"]
                    target_price = entry_price - target_mult * curr_atr
                    sl_price = entry_price + sl_mult * curr_atr

    return trades


def load_db_candles(symbol: str, start_date: str, end_date: str, engine) -> pd.DataFrame:
    """Fetch candles from SQLite database."""
    query = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_15m
        WHERE symbol = :sym AND timestamp >= :start AND timestamp <= :end
        ORDER BY timestamp ASC
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "sym": symbol,
                "start": f"{start_date} 00:00:00",
                "end": f"{end_date} 23:59:59"
            })
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        LOGGER.error(f"Failed to load candles from database for {symbol}: {e}")
        return pd.DataFrame()


def main():
    engine = get_engine()
    
    # We backtest over the actual option chain dates stored in DB (2026-03-25 to 2026-06-25)
    start_date = "2026-03-25"
    end_date = "2026-06-25"
    
    symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
    
    print(f"Starting GEX + EMA Stack backtest on Database (15m charts)...")
    print(f"Window: {start_date} to {end_date}")
    print("="*90)
    print(f"{'Symbol':<20}{'Total Trades':<15}{'Win Rate':<15}{'Net PnL Pts':<15}{'Profit Factor':<15}")
    print("="*90)
    
    all_trades = []
    for sym in symbols:
        df = load_db_candles(sym, start_date, end_date, engine)
        trades = run_gex_backtest(df, sym, engine)
        all_trades.extend(trades)
        
        clean_sym = sym.replace("NSE:", "").replace("-INDEX", "")
        if trades:
            tdf = pd.DataFrame(trades)
            total = len(tdf)
            win_rate = (tdf["points_captured"] > 0).sum() / total * 100
            pnl = tdf["points_captured"].sum()
            gross_p = tdf[tdf["points_captured"] > 0]["points_captured"].sum()
            gross_l = abs(tdf[tdf["points_captured"] <= 0]["points_captured"].sum())
            pf = gross_p / gross_l if gross_l > 0 else float("inf")
            
            win_rate_str = f"{win_rate:.2f}%"
            pnl_str = f"{pnl:+.2f}"
            pf_str = f"{pf:.2f}"
            print(f"{clean_sym:<20}{total:<15}{win_rate_str:<15}{pnl_str:<15}{pf_str:<15}")
        else:
            print(f"{clean_sym:<20}{0:<15}{'—':<15}{'0.00':<15}{'—':<15}")
            
    print("="*90)
    if all_trades:
        df_all = pd.DataFrame(all_trades)
        df_all["entry_time"] = pd.to_datetime(df_all["entry_time"])
        df_all["year"] = df_all["entry_time"].dt.year
        
        # Save Trade Files & Summary
        report_dir = ROOT / "reports" / "gamma_ema_stack"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        summary_rows = []
        for y, group in df_all.groupby("year"):
            total_t = len(group)
            win_t = (group["points_captured"] > 0).sum()
            loss_t = total_t - win_t
            wr = win_t / total_t * 100
            net_pts = group["points_captured"].sum()
            avg_pts = group["points_captured"].mean()
            
            wins = group[group["points_captured"] > 0]["points_captured"]
            losses = group[group["points_captured"] <= 0]["points_captured"]
            
            avg_win = wins.mean() if not wins.empty else 0.0
            avg_loss = losses.mean() if not losses.empty else 0.0
            pf = wins.sum() / abs(losses.sum()) if not losses.empty and losses.sum() != 0 else float("inf")
            
            target_hits = (group["exit_reason"] == "TARGET_HIT").sum()
            sl_hits = (group["exit_reason"] == "SL_HIT").sum()
            eod_exits = (group["exit_reason"] == "EOD_EXIT").sum()
            
            # Drawdown
            group = group.sort_values("entry_time").reset_index(drop=True)
            group["cum_pnl"] = group["points_captured"].cumsum()
            cum_max = group["cum_pnl"].cummax()
            dd = group["cum_pnl"] - cum_max
            max_dd = dd.min()
            
            max_win = wins.max() if not wins.empty else 0.0
            max_loss = losses.min() if not losses.empty else 0.0
            
            # Save trades for this year
            trades_path = report_dir / f"gamma_ema_stack_{y}_trades.csv"
            group.to_csv(trades_path, index=False)
            
            summary_rows.append({
                "year": y, "trades": total_t, "wins": win_t, "losses": loss_t, "win_rate": wr,
                "net_points": net_pts, "avg_points": avg_pts, "avg_win": avg_win, "avg_loss": avg_loss,
                "profit_factor": pf, "expectancy": avg_pts, "total_r": group["r_multiple"].sum(),
                "avg_r": group["r_multiple"].mean(), "target_hits": target_hits, "sl_hits": sl_hits,
                "eod_exits": eod_exits, "max_drawdown_points": max_dd, "max_win": max_win, "max_loss": max_loss
            })
            
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(report_dir / "gamma_ema_stack_summary.csv", index=False)
        print(f"\n📂 Reports generated and saved in {report_dir}")
        
        # Combined Console Output
        total = len(df_all)
        win_rate = (df_all["points_captured"] > 0).sum() / total * 100
        pnl = df_all["points_captured"].sum()
        
        gross_p = df_all[df_all["points_captured"] > 0]["points_captured"].sum()
        gross_l = abs(df_all[df_all["points_captured"] <= 0]["points_captured"].sum())
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        
        win_rate_str = f"{win_rate:.2f}%"
        pnl_str = f"{pnl:+.2f}"
        pf_str = f"{pf:.2f}"
        
        print("\n📊 COMPOSITE SUMMARY:")
        print(f"• Total Trades Run : {total}")
        print(f"• Combined Win Rate: {win_rate_str}")
        print(f"• Cumulative PnL   : {pnl_str} pts")
        print(f"• Profit Factor     : {pf_str}")
        
        # Breakdown by GEX label
        print("\n🔬 PERFORMANCE BY GEX REGIME:")
        for regime in ["POSITIVE_GEX", "NEGATIVE_GEX"]:
            rdf = df_all[df_all["gex"] == regime]
            if not rdf.empty:
                r_total = len(rdf)
                r_win = (rdf["points_captured"] > 0).sum() / r_total * 100
                r_pnl = rdf["points_captured"].sum()
                r_gross_p = rdf[rdf["points_captured"] > 0]["points_captured"].sum()
                r_gross_l = abs(rdf[rdf["points_captured"] <= 0]["points_captured"].sum())
                r_pf = r_gross_p / r_gross_l if r_gross_l > 0 else float("inf")
                print(f"  [{regime}]: Trades={r_total}, Win Rate={r_win:.2f}%, Net PnL={r_pnl:+.2f} pts, PF={r_pf:.2f}")
            else:
                print(f"  [{regime}]: No trades recorded.")
    else:
        print("\nNo setups triggered during the backtest window.")


if __name__ == "__main__":
    main()
