import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from sqlalchemy import text
from trade_system.infrastructure.database.connection import get_engine

logger = logging.getLogger(__name__)

class CandlestickPatternAgent:
    """
    Agent responsible for detecting daily candlestick patterns,
    calculating historical success probabilities, scanning for active setups,
    and capturing live charts from TradingView using Playwright.
    """
    
    def __init__(self):
        self.engine = get_engine()
        
    def detect_patterns(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Detects standard candlestick patterns on daily data.
        Returns a copy of the DataFrame with pattern columns.
        """
        df = df_daily.sort_values("timestamp").copy().reset_index(drop=True)
        if len(df) < 5:
            return df
            
        # Initialize pattern columns (0 = No pattern, 1 = Bullish/True, -1 = Bearish)
        df["bullish_engulfing"] = 0
        df["bearish_engulfing"] = 0
        df["hammer"] = 0
        df["shooting_star"] = 0
        df["doji"] = 0
        df["morning_star"] = 0
        df["evening_star"] = 0
        
        close = df["close"]
        open_p = df["open"]
        high = df["high"]
        low = df["low"]
        
        for i in range(2, len(df)):
            curr_c, curr_o, curr_h, curr_l = close.iloc[i], open_p.iloc[i], high.iloc[i], low.iloc[i]
            prev_c, prev_o, prev_h, prev_l = close.iloc[i-1], open_p.iloc[i-1], high.iloc[i-1], low.iloc[i-1]
            prev2_c, prev2_o, prev2_h, prev2_l = close.iloc[i-2], open_p.iloc[i-2], high.iloc[i-2], low.iloc[i-2]
            
            curr_body = abs(curr_c - curr_o)
            prev_body = abs(prev_c - prev_o)
            prev2_body = abs(prev2_c - prev2_o)
            
            curr_range = curr_h - curr_l
            
            # 1. Doji
            if curr_range > 0 and (curr_body / curr_range) <= 0.05:
                df.loc[i, "doji"] = 1
                
            # 2. Bullish Engulfing
            if prev_c < prev_o and curr_c > curr_o:
                if curr_c >= prev_o and curr_o <= prev_c and curr_body > prev_body:
                    df.loc[i, "bullish_engulfing"] = 1
                    
            # 3. Bearish Engulfing
            if prev_c > prev_o and curr_c < curr_o:
                if curr_c <= prev_o and curr_o >= prev_c and curr_body > prev_body:
                    df.loc[i, "bearish_engulfing"] = 1
                    
            # 4. Hammer (Bullish pinbar)
            if curr_range > 0:
                upper_shadow = curr_h - max(curr_c, curr_o)
                lower_shadow = min(curr_c, curr_o) - curr_l
                body_ratio = curr_body / curr_range
                
                # Lower shadow is at least 2x the body, upper shadow is small, body is small
                if lower_shadow >= 2 * curr_body and upper_shadow <= 0.15 * curr_range and body_ratio <= 0.35:
                    # Occurs near recent lows (lower 30% of past 10 days)
                    recent_low = low.iloc[max(0, i-9):i+1].min()
                    recent_high = high.iloc[max(0, i-9):i+1].max()
                    recent_range = recent_high - recent_low
                    if recent_range > 0 and (curr_c - recent_low) / recent_range <= 0.35:
                        df.loc[i, "hammer"] = 1
                        
            # 5. Shooting Star (Bearish pinbar)
            if curr_range > 0:
                upper_shadow = curr_h - max(curr_c, curr_o)
                lower_shadow = min(curr_c, curr_o) - curr_l
                body_ratio = curr_body / curr_range
                
                # Upper shadow is at least 2x the body, lower shadow is small, body is small
                if upper_shadow >= 2 * curr_body and lower_shadow <= 0.15 * curr_range and body_ratio <= 0.35:
                    # Occurs near recent highs (upper 30% of past 10 days)
                    recent_low = low.iloc[max(0, i-9):i+1].min()
                    recent_high = high.iloc[max(0, i-9):i+1].max()
                    recent_range = recent_high - recent_low
                    if recent_range > 0 and (recent_high - curr_c) / recent_range <= 0.35:
                        df.loc[i, "shooting_star"] = 1
                        
            # 6. Morning Star (Bullish 3-candle reversal)
            if prev2_c < prev2_o and curr_c > curr_o:
                # prev1 is small body
                if prev_body < abs(prev2_c - prev2_o) * 0.35 and curr_body > abs(prev2_c - prev2_o) * 0.4:
                    # curr closes above midpoint of prev2
                    if curr_c >= prev2_c + abs(prev2_c - prev2_o) * 0.5:
                        # prev1 gaps down
                        if min(prev_o, prev_c) < prev2_c:
                            df.loc[i, "morning_star"] = 1
                            
            # 7. Evening Star (Bearish 3-candle reversal)
            if prev2_c > prev2_o and curr_c < curr_o:
                # prev1 is small body
                if prev_body < abs(prev2_c - prev2_o) * 0.35 and curr_body > abs(prev2_c - prev2_o) * 0.4:
                    # curr closes below midpoint of prev2
                    if curr_c <= prev2_c - abs(prev2_c - prev2_o) * 0.5:
                        # prev1 gaps up
                        if max(prev_o, prev_c) > prev2_c:
                            df.loc[i, "evening_star"] = 1
                            
        return df

    def _load_combined_daily_candles(self, symbol: str = None) -> pd.DataFrame:
        """Loads combined daily candles from daily and 15m tables to avoid missing recent days."""
        with self.engine.connect() as conn:
            if symbol:
                df_daily = pd.read_sql(text("""
                    SELECT symbol, timestamp, open, high, low, close, volume 
                    FROM ohlcv_daily 
                    WHERE symbol = :sym
                """), conn, params={"sym": symbol})
                df_15m = pd.read_sql(text("""
                    SELECT symbol, timestamp, open, high, low, close, volume 
                    FROM ohlcv_15m 
                    WHERE symbol = :sym AND timestamp >= '2026-05-05 00:00:00'
                """), conn, params={"sym": symbol})
            else:
                df_daily = pd.read_sql(text("""
                    SELECT symbol, timestamp, open, high, low, close, volume 
                    FROM ohlcv_daily
                """), conn)
                df_15m = pd.read_sql(text("""
                    SELECT symbol, timestamp, open, high, low, close, volume 
                    FROM ohlcv_15m 
                    WHERE timestamp >= '2026-05-05 00:00:00'
                """), conn)
                
        df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"], format="mixed")
        if df_15m.empty:
            return df_daily
            
        df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"], format="mixed")
        
        # Aggregate 15m to daily
        df_15m["date"] = df_15m["timestamp"].dt.date
        df_15m_daily = df_15m.groupby(["symbol", "date"]).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum')
        ).reset_index()
        df_15m_daily["timestamp"] = pd.to_datetime(df_15m_daily["date"])
        df_15m_daily = df_15m_daily.drop(columns=["date"])
        
        # Merge
        df_daily["date"] = df_daily["timestamp"].dt.date
        df_15m_daily["date"] = df_15m_daily["timestamp"].dt.date
        dates_15m = set(df_15m_daily["date"])
        df_daily_filtered = df_daily[~df_daily["date"].isin(dates_15m)].copy()
        
        df_combined = pd.concat([df_daily_filtered, df_15m_daily], ignore_index=True)
        return df_combined.drop(columns=["date"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    def calculate_probabilities(self, symbol: str) -> dict:
        """
        Detects patterns and computes next-day directional outcomes.
        Returns a dictionary summarizing total signals, win rate, avg pnl, and t-stats.
        """
        # Load daily candles (combined daily + 15m)
        df = self._load_combined_daily_candles(symbol)
            
        if df.empty or len(df) < 50:
            return {}
            
        df = self.detect_patterns(df)
        
        patterns_list = [
            ("Bullish Engulfing", "bullish_engulfing", "LONG"),
            ("Bearish Engulfing", "bearish_engulfing", "SHORT"),
            ("Hammer", "hammer", "LONG"),
            ("Shooting Star", "shooting_star", "SHORT"),
            ("Doji", "doji", "NEUTRAL"),
            ("Morning Star", "morning_star", "LONG"),
            ("Evening Star", "evening_star", "SHORT")
        ]
        
        results = {}
        
        for name, col, direction in patterns_list:
            indices = df[df[col] == 1].index.tolist()
            # Filter out index at the very end (cannot check next candle)
            indices = [idx for idx in indices if idx + 1 < len(df)]
            
            count = len(indices)
            if count == 0:
                results[name] = {"count": 0, "win_rate": 0.0, "avg_pnl": 0.0, "t_stat": 0.0, "p_value": 1.0}
                continue
                
            pnls = []
            wins = 0
            
            for idx in indices:
                curr_close = df.iloc[idx]["close"]
                next_close = df.iloc[idx+1]["close"]
                next_open = df.iloc[idx+1]["open"]
                
                # Check next-day close change
                pnl = (next_close - curr_close) / curr_close * 100
                
                if direction == "LONG":
                    wins += 1 if pnl > 0 else 0
                    pnls.append(pnl)
                elif direction == "SHORT":
                    wins += 1 if pnl < 0 else 0
                    pnls.append(-pnl) # return is positive if stock drops
                else: # Doji
                    wins += 1 if abs(pnl) >= 1.0 else 0 # 1% volatility win
                    pnls.append(abs(pnl))
                    
            pnls = np.array(pnls)
            win_rate = (wins / count) * 100
            avg_pnl = np.mean(pnls)
            
            # Compute t-statistic directly
            t_stat = 0.0
            if count > 1:
                std_dev = np.std(pnls, ddof=1)
                if std_dev > 0:
                    t_stat = avg_pnl / (std_dev / np.sqrt(count))
            
            results[name] = {
                "count": count,
                "win_rate": round(win_rate, 2),
                "avg_pnl": round(avg_pnl, 3),
                "t_stat": round(t_stat, 3)
            }
            
        return results

    def scan_active_setups(self, target_date: str | date | None = None) -> list:
        """
        Scans all F&O stocks as of the target (or latest) trading session to identify
        active candlestick pattern signals and historical probability estimates.
        """
        # Load all daily candles combined
        df_all = self._load_combined_daily_candles()
            
        if df_all.empty:
            return []
            
        if target_date is not None:
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            target_date = df_all["timestamp"].max().date()
        
        active_signals = []
        
        # Process symbol by symbol
        for symbol, grp in df_all.groupby("symbol"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            if len(grp) < 20:
                continue
                
            # Run pattern detection
            res = self.detect_patterns(grp)
            latest_row = res[res["timestamp"].dt.date == target_date]
            
            if latest_row.empty:
                continue
                
            idx = latest_row.index[0]
            close_price = latest_row.iloc[0]["close"]
            
            patterns_cols = [
                ("Bullish Engulfing", "bullish_engulfing", "LONG"),
                ("Bearish Engulfing", "bearish_engulfing", "SHORT"),
                ("Hammer", "hammer", "LONG"),
                ("Shooting Star", "shooting_star", "SHORT"),
                ("Doji", "doji", "NEUTRAL"),
                ("Morning Star", "morning_star", "LONG"),
                ("Evening Star", "evening_star", "SHORT")
            ]
            
            for name, col, direction in patterns_cols:
                if latest_row.iloc[0][col] == 1:
                    clean_sym = symbol.replace("NSE:", "").replace("-EQ", "")
                    
                    # Fetch historical statistics for this pattern on this stock
                    hist_stats = self.calculate_probabilities(symbol).get(name, {})
                    
                    active_signals.append({
                        "Symbol": clean_sym,
                        "Pattern": name,
                        "Direction": direction,
                        "Close Price": f"₹{close_price:.2f}",
                        "Hist Count": hist_stats.get("count", 0),
                        "Win Rate": f"{hist_stats.get('win_rate', 0.0):.1f}%",
                        "Avg Next-Day return": f"{hist_stats.get('avg_pnl', 0.0):+.2f}%",
                        "t-stat": hist_stats.get("t_stat", 0.0),
                        "Date": target_date.strftime("%Y-%m-%d")
                    })
                    
        return active_signals

    def capture_tradingview_screenshot(self, symbol: str, out_path: str) -> bool:
        """
        Launches Playwright and captures a screenshot of a TradingView chart.
        Handles dismissing accept/cookie popups.
        """
        clean_symbol = symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
        # Nifty formatting mapping
        if "NIFTY50" in clean_symbol:
            tv_symbol = "NSE:NIFTY"
        elif "NIFTYBANK" in clean_symbol:
            tv_symbol = "NSE:BANKNIFTY"
        else:
            tv_symbol = f"NSE:{clean_symbol}"
            
        print(f"Capturing TradingView screenshot for: {tv_symbol} to {out_path}...")
        
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1280, 'height': 720},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                url = f"https://in.tradingview.com/chart/?symbol={tv_symbol}"
                page.goto(url, timeout=45000, wait_until="networkidle")
                
                # Wait for chart gui wrapper
                page.wait_for_selector(".chart-gui-wrapper", timeout=15000)
                time.sleep(3) # Let charts draw fully
                
                # Attempt to close popups
                try:
                    page.click("button[data-name='dismiss']", timeout=2000)
                except:
                    pass
                try:
                    page.click(".tv-dialog__close", timeout=1500)
                except:
                    pass
                    
                chart_element = page.locator(".chart-gui-wrapper")
                chart_element.screenshot(path=out_path)
                browser.close()
                return True
        except Exception as e:
            print(f"Failed to capture TradingView screenshot for {symbol}: {e}")
            return False

if __name__ == "__main__":
    agent = CandlestickPatternAgent()
    print("Active setups:")
    print(agent.scan_active_setups())
