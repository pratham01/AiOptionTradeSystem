import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import text
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
from trade_system.domains.advisory.application.agent.candlestick_pattern_agent import CandlestickPatternAgent

LOGGER = logging.getLogger(__name__)

class TopGainerAnalysisAgent:
    """
    Autonomous agent that:
    1. Scans the database to identify top performing stocks for a given date.
    2. Runs technical analysis (volume ratios, indicators, candlestick patterns).
    3. Synthesizes a technical catalyst / reason for the move using the LLM.
    4. Generates a detailed markdown performance report.
    """
    
    def __init__(self, llm_client: LlmAdvisorClient | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()
        self.engine = get_engine()
        self.candlestick_agent = CandlestickPatternAgent()
        
    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        if len(prices) < period + 1:
            return pd.Series(50.0, index=prices.index)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        if len(close) < period * 2:
            return pd.Series(20.0, index=close.index)
        try:
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            dm_plus = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
            dm_minus = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
            atr = tr.ewm(span=period).mean()
            di_plus = 100 * dm_plus.ewm(span=period).mean() / (atr + 1e-9)
            di_minus = 100 * dm_minus.ewm(span=period).mean() / (atr + 1e-9)
            dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
            adx = dx.ewm(span=period).mean()
            return adx
        except Exception:
            return pd.Series(20.0, index=close.index)
            
    def _get_daily_candles_from_15m(self, df_15m):
        df = df_15m.copy()
        df["date"] = pd.to_datetime(df["timestamp"], format="mixed").dt.date
        daily = df.groupby(["symbol", "date"]).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum')
        ).reset_index()
        daily["timestamp"] = pd.to_datetime(daily["date"], format="mixed")
        return daily.drop(columns=["date"])

    def _load_combined_daily(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df_daily_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_daily"), conn)
            df_15m_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE timestamp >= '2026-05-05 00:00:00'"), conn)
            
        df_15m_raw["timestamp"] = pd.to_datetime(df_15m_raw["timestamp"], format="mixed")
        df_daily_raw["timestamp"] = pd.to_datetime(df_daily_raw["timestamp"], format="mixed")
        
        df_15m_daily = self._get_daily_candles_from_15m(df_15m_raw)
        df_daily_raw["date"] = df_daily_raw["timestamp"].dt.date
        df_15m_daily["date"] = df_15m_daily["timestamp"].dt.date
        
        dates_15m = set(df_15m_daily["date"])
        df_daily_filtered = df_daily_raw[~df_daily_raw["date"].isin(dates_15m)].copy()
        
        df_combined_daily = pd.concat([df_daily_filtered, df_15m_daily], ignore_index=True)
        df_combined_daily = df_combined_daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return df_combined_daily

    async def analyze_top_gainers(self, target_date: str | date | None = None, top_n: int = 5) -> dict:
        """
        Runs the daily gainer analysis.
        """
        df_all = self._load_combined_daily()
        if df_all.empty:
            return {"success": False, "error": "No daily data found in database."}
            
        if target_date is not None:
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            target_date = df_all["timestamp"].max().date()
            
        # Calculate daily change percentage for all symbols on this date
        gainer_records = []
        
        print(f"Analyzing top gainers as of {target_date.strftime('%Y-%m-%d')}...")
        
        for symbol, grp in df_all.groupby("symbol"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            if len(grp) < 20:
                continue
                
            # Find the row of target_date
            target_rows = grp[grp["timestamp"].dt.date == target_date]
            if target_rows.empty:
                continue
                
            target_idx = target_rows.index[0]
            if target_idx == 0:
                continue # no previous day to calculate change
                
            prev_row = grp.iloc[target_idx - 1]
            curr_row = target_rows.iloc[0]
            
            change_pct = (curr_row["close"] - prev_row["close"]) / prev_row["close"] * 100
            
            gainer_records.append({
                "symbol": symbol,
                "close": curr_row["close"],
                "change_pct": change_pct,
                "volume": curr_row["volume"],
                "history": grp.iloc[:target_idx + 1] # history up to target_date
            })
            
        if not gainer_records:
            return {"success": False, "error": f"No data found for date {target_date.strftime('%Y-%m-%d')}."}
            
        # Sort by change_pct descending and pick top_n
        top_gainers = sorted(gainer_records, key=lambda x: x["change_pct"], reverse=True)[:top_n]
        
        gainer_analyses = []
        
        for idx, item in enumerate(top_gainers, 1):
            sym = item["symbol"]
            hist = item["history"].copy().reset_index(drop=True)
            close_val = item["close"]
            chg_pct = item["change_pct"]
            vol_val = item["volume"]
            
            # Technical Indicators
            hist["rsi"] = self._calc_rsi(hist["close"])
            hist["adx"] = self._calc_adx(hist["high"], hist["low"], hist["close"])
            
            # Volume surge ratio
            hist["vol_ma"] = hist["volume"].rolling(20).mean()
            vol_ratio = vol_val / hist["vol_ma"].iloc[-1] if hist["vol_ma"].iloc[-1] > 0 else 1.0
            
            latest_rsi = hist["rsi"].iloc[-1]
            latest_adx = hist["adx"].iloc[-1]
            
            # Candlestick pattern check
            res_patterns = self.candlestick_agent.detect_patterns(hist)
            active_patterns = []
            patterns_cols = [
                ("Bullish Engulfing", "bullish_engulfing"),
                ("Bearish Engulfing", "bearish_engulfing"),
                ("Hammer", "hammer"),
                ("Shooting Star", "shooting_star"),
                ("Doji", "doji"),
                ("Morning Star", "morning_star"),
                ("Evening Star", "evening_star")
            ]
            for name, col in patterns_cols:
                if res_patterns[col].iloc[-1] == 1:
                    active_patterns.append(name)
            
            pattern_str = ", ".join(active_patterns) if active_patterns else "None Detected"
            
            # Squeeze / Breakout metrics
            close_series = hist["close"]
            high_20 = float(hist["high"].tail(20).max())
            low_20 = float(hist["low"].tail(20).min())
            range_pct = (high_20 - low_20) / low_20 * 100
            
            # Let the LLM explain the move based on technical details
            clean_sym = sym.replace("NSE:", "").replace("-EQ", "")
            
            prompt = (
                f"You are an expert quantitative technical analyst.\n"
                f"Analyze the following technical snapshot for the top gainer stock {clean_sym} on {target_date.strftime('%Y-%m-%d')}:\n"
                f"- **Daily Close Price**: ₹{close_val:.2f} ({chg_pct:+.2f}% change from yesterday)\n"
                f"- **Volume Ratio**: {vol_ratio:.2f}x the 20-day average volume (Current Volume: {vol_val:,})\n"
                f"- **RSI (14)**: {latest_rsi:.1f}\n"
                f"- **ADX (14)**: {latest_adx:.1f}\n"
                f"- **Candlestick Patterns Detected today**: {pattern_str}\n"
                f"- **20-day price range contraction**: {range_pct:.1f}%\n\n"
                f"Explain the technical 'reason for the move' in 2-3 concise sentences. Focus on breakout, volume surge, RSI momentum, or reversal patterns. Do not speculate on news or rumors. Be professional."
            )
            
            reason = await self.llm.complete(prompt)
            
            gainer_analyses.append({
                "Rank": idx,
                "Symbol": clean_sym,
                "Close Price": f"₹{close_val:.2f}",
                "Change %": f"{chg_pct:+.2f}%",
                "Volume Ratio": f"{vol_ratio:.1f}x",
                "RSI (14)": round(latest_rsi, 1),
                "ADX (14)": round(latest_adx, 1),
                "Pattern": pattern_str,
                "Reason for Move": reason.strip()
            })
            
        # Build Markdown Report
        markdown_content = f"""# 🏆 Daily Top Gainers Technical Analysis Report
        
**Scan Date**: {target_date.strftime('%Y-%m-%d')}  
**Target universe**: Active F&O and Index stocks in database.

This report is autonomously generated by the `TopGainerAnalysisAgent`. It analyzes the top performing stocks of the session, identifying technical catalysts, volume surges, trend regimes (ADX), and candlestick setups.

## 📈 Top performing stocks

| Rank | Symbol | Close Price | Change % | Vol Ratio | RSI (14) | ADX (14) | Candlestick Pattern |
|:-----|:-------|:------------|:---------|:----------|:---------|:---------|:--------------------|
"""
        for row in gainer_analyses:
            markdown_content += f"| {row['Rank']} | **{row['Symbol']}** | {row['Close Price']} | `{row['Change %']}` | {row['Volume Ratio']} | {row['RSI (14)']} | {row['ADX (14)']} | {row['Pattern']} |\n"
            
        markdown_content += "\n## 🧠 Deep-Dive Technical Catalysts\n\n"
        for row in gainer_analyses:
            markdown_content += f"### {row['Rank']}. **{row['Symbol']}** ({row['Change %']})\n"
            markdown_content += f"* **Indicators**: RSI-14 is `{row['RSI (14)']}`, ADX-14 is `{row['ADX (14)']}` (Trend strength).\n"
            markdown_content += f"* **Volume**: Massive surge of `{row['Volume Ratio']}` over 20-day average.\n"
            markdown_content += f"* **Candlestick Pattern**: `{row['Pattern']}`\n"
            markdown_content += f"* **Analysis**: {row['Reason for Move']}\n\n"
            
        markdown_content += """---
### 💡 Core Trading Guidelines for Gainers
1. **Chasing the Momentum (ADX > 25, RSI < 70)**: High change accompanied by volume ratio > 2.0 and strong ADX indicates institutional buying. These have high follow-through odds for next-day continuation.
2. **Exhaustion Risk (RSI > 75)**: If the stock has run into an extreme overbought state, caution is advised as it is vulnerable to profit-booking or gap-down reversals.
3. **Breakout confirmations**: If the volume surge occurs right after a range contraction (< 5-7% over 20 days), it marks a high-conviction breakout entry.
"""
        
        report_path = f"scratch/top_gainers_report_{target_date.strftime('%Y%m%d')}.md"
        with open(report_path, "w") as f:
            f.write(markdown_content)
            
        with open("scratch/daily_gainer_analysis_report.md", "w") as f:
            f.write(markdown_content)
            
        return {
            "success": True,
            "report_path": report_path,
            "gainers": gainer_analyses,
            "markdown": markdown_content
        }
