from __future__ import annotations

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from sqlalchemy import text
import scipy.stats as stats

from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)

class OptionResearchAgent:
    """
    Autonomous options research agent that:
    1. Researches top traders, books, and event pricing mechanics.
    2. Runs quantitative backtests to evaluate next-day direction predictability.
    3. Journals trade performance, scans for missed opportunities, and suggests enhancements.
    """

    def __init__(self, llm_client: LlmAdvisorClient | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()
        self.engine = get_engine()

    async def research_option_wisdom(self) -> str:
        """
        Queries LLM for synthesized scholar report on option trading.
        """
        prompt = (
            "You are a world-class Quantitative Option Trading Scholar and Advisor.\n"
            "Produce a premium, deeply technical Option Trading Masterclass Report covering:\n\n"
            "1. **The Giants of Option Trading**: Critically analyze the core strategies of the world's most profitable "
            "options traders (e.g., Jim Simons' statistical arbitrage, Ed Thorp's warrants/convertible hedging, Nassim Taleb's "
            "convexity/tail-risk premium, Dan Harvey's options structures).\n"
            "2. **The Option Buyer's Canonical Reading List**: Summarize the critical takeaways for option buyers from "
            "Sheldon Natenberg's *Option Volatility and Pricing*, Nassim Taleb's *Dynamic Hedging*, and Lawrence McMillan's "
            "*Options as a Strategic Investment*.\n"
            "3. **Greeks & Event Pricing Physics**: Explain *what*, *how*, and *when* specific events impact option prices. "
            "Specifically explain:\n"
            "   - **Theta vs. Gamma Tradeoff**: The risk-reward dynamics of short-dated options buying.\n"
            "   - **Earnings IV Crush**: Why buying options before major corporate earnings often leads to instant losses "
            "even if the direction is guessed correctly (volatility collapse).\n"
            "   - **VIX & regime shifts**: How high vs low VIX environments shift option pricing and model assumptions.\n\n"
            "Format the report in clean, professional Markdown with headers, bold points, and structured bullet lists."
        )
        return await self.llm.complete(prompt)

    async def run_predictability_backtest(
        self, symbol: str, indicator: str, threshold: float
    ) -> dict:
        """
        Backtests whether next-day return direction is statistically predictable using historical SQLite daily data.
        """
        try:
            with self.engine.connect() as conn:
                # Load daily candles
                query = text("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM ohlcv_daily 
                    WHERE symbol = :symbol 
                    ORDER BY timestamp ASC
                """)
                df = pd.read_sql(query, conn, params={"symbol": symbol})

            if df.empty or len(df) < 50:
                return {
                    "success": False,
                    "error": f"Insufficient historical data for {symbol}. Found {len(df)} rows, minimum required is 50."
                }

            # 1. Calculate next-day close return (target variable)
            df["next_day_return"] = df["close"].pct_change().shift(-1) * 100
            
            # 2. Compute selected indicator and signal masks
            df["close_change"] = df["close"].pct_change() * 100
            df["volume_ma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df.apply(lambda r: r["volume_ma"] if r["volume_ma"] > 0 else 1, axis=1)

            # RSI calculation
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss.replace(0, np.nan))
            df["rsi"] = 100 - (100 / (1 + rs))
            df["rsi"] = df["rsi"].fillna(50)

            # Signal conditions
            if indicator == "Momentum Surge":
                condition = df["close_change"] >= threshold
                desc = f"Today's Close Change >= {threshold}%"
            elif indicator == "Volume Ratio":
                condition = df["volume_ratio"] >= threshold
                desc = f"Today's Volume >= {threshold}x the 20-day Average"
            elif indicator == "RSI Oversold":
                condition = df["rsi"] <= threshold
                desc = f"Today's RSI <= {threshold} (Oversold Buy)"
            elif indicator == "RSI Overbought":
                condition = df["rsi"] >= threshold
                desc = f"Today's RSI >= {threshold} (Overbought Sell)"
            elif indicator == "High-Volume Breakout":
                condition = (df["close_change"] >= 1.5) & (df["volume_ratio"] >= threshold)
                desc = f"Close Change >= 1.5% and Volume >= {threshold}x Average"
            else:
                return {"success": False, "error": f"Unknown indicator type: {indicator}"}

            df["signal"] = condition
            
            # Filter clean records (remove NaNs from indicators or shift)
            df_clean = df.dropna(subset=["next_day_return", "rsi", "volume_ratio"]).copy()
            
            # 3. Partition returns based on signal
            signal_returns = df_clean[df_clean["signal"]]["next_day_return"]
            baseline_returns = df_clean[~df_clean["signal"]]["next_day_return"]

            n_signal = len(signal_returns)
            n_baseline = len(baseline_returns)

            if n_signal < 5:
                return {
                    "success": False,
                    "error": f"The indicator condition met only {n_signal} times in history. Increase lookback or lower threshold."
                }

            # 4. Statistical parameters
            win_rate_signal = (signal_returns > 0).sum() / n_signal
            win_rate_baseline = (baseline_returns > 0).sum() / n_baseline

            mean_sig = float(signal_returns.mean())
            mean_base = float(baseline_returns.mean())
            std_sig = float(signal_returns.std())
            std_base = float(baseline_returns.std())

            # Perform independent two-sample t-test
            t_stat, p_value = stats.ttest_ind(signal_returns, baseline_returns, equal_var=False)

            is_significant = p_value < 0.05
            verdict = (
                "STATISTICALLY SIGNIFICANT PREDICTABILITY" if is_significant 
                else "NOT STATISTICALLY SIGNIFICANT (Likely Random Walk)"
            )

            # Generate monthly return breakdown to plot
            df_clean["year_month"] = pd.to_datetime(df_clean["timestamp"], format="mixed").dt.to_period("M").astype(str)
            monthly_perf = df_clean[df_clean["signal"]].groupby("year_month")["next_day_return"].mean().reset_index()
            monthly_data = monthly_perf.to_dict("records")

            return {
                "success": True,
                "symbol": symbol,
                "indicator": indicator,
                "threshold": threshold,
                "description": desc,
                "total_days_analyzed": len(df_clean),
                "condition_met_count": n_signal,
                "win_rate_when_signal": win_rate_signal,
                "win_rate_baseline": win_rate_baseline,
                "avg_return_when_signal": mean_sig,
                "avg_return_baseline": mean_base,
                "std_return_when_signal": std_sig,
                "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                "verdict": verdict,
                "is_significant": bool(is_significant),
                "monthly_performance": monthly_data
            }

        except Exception as e:
            LOGGER.error(f"Backtest failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_autonomous_journal(self) -> dict:
        """
        Scans suggested trades, checks performance, finds missed opportunities, and writes journal entries.
        """
        try:
            # 1. Query past suggested trades summary
            with self.engine.connect() as conn:
                trade_query = text("""
                    SELECT date, symbol, direction, outcome, actual_pnl_pct, narrative 
                    FROM suggested_trades 
                    ORDER BY date DESC LIMIT 20
                """)
                trades_df = pd.read_sql(trade_query, conn)
                
                # Check for recent active F&O stocks to scan for missed opportunities
                stocks_query = text("SELECT DISTINCT symbol FROM ohlcv_daily LIMIT 180")
                stocks_df = pd.read_sql(stocks_query, conn)

            # 2. Scan for Missed Breakout Opportunities in ohlcv_daily over the last 10 trading days
            # A missed opportunity is defined as close change >= 1.5% and volume ratio >= 2.0
            # which did not have a corresponding suggested_trade.
            missed_ops = []
            if not stocks_df.empty:
                # Find the last 5 unique dates in ohlcv_daily
                with self.engine.connect() as conn:
                    dates_query = text("SELECT DISTINCT date(timestamp) as dt FROM ohlcv_daily ORDER BY dt DESC LIMIT 5")
                    dates_df = pd.read_sql(dates_query, conn)
                
                recent_dates = dates_df["dt"].tolist() if not dates_df.empty else []
                
                if recent_dates:
                    for d_str in recent_dates:
                        # Fetch all candles for this date
                        with self.engine.connect() as conn:
                            # We check the close change and calculate volume MA
                            # To be fast, we query database with simple SQL, joining current candle with previous
                            candle_query = text("""
                                SELECT c.symbol, c.close, c.volume, 
                                       ((c.close - p.close) / p.close) * 100 as change_pct
                                FROM ohlcv_daily c
                                JOIN ohlcv_daily p ON c.symbol = p.symbol 
                                  AND date(p.timestamp) = date(c.timestamp, '-1 day')
                                WHERE date(c.timestamp) = :date_str
                            """)
                            daily_df = pd.read_sql(candle_query, conn, params={"date_str": d_str})
                            
                            # Filter suggested trades on this date
                            sugg_query = text("SELECT symbol FROM suggested_trades WHERE date = :date_str")
                            sugg_df = pd.read_sql(sugg_query, conn, params={"date_str": d_str})
                            sugg_symbols = set(sugg_df["symbol"].tolist()) if not sugg_df.empty else set()
                            
                        for _, row in daily_df.iterrows():
                            sym = row["symbol"]
                            chg = row["change_pct"]
                            vol = row["volume"]
                            
                            # Standard breakout thresholds
                            if chg >= 1.5 and sym not in sugg_symbols:
                                # Fetch previous 20 days volumes to compute ratio
                                with self.engine.connect() as conn:
                                    vol_query = text("""
                                        SELECT AVG(volume) as avg_vol 
                                        FROM ohlcv_daily 
                                        WHERE symbol = :symbol AND timestamp < :d_str 
                                        ORDER BY timestamp DESC LIMIT 20
                                    """)
                                    v_res = conn.execute(vol_query, {"symbol": sym, "d_str": d_str}).first()
                                    avg_vol = v_res[0] if v_res and v_res[0] else 1
                                    
                                vol_ratio = vol / avg_vol
                                if vol_ratio >= 1.8:
                                    # Fetch next day close return if available (to see if it would have worked)
                                    with self.engine.connect() as conn:
                                        next_query = text("""
                                            SELECT ((n.close - :close) / :close) * 100 as next_ret
                                            FROM ohlcv_daily n
                                            WHERE symbol = :symbol AND date(timestamp) = date(:d_str, '+1 day')
                                        """)
                                        n_res = conn.execute(next_query, {"symbol": sym, "close": row["close"], "d_str": d_str}).first()
                                        next_ret = n_res[0] if n_res and n_res[0] is not None else None
                                        
                                    missed_ops.append({
                                        "date": d_str,
                                        "symbol": sym,
                                        "close_change": float(chg),
                                        "volume_ratio": float(vol_ratio),
                                        "next_day_return": next_ret
                                    })
            
            # Sort missed opportunities by absolute close change descending and limit to top 15
            missed_ops = sorted(missed_ops, key=lambda x: abs(x["close_change"]), reverse=True)[:15]

            # 3. Call LLM to synthesize the Trade Journal
            trades_summary = ""
            if not trades_df.empty:
                for _, r in trades_df.iterrows():
                    trades_summary += f"- Date: {r['date']} | Stock: {r['symbol']} | Dir: {r['direction']} | Outcome: {r['outcome']} | PnL: {r['actual_pnl_pct'] or 0.0}% | Reasoning: {r['narrative'][:80]}...\n"
            else:
                trades_summary = "No recommended trades found in history.\n"

            missed_summary = ""
            if missed_ops:
                for m in missed_ops:
                    next_ret_str = f"{m['next_day_return']:.2f}%" if m['next_day_return'] is not None else "N/A"
                    missed_summary += f"- Date: {m['date']} | Stock: {m['symbol']} | Close Chg: {m['close_change']:.2f}% | Vol Surge: {m['volume_ratio']:.1f}x | Next Day Return: {next_ret_str}\n"
            else:
                missed_summary = "No significant missed breakout opportunities detected.\n"

            prompt = (
                "You are an expert autonomous Options Trading Desk Journaler and Auditor.\n"
                "Analyze the following data from our automated multi-agent trading system:\n\n"
                "### 📈 SYSTEM RECOMMENDED TRADES RECORD:\n" + trades_summary + "\n"
                "### 🔭 DETECTED MISSED BREAKOUT OPPORTUNITIES:\n" + missed_summary + "\n\n"
                "Provide an EOD Trade Journal & System Performance Enhancement Audit. Please structure it with:\n"
                "1. **Execution Journal & Review**: Summarize our win rate and performance. Pinpoint specific strengths and weaknesses in our entries/targets based on the recommended trade outcomes.\n"
                "2. **Missed Opportunities Audit**: Analyze why we missed the listed breakout opportunities. For example, did we filter them out due to high daily RSI? Did the conviction fuser reject them? Explain how option buyers lose premium if they ignore these setups.\n"
                "3. **Proposed System Enhancements**: Recommend concrete parameters to modify in the indicator configs (e.g. adjust volume surge thresholds, change Supertrend multiplier bounds) to capture these missed breakouts while keeping risk minimal.\n"
                "Format this report in premium, clean Markdown with matching emojis."
            )

            journal_text = await self.llm.complete(prompt)

            return {
                "success": True,
                "trades_audited_count": len(trades_df),
                "missed_opportunities_count": len(missed_ops),
                "journal_text": journal_text,
                "missed_opportunities": missed_ops
            }

        except Exception as e:
            LOGGER.error(f"Failed to generate journal: {e}")
            return {"success": False, "error": str(e)}
