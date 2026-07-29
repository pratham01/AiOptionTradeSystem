"""
Agentic AI Workflow: Screener, Validation & Performance Evaluator.

This script implements a complete agentic AI loop:
1. Candidate Suggestion (Rule-Based & Multi-Strategy Screener)
2. Agentic Validation (SetupValidatorAgent with Gemini LLM Critique & PCR Check)
3. Performance Tracking (Simulating next-day trade outcomes)
4. Comparative Analysis (Rules vs. Agentic validation performance)
"""
import sys
import os
from datetime import datetime, date, timedelta
import json
import asyncio
import warnings

sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
import numpy as np
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.advisory.application.agent.setup_validator_agent import SetupValidatorAgent
from trade_system.domains.advisory.application.agent.candidate_screener_agent import CandidateScore
from trade_system.shared import MarketContext, MarketRegime, TradeHorizon, OptionChainSnapshot
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient

warnings.filterwarnings('ignore')

# 1. Aggregate daily data from 15m to get accurate daily parameters
def get_daily_candles_from_15m(df_15m):
    df = df_15m.copy()
    df["date"] = pd.to_datetime(df["timestamp"], format="mixed").dt.date
    daily = df.groupby(["symbol", "date"]).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    daily["timestamp"] = pd.to_datetime(daily["date"])
    return daily.drop(columns=["date"])

def add_supertrend(df, period=10, multiplier=3.0):
    def calc_st(grp):
        high, low, close = grp['high'], grp['low'], grp['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        direction = pd.Series(1, index=grp.index)
        for i in range(1, len(grp)):
            if close.iloc[i] > upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
        grp["st_dir"] = direction
        return grp
    return df.groupby("symbol", group_keys=False).apply(calc_st)

def compute_all_daily_indicators(df):
    df = df.sort_values(["symbol", "timestamp"]).copy()
    
    # ATR-14
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["tr"] = np.maximum(df["high"] - df["low"], 
                          np.maximum((df["high"] - df["prev_close"]).abs(), 
                                     (df["low"] - df["prev_close"]).abs()))
    df["atr14"] = df.groupby("symbol")["tr"].transform(lambda x: x.rolling(14, min_periods=5).mean())
    df["atr_pct"] = (df["atr14"] / df["close"]) * 100
    
    # BB Width
    df["close_ma"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["close_std"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20, min_periods=5).std())
    df["bb_width"] = (4 * df["close_std"]) / df["close_ma"].replace(0, 1) * 100
    
    # RSI-14
    df["delta"] = df.groupby("symbol")["close"].diff()
    df["gain"] = df["delta"].clip(lower=0)
    df["loss"] = -df["delta"].clip(upper=0)
    df["avg_gain"] = df.groupby("symbol")["gain"].transform(lambda x: x.rolling(14, min_periods=5).mean())
    df["avg_loss"] = df.groupby("symbol")["loss"].transform(lambda x: x.rolling(14, min_periods=5).mean())
    df["rs"] = df["avg_gain"] / df["avg_loss"].replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + df["rs"])
    
    # Volume MA
    df["vol_ma"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, 1)
    
    # 52w High
    df["w52_high"] = df.groupby("symbol")["high"].transform(lambda x: x.rolling(252, min_periods=100).max())
    
    # Supertrend
    df = add_supertrend(df)
    return df

async def main():
    print("🤖 STARTING AGENTIC AI WORKFLOW: SCREENER, VALIDATION & PERFORMANCE EVALUATOR\n")
    engine = get_engine()
    
    # Load raw historical data from SQLite
    print("📦 Loading historical candles from DB...")
    with engine.connect() as conn:
        df_daily_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_daily"), conn)
        df_15m_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE timestamp >= '2026-05-01 00:00:00'"), conn)
        
    df_15m_raw["timestamp"] = pd.to_datetime(df_15m_raw["timestamp"], format="mixed")
    df_daily_raw["timestamp"] = pd.to_datetime(df_daily_raw["timestamp"], format="mixed")
    
    # Combine daily data
    df_15m_daily = get_daily_candles_from_15m(df_15m_raw)
    df_daily_raw["date"] = df_daily_raw["timestamp"].dt.date
    df_15m_daily["date"] = df_15m_daily["timestamp"].dt.date
    
    dates_15m = set(df_15m_daily["date"])
    df_daily_filtered = df_daily_raw[~df_daily_raw["date"].isin(dates_15m)].copy()
    
    df_combined_daily = pd.concat([df_daily_filtered, df_15m_daily], ignore_index=True)
    df_combined_daily = df_combined_daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df_combined_daily = compute_all_daily_indicators(df_combined_daily)
    
    all_dates = sorted(df_combined_daily['timestamp'].dt.date.unique())
    
    # Select last week's trading sessions for comparison
    test_dates = all_dates[-4:]
    print(f"🗓️ Evaluation Dates: {[str(d) for d in test_dates]}")
    
    # Configure Agents
    llm = LlmAdvisorClient()
    validator = SetupValidatorAgent(llm_client=llm)
    
    # Track statistics
    rules_trades = []
    agent_validated_trades = []
    
    for t_date in test_dates:
        print(f"\n--- Processing Session: {t_date} ---")
        df_scan = df_combined_daily[df_combined_daily['timestamp'].dt.date <= t_date]
        df_future = df_combined_daily[df_combined_daily['timestamp'].dt.date > t_date]
        
        # Determine Nifty state for context
        market_ctx = MarketContext(
            timestamp=datetime.combine(t_date, datetime.min.time()),
            regime=MarketRegime.TRENDING_BULL,
            bias="BULLISH",
            vix=12.5,
            pcr=1.05,
            narrative="Dynamic agentic evaluation session."
        )
        
        df_curr = df_scan[df_scan['timestamp'].dt.date == t_date]
        
        for _, row in df_curr.iterrows():
            sym = row["symbol"]
            sym_hist = df_scan[df_scan["symbol"] == sym]
            if len(sym_hist) < 20: continue
            
            bbw_limit = sym_hist["bb_width"].quantile(0.25)
            is_compressed = row["bb_width"] <= bbw_limit
            change_pct = ((row["close"] - row["prev_close"])/row["prev_close"]*100) if not pd.isna(row["prev_close"]) else 0.0
            high_range = row["high"] - row["low"]
            close_high_ratio = (row["close"] - row["low"])/high_range if high_range > 0 else 0
            
            # Match Strategies (including the BTST & Volume shocks we researched)
            reasons = []
            direction = "CALL"
            
            # BTST Squeeze Breakout
            if is_compressed and row["atr_pct"] >= 1.4 and change_pct >= 1.5 and close_high_ratio >= 0.92 and row["volume"] >= 1.5 * row["vol_ma"]:
                reasons.append("BTST_Squeeze_BO")
            # 52W High Breakout
            if row["close"] >= row["w52_high"] * 0.985 and row["volume"] >= 1.5 * row["vol_ma"] and change_pct > 0:
                reasons.append("52W_High_BO")
            # Supertrend Flip
            st_prev = sym_hist.iloc[-2]["st_dir"] if len(sym_hist) >= 2 else 1
            if st_prev == -1 and row["st_dir"] == 1:
                reasons.append("ST_Flip_Up")
                
            # Intraday ORB Morning Shock
            if row["volume"] >= 2.5 * row["vol_ma"] and change_pct >= 3.0:
                reasons.append("StartOfDay_Volume_Shock")
            
            if not reasons: continue
            
            # Wrap as CandidateScore
            candidate = CandidateScore(
                symbol=sym,
                intraday_score=0.75,
                swing_score=0.80,
                horizon=TradeHorizon.INTRADAY if "StartOfDay" in "".join(reasons) else TradeHorizon.SWING,
                direction=direction,
                sector="Unknown",
                entry_price=row["close"],
                rsi_daily=row["rsi"],
                volume_surge=row["volume"] / row["vol_ma"] if row["vol_ma"] else 1.0,
                atr_pct=row["atr_pct"],
                is_compressed=is_compressed,
                vol_delta_positive=True,
                above_poc=True,
                alignment_score=0.8,
                breakout_type=reasons[0],
                pattern=" + ".join(reasons)
            )
            candidate.component_scores = {"volume_delta": 0.8, "value_area": 0.85, "momentum": 0.8}
            
            # Evaluate Future price action (next day)
            fut = df_future[df_future['symbol'] == sym].sort_values('timestamp')
            if fut.empty: continue
            next_day = fut.iloc[0]
            
            entry = float(row['close'])
            high_target = entry * (1 + 1.5 * (row['atr_pct']/100))
            low_sl = entry * (1 - 0.7 * (row['atr_pct']/100))
            
            # Simulating outcome
            pnl_pct = 0.0
            outcome = "EXPIRED"
            if next_day['high'] >= high_target:
                outcome = "WIN"
                pnl_pct = (high_target - entry)/entry * 100
            elif next_day['low'] <= low_sl:
                outcome = "LOSS"
                pnl_pct = (low_sl - entry)/entry * 100
            else:
                pnl_pct = (next_day['close'] - entry)/entry * 100
                outcome = "NEUTRAL"
            
            trade_record = {
                "date": str(t_date),
                "symbol": sym,
                "strategy": reasons[0],
                "direction": direction,
                "entry": entry,
                "pnl": pnl_pct,
                "outcome": outcome
            }
            
            # Rules-based logs get ALL candidates
            rules_trades.append(trade_record)
            
            # RUN AGENTIC AI VALIDATION (EVERY TIME)
            print(f"  🔍 Agent Validating: {sym} ({reasons[0]}) ...")
            try:
                # Fetch recent 15m bars to pass to validator
                bars_15m = df_15m_raw[(df_15m_raw['symbol'] == sym) & (df_15m_raw['timestamp'].dt.date <= t_date)].tail(30)
                validated_sugg = await validator.validate(candidate, market_ctx, current_bars=bars_15m)
                
                if validated_sugg:
                    # Log validated trade
                    validated_record = trade_record.copy()
                    validated_record["ai_confidence"] = validated_sugg.confidence
                    validated_record["critique"] = validated_sugg.narrative
                    agent_validated_trades.append(validated_record)
                    print(f"    ✅ APPROVED by LLM. Confidence: {validated_sugg.confidence:.2f} | Critique: {validated_sugg.narrative[:120]}...")
                else:
                    print("    ❌ REJECTED by LLM (Risk/Reward, Option Chain, or Critic mismatch).")
            except Exception as ex:
                print(f"    ⚠️ Agent validation skipped due to error: {ex}")
                
    # Compile comparative stats
    print("\n" + "="*80)
    print("📈 PERFORMANCE COMPARATIVE SUMMARY: RULES VS. AGENTIC AI VALIDATION")
    print("="*80)
    
    def get_stats(trades_list):
        if not trades_list:
            return 0, 0.0, 0.0, 0.0
        df = pd.DataFrame(trades_list)
        total = len(df)
        wins = len(df[df["outcome"] == "WIN"])
        losses = len(df[df["outcome"] == "LOSS"])
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 50.0
        avg_pnl = df["pnl"].mean()
        
        profit_factor = 1.0
        gross_profit = df[df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(df[df["pnl"] < 0]["pnl"].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 999.0 # Infinity
            
        return total, win_rate, avg_pnl, profit_factor

    r_total, r_win, r_pnl, r_pf = get_stats(rules_trades)
    a_total, a_win, a_pnl, a_pf = get_stats(agent_validated_trades)
    
    print(f"{'Metric':<25} | {'Rules-Based Scan':<20} | {'Agentic AI Validated':<25}")
    print("-" * 78)
    print(f"{'Total Suggestions':<25} | {r_total:<20d} | {a_total:<25d}")
    print(f"{'Win Rate':<25} | {r_win:<19.1f}% | {a_win:<24.1f}%")
    print(f"{'Average PnL':<25} | {r_pnl:<+19.2f}% | {a_pnl:<+24.2f}%")
    print(f"{'Profit Factor':<25} | {r_pf:<19.2f} | {a_pf:<25.2f}")
    print("="*80)
    
    # Save a detailed markdown evaluation report
    report_path = "reports/agentic_validation_performance.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Agentic AI Validation Performance Report\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Summary Performance Metrics\n\n")
        f.write("| Metric | Rules-Based Scan | Agentic AI Validated |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Total Suggestions** | {r_total} | {a_total} |\n")
        f.write(f"| **Win Rate** | {r_win:.1f}% | {a_win:.1f}% |\n")
        f.write(f"| **Average PnL** | {r_pnl:+.2f}% | {a_pnl:+.2f}% |\n")
        f.write(f"| **Profit Factor** | {r_pf:.2f} | {a_pf:.2f} |\n\n")
        
        f.write("## 2. Agentic Validation Logs\n\n")
        f.write("| Date | Symbol | Strategy | AI Conf | Outcome | PnL | AI Critique |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for t in agent_validated_trades:
            crit = t.get("critique", "").replace("\n", " ")
            f.write(f"| {t['date']} | {t['symbol']} | {t['strategy']} | {t['ai_confidence']:.2f} | {t['outcome']} | {t['pnl']:+.2f}% | {crit} |\n")
            
    print(f"\n📁 Performance report successfully compiled and saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
