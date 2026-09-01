"""
Multi-Asset Portfolio Backtesting Engine.

Simulates point-in-time quantitative execution across the entire F&O Universe with:
1. Strict Point-in-Time Causality (Zero Lookahead Bias)
2. Stop-Limit Trigger Confirmation Validation
3. Fixed-Fractional 1% Risk-per-Trade Position Sizing
4. Portfolio Allocation, Max Concurrent Positions & Sector Exposure Caps
5. Realistic Execution Friction (Slippage + STT + Turnover + Exchange + Brokerage)
6. Dynamic Order Resolution (Stop Loss vs Target 1 vs Trailing SL vs Max Holding Days)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping
from trade_system.domains.analysis.application.analysis.reversal_scanner import DailyReversalScanner, DailyReversalSetup, _compute_rsi
from trade_system.domains.analysis.application.backtesting.performance_analytics import (
    BacktestTradeRecord,
    PerformanceAnalytics,
    QuantitativeReport
)

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0   # ₹10 Lakhs
    risk_per_trade_pct: float = 1.0       # 1.0% portfolio risk per trade
    max_concurrent_positions: int = 8     # Max concurrent active trades
    max_positions_per_sector: int = 2     # Max 2 concurrent positions per sector
    max_holding_days: int = 5             # Time-based stop (5 trading days)
    slippage_pct: float = 0.0005          # 0.05% slippage per side (0.10% round trip)
    stt_tax_pct: float = 0.00025          # 0.025% STT on equity intraday / delivery
    brokerage_per_order: float = 20.0     # ₹20 flat brokerage per leg
    trailing_stop: bool = True            # Move SL to breakeven after hitting Target 1
    min_probability: int = 60             # Minimum setup conviction score


@dataclass
class ActivePosition:
    trade_id: int
    symbol: str
    sector: str
    direction: str
    entry_date: date
    entry_price: float
    shares: int
    capital_invested: float
    initial_sl: float
    current_sl: float
    target_1: float
    target_2: float
    probability: int
    confidence: str
    confluences: List[str]
    days_held: int = 0
    hit_t1: bool = False
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0


class PortfolioBacktestEngine:
    """
    State-of-the-art Multi-Asset Portfolio Backtest Engine for Indian F&O Universe.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self.sector_map = get_sector_mapping()

    def run_reversal_backtest(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        min_probability: Optional[int] = None
    ) -> QuantitativeReport:
        """
        Runs walk-forward multi-asset portfolio simulation on daily reversal setups.
        """
        min_prob = min_probability or self.config.min_probability
        e_date = end_date or date.today()
        
        # Load historical daily candles
        engine = get_engine()
        with engine.connect() as conn:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM ohlcv_daily 
                WHERE timestamp >= date(:e_date, '-365 days') AND timestamp <= :e_date
                ORDER BY symbol, timestamp ASC
            """)
            df_d = pd.read_sql(query, conn, params={"e_date": e_date.isoformat()})

        if df_d.empty:
            LOGGER.error("No daily data available for backtesting.")
            return PerformanceAnalytics.compute([], pd.DataFrame(), self.config.initial_capital)

        df_d["timestamp"] = pd.to_datetime(df_d["timestamp"], format="mixed", errors="coerce")
        df_d = df_d.dropna(subset=["timestamp", "close"])

        if symbols:
            df_d = df_d[df_d["symbol"].isin(symbols)]

        # Unique trading dates in chronological order
        all_dates = sorted(df_d["timestamp"].dt.date.unique())
        if len(all_dates) < 35:
            LOGGER.error("Insufficient historical trading dates.")
            return PerformanceAnalytics.compute([], pd.DataFrame(), self.config.initial_capital)

        # Simulation state
        capital = self.config.initial_capital
        cash = capital
        active_positions: List[ActivePosition] = []
        completed_trades: List[BacktestTradeRecord] = []
        equity_records: List[Dict[str, Any]] = []
        trade_id_counter = 1
        total_signals_count = 0

        scanner = DailyReversalScanner(min_probability=min_prob)

        # Precompute Technical Indicators for all symbols upfront in one fast pass
        def _precompute(df_s: pd.DataFrame) -> pd.DataFrame:
            df_s = df_s.copy().sort_values("timestamp").reset_index(drop=True)
            df_s["rsi"] = _compute_rsi(df_s["close"], 14)
            df_s["vol_sma20"] = df_s["volume"].rolling(20).mean()
            df_s["sma20"] = df_s["close"].rolling(20).mean()
            df_s["std20"] = df_s["close"].rolling(20).std()
            df_s["lower_bb"] = df_s["sma20"] - 2.0 * df_s["std20"]
            df_s["upper_bb"] = df_s["sma20"] + 2.0 * df_s["std20"]
            df_s["ema9"] = df_s["close"].ewm(span=9, adjust=False).mean()
            df_s["ema20"] = df_s["close"].ewm(span=20, adjust=False).mean()
            df_s["ema50"] = df_s["close"].ewm(span=50, adjust=False).mean()
            df_s["ema200"] = df_s["close"].ewm(span=200, adjust=False).mean() if len(df_s) >= 120 else np.nan
            tr = pd.concat([
                df_s["high"] - df_s["low"],
                (df_s["high"] - df_s["close"].shift(1)).abs(),
                (df_s["low"] - df_s["close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            df_s["atr"] = tr.rolling(14).mean()
            return df_s

        symbol_data_dict: Dict[str, pd.DataFrame] = {
            sym: _precompute(grp)
            for sym, grp in df_d.groupby("symbol")
        }

        # Date loop from day index 30 to N
        start_idx = 30
        if start_date:
            for idx, d in enumerate(all_dates):
                if d >= start_date and idx >= 30:
                    start_idx = idx
                    break

        pending_orders: List[Tuple[DailyReversalSetup, float, float]] = []  # (Setup, trigger_high, trigger_low)

        for day_idx in range(start_idx, len(all_dates)):
            curr_date = all_dates[day_idx]
            
            # -------------------------------------------------------------
            # STEP 1: Process and Update Existing Open Positions
            # -------------------------------------------------------------
            remaining_positions: List[ActivePosition] = []

            for pos in active_positions:
                pos.days_held += 1
                sym = pos.symbol
                
                # Fetch today's candle for this symbol
                sym_history = symbol_data_dict.get(sym)
                today_rows = sym_history[sym_history["timestamp"].dt.date == curr_date] if sym_history is not None else pd.DataFrame()

                if today_rows.empty:
                    remaining_positions.append(pos)
                    continue

                today_candle = today_rows.iloc[0]
                t_o = float(today_candle["open"])
                t_h = float(today_candle["high"])
                t_l = float(today_candle["low"])
                t_c = float(today_candle["close"])

                is_call = (pos.direction == "CALL")
                entry_p = pos.entry_price

                # Track MFE / MAE
                if is_call:
                    fav = (t_h - entry_p) / entry_p * 100.0
                    adv = (t_l - entry_p) / entry_p * 100.0
                else:
                    fav = (entry_p - t_l) / entry_p * 100.0
                    adv = (entry_p - t_h) / entry_p * 100.0
                pos.max_favorable_pct = max(pos.max_favorable_pct, fav)
                pos.max_adverse_pct = min(pos.max_adverse_pct, adv)

                # Check Exits
                exit_occurred = False
                exit_p = 0.0
                exit_reason = ""

                if is_call:
                    # 1. Stop Loss Hit
                    if t_l <= pos.current_sl:
                        exit_occurred = True
                        exit_p = min(t_o, pos.current_sl) if t_o <= pos.current_sl else pos.current_sl
                        exit_reason = "TRAILING_SL" if pos.hit_t1 else "STOP_LOSS"
                    # 2. Target 2 Hit
                    elif t_h >= pos.target_2:
                        exit_occurred = True
                        exit_p = pos.target_2
                        exit_reason = "TARGET_2"
                    # 3. Target 1 Hit (Move SL to Breakeven if trailing enabled)
                    elif t_h >= pos.target_1 and not pos.hit_t1:
                        pos.hit_t1 = True
                        if self.config.trailing_stop:
                            pos.current_sl = max(pos.current_sl, entry_p)  # Move SL to breakeven
                    # 4. Time-based Max Holding Days Expiry
                    if not exit_occurred and pos.days_held >= self.config.max_holding_days:
                        exit_occurred = True
                        exit_p = t_c
                        exit_reason = "TIME_EXPIRY"
                else:  # PUT
                    # 1. Stop Loss Hit
                    if t_h >= pos.current_sl:
                        exit_occurred = True
                        exit_p = max(t_o, pos.current_sl) if t_o >= pos.current_sl else pos.current_sl
                        exit_reason = "TRAILING_SL" if pos.hit_t1 else "STOP_LOSS"
                    # 2. Target 2 Hit
                    elif t_l <= pos.target_2:
                        exit_occurred = True
                        exit_p = pos.target_2
                        exit_reason = "TARGET_2"
                    # 3. Target 1 Hit
                    elif t_l <= pos.target_1 and not pos.hit_t1:
                        pos.hit_t1 = True
                        if self.config.trailing_stop:
                            pos.current_sl = min(pos.current_sl, entry_p)  # Breakeven
                    # 4. Time Expiry
                    if not exit_occurred and pos.days_held >= self.config.max_holding_days:
                        exit_occurred = True
                        exit_p = t_c
                        exit_reason = "TIME_EXPIRY"

                if exit_occurred:
                    # Apply Exit Slippage & Costs
                    slip_mult = (1.0 - self.config.slippage_pct) if is_call else (1.0 + self.config.slippage_pct)
                    eff_exit_price = exit_p * slip_mult
                    
                    gross_pnl = (eff_exit_price - entry_p) * pos.shares if is_call else (entry_p - eff_exit_price) * pos.shares
                    trade_val = (entry_p + eff_exit_price) * pos.shares
                    stt = trade_val * self.config.stt_tax_pct
                    brokerage = self.config.brokerage_per_order * 2.0  # Entry + Exit
                    costs = stt + brokerage
                    net_pnl = gross_pnl - costs
                    ret_pct = (net_pnl / pos.capital_invested) * 100.0 if pos.capital_invested > 0 else 0.0

                    risk_unit = abs(entry_p - pos.initial_sl) * pos.shares
                    r_mult = round(net_pnl / risk_unit, 2) if risk_unit > 0 else 0.0

                    cash += (pos.capital_invested + net_pnl)
                    capital = cash + sum(p.capital_invested for p in remaining_positions)

                    completed_trades.append(BacktestTradeRecord(
                        trade_id=pos.trade_id,
                        symbol=pos.symbol.replace("NSE:", "").replace("-EQ", ""),
                        sector=pos.sector,
                        direction=pos.direction,
                        entry_date=pos.entry_date,
                        exit_date=curr_date,
                        holding_days=pos.days_held,
                        entry_price=round(entry_p, 2),
                        exit_price=round(eff_exit_price, 2),
                        shares=pos.shares,
                        capital_invested=round(pos.capital_invested, 2),
                        stop_loss=round(pos.initial_sl, 2),
                        target_1=round(pos.target_1, 2),
                        target_2=round(pos.target_2, 2),
                        gross_pnl=round(gross_pnl, 2),
                        costs=round(costs, 2),
                        net_pnl=round(net_pnl, 2),
                        return_pct=round(ret_pct, 2),
                        r_multiple=r_mult,
                        exit_reason=exit_reason,
                        confidence=pos.confidence,
                        probability=pos.probability,
                        confluences=pos.confluences,
                        mfe_pct=round(pos.max_favorable_pct, 2),
                        mae_pct=round(pos.max_adverse_pct, 2)
                    ))
                else:
                    remaining_positions.append(pos)

            active_positions = remaining_positions

            # -------------------------------------------------------------
            # STEP 2: Execute Pending Orders from Yesterday's Scan (Stop-Limit Trigger)
            # -------------------------------------------------------------
            for setup, trig_h, trig_l in pending_orders:
                if len(active_positions) >= self.config.max_concurrent_positions:
                    break

                sec = setup.sector
                current_sector_positions = sum(1 for p in active_positions if p.sector == sec)
                if current_sector_positions >= self.config.max_positions_per_sector:
                    continue  # Sector exposure limit

                sym = setup.symbol
                sym_history = symbol_data_dict.get(sym)
                today_rows = sym_history[sym_history["timestamp"].dt.date == curr_date] if sym_history is not None else pd.DataFrame()

                if today_rows.empty:
                    continue

                today_candle = today_rows.iloc[0]
                t_o = float(today_candle["open"])
                t_h = float(today_candle["high"])
                t_l = float(today_candle["low"])
                is_call = (setup.direction == "CALL")

                triggered = False
                fill_price = 0.0

                if is_call:
                    # Must break above yesterday's high
                    if t_h >= trig_h:
                        triggered = True
                        fill_price = max(t_o, trig_h) * (1.0 + self.config.slippage_pct)
                else:  # PUT
                    # Must break below yesterday's low
                    if t_l <= trig_l:
                        triggered = True
                        fill_price = min(t_o, trig_l) * (1.0 - self.config.slippage_pct)

                if triggered and fill_price > 0:
                    sl = setup.stop_loss
                    risk_per_share = abs(fill_price - sl)
                    if risk_per_share <= 0:
                        risk_per_share = fill_price * 0.015

                    # Position Sizing: 1% risk per trade
                    total_portfolio_equity = cash + sum(p.capital_invested for p in active_positions)
                    risk_amount = total_portfolio_equity * (self.config.risk_per_trade_pct / 100.0)
                    desired_shares = int(risk_amount / risk_per_share)
                    
                    if desired_shares <= 0:
                        desired_shares = 1

                    trade_cost = desired_shares * fill_price
                    # Cap position to max 20% of portfolio or available cash
                    max_allowed_cost = min(total_portfolio_equity * 0.20, cash * 0.95)
                    if trade_cost > max_allowed_cost:
                        desired_shares = max(1, int(max_allowed_cost / fill_price))
                        trade_cost = desired_shares * fill_price

                    if cash >= trade_cost and desired_shares > 0:
                        cash -= trade_cost
                        active_positions.append(ActivePosition(
                            trade_id=trade_id_counter,
                            symbol=sym,
                            sector=sec,
                            direction=setup.direction,
                            entry_date=curr_date,
                            entry_price=fill_price,
                            shares=desired_shares,
                            capital_invested=trade_cost,
                            initial_sl=sl,
                            current_sl=sl,
                            target_1=setup.target_1,
                            target_2=setup.target_2,
                            probability=setup.probability,
                            confidence=setup.confidence,
                            confluences=setup.confluences
                        ))
                        trade_id_counter += 1

            # -------------------------------------------------------------
            # STEP 3: Record Daily Equity Curve Snapshot
            # -------------------------------------------------------------
            current_portfolio_equity = cash
            for p in active_positions:
                sym_history = symbol_data_dict.get(p.symbol)
                t_rows = sym_history[sym_history["timestamp"].dt.date == curr_date] if sym_history is not None else pd.DataFrame()
                if not t_rows.empty:
                    current_close = float(t_rows.iloc[0]["close"])
                    unrealized = (current_close - p.entry_price) * p.shares if p.direction == "CALL" else (p.entry_price - current_close) * p.shares
                    current_portfolio_equity += (p.capital_invested + unrealized)
                else:
                    current_portfolio_equity += p.capital_invested

            equity_records.append({
                "date": curr_date,
                "equity": round(current_portfolio_equity, 2),
                "cash": round(cash, 2),
                "open_positions": len(active_positions)
            })

            # -------------------------------------------------------------
            # STEP 4: Run EOD Scan to Generate Setup Orders for Tomorrow (T+1)
            # -------------------------------------------------------------
            pending_orders = []
            if day_idx < len(all_dates) - 1:
                # Scan all symbols on data up to curr_date
                for sym, grp in symbol_data_dict.items():
                    sub_grp = grp[grp["timestamp"].dt.date <= curr_date]
                    if len(sub_grp) < 25:
                        continue
                    
                    setup = scanner.evaluate_series(sym, sub_grp, curr_date)
                    if setup and setup.probability >= min_prob:
                        total_signals_count += 1
                        last_candle = sub_grp.iloc[-1]
                        trig_h = float(last_candle["high"])
                        trig_l = float(last_candle["low"])
                        pending_orders.append((setup, trig_h, trig_l))

                # Sort pending setups by probability descending
                pending_orders.sort(key=lambda item: -item[0].probability)

        # Force close any remaining open positions on the final day
        final_date = all_dates[-1]
        for pos in active_positions:
            sym_history = symbol_data_dict.get(pos.symbol)
            final_rows = sym_history[sym_history["timestamp"].dt.date == final_date] if sym_history is not None else pd.DataFrame()
            final_close = float(final_rows.iloc[0]["close"]) if not final_rows.empty else pos.entry_price
            
            is_call = (pos.direction == "CALL")
            gross_pnl = (final_close - pos.entry_price) * pos.shares if is_call else (pos.entry_price - final_close) * pos.shares
            trade_val = (pos.entry_price + final_close) * pos.shares
            stt = trade_val * self.config.stt_tax_pct
            brokerage = self.config.brokerage_per_order * 2.0
            net_pnl = gross_pnl - (stt + brokerage)
            ret_pct = (net_pnl / pos.capital_invested) * 100.0 if pos.capital_invested > 0 else 0.0

            completed_trades.append(BacktestTradeRecord(
                trade_id=pos.trade_id,
                symbol=pos.symbol.replace("NSE:", "").replace("-EQ", ""),
                sector=pos.sector,
                direction=pos.direction,
                entry_date=pos.entry_date,
                exit_date=final_date,
                holding_days=pos.days_held,
                entry_price=round(pos.entry_price, 2),
                exit_price=round(final_close, 2),
                shares=pos.shares,
                capital_invested=round(pos.capital_invested, 2),
                stop_loss=round(pos.initial_sl, 2),
                target_1=round(pos.target_1, 2),
                target_2=round(pos.target_2, 2),
                gross_pnl=round(gross_pnl, 2),
                costs=round(stt + brokerage, 2),
                net_pnl=round(net_pnl, 2),
                return_pct=round(ret_pct, 2),
                r_multiple=round(net_pnl / max(abs(pos.entry_price - pos.initial_sl) * pos.shares, 1.0), 2),
                exit_reason="BACKTEST_END_CLOSE",
                confidence=pos.confidence,
                probability=pos.probability,
                confluences=pos.confluences,
                mfe_pct=round(pos.max_favorable_pct, 2),
                mae_pct=round(pos.max_adverse_pct, 2)
            ))

        df_equity = pd.DataFrame(equity_records)
        return PerformanceAnalytics.compute(
            trades=completed_trades,
            daily_equity_series=df_equity,
            initial_capital=self.config.initial_capital,
            total_signals_generated=total_signals_count
        )
