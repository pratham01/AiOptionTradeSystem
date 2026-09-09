"""
Modular Multi-Timeframe Quantitative Backtesting Framework.

Provides:
1. Multi-timeframe data retrieval (SQLite DB -> CSV Catalog -> Broker API Fallback).
2. Pluggable multi-indicator engine.
3. Configurable Stop Loss and Target models (Fixed %, ATR, Swing, Trailing, Breakeven, Partial Tiers).
4. Duration and time-in-trade management (Intraday auto-square-off, max bars, MFE/MAE tracking).
5. Comprehensive quantitative statistics (Profit factor, win rate, expectancy, drawdown, Sharpe, Sortino).
6. Multi-format reporting and visualization exports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from trade_system.domains.analysis.application.backtesting.indicators_library import (
    apply_indicator_suite,
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    calculate_supertrend,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_vwap,
)

LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Configurations
# ─────────────────────────────────────────────────────────────────────────────

class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class EntrySignal:
    """Explicit entry signal structure with directional intent and custom SL/TP levels."""
    side: TradeSide
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    confidence: float = 80.0
    notes: str = ""


class StopLossType(str, Enum):
    PERCENT = "PERCENT"           # e.g., 1.5% below entry
    ATR = "ATR"                   # e.g., 1.5x ATR below entry
    SWING = "SWING"               # lowest low (or highest high) of last N bars
    TRAILING_ATR = "TRAILING_ATR" # trails price at N x ATR
    SUPERTREND = "SUPERTREND"     # trailing with SuperTrend line


class TargetType(str, Enum):
    PERCENT = "PERCENT"           # e.g., 3.0% above entry
    RR_RATIO = "RR_RATIO"         # e.g., 1:2 or 1:3 of initial risk
    ATR = "ATR"                   # e.g., 2.5x ATR above entry
    INDICATOR_REVERSAL = "INDICATOR_REVERSAL" # exit when indicator condition reverses


class ExitReason(str, Enum):
    TARGET = "TARGET"
    TARGET_PARTIAL = "TARGET_PARTIAL"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    DURATION_MAX_BARS = "DURATION_MAX_BARS"
    INTRADAY_CUTOFF = "INTRADAY_CUTOFF"
    INDICATOR_EXIT = "INDICATOR_EXIT"
    END_OF_DATA = "END_OF_DATA"

    # Convenient aliases
    TARGET_HIT = "TARGET"
    STOP_LOSS_HIT = "STOP_LOSS"
    BREAKEVEN_HIT = "BREAKEVEN_STOP"
    MAX_BARS_TIMEOUT = "DURATION_MAX_BARS"



@dataclass
class StopLossConfig:
    sl_type: StopLossType = StopLossType.ATR
    value: float = 1.5              # 1.5% or 1.5 ATR
    swing_bars: int = 5             # lookback bars if SWING
    breakeven_at_rr: Optional[float] = 1.0 # Move SL to entry when trade reaches 1.0 R
    trailing_trigger_rr: Optional[float] = None # Start trailing after 1.5 R


@dataclass
class TargetConfig:
    target_type: TargetType = TargetType.RR_RATIO
    value: float = 2.0              # 2.0 R:R or 2.0% or 2.0x ATR
    partial_exit_pct: float = 0.0   # e.g., 0.5 (50% book at T1, remainder trails)
    target_2_value: Optional[float] = None


@dataclass
class DurationConfig:
    max_bars: Optional[int] = 50           # Exit after N bars if neither SL nor TP hit
    intraday_cutoff_time: Optional[dtime] = dtime(15, 15) # Auto square-off at 15:15 IST
    max_calendar_days: Optional[int] = None # Swing duration timeout


@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str = "15"           # "1", "3", "5", "15", "30", "60", "D"
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    initial_capital: float = 100_000.0
    position_sizing: str = "CAPITAL" # "CAPITAL" (fixed INR per trade) or "FIXED_SHARES" or "RISK_PCT"
    position_size_value: float = 100_000.0 # INR per trade or risk % or share count
    slippage_pct: float = 0.05      # 0.05% slippage per fill
    brokerage_cost_per_trade: float = 20.0 # ₹20 per trade round-trip STT/brokerage


# ─────────────────────────────────────────────────────────────────────────────
# Trade Records & Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulatedTrade:
    trade_id: int
    symbol: str
    side: TradeSide
    entry_index: int
    entry_time: datetime
    entry_price: float
    shares: int
    capital_invested: float
    initial_stop_loss: float
    current_stop_loss: float
    target_price: float
    initial_risk_inr: float
    target_2_price: Optional[float] = None

    # Exit fields
    exit_index: Optional[int] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    bars_held: int = 0
    duration_minutes: float = 0.0

    # PnL & Excursions
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    r_multiple: float = 0.0
    mfe_pct: float = 0.0            # Max Favorable Excursion %
    mae_pct: float = 0.0            # Max Adverse Excursion %
    breakeven_activated: bool = False
    notes: str = ""
    market_regime: str = ""
    vix_at_entry: float = 0.0

    def is_open(self) -> bool:
        return self.exit_time is None



@dataclass
class BacktestResult:
    config: BacktestConfig
    candles_count: int
    trades: List[SimulatedTrade]
    trades_df: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict[str, Any]

    def summary_terminal(self) -> str:
        from trade_system.domains.analysis.application.backtesting.backtest_reporter import (
            BacktestReporter,
        )
        return BacktestReporter.generate_ascii_summary(self)

    def summary_markdown(self, title: str = "Backtest Performance Report") -> str:
        from trade_system.domains.analysis.application.backtesting.backtest_reporter import (
            BacktestReporter,
        )
        return BacktestReporter.generate_markdown_report(self, title=title)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Timeframe Data Service
# ─────────────────────────────────────────────────────────────────────────────

class TimeframeDataService:
    """
    Robust multi-timeframe data fetcher:
    Priority:
    1. SQLite Database (`trade_system.db`)
    2. Local CSV Data Catalog (`data/fo_historical`)
    3. Fyers Broker Client (on-demand API fallback + automatic cache)
    """

    SUPPORTED_RESOLUTIONS = ["1", "3", "5", "15", "30", "60", "D"]

    def __init__(self, db_path: Optional[Path] = None, data_dir: Optional[Path] = None, broker: Any = None):
        try:
            from trade_system.shared.config import Settings
            settings = Settings.load()
            default_db = settings.data_dir / "trade_system.db"
            default_data = settings.data_dir / "fo_historical"
        except Exception:
            root_dir = Path(__file__).resolve().parents[6]
            default_db = root_dir / "data" / "trade_system.db"
            default_data = root_dir / "data" / "fo_historical"

        self.db_path = db_path or default_db
        self.data_dir = data_dir or default_data
        self.broker = broker
        self._init_session_factory()

    def _init_session_factory(self):
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        except Exception as e:
            LOGGER.warning("SQLite engine initialization failed: %s", e)
            self.SessionLocal = None

    @staticmethod
    def normalize_resolution(res: str) -> str:
        r = str(res).strip().upper()
        if r in ["1", "1M", "1MIN"]:
            return "1"
        if r in ["3", "3M", "3MIN"]:
            return "3"
        if r in ["5", "5M", "5MIN"]:
            return "5"
        if r in ["15", "15M", "15MIN"]:
            return "15"
        if r in ["30", "30M", "30MIN"]:
            return "30"
        if r in ["60", "60M", "1H", "60MIN"]:
            return "60"
        if r in ["D", "DAY", "DAILY"]:
            return "D"
        return res

    def get_data(
        self,
        symbol: str,
        timeframe: str = "15",
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> pd.DataFrame:
        res = self.normalize_resolution(timeframe)
        from_dt = datetime.combine(from_date, dtime.min) if from_date else None
        to_dt = datetime.combine(to_date, dtime.max) if to_date else None

        df = pd.DataFrame()

        # 1. Check SQLite database
        if self.SessionLocal:
            try:
                from trade_system.domains.market_data.infrastructure.database.repository import (
                    get_market_data,
                )
                with self.SessionLocal() as session:
                    records = get_market_data(
                        session=session,
                        symbol=symbol,
                        resolution=res,
                        from_date=from_dt,
                        to_date=to_dt,
                    )
                    if records:
                        data_rows = [
                            {
                                "timestamp": r.timestamp,
                                "open": float(r.open),
                                "high": float(r.high),
                                "low": float(r.low),
                                "close": float(r.close),
                                "volume": float(r.volume or 0),
                            }
                            for r in records
                        ]
                        df = pd.DataFrame(data_rows)
                        LOGGER.info("Retrieved %d %s bars for %s from SQLite DB", len(df), res, symbol)
            except Exception as e:
                LOGGER.debug("SQLite read error for %s (%s): %s", symbol, res, e)

        # 2. Check CSV data catalog if SQLite empty or missing
        if df.empty and self.data_dir.exists():
            df = self._read_from_csv(symbol, res, from_date, to_date)
            if not df.empty:
                LOGGER.info("Retrieved %d %s bars for %s from CSV Catalog", len(df), res, symbol)

        # 3. Check Broker API Fallback if still empty
        if df.empty and self.broker:
            df = self._fetch_from_broker(symbol, res, from_date, to_date)

        # 4. If target resolution is missing but 1m or 5m data exists, resample
        if df.empty and res in ["3", "5", "15", "30", "60"]:
            df_base = self.get_data(symbol, timeframe="1", from_date=from_date, to_date=to_date)
            if not df_base.empty:
                df = self.resample_candles(df_base, target_minutes=int(res))
                LOGGER.info("Resampled %d bars from 1m to %sm for %s", len(df), res, symbol)

        if df.empty:
            LOGGER.warning("No data found for %s (%s) from %s to %s", symbol, res, from_date, to_date)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Format and clean
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _read_from_csv(
        self, symbol: str, res: str, from_date: Optional[date], to_date: Optional[date]
    ) -> pd.DataFrame:
        clean_sym = symbol.replace(":", "_").replace("-", "_")
        candidates = list(self.data_dir.glob(f"{clean_sym}_{res}*.csv")) + list(
            self.data_dir.glob(f"*{clean_sym}*.csv")
        )
        if not candidates:
            return pd.DataFrame()

        dfs = []
        for p in candidates:
            try:
                sub = pd.read_csv(p)
                if "timestamp" in sub.columns and "close" in sub.columns:
                    dfs.append(sub)
            except Exception:
                pass
        if not dfs:
            return pd.DataFrame()

        comb = pd.concat(dfs, ignore_index=True)
        comb["timestamp"] = pd.to_datetime(comb["timestamp"])
        if from_date:
            comb = comb[comb["timestamp"].dt.date >= from_date]
        if to_date:
            comb = comb[comb["timestamp"].dt.date <= to_date]
        return comb

    def _fetch_from_broker(
        self, symbol: str, res: str, from_date: Optional[date], to_date: Optional[date]
    ) -> pd.DataFrame:
        try:
            f_str = from_date.isoformat() if from_date else (date.today() - timedelta(days=30)).isoformat()
            t_str = to_date.isoformat() if to_date else date.today().isoformat()
            LOGGER.info("Fetching on-demand broker candles for %s (%s) [%s to %s]", symbol, res, f_str, t_str)
            df = self.broker.fetch_history(symbol, res, f_str, t_str)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            LOGGER.error("Broker fallback fetch error: %s", e)
            return pd.DataFrame()

    @staticmethod
    def resample_candles(df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
        if df.empty:
            return df
        work = df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        work = work.set_index("timestamp").sort_index()

        rule = f"{target_minutes}min"
        resampled = work.resample(rule, closed="left", label="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna().reset_index()
        return resampled


# ─────────────────────────────────────────────────────────────────────────────
# 2. Backtest Simulation Engine
# ─────────────────────────────────────────────────────────────────────────────

class ModularBacktester:
    """
    Core event-driven simulation engine.
    Applies multiple indicators, checks entry rules, tracks active positions,
    evaluates Stop Loss & Targets, monitors duration, and generates rich statistics.
    """

    def __init__(
        self,
        config: BacktestConfig,
        sl_config: Optional[StopLossConfig] = None,
        target_config: Optional[TargetConfig] = None,
        duration_config: Optional[DurationConfig] = None,
        data_service: Optional[TimeframeDataService] = None,
    ):
        self.config = config
        self.sl_config = sl_config or StopLossConfig()
        self.target_config = target_config or TargetConfig()
        self.duration_config = duration_config or DurationConfig()
        self.data_service = data_service or TimeframeDataService()

    def run(
        self,
        entry_signal_fn: Callable[[pd.DataFrame, int], Optional[TradeSide]],
        exit_signal_fn: Optional[Callable[[pd.DataFrame, int, SimulatedTrade], bool]] = None,
        custom_candles: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Executes backtest with user-provided entry/exit logic.
        `entry_signal_fn(df, i)` receives DataFrame and current bar index `i`.
        Returns TradeSide.LONG, TradeSide.SHORT, or None.
        """
        # 1. Fetch data
        if custom_candles is not None and not custom_candles.empty:
            candles = custom_candles.copy()
        else:
            candles = self.data_service.get_data(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                from_date=self.config.from_date,
                to_date=self.config.to_date,
            )

        if candles.empty or len(candles) < 20:
            LOGGER.warning("Insufficient candles (%d) to run backtest.", len(candles))
            return self._empty_result(candles)

        # 2. Apply indicator suite (with VIX data if available)
        if "adx" not in candles.columns:
            vix_df = None
            try:
                vix_df = self.data_service.get_data(
                    symbol="NSE:INDIAVIX-INDEX",
                    timeframe=self.config.timeframe,
                    from_date=self.config.from_date,
                    to_date=self.config.to_date,
                )
            except Exception as e:
                LOGGER.debug("VIX data loading error: %s", e)
            df = apply_indicator_suite(candles, vix_df=vix_df)
        else:
            df = candles

        # 3. Simulate execution bar-by-bar
        trades: List[SimulatedTrade] = []
        open_trade: Optional[SimulatedTrade] = None
        equity_records: List[Dict[str, Any]] = []
        current_equity = self.config.initial_capital
        trade_counter = 1

        n_bars = len(df)
        for i in range(1, n_bars):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            bar_time: datetime = pd.to_datetime(row["timestamp"]).to_pydatetime()
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])
            bar_atr = float(row["atr"]) if not np.isnan(row["atr"]) else bar_close * 0.01

            # A. Manage Active Trade
            if open_trade is not None:
                open_trade.bars_held += 1
                open_trade.duration_minutes = (bar_time - open_trade.entry_time).total_seconds() / 60.0

                # Track MFE & MAE
                if open_trade.side == TradeSide.LONG:
                    run_up = (bar_high - open_trade.entry_price) / open_trade.entry_price * 100.0
                    drawdown = (open_trade.entry_price - bar_low) / open_trade.entry_price * 100.0
                    open_trade.mfe_pct = max(open_trade.mfe_pct, run_up)
                    open_trade.mae_pct = max(open_trade.mae_pct, drawdown)
                else:
                    run_up = (open_trade.entry_price - bar_low) / open_trade.entry_price * 100.0
                    drawdown = (bar_high - open_trade.entry_price) / open_trade.entry_price * 100.0
                    open_trade.mfe_pct = max(open_trade.mfe_pct, run_up)
                    open_trade.mae_pct = max(open_trade.mae_pct, drawdown)

                # Check Breakeven SL Adjustment
                if self.sl_config.breakeven_at_rr and not open_trade.breakeven_activated:
                    be_threshold = open_trade.entry_price + (
                        open_trade.initial_risk_inr / open_trade.shares * self.sl_config.breakeven_at_rr
                        if open_trade.side == TradeSide.LONG
                        else -(open_trade.initial_risk_inr / open_trade.shares * self.sl_config.breakeven_at_rr)
                    )
                    if (open_trade.side == TradeSide.LONG and bar_high >= be_threshold) or (
                        open_trade.side == TradeSide.SHORT and bar_low <= be_threshold
                    ):
                        open_trade.current_stop_loss = open_trade.entry_price
                        open_trade.breakeven_activated = True

                # Trailing ATR SL check
                if self.sl_config.sl_type == StopLossType.TRAILING_ATR:
                    if open_trade.side == TradeSide.LONG:
                        new_trail = bar_close - (bar_atr * self.sl_config.value)
                        if new_trail > open_trade.current_stop_loss:
                            open_trade.current_stop_loss = round(new_trail, 2)
                    else:
                        new_trail = bar_close + (bar_atr * self.sl_config.value)
                        if new_trail < open_trade.current_stop_loss:
                            open_trade.current_stop_loss = round(new_trail, 2)

                # Check Exits: Target, Stop Loss, Cutoff, or Duration
                exit_reason: Optional[ExitReason] = None
                exit_price: float = 0.0

                if open_trade.side == TradeSide.LONG:
                    if bar_high >= open_trade.target_price:
                        exit_reason = ExitReason.TARGET_HIT
                        exit_price = open_trade.target_price
                    elif bar_low <= open_trade.current_stop_loss:
                        exit_reason = (
                            ExitReason.BREAKEVEN_HIT
                            if open_trade.breakeven_activated
                            else ExitReason.STOP_LOSS_HIT
                        )
                        exit_price = min(bar_open, open_trade.current_stop_loss)
                else:  # SHORT
                    if bar_low <= open_trade.target_price:
                        exit_reason = ExitReason.TARGET_HIT
                        exit_price = open_trade.target_price
                    elif bar_high >= open_trade.current_stop_loss:
                        exit_reason = (
                            ExitReason.BREAKEVEN_HIT
                            if open_trade.breakeven_activated
                            else ExitReason.STOP_LOSS_HIT
                        )
                        exit_price = max(bar_open, open_trade.current_stop_loss)

                # Intraday Cutoff check (15:15 IST)
                is_intraday = self.config.timeframe not in ["D", "DAY", "DAILY"]
                if not exit_reason and is_intraday and self.duration_config.intraday_cutoff_time:
                    if bar_time.time() >= self.duration_config.intraday_cutoff_time:
                        exit_reason = ExitReason.INTRADAY_CUTOFF
                        exit_price = bar_close

                # Max Bars timeout
                if not exit_reason and self.duration_config.max_bars:
                    if open_trade.bars_held >= self.duration_config.max_bars:
                        exit_reason = ExitReason.MAX_BARS_TIMEOUT
                        exit_price = bar_close

                if exit_reason:
                    open_trade.exit_index = i
                    open_trade.exit_time = bar_time
                    # Add exit slippage
                    slip = exit_price * (self.config.slippage_pct / 100.0)
                    open_trade.exit_price = (
                        round(exit_price - slip, 2)
                        if open_trade.side == TradeSide.LONG
                        else round(exit_price + slip, 2)
                    )
                    open_trade.exit_reason = exit_reason

                    if open_trade.side == TradeSide.LONG:
                        open_trade.gross_pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.shares
                    else:
                        open_trade.gross_pnl = (open_trade.entry_price - open_trade.exit_price) * open_trade.shares

                    open_trade.costs = self.config.brokerage_cost_per_trade
                    open_trade.net_pnl = round(open_trade.gross_pnl - open_trade.costs, 2)
                    open_trade.return_pct = round((open_trade.net_pnl / open_trade.capital_invested) * 100.0, 2)
                    open_trade.r_multiple = round(open_trade.net_pnl / open_trade.initial_risk_inr, 2) if open_trade.initial_risk_inr > 0 else 0.0

                    current_equity += open_trade.net_pnl
                    trades.append(open_trade)
                    open_trade = None

            # B. Check Entry Signal if no position is open
            if open_trade is None and i < n_bars - 1:
                sig_raw = entry_signal_fn(df, i)
                if sig_raw:
                    sig_notes = ""
                    custom_sl = None
                    custom_tp = None
                    if isinstance(sig_raw, EntrySignal):
                        signal_side = sig_raw.side
                        sig_notes = sig_raw.notes
                        custom_sl = sig_raw.stop_loss
                        custom_tp = sig_raw.target
                    else:
                        signal_side = sig_raw

                    # Check intraday cutoff: do not enter if already past cutoff or within 30 min of cutoff
                    can_enter = True
                    is_intraday = self.config.timeframe not in ["D", "DAY", "DAILY"]
                    if is_intraday and self.duration_config.intraday_cutoff_time:
                        cutoff = self.duration_config.intraday_cutoff_time
                        if bar_time.time() >= dtime(cutoff.hour, max(0, cutoff.minute - 30)):
                            can_enter = False

                    if can_enter:
                        entry_price = bar_close
                        # Add entry slippage
                        slip = entry_price * (self.config.slippage_pct / 100.0)
                        fill_entry_price = entry_price + slip if signal_side == TradeSide.LONG else entry_price - slip

                        # Calculate Stop Loss Price
                        if custom_sl is not None:
                            sl_price = custom_sl
                            sl_dist = abs(fill_entry_price - sl_price)
                        elif self.sl_config.sl_type == StopLossType.PERCENT:
                            sl_dist = fill_entry_price * (self.sl_config.value / 100.0)
                            sl_price = fill_entry_price - sl_dist if signal_side == TradeSide.LONG else fill_entry_price + sl_dist
                        elif self.sl_config.sl_type in [StopLossType.ATR, StopLossType.TRAILING_ATR]:
                            sl_dist = bar_atr * self.sl_config.value
                            sl_price = fill_entry_price - sl_dist if signal_side == TradeSide.LONG else fill_entry_price + sl_dist
                        elif self.sl_config.sl_type == StopLossType.SWING:
                            lookback = max(2, self.sl_config.swing_bars)
                            if signal_side == TradeSide.LONG:
                                sl_price = df["low"].iloc[max(0, i - lookback) : i].min()
                                sl_dist = max(fill_entry_price - sl_price, fill_entry_price * 0.005)
                            else:
                                sl_price = df["high"].iloc[max(0, i - lookback) : i].max()
                                sl_dist = max(sl_price - fill_entry_price, fill_entry_price * 0.005)
                        else:
                            sl_dist = bar_atr * 1.5
                            sl_price = fill_entry_price - sl_dist if signal_side == TradeSide.LONG else fill_entry_price + sl_dist

                        # Calculate Target Price
                        if custom_tp is not None:
                            tp_price = custom_tp
                            tp_dist = abs(tp_price - fill_entry_price)
                        elif self.target_config.target_type == TargetType.RR_RATIO:
                            tp_dist = sl_dist * self.target_config.value
                            tp_price = fill_entry_price + tp_dist if signal_side == TradeSide.LONG else fill_entry_price - tp_dist
                        elif self.target_config.target_type == TargetType.PERCENT:
                            tp_dist = fill_entry_price * (self.target_config.value / 100.0)
                            tp_price = fill_entry_price + tp_dist if signal_side == TradeSide.LONG else fill_entry_price - tp_dist
                        elif self.target_config.target_type == TargetType.ATR:
                            tp_dist = bar_atr * self.target_config.value
                            tp_price = fill_entry_price + tp_dist if signal_side == TradeSide.LONG else fill_entry_price - tp_dist
                        else:
                            tp_dist = sl_dist * 2.0
                            tp_price = fill_entry_price + tp_dist if signal_side == TradeSide.LONG else fill_entry_price - tp_dist

                        # Position Sizing
                        if self.config.position_sizing == "CAPITAL":
                            capital_alloc = min(self.config.position_size_value, current_equity)
                            shares = max(1, int(capital_alloc / fill_entry_price))
                        elif self.config.position_sizing == "RISK_PCT":
                            risk_capital = current_equity * (self.config.position_size_value / 100.0)
                            shares = max(1, int(risk_capital / sl_dist)) if sl_dist > 0 else 1
                            capital_alloc = shares * fill_entry_price
                        else:
                            shares = max(1, int(self.config.position_size_value))
                            capital_alloc = shares * fill_entry_price

                        initial_risk = sl_dist * shares

                        open_trade = SimulatedTrade(
                            trade_id=trade_counter,
                            symbol=self.config.symbol,
                            side=signal_side,
                            entry_index=i,
                            entry_time=bar_time,
                            entry_price=round(fill_entry_price, 2),
                            shares=shares,
                            capital_invested=round(capital_alloc, 2),
                            initial_stop_loss=round(sl_price, 2),
                            current_stop_loss=round(sl_price, 2),
                            target_price=round(tp_price, 2),
                            initial_risk_inr=round(initial_risk, 2),
                            notes=sig_notes,
                            market_regime=str(row.get("market_regime", "")),
                            vix_at_entry=float(row.get("vix_level", 0.0)),
                        )
                        trade_counter += 1

            # Track equity curve point
            unrealized = 0.0
            if open_trade is not None:
                if open_trade.side == TradeSide.LONG:
                    unrealized = (bar_close - open_trade.entry_price) * open_trade.shares
                else:
                    unrealized = (open_trade.entry_price - bar_close) * open_trade.shares

            equity_records.append(
                {
                    "timestamp": bar_time,
                    "close": bar_close,
                    "equity": round(current_equity + unrealized, 2),
                    "cash": round(current_equity, 2),
                    "open_trades": 1 if open_trade else 0,
                }
            )

        # Close open trade on last bar
        if open_trade is not None:
            last_row = df.iloc[-1]
            last_time = pd.to_datetime(last_row["timestamp"]).to_pydatetime()
            last_close = float(last_row["close"])
            open_trade.exit_index = n_bars - 1
            open_trade.exit_time = last_time
            open_trade.exit_price = last_close
            open_trade.exit_reason = ExitReason.END_OF_DATA
            open_trade.gross_pnl = (
                (open_trade.exit_price - open_trade.entry_price) * open_trade.shares
                if open_trade.side == TradeSide.LONG
                else (open_trade.entry_price - open_trade.exit_price) * open_trade.shares
            )
            open_trade.costs = self.config.brokerage_cost_per_trade
            open_trade.net_pnl = round(open_trade.gross_pnl - open_trade.costs, 2)
            open_trade.return_pct = round((open_trade.net_pnl / open_trade.capital_invested) * 100.0, 2)
            open_trade.r_multiple = round(open_trade.net_pnl / open_trade.initial_risk_inr, 2) if open_trade.initial_risk_inr > 0 else 0.0
            current_equity += open_trade.net_pnl
            trades.append(open_trade)

        # 4. Generate DataFrames & Metrics
        trades_df = self._trades_to_dataframe(trades)
        equity_df = pd.DataFrame(equity_records)
        metrics = BacktestStatisticsEngine.calculate_statistics(trades_df, equity_df, self.config.initial_capital)

        return BacktestResult(
            config=self.config,
            candles_count=len(df),
            trades=trades,
            trades_df=trades_df,
            equity_curve=equity_df,
            metrics=metrics,
        )

    def _trades_to_dataframe(self, trades: List[SimulatedTrade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        records = []
        for t in trades:
            records.append(
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "shares": t.shares,
                    "capital": t.capital_invested,
                    "initial_sl": t.initial_stop_loss,
                    "exit_sl": t.current_stop_loss,
                    "target": t.target_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason.value if t.exit_reason else "",
                    "bars_held": t.bars_held,
                    "duration_mins": round(t.duration_minutes, 1),
                    "regime": t.market_regime,
                    "vix": round(t.vix_at_entry, 2),
                    "notes": t.notes,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct,
                    "r_multiple": t.r_multiple,
                    "mfe_pct": round(t.mfe_pct, 2),
                    "mae_pct": round(t.mae_pct, 2),
                }
            )
        return pd.DataFrame(records)

    def _empty_result(self, candles: pd.DataFrame) -> BacktestResult:
        empty_metrics = BacktestStatisticsEngine.empty_metrics(self.config.initial_capital)
        return BacktestResult(
            config=self.config,
            candles_count=len(candles),
            trades=[],
            trades_df=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            metrics=empty_metrics,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantitative Statistics Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestStatisticsEngine:
    """Calculates institutional risk-adjusted performance metrics."""

    @classmethod
    def calculate_statistics(
        cls, trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float
    ) -> Dict[str, Any]:
        if trades_df.empty:
            return cls.empty_metrics(initial_capital)

        total_trades = len(trades_df)
        winning_trades = int((trades_df["net_pnl"] > 0).sum())
        losing_trades = int((trades_df["net_pnl"] < 0).sum())
        breakeven_trades = int((trades_df["net_pnl"] == 0).sum())

        win_rate_pct = round((winning_trades / total_trades) * 100.0, 2)
        loss_rate_pct = round((losing_trades / total_trades) * 100.0, 2)

        gross_profit = float(trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum())
        gross_loss = float(abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum()))
        net_profit = round(gross_profit - gross_loss, 2)

        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        final_equity = round(initial_capital + net_profit, 2)
        total_return_pct = round((net_profit / initial_capital) * 100.0, 2)

        avg_trade_pnl = round(float(trades_df["net_pnl"].mean()), 2)
        avg_win_pnl = round(float(trades_df[trades_df["net_pnl"] > 0]["net_pnl"].mean()), 2) if winning_trades > 0 else 0.0
        avg_loss_pnl = round(float(abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].mean())), 2) if losing_trades > 0 else 0.0

        payoff_ratio = round(avg_win_pnl / avg_loss_pnl, 2) if avg_loss_pnl > 0 else 0.0
        expectancy_inr = round(
            (win_rate_pct / 100.0 * avg_win_pnl) - (loss_rate_pct / 100.0 * avg_loss_pnl), 2
        )
        avg_r_multiple = round(float(trades_df["r_multiple"].mean()), 2)

        # Drawdown calculations
        max_drawdown_inr, max_drawdown_pct, max_dd_bars = cls._calculate_drawdowns(equity_df)

        # Consecutive streaks
        max_cons_wins, max_cons_losses = cls._calculate_streaks(trades_df["net_pnl"].values)

        # Duration stats
        avg_bars_held = round(float(trades_df["bars_held"].mean()), 1)
        avg_duration_mins = round(float(trades_df["duration_mins"].mean()), 1)
        win_bars = round(float(trades_df[trades_df["net_pnl"] > 0]["bars_held"].mean()), 1) if winning_trades > 0 else 0.0
        loss_bars = round(float(trades_df[trades_df["net_pnl"] < 0]["bars_held"].mean()), 1) if losing_trades > 0 else 0.0

        # Excursion stats
        avg_mfe_pct = round(float(trades_df["mfe_pct"].mean()), 2)
        avg_mae_pct = round(float(trades_df["mae_pct"].mean()), 2)

        # Risk-adjusted ratios
        sharpe, sortino, calmar = cls._calculate_risk_adjusted_ratios(equity_df, total_return_pct, max_drawdown_pct)

        # Exit reasons distribution
        exit_breakdown = trades_df["exit_reason"].value_counts().to_dict()

        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "net_profit": net_profit,
            "total_return_pct": total_return_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "breakeven_trades": breakeven_trades,
            "win_rate_pct": win_rate_pct,
            "loss_rate_pct": loss_rate_pct,
            "profit_factor": profit_factor,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_trade_pnl": avg_trade_pnl,
            "avg_win_pnl": avg_win_pnl,
            "avg_loss_pnl": avg_loss_pnl,
            "payoff_ratio": payoff_ratio,
            "expectancy_inr": expectancy_inr,
            "avg_r_multiple": avg_r_multiple,
            "max_drawdown_inr": max_drawdown_inr,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_bars": max_dd_bars,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_consecutive_wins": max_cons_wins,
            "max_consecutive_losses": max_cons_losses,
            "avg_bars_held": avg_bars_held,
            "avg_duration_mins": avg_duration_mins,
            "avg_win_bars": win_bars,
            "avg_loss_bars": loss_bars,
            "avg_mfe_pct": avg_mfe_pct,
            "avg_mae_pct": avg_mae_pct,
            "exit_reasons": exit_breakdown,
        }

    @staticmethod
    def _calculate_drawdowns(equity_df: pd.DataFrame) -> Tuple[float, float, int]:
        if equity_df.empty or "equity" not in equity_df.columns:
            return 0.0, 0.0, 0
        eq = equity_df["equity"].values
        peak = np.maximum.accumulate(eq)
        drawdowns = peak - eq
        drawdown_pcts = (drawdowns / peak) * 100.0

        max_dd_inr = round(float(np.max(drawdowns)), 2)
        max_dd_pct = round(float(np.max(drawdown_pcts)), 2)

        # Max drawdown duration (bars)
        duration = 0
        max_duration = 0
        for dd in drawdowns:
            if dd > 0:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0

        return max_dd_inr, max_dd_pct, max_duration

    @staticmethod
    def _calculate_streaks(pnls: np.ndarray) -> Tuple[int, int]:
        max_w = curr_w = 0
        max_l = curr_l = 0
        for p in pnls:
            if p > 0:
                curr_w += 1
                curr_l = 0
                max_w = max(max_w, curr_w)
            elif p < 0:
                curr_l += 1
                curr_w = 0
                max_l = max(max_l, curr_l)
            else:
                curr_w = 0
                curr_l = 0
        return max_w, max_l

    @staticmethod
    def _calculate_risk_adjusted_ratios(
        equity_df: pd.DataFrame, total_return_pct: float, max_drawdown_pct: float
    ) -> Tuple[float, float, float]:
        if equity_df.empty or len(equity_df) < 5:
            return 0.0, 0.0, 0.0

        returns = equity_df["equity"].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0.0, 0.0, 0.0

        # Annualized factor (assuming ~252 trading days, ~25 bars/day for 15m)
        ann_factor = np.sqrt(252 * 25)
        mean_ret = returns.mean()
        std_ret = returns.std()
        downside_std = returns[returns < 0].std()

        sharpe = round(float((mean_ret / std_ret) * ann_factor), 2)
        sortino = round(float((mean_ret / downside_std) * ann_factor), 2) if downside_std > 0 else 0.0
        calmar = round(total_return_pct / max_drawdown_pct, 2) if max_drawdown_pct > 0 else 0.0

        return sharpe, sortino, calmar

    @classmethod
    def empty_metrics(cls, initial_capital: float) -> Dict[str, Any]:
        return {
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "loss_rate_pct": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_win_pnl": 0.0,
            "avg_loss_pnl": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_inr": 0.0,
            "avg_r_multiple": 0.0,
            "max_drawdown_inr": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_bars": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_bars_held": 0.0,
            "avg_duration_mins": 0.0,
            "avg_win_bars": 0.0,
            "avg_loss_bars": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "exit_reasons": {},
        }
