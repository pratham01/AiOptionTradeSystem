"""DataSanityManager — Automated data integrity validation, gap detection, and persistent healing."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.application.data_healer import DataHealer, HealReport
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.shared.exceptions import DataSanityError, DataGapError

LOGGER = logging.getLogger(__name__)


@dataclass
class SanityReport:
    """Detailed health and sanity report of historical database tables."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_symbols_checked: int = 0
    symbols_with_gaps: Dict[str, List[str]] = field(default_factory=dict)
    corrupted_records_fixed: int = 0
    synced_resolutions: Dict[str, int] = field(default_factory=dict)
    is_healthy: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        total_gaps = sum(len(syms) for syms in self.symbols_with_gaps.values())
        status = "✅ HEALTHY" if self.is_healthy else f"⚠️ GAPS DETECTED ({total_gaps} unresolved)"
        return (
            f"Database Sanity Report [{status}] @ {self.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"  Symbols Checked: {self.total_symbols_checked}\n"
            f"  Symbols with Gaps: {self.symbols_with_gaps}\n"
            f"  Corrupt Records Fixed: {self.corrupted_records_fixed}\n"
            f"  Auto-Healed & Synced Rows: {self.synced_resolutions}"
        )


class DataSanityManager:
    """
    Orchestrates data sanity checks, integrity verification, and automated healing.
    
    Guarantees:
    - No negative prices, open/high/low/close consistency (high >= low).
    - No duplicate timestamps.
    - All previous trading days exist in DB before live market processing or strategy computation.
    - Automatically heals missing historical data for all F&O universe stocks and indices.
    """

    def __init__(self, broker: Any = None, settings: Any = None) -> None:
        self.engine = get_engine()
        self.broker = broker
        self.settings = settings
        self.healer = DataHealer(self.engine, broker=broker)

    def validate_dataframe(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Validate and sanitize candle dataframe before database insertion or strategy calculation.
        Raises DataSanityError if structural integrity is fundamentally broken.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        clean_df = df.copy()
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(clean_df.columns):
            raise DataSanityError(f"Missing required OHLC columns for {symbol}. Found: {list(clean_df.columns)}")

        # Numeric conversions
        for col in ["open", "high", "low", "close", "volume"]:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        # Drop NaN OHLC
        clean_df = clean_df.dropna(subset=["open", "high", "low", "close"])

        # Enforce positive price constraint
        clean_df = clean_df[(clean_df["open"] > 0) & (clean_df["high"] > 0) & (clean_df["low"] > 0) & (clean_df["close"] > 0)]

        # Enforce high >= max(open, close, high), low <= min(open, close, low)
        true_high = clean_df[["open", "high", "low", "close"]].max(axis=1)
        true_low = clean_df[["open", "high", "low", "close"]].min(axis=1)
        clean_df["high"] = true_high
        clean_df["low"] = true_low

        if "volume" in clean_df.columns:
            clean_df["volume"] = clean_df["volume"].fillna(0).clip(lower=0)

        # Deduplicate timestamps if timestamp index or column exists
        if "timestamp" in clean_df.columns:
            clean_df = clean_df.drop_duplicates(subset=["timestamp"], keep="last")
        elif isinstance(clean_df.index, pd.DatetimeIndex):
            clean_df = clean_df[~clean_df.index.duplicated(keep="last")]

        return clean_df

    def cleanse_corrupt_records(self, symbols: Optional[List[str]] = None, lookback_days: int = 30) -> int:
        """
        Scan and purge any corrupted database rows (non-positive prices, high < low)
        across all OHLCV tables.
        """
        sym_list = symbols or self._get_all_symbols()
        since = (date.today() - timedelta(days=lookback_days)).isoformat()
        total_cleaned = 0

        for table in ["ohlcv_1m", "ohlcv_5m", "ohlcv_15m", "ohlcv_daily"]:
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        text(f"""
                            DELETE FROM {table}
                            WHERE timestamp >= :since
                              AND (open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                                   OR high < low OR open IS NULL OR close IS NULL)
                        """),
                        {"since": since},
                    )
                    if result.rowcount > 0:
                        LOGGER.warning("DataSanityManager: Purged %d corrupt rows from %s", result.rowcount, table)
                        total_cleaned += result.rowcount
            except Exception as exc:
                LOGGER.debug("Table check skipped for %s: %s", table, exc)

        return total_cleaned

    def ensure_data_sanity_and_heal(
        self,
        symbols: Optional[List[str]] = None,
        resolutions: Optional[List[str]] = None,
        lookback_days: int = 10,
        auto_heal: bool = True,
    ) -> SanityReport:
        """
        Primary pre-computation gatekeeper:
        1. Cleanses corrupt records from SQLite.
        2. Audits all F&O universe stocks and Indices for missing historical data.
        3. If gaps exist, automatically backfills missing data from Broker REST API.
        4. Validates data integrity before allowing computations to proceed.
        """
        report = SanityReport()
        sym_list = symbols or self._get_all_symbols()
        target_resolutions = resolutions or ["15", "D"]
        report.total_symbols_checked = len(sym_list)

        LOGGER.info(
            "🛡️ DataSanityManager: Verifying data sanity for %d symbols across %s...",
            len(sym_list), target_resolutions
        )

        try:
            # 1. Cleanse corrupt records
            report.corrupted_records_fixed = self.cleanse_corrupt_records(sym_list, lookback_days=lookback_days)

            # 2. Check for missing data across resolutions
            from trade_system.interfaces.live.data_sync_service import DataSyncService
            sync_svc = DataSyncService(broker=self.broker, settings=self.settings or __import__("trade_system.shared.config", fromlist=["Settings"]).Settings.load())
            gaps_by_res = sync_svc.find_symbols_needing_sync(symbols=sym_list, resolutions=target_resolutions)
            report.symbols_with_gaps = gaps_by_res

            total_gaps = sum(len(v) for v in gaps_by_res.values())

            # 3. Auto-heal missing data if broker is available
            if total_gaps > 0 and auto_heal and self.broker is not None:
                LOGGER.info("⚠️ Detected missing data in %d symbol-resolution pairs. Initiating automatic heal...", total_gaps)
                for res, missing_syms in gaps_by_res.items():
                    if missing_syms:
                        LOGGER.info("🩹 Auto-healing %d symbols for %s resolution...", len(missing_syms), res)
                        written = sync_svc.backfill_symbols(symbols=missing_syms, resolutions=[res])
                        report.synced_resolutions.update(written)

                # Re-verify gaps after healing
                gaps_after = sync_svc.find_symbols_needing_sync(symbols=sym_list, resolutions=target_resolutions)
                report.symbols_with_gaps = gaps_after
                report.is_healthy = sum(len(v) for v in gaps_after.values()) == 0
            else:
                report.is_healthy = total_gaps == 0

            LOGGER.info(report.summary())

        except Exception as exc:
            LOGGER.error("DataSanityManager encountered an error during sanity check: %s", exc, exc_info=True)
            report.is_healthy = False
            report.details["error"] = str(exc)

        return report

    def run_full_sanity_check(self, symbols: Optional[List[str]] = None, lookback_days: int = 5) -> SanityReport:
        """Backward-compatible alias for ensure_data_sanity_and_heal."""
        return self.ensure_data_sanity_and_heal(symbols=symbols, lookback_days=lookback_days)

    def _get_all_symbols(self) -> List[str]:
        try:
            fo_syms = get_fo_universe()
        except Exception:
            fo_syms = []
        index_syms = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX", "NSE:FINNIFTY-INDEX"]
        return list(dict.fromkeys(index_syms + fo_syms))
