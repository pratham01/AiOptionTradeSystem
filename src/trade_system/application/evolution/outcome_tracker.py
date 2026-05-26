"""
OutcomeTracker — marks PENDING trades as WIN/LOSS/NEUTRAL at EOD.

At end-of-day, this fetches the actual closing data for suggested trades
and determines whether the trade would have hit target, stopped out, or expired.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from trade_system.infrastructure.database.models import SuggestedTrade
from trade_system.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)

# Win/Loss thresholds in underlying price movement %
WIN_TARGET_PCT = 1.0        # Target considered hit if stock moved >= 1% toward target
LOSS_SL_PCT = 0.7           # SL considered hit if stock moved >= 0.7% against trade


class OutcomeTracker:
    """
    Evaluates pending trade suggestions and marks their outcomes.

    Logic:
    - For CALL suggestions: WIN if high >= target, LOSS if low <= stop_loss, else NEUTRAL
    - For PUT suggestions: WIN if low <= target, LOSS if high >= stop_loss, else NEUTRAL
    - Swing trades check next N days, intraday checks same day only
    """

    def __init__(self, broker: Any = None, engine: Any = None) -> None:
        """
        Args:
            broker: FyersBroker instance (for live data fetching)
            engine: SQLAlchemy engine (uses default if None)
        """
        self.broker = broker
        self.engine = engine or get_engine()

    def check_and_mark_outcomes(self, target_date: str | None = None) -> dict[str, int]:
        """
        Check all PENDING trades and mark outcomes.

        Args:
            target_date: Date string (YYYY-MM-DD). Defaults to yesterday.

        Returns:
            dict with counts: {"wins": N, "losses": N, "neutrals": N, "skipped": N}
        """
        if target_date is None:
            target_date = (date.today() - timedelta(days=1)).isoformat()

        counts = {"wins": 0, "losses": 0, "neutrals": 0, "skipped": 0}

        with Session(self.engine) as session:
            pending = (
                session.query(SuggestedTrade)
                .filter(SuggestedTrade.outcome == "PENDING")
                .filter(SuggestedTrade.date <= target_date)
                .all()
            )

            LOGGER.info("Checking outcomes for %d pending trades up to %s.",
                        len(pending), target_date)

            for trade in pending:
                try:
                    outcome, pnl_pct = self._evaluate_trade(trade, target_date)
                    trade.outcome = outcome
                    trade.actual_pnl_pct = pnl_pct
                    trade.outcome_checked_at = datetime.utcnow()

                    if outcome == "WIN":
                        counts["wins"] += 1
                    elif outcome == "LOSS":
                        counts["losses"] += 1
                    elif outcome == "NEUTRAL":
                        counts["neutrals"] += 1
                    else:
                        counts["skipped"] += 1

                    LOGGER.debug("Trade %s (%s %s) → %s (PnL: %s%%)",
                                 trade.id, trade.symbol, trade.direction,
                                 outcome, round(pnl_pct, 2) if pnl_pct else "N/A")
                except Exception as exc:
                    LOGGER.error("Error evaluating trade %s: %s", trade.id, exc)
                    counts["skipped"] += 1

            session.commit()

        LOGGER.info("Outcome results: %s", counts)
        return counts

    def _evaluate_trade(self, trade: SuggestedTrade, as_of_date: str) -> tuple[str, float | None]:
        """
        Evaluate outcome for a single trade.

        Returns: (outcome_str, pnl_pct or None)
        """
        ohlc = self._fetch_ohlc(trade.symbol, trade.date, as_of_date, trade.horizon)
        if not ohlc:
            return "PENDING", None

        entry = trade.entry_zone_high  # Conservative — use top of entry zone
        target = trade.target
        sl = trade.stop_loss
        direction = trade.direction.upper()

        if direction == "CALL":
            # Bullish — WIN if high hits target, LOSS if low hits SL
            hit_target = any(bar["high"] >= target for bar in ohlc)
            hit_sl = any(bar["low"] <= sl for bar in ohlc)
            last_close = ohlc[-1]["close"]
            pnl_pct = (last_close - entry) / entry * 100.0

        elif direction == "PUT":
            # Bearish — WIN if low hits target, LOSS if high hits SL
            hit_target = any(bar["low"] <= target for bar in ohlc)
            hit_sl = any(bar["high"] >= sl for bar in ohlc)
            last_close = ohlc[-1]["close"]
            pnl_pct = (entry - last_close) / entry * 100.0
        else:
            return "NEUTRAL", None

        # Determine outcome: first event wins (SL hit before target is a LOSS)
        if hit_sl and not hit_target:
            return "LOSS", -abs(sl - entry) / entry * 100.0
        elif hit_target:
            return "WIN", abs(target - entry) / entry * 100.0
        elif abs(pnl_pct) < 0.3:
            return "NEUTRAL", pnl_pct
        elif pnl_pct > 0:
            return "WIN", pnl_pct
        else:
            return "LOSS", pnl_pct

    def _fetch_ohlc(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        horizon: str,
    ) -> list[dict[str, float]]:
        """
        Fetch OHLC bars from broker. Returns list of {open, high, low, close} dicts.
        Falls back to CSV data catalog if broker unavailable.
        """
        if self.broker is None:
            return self._fetch_from_csv(symbol, from_date, to_date)

        try:
            resolution = "D" if horizon == "SWING" else "15"
            df = self.broker.fetch_history(
                symbol=symbol,
                resolution=resolution,
                range_from=from_date,
                range_to=to_date,
            )
            if df is None or df.empty:
                return []
            return df[["open", "high", "low", "close"]].to_dict("records")
        except Exception as exc:
            LOGGER.warning("Broker fetch failed for %s: %s. Trying CSV.", symbol, exc)
            return self._fetch_from_csv(symbol, from_date, to_date)

    def _fetch_from_csv(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> list[dict[str, float]]:
        """Fallback: try to load from local CSV data catalog."""
        try:
            import pandas as pd
            from pathlib import Path

            data_dir = Path("data")
            # Try to find matching daily CSV
            for year in [from_date[:4]]:
                csv_path = data_dir / f"{symbol.replace(':', '_')}_d_{year}.csv"
                if not csv_path.exists():
                    # Try alternate naming
                    sym_clean = symbol.split(":")[-1].replace("-", "_")
                    for f in data_dir.glob(f"*{sym_clean}*_d_{year}*.csv"):
                        csv_path = f
                        break

                if csv_path.exists():
                    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                    df = df[
                        (df["timestamp"].dt.date >= pd.to_datetime(from_date).date()) &
                        (df["timestamp"].dt.date <= pd.to_datetime(to_date).date())
                    ]
                    if not df.empty:
                        return df[["open", "high", "low", "close"]].to_dict("records")
        except Exception as exc:
            LOGGER.debug("CSV fallback failed for %s: %s", symbol, exc)
        return []
