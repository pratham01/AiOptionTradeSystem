"""
TradeLogger — logs every suggested trade to SQLite for evolution tracking.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from trade_system.infrastructure.database.models import SuggestedTrade, FeatureSnapshot
from trade_system.infrastructure.database.connection import get_engine
from trade_system.core import TradeSuggestion

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

LOGGER = logging.getLogger(__name__)


class TradeLogger:
    """
    Persists every TradeSuggestion to the database so the evolution
    engine can later match outcomes to features.
    """

    def __init__(self, engine: "Engine | None" = None) -> None:
        self.engine = engine or get_engine()
        # Ensure tables exist
        from trade_system.infrastructure.database.models import Base
        Base.metadata.create_all(self.engine)

    def log(self, suggestion: TradeSuggestion) -> None:
        """Persist a TradeSuggestion and its feature snapshot."""
        with Session(self.engine) as session:
            # Avoid duplicates
            existing = session.get(SuggestedTrade, suggestion.id)
            if existing:
                LOGGER.debug("Trade %s already logged — skipping.", suggestion.id)
                return

            db_trade = SuggestedTrade(
                id=suggestion.id,
                date=suggestion.timestamp.strftime("%Y-%m-%d"),
                symbol=suggestion.symbol,
                direction=suggestion.direction.value,
                horizon=suggestion.horizon.value,
                is_nifty=int(suggestion.is_nifty),
                sector=suggestion.sector,
                entry_zone_low=suggestion.entry_zone_low,
                entry_zone_high=suggestion.entry_zone_high,
                target=suggestion.target,
                stop_loss=suggestion.stop_loss,
                confidence=suggestion.confidence,
                option_direction=suggestion.option_params.direction.value,
                option_strike=suggestion.option_params.suggested_strike,
                option_expiry_type=suggestion.option_params.expiry_type,
                option_atm_offset=suggestion.option_params.atm_offset,
                narrative=suggestion.narrative,
                tags=json.dumps(suggestion.tags),
                outcome=suggestion.outcome.value,
                generated_at=suggestion.timestamp,
            )

            f = suggestion.setup_features
            if f:
                db_feat = FeatureSnapshot(
                    trade_id=suggestion.id,
                    rsi_daily=f.rsi_daily,
                    rsi_hourly=f.rsi_hourly,
                    adx=f.adx,
                    volume_surge=f.volume_surge,
                    atr_pct=f.atr_pct,
                    vix=f.vix,
                    pcr=f.pcr,
                    near_support=int(f.near_support),
                    near_resistance=int(f.near_resistance),
                    is_compressed=int(f.is_compressed),
                    vol_delta_positive=int(f.vol_delta_positive),
                    above_vwap=int(f.above_vwap) if f.above_vwap is not None else None,
                    above_poc=int(f.above_poc) if f.above_poc is not None else None,
                    breakout_type=f.breakout_type,
                    pattern=f.pattern,
                    market_regime=f.market_regime,
                    alignment_score=f.alignment_score,
                )
                session.add(db_feat)
            else:
                LOGGER.warning("Trade %s has no setup features — logging trade only.", suggestion.id)

            session.add(db_trade)
            session.commit()
            LOGGER.info("Logged trade suggestion %s for %s (%s %s).",
                        suggestion.id, suggestion.symbol,
                        suggestion.direction.value, suggestion.horizon.value)

    def log_many(self, suggestions: list[TradeSuggestion]) -> None:
        """Log a list of suggestions."""
        for s in suggestions:
            try:
                self.log(s)
            except Exception as exc:
                LOGGER.error("Failed to log trade %s: %s", s.id, exc)

    def get_pending_trades(self, date: str | None = None) -> list[SuggestedTrade]:
        """Return all trades with PENDING outcome, optionally filtered by date."""
        with Session(self.engine) as session:
            q = session.query(SuggestedTrade).filter(
                SuggestedTrade.outcome == "PENDING"
            )
            if date:
                q = q.filter(SuggestedTrade.date == date)
            return q.all()

    def get_closed_trades(self, lookback_days: int = 30) -> list[SuggestedTrade]:
        """Return all non-pending trades within lookback window."""
        from datetime import date, timedelta
        cutoff = (datetime.utcnow().date() - timedelta(days=lookback_days)).isoformat()
        with Session(self.engine) as session:
            return (
                session.query(SuggestedTrade)
                .filter(SuggestedTrade.outcome != "PENDING")
                .filter(SuggestedTrade.date >= cutoff)
                .order_by(SuggestedTrade.date.desc())
                .all()
            )
