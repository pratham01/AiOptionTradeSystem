"""Database models for trade system."""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class TopGainersSnapshot(Base):
    """Snapshot of top gainers for a specific date."""
    
    __tablename__ = "top_gainers_snapshots"
    
    id = Column(Integer, primary_key=True)
    date = Column(String, unique=True, nullable=False)  # YYYY-MM-DD
    total_stocks_analyzed = Column(Integer, nullable=False)
    stocks_passing_filters = Column(Integer, nullable=False)
    top_n = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to stocks
    stocks = relationship("TopGainersStock", back_populates="snapshot", cascade="all, delete-orphan")
    
    __table_args__ = (Index("idx_top_gainers_date", "date"),)


class TopGainersStock(Base):
    """Individual stock in top gainers snapshot."""
    
    __tablename__ = "top_gainers_stocks"
    
    id = Column(Integer, primary_key=True)
    snapshot_id = Column(Integer, ForeignKey("top_gainers_snapshots.id"), nullable=False)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)
    close = Column(Float, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    prev_close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    change = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # Relationship
    snapshot = relationship("TopGainersSnapshot", back_populates="stocks")
    
    __table_args__ = (
        Index("idx_top_gainers_snapshot", "snapshot_id"),
        Index("idx_top_gainers_symbol", "symbol"),
    )


class FOSelectionSnapshot(Base):
    """Snapshot of FO selection for a specific date."""
    
    __tablename__ = "fo_selection_snapshots"
    
    id = Column(Integer, primary_key=True)
    date = Column(String, unique=True, nullable=False)  # YYYY-MM-DD
    total_stocks_analyzed = Column(Integer, nullable=False)
    stocks_with_data = Column(Integer, nullable=False)
    stocks_passing_filters = Column(Integer, nullable=False)
    top_n = Column(Integer, nullable=False)
    min_score_threshold = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to stocks
    stocks = relationship("FOSelectionStock", back_populates="snapshot", cascade="all, delete-orphan")
    
    __table_args__ = (Index("idx_fo_selection_date", "date"),)


class FOSelectionStock(Base):
    """Individual stock in FO selection snapshot."""
    
    __tablename__ = "fo_selection_stocks"
    
    id = Column(Integer, primary_key=True)
    snapshot_id = Column(Integer, ForeignKey("fo_selection_snapshots.id"), nullable=False)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # Relationship
    snapshot = relationship("FOSelectionSnapshot", back_populates="stocks")
    
    __table_args__ = (
        Index("idx_fo_selection_snapshot", "snapshot_id"),
        Index("idx_fo_selection_symbol", "symbol"),
    )


class FOTopStocksSnapshot(Base):
    """Snapshot of FO top stocks for a specific date."""
    
    __tablename__ = "fo_top_stocks_snapshots"
    
    id = Column(Integer, primary_key=True)
    date = Column(String, unique=True, nullable=False)  # YYYY-MM-DD
    total_stocks_analyzed = Column(Integer, nullable=False)
    stocks_passing_filters = Column(Integer, nullable=False)
    top_n = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to stocks
    stocks = relationship("FOTopStock", back_populates="snapshot", cascade="all, delete-orphan")
    
    __table_args__ = (Index("idx_fo_top_stocks_date", "date"),)


class FOTopStock(Base):
    """Individual stock in FO top stocks snapshot."""
    
    __tablename__ = "fo_top_stocks"
    
    id = Column(Integer, primary_key=True)
    snapshot_id = Column(Integer, ForeignKey("fo_top_stocks_snapshots.id"), nullable=False)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)
    close = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    avg_volume = Column(Float, nullable=False)
    volume_ratio = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    atr = Column(Float, nullable=False)
    volume_divergence = Column(String, nullable=False)
    divergence_strength = Column(Float, nullable=False)
    price_action_score = Column(Float, nullable=False)
    trend_score = Column(Float, nullable=False)
    liquidity_score = Column(Float, nullable=False)
    total_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # Relationship
    snapshot = relationship("FOTopStocksSnapshot", back_populates="stocks")
    
    __table_args__ = (
        Index("idx_fo_top_stocks_snapshot", "snapshot_id"),
        Index("idx_fo_top_stocks_symbol", "symbol"),
    )


class SectorPerformanceSnapshot(Base):
    """Snapshot of sector performance for a specific date."""
    
    __tablename__ = "sector_performance_snapshots"
    
    id = Column(Integer, primary_key=True)
    date = Column(String, unique=True, nullable=False)  # YYYY-MM-DD
    total_stocks_analyzed = Column(Integer, nullable=False)
    stocks_with_sector = Column(Integer, nullable=False)
    top_n = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    top_sectors = relationship("SectorPerformanceSector", back_populates="snapshot", foreign_keys="SectorPerformanceSector.snapshot_id", cascade="all, delete-orphan")
    worst_sectors = relationship("SectorPerformanceSector", back_populates="snapshot_worst", foreign_keys="SectorPerformanceSector.snapshot_id_worst", cascade="all, delete-orphan")
    top_stocks_overall = relationship("SectorPerformanceStock", back_populates="snapshot_top", foreign_keys="SectorPerformanceStock.snapshot_id_top", cascade="all, delete-orphan")
    worst_stocks_overall = relationship("SectorPerformanceStock", back_populates="snapshot_worst", foreign_keys="SectorPerformanceStock.snapshot_id_worst", cascade="all, delete-orphan")
    
    __table_args__ = (Index("idx_sector_perf_date", "date"),)


class SectorPerformanceSector(Base):
    """Sector in sector performance snapshot."""
    
    __tablename__ = "sector_performance_sectors"
    
    id = Column(Integer, primary_key=True)
    snapshot_id = Column(Integer, ForeignKey("sector_performance_snapshots.id"), nullable=True)
    snapshot_id_worst = Column(Integer, ForeignKey("sector_performance_snapshots.id"), nullable=True)
    sector = Column(String, nullable=False)
    stock_count = Column(Integer, nullable=False)
    avg_change_pct = Column(Float, nullable=False)
    is_top_sector = Column(Integer, default=1)  # 1 for top, 0 for worst
    
    # Relationships
    snapshot = relationship("SectorPerformanceSnapshot", back_populates="top_sectors", foreign_keys=[snapshot_id])
    snapshot_worst = relationship("SectorPerformanceSnapshot", back_populates="worst_sectors", foreign_keys=[snapshot_id_worst])
    top_stocks = relationship("SectorPerformanceStock", back_populates="sector", foreign_keys="SectorPerformanceStock.sector_id", cascade="all, delete-orphan")
    worst_stocks = relationship("SectorPerformanceStock", back_populates="sector_worst", foreign_keys="SectorPerformanceStock.sector_id_worst", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_sector_perf_snapshot", "snapshot_id"),
        Index("idx_sector_perf_snapshot_worst", "snapshot_id_worst"),
    )


class SectorPerformanceStock(Base):
    """Stock in sector performance snapshot."""
    
    __tablename__ = "sector_performance_stocks"
    
    id = Column(Integer, primary_key=True)
    snapshot_id_top = Column(Integer, ForeignKey("sector_performance_snapshots.id"), nullable=True)
    snapshot_id_worst = Column(Integer, ForeignKey("sector_performance_snapshots.id"), nullable=True)
    sector_id = Column(Integer, ForeignKey("sector_performance_sectors.id"), nullable=True)
    sector_id_worst = Column(Integer, ForeignKey("sector_performance_sectors.id"), nullable=True)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)
    sector = Column(String, nullable=False)
    close = Column(Float, nullable=False)
    prev_close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    change = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    is_top_stock = Column(Integer, default=1)  # 1 for top, 0 for worst
    
    # Relationships
    snapshot_top = relationship("SectorPerformanceSnapshot", back_populates="top_stocks_overall", foreign_keys=[snapshot_id_top])
    snapshot_worst = relationship("SectorPerformanceSnapshot", back_populates="worst_stocks_overall", foreign_keys=[snapshot_id_worst])
    sector = relationship("SectorPerformanceSector", back_populates="top_stocks", foreign_keys=[sector_id])
    sector_worst = relationship("SectorPerformanceSector", back_populates="worst_stocks", foreign_keys=[sector_id_worst])
    
    __table_args__ = (
        Index("idx_sector_perf_stock_snapshot_top", "snapshot_id_top"),
        Index("idx_sector_perf_stock_snapshot_worst", "snapshot_id_worst"),
        Index("idx_sector_perf_stock_symbol", "symbol"),
    )


# ---------------------------------------------------------------------------
# Evolution Engine Tables
# ---------------------------------------------------------------------------

class SuggestedTrade(Base):
    """
    Every trade suggestion produced by the TradeOrchestrator.
    Used by the self-evolution engine to track what was suggested and what happened.
    """
    __tablename__ = "suggested_trades"

    id = Column(String, primary_key=True)           # UUID
    date = Column(String, nullable=False)            # YYYY-MM-DD
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)       # "CALL" or "PUT"
    horizon = Column(String, nullable=False)         # "INTRADAY" or "SWING"
    is_nifty = Column(Integer, default=0)
    sector = Column(String, default="Unknown")

    # Pricing
    entry_zone_low = Column(Float, nullable=False)
    entry_zone_high = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    # Option params
    option_direction = Column(String, nullable=True)
    option_strike = Column(Float, nullable=True)
    option_expiry_type = Column(String, nullable=True)
    option_atm_offset = Column(Integer, nullable=True)

    # Agent reasoning
    narrative = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)              # JSON list

    # Outcome (filled by OutcomeTracker at EOD)
    outcome = Column(String, default="PENDING")     # PENDING/WIN/LOSS/NEUTRAL/EXPIRED
    actual_entry = Column(Float, nullable=True)
    actual_exit = Column(Float, nullable=True)
    actual_pnl_pct = Column(Float, nullable=True)
    outcome_checked_at = Column(DateTime, nullable=True)

    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    features = relationship("FeatureSnapshot", back_populates="trade",
                            uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_suggested_trades_date", "date"),
        Index("idx_suggested_trades_symbol", "symbol"),
        Index("idx_suggested_trades_outcome", "outcome"),
    )


class FeatureSnapshot(Base):
    """
    Feature values at the time a trade was suggested.
    Used by FeatureAnalyzer to learn what predicts wins.
    """
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String, ForeignKey("suggested_trades.id"), nullable=False, unique=True)

    # Technical features
    rsi_daily = Column(Float, nullable=True)
    rsi_hourly = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    volume_surge = Column(Float, nullable=True)
    atr_pct = Column(Float, nullable=True)
    vix = Column(Float, nullable=True)
    pcr = Column(Float, nullable=True)

    # Boolean features (stored as 0/1)
    near_support = Column(Integer, default=0)
    near_resistance = Column(Integer, default=0)
    is_compressed = Column(Integer, default=0)
    vol_delta_positive = Column(Integer, default=0)
    above_vwap = Column(Integer, nullable=True)
    above_poc = Column(Integer, nullable=True)

    # Categorical
    breakout_type = Column(String, nullable=True)
    pattern = Column(String, nullable=True)
    market_regime = Column(String, nullable=True)

    # Composite
    alignment_score = Column(Float, nullable=True)

    # Relationship
    trade = relationship("SuggestedTrade", back_populates="features")

    __table_args__ = (Index("idx_feature_trade_id", "trade_id"),)


class AgentWeightHistory(Base):
    """
    Tracks how agent scoring weights change over time.
    Enables rollback if new weights degrade performance.
    """
    __tablename__ = "agent_weight_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    agent_name = Column(String, nullable=False)     # "candidate_screener", "setup_validator"
    weights_json = Column(Text, nullable=False)     # JSON dict of weights
    trigger = Column(String, nullable=True)         # "evolution_loop", "manual", "rollback"

    # Performance at time of update
    win_rate = Column(Float, nullable=True)
    avg_pnl_pct = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_agent_weight_agent", "agent_name"),
        Index("idx_agent_weight_recorded_at", "recorded_at"),
    )


class EvolutionReport(Base):
    """
    Summary report of each evolution loop run.
    """
    __tablename__ = "evolution_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    lookback_days = Column(Integer, nullable=False)
    total_trades_analyzed = Column(Integer, nullable=False)
    win_count = Column(Integer, nullable=False)
    loss_count = Column(Integer, nullable=False)
    neutral_count = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=True)
    avg_pnl_pct = Column(Float, nullable=True)
    weights_updated = Column(Integer, default=0)    # 1 if weights were changed
    report_text = Column(Text, nullable=True)       # Narrative summary

    __table_args__ = (Index("idx_evolution_run_at", "run_at"),)


class AbstractOhlcv:
    """Base class for all OHLCV tables to ensure consistent schema."""
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    vix = Column(Float, nullable=True)


class Ohlcv1m(Base, AbstractOhlcv):
    __tablename__ = "ohlcv_1m"
    __table_args__ = (
        Index("idx_ohlcv_1m_lookup", "symbol", "timestamp", unique=True),
        Index("idx_ohlcv_1m_timestamp", "timestamp"),
    )


class Ohlcv3m(Base, AbstractOhlcv):
    __tablename__ = "ohlcv_3m"
    __table_args__ = (
        Index("idx_ohlcv_3m_lookup", "symbol", "timestamp", unique=True),
        Index("idx_ohlcv_3m_timestamp", "timestamp"),
    )


class Ohlcv5m(Base, AbstractOhlcv):
    __tablename__ = "ohlcv_5m"
    __table_args__ = (
        Index("idx_ohlcv_5m_lookup", "symbol", "timestamp", unique=True),
        Index("idx_ohlcv_5m_timestamp", "timestamp"),
    )


class Ohlcv15m(Base, AbstractOhlcv):
    __tablename__ = "ohlcv_15m"
    __table_args__ = (
        Index("idx_ohlcv_15m_lookup", "symbol", "timestamp", unique=True),
        Index("idx_ohlcv_15m_timestamp", "timestamp"),
    )


class OhlcvDaily(Base, AbstractOhlcv):
    __tablename__ = "ohlcv_daily"
    __table_args__ = (
        Index("idx_ohlcv_daily_lookup", "symbol", "timestamp", unique=True),
        Index("idx_ohlcv_daily_timestamp", "timestamp"),
    )


class OptionChainSnapshot(Base):
    """
    Historical option chain snapshots including Greeks and OI.
    """
    __tablename__ = "option_chain_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    underlying_symbol = Column(String, nullable=False)
    symbol = Column(String, nullable=False)  # Option symbol e.g. NIFTY24MAY22000CE
    expiry = Column(String, nullable=False)
    strike = Column(Float, nullable=False)
    option_type = Column(String, nullable=False)  # CE or PE
    
    # Pricing & Volume
    ltp = Column(Float, nullable=True)
    oi = Column(Integer, nullable=True)
    oi_change = Column(Integer, nullable=True)
    volume = Column(Integer, nullable=True)
    iv = Column(Float, nullable=True)
    
    # Greeks
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_option_lookup", "underlying_symbol", "timestamp"),
        Index("idx_option_symbol", "symbol"),
        Index("idx_option_timestamp", "timestamp"),
    )


class AgentThought(Base):
    """
    Real-time 'Thought Stream' logs from the AI Swarm.
    Makes the agents' internal processes visible in the UI.
    """
    __tablename__ = "agent_thought_stream"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    agent_name = Column(String, nullable=False)
    symbol = Column(String, nullable=True)
    action = Column(String, nullable=True) # e.g. SCANNING, EVALUATING, REJECTED, APPROVED
    message = Column(Text, nullable=False)

    __table_args__ = (
        Index("idx_thought_time", "timestamp"),
        Index("idx_thought_agent", "agent_name"),
    )


class ConsolidationWatchlistStock(Base):
    """Stocks identified as squeezed/consolidating, to be monitored for live breakouts."""
    
    __tablename__ = "consolidation_watchlist_stocks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False)  # YYYY-MM-DD (the date it was scanned/identified)
    symbol = Column(String, nullable=False)
    resistance = Column(Float, nullable=False)
    support = Column(Float, nullable=False)
    bbw = Column(Float, nullable=False)
    atr = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_consolidation_watchlist_lookup", "date", "symbol", unique=True),
        Index("idx_consolidation_watchlist_symbol", "symbol"),
    )


