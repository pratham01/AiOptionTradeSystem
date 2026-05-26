"""Repository functions for database operations."""

from datetime import datetime, date
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from trade_system.core.ports.repository import TradeRepository, LogRepository
from trade_system.infrastructure.database.models import (
    SuggestedTrade,
    TopGainersSnapshot,
    TopGainersStock,
    FOSelectionSnapshot,
    FOSelectionStock,
    FOTopStocksSnapshot,
    FOTopStock,
    SectorPerformanceSnapshot,
    SectorPerformanceSector,
    SectorPerformanceStock,
    Ohlcv1m,
    Ohlcv3m,
    Ohlcv5m,
    Ohlcv15m,
    OhlcvDaily,
    OptionChainSnapshot,
    AgentThought,
)

RESOLUTION_TO_MODEL = {
    "1": Ohlcv1m,
    "3": Ohlcv3m,
    "5": Ohlcv5m,
    "15": Ohlcv15m,
    "D": OhlcvDaily,
    "DAY": OhlcvDaily,
}


# Market Data Repository
def save_market_data_batch(
    session: Session,
    symbol: str,
    resolution: str,
    candles: List[Dict[str, Any]],
) -> int:
    """
    Save a batch of candles to the correct timeframe table.
    """
    model = RESOLUTION_TO_MODEL.get(resolution.upper())
    if not model:
        raise ValueError(f"Unsupported resolution: {resolution}")

    count = 0
    for c_data in candles:
        ts = c_data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        # Check for existing
        existing = session.query(model).filter_by(symbol=symbol, timestamp=ts).first()

        if existing:
            existing.open = c_data["open"]
            existing.high = c_data["high"]
            existing.low = c_data["low"]
            existing.close = c_data["close"]
            existing.volume = c_data.get("volume", 0.0)
            existing.vix = c_data.get("vix")
        else:
            candle = model(
                symbol=symbol,
                timestamp=ts,
                open=c_data["open"],
                high=c_data["high"],
                low=c_data["low"],
                close=c_data["close"],
                volume=c_data.get("volume", 0.0),
                vix=c_data.get("vix"),
            )
            session.add(candle)
        count += 1

    session.commit()
    return count


def get_market_data(
    session: Session,
    symbol: str,
    resolution: str,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Any]:
    """Retrieve market data candles from the correct timeframe table."""
    model = RESOLUTION_TO_MODEL.get(resolution.upper())
    if not model:
        raise ValueError(f"Unsupported resolution: {resolution}")

    query = session.query(model).filter_by(symbol=symbol)

    if from_date:
        query = query.filter(model.timestamp >= from_date)
    if to_date:
        query = query.filter(model.timestamp <= to_date)

    query = query.order_by(model.timestamp.asc())

    if limit:
        query = query.limit(limit)

    return query.all()


def save_option_chain_batch(
    session: Session,
    underlying: str,
    timestamp: datetime,
    snapshots: List[Dict[str, Any]],
) -> int:
    """Save a batch of option chain snapshots."""
    count = 0
    for s in snapshots:
        snapshot = OptionChainSnapshot(
            timestamp=timestamp,
            underlying_symbol=underlying,
            symbol=s["symbol"],
            expiry=s["expiry"],
            strike=s["strike"],
            option_type=s["option_type"],
            ltp=s.get("ltp"),
            oi=s.get("oi"),
            oi_change=s.get("oi_change"),
            volume=s.get("volume"),
            iv=s.get("iv"),
            delta=s.get("delta"),
            gamma=s.get("gamma"),
            theta=s.get("theta"),
            vega=s.get("vega"),
        )
        session.add(snapshot)
        count += 1
    session.commit()
    return count


def log_agent_thought(
    session: Session,
    agent_name: str,
    message: str,
    symbol: str | None = None,
    action: str | None = None,
) -> None:
    """Log an internal thought or deliberation from an agent."""
    thought = AgentThought(
        agent_name=agent_name,
        message=message,
        symbol=symbol,
        action=action,
        timestamp=datetime.now()
    )
    session.add(thought)
    session.commit()


def get_latest_thoughts(session: Session, limit: int = 50) -> List[AgentThought]:
    """Retrieve the latest entries from the thought stream."""
    return (
        session.query(AgentThought)
        .order_by(AgentThought.timestamp.desc())
        .limit(limit)
        .all()
    )


# Top Gainers Repository
def save_top_gainers(
    session: Session,
    date_str: str,
    total_stocks_analyzed: int,
    stocks_passing_filters: int,
    top_n: int,
    gainers: List[Dict[str, Any]],
) -> TopGainersSnapshot:
    """Save top gainers data to database."""
    # Check if snapshot exists
    snapshot = session.query(TopGainersSnapshot).filter_by(date=date_str).first()
    
    if snapshot:
        # Delete existing stocks
        session.query(TopGainersStock).filter_by(snapshot_id=snapshot.id).delete()
        snapshot.total_stocks_analyzed = total_stocks_analyzed
        snapshot.stocks_passing_filters = stocks_passing_filters
        snapshot.top_n = top_n
    else:
        snapshot = TopGainersSnapshot(
            date=date_str,
            total_stocks_analyzed=total_stocks_analyzed,
            stocks_passing_filters=stocks_passing_filters,
            top_n=top_n,
        )
        session.add(snapshot)
    
    session.flush()
    
    # Add stocks
    for stock_data in gainers:
        stock = TopGainersStock(
            snapshot_id=snapshot.id,
            symbol=stock_data["symbol"],
            name=stock_data["name"],
            close=stock_data["close"],
            open=stock_data.get("open"),
            high=stock_data.get("high"),
            low=stock_data.get("low"),
            prev_close=stock_data.get("prev_close"),
            volume=stock_data["volume"],
            change=stock_data.get("change"),
            change_pct=stock_data.get("change_pct"),
            timestamp=datetime.fromisoformat(stock_data["timestamp"]) if isinstance(stock_data["timestamp"], str) else stock_data["timestamp"],
        )
        session.add(stock)
    
    session.commit()
    return snapshot


def get_top_gainers(session: Session, date_str: str) -> Optional[Dict[str, Any]]:
    """Get top gainers data for a specific date."""
    snapshot = session.query(TopGainersSnapshot).filter_by(date=date_str).first()
    if not snapshot:
        return None
    
    stocks = session.query(TopGainersStock).filter_by(snapshot_id=snapshot.id).all()
    
    return {
        "date": snapshot.date,
        "total_stocks_analyzed": snapshot.total_stocks_analyzed,
        "stocks_passing_filters": snapshot.stocks_passing_filters,
        "top_n": snapshot.top_n,
        "gainers": [
            {
                "symbol": s.symbol,
                "name": s.name,
                "close": s.close,
                "open": s.open,
                "high": s.high,
                "low": s.low,
                "prev_close": s.prev_close,
                "volume": s.volume,
                "change": s.change,
                "change_pct": s.change_pct,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in stocks
        ],
    }


def get_top_gainers_history(session: Session, limit: int = 30) -> List[Dict[str, Any]]:
    """Get top gainers history."""
    snapshots = session.query(TopGainersSnapshot).order_by(TopGainersSnapshot.date.desc()).limit(limit).all()
    
    history = []
    for snapshot in snapshots:
        data = get_top_gainers(session, snapshot.date)
        if data:
            history.append(data)
    
    return history


# FO Selection Repository
def save_fo_selection(
    session: Session,
    date_str: str,
    total_stocks_analyzed: int,
    stocks_with_data: int,
    stocks_passing_filters: int,
    top_n: int,
    min_score_threshold: float,
    selected_stocks: List[Dict[str, Any]],
) -> FOSelectionSnapshot:
    """Save FO selection data to database."""
    # Check if snapshot exists
    snapshot = session.query(FOSelectionSnapshot).filter_by(date=date_str).first()
    
    if snapshot:
        # Delete existing stocks
        session.query(FOSelectionStock).filter_by(snapshot_id=snapshot.id).delete()
        snapshot.total_stocks_analyzed = total_stocks_analyzed
        snapshot.stocks_with_data = stocks_with_data
        snapshot.stocks_passing_filters = stocks_passing_filters
        snapshot.top_n = top_n
        snapshot.min_score_threshold = min_score_threshold
    else:
        snapshot = FOSelectionSnapshot(
            date=date_str,
            total_stocks_analyzed=total_stocks_analyzed,
            stocks_with_data=stocks_with_data,
            stocks_passing_filters=stocks_passing_filters,
            top_n=top_n,
            min_score_threshold=min_score_threshold,
        )
        session.add(snapshot)
    
    session.flush()
    
    # Add stocks
    for stock_data in selected_stocks:
        stock = FOSelectionStock(
            snapshot_id=snapshot.id,
            symbol=stock_data["symbol"],
            name=stock_data["name"],
            score=stock_data["score"],
            timestamp=datetime.fromisoformat(stock_data["timestamp"]) if isinstance(stock_data["timestamp"], str) else stock_data["timestamp"],
        )
        session.add(stock)
    
    session.commit()
    return snapshot


def get_fo_selection(session: Session, date_str: str) -> Optional[Dict[str, Any]]:
    """Get FO selection data for a specific date."""
    snapshot = session.query(FOSelectionSnapshot).filter_by(date=date_str).first()
    if not snapshot:
        return None
    
    stocks = session.query(FOSelectionStock).filter_by(snapshot_id=snapshot.id).all()
    
    return {
        "date": snapshot.date,
        "total_stocks_analyzed": snapshot.total_stocks_analyzed,
        "stocks_with_data": snapshot.stocks_with_data,
        "stocks_passing_filters": snapshot.stocks_passing_filters,
        "top_n": snapshot.top_n,
        "min_score_threshold": snapshot.min_score_threshold,
        "selected_stocks": [
            {
                "symbol": s.symbol,
                "name": s.name,
                "score": s.score,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in stocks
        ],
    }


def get_fo_selection_history(session: Session, limit: int = 30) -> List[Dict[str, Any]]:
    """Get FO selection history."""
    snapshots = session.query(FOSelectionSnapshot).order_by(FOSelectionSnapshot.date.desc()).limit(limit).all()
    
    history = []
    for snapshot in snapshots:
        data = get_fo_selection(session, snapshot.date)
        if data:
            history.append(data)
    
    return history


# FO Top Stocks Repository
def save_fo_top_stocks(
    session: Session,
    date_str: str,
    total_stocks_analyzed: int,
    stocks_passing_filters: int,
    top_n: int,
    stocks: List[Dict[str, Any]],
) -> FOTopStocksSnapshot:
    """Save FO top stocks data to database."""
    # Check if snapshot exists
    snapshot = session.query(FOTopStocksSnapshot).filter_by(date=date_str).first()
    
    if snapshot:
        # Delete existing stocks
        session.query(FOTopStock).filter_by(snapshot_id=snapshot.id).delete()
        snapshot.total_stocks_analyzed = total_stocks_analyzed
        snapshot.stocks_passing_filters = stocks_passing_filters
        snapshot.top_n = top_n
    else:
        snapshot = FOTopStocksSnapshot(
            date=date_str,
            total_stocks_analyzed=total_stocks_analyzed,
            stocks_passing_filters=stocks_passing_filters,
            top_n=top_n,
        )
        session.add(snapshot)
    
    session.flush()
    
    # Add stocks
    for stock_data in stocks:
        stock = FOTopStock(
            snapshot_id=snapshot.id,
            symbol=stock_data["symbol"],
            name=stock_data["name"],
            close=stock_data["close"],
            change_pct=stock_data["change_pct"],
            volume=stock_data["volume"],
            avg_volume=stock_data["avg_volume"],
            volume_ratio=stock_data["volume_ratio"],
            volatility=stock_data["volatility"],
            atr=stock_data["atr"],
            volume_divergence=stock_data.get("volume_diverggence", "none"),
            divergence_strength=stock_data["divergence_strength"],
            price_action_score=stock_data["price_action_score"],
            trend_score=stock_data["trend_score"],
            liquidity_score=stock_data["liquidity_score"],
            total_score=stock_data["total_score"],
            timestamp=datetime.fromisoformat(stock_data["timestamp"]) if isinstance(stock_data["timestamp"], str) else stock_data["timestamp"],
        )
        session.add(stock)
    
    session.commit()
    return snapshot


def get_fo_top_stocks(session: Session, date_str: str) -> Optional[Dict[str, Any]]:
    """Get FO top stocks data for a specific date."""
    snapshot = session.query(FOTopStocksSnapshot).filter_by(date=date_str).first()
    if not snapshot:
        return None
    
    stocks = session.query(FOTopStock).filter_by(snapshot_id=snapshot.id).all()
    
    return {
        "date": snapshot.date,
        "total_stocks_analyzed": snapshot.total_stocks_analyzed,
        "stocks_passing_filters": snapshot.stocks_passing_filters,
        "top_n": snapshot.top_n,
        "stocks": [
            {
                "symbol": s.symbol,
                "name": s.name,
                "close": s.close,
                "change_pct": s.change_pct,
                "volume": s.volume,
                "avg_volume": s.avg_volume,
                "volume_ratio": s.volume_ratio,
                "volatility": s.volatility,
                "atr": s.atr,
                "volume_diverggence": s.volume_divergence,
                "divergence_strength": s.divergence_strength,
                "price_action_score": s.price_action_score,
                "trend_score": s.trend_score,
                "liquidity_score": s.liquidity_score,
                "total_score": s.total_score,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in stocks
        ],
    }


def get_fo_top_stocks_history(session: Session, limit: int = 30) -> List[Dict[str, Any]]:
    """Get FO top stocks history."""
    snapshots = session.query(FOTopStocksSnapshot).order_by(FOTopStocksSnapshot.date.desc()).limit(limit).all()
    
    history = []
    for snapshot in snapshots:
        data = get_fo_top_stocks(session, snapshot.date)
        if data:
            history.append(data)
    
    return history


# Sector Performance Repository
def save_sector_performance(
    session: Session,
    date_str: str,
    total_stocks_analyzed: int,
    stocks_with_sector: int,
    top_n: int,
    top_sectors: List[Dict[str, Any]],
    worst_sectors: List[Dict[str, Any]],
    top_stocks_overall: List[Dict[str, Any]],
    worst_stocks_overall: List[Dict[str, Any]],
) -> SectorPerformanceSnapshot:
    """Save sector performance data to database."""
    # Check if snapshot exists
    snapshot = session.query(SectorPerformanceSnapshot).filter_by(date=date_str).first()
    
    if snapshot:
        # Delete existing data
        session.query(SectorPerformanceStock).filter_by(snapshot_id_top=snapshot.id).delete()
        session.query(SectorPerformanceStock).filter_by(snapshot_id_worst=snapshot.id).delete()
        session.query(SectorPerformanceSector).filter_by(snapshot_id=snapshot.id).delete()
        session.query(SectorPerformanceSector).filter_by(snapshot_id_worst=snapshot.id).delete()
        snapshot.total_stocks_analyzed = total_stocks_analyzed
        snapshot.stocks_with_sector = stocks_with_sector
        snapshot.top_n = top_n
    else:
        snapshot = SectorPerformanceSnapshot(
            date=date_str,
            total_stocks_analyzed=total_stocks_analyzed,
            stocks_with_sector=stocks_with_sector,
            top_n=top_n,
        )
        session.add(snapshot)
    
    session.flush()
    
    # Add top sectors
    for sector_data in top_sectors:
        sector = SectorPerformanceSector(
            snapshot_id=snapshot.id,
            sector=sector_data["sector"],
            stock_count=sector_data["stock_count"],
            avg_change_pct=sector_data["avg_change_pct"],
            is_top_sector=1,
        )
        session.add(sector)
        session.flush()
        
        # Add top stocks for this sector
        for stock_data in sector_data.get("top_stocks", []):
            stock = SectorPerformanceStock(
                snapshot_id_top=snapshot.id,
                sector_id=sector.id,
                symbol=stock_data["symbol"],
                name=stock_data["name"],
                sector=stock_data["sector"],
                close=stock_data["close"],
                prev_close=stock_data["prev_close"],
                volume=stock_data["volume"],
                change=stock_data["change"],
                change_pct=stock_data["change_pct"],
                timestamp=datetime.fromisoformat(stock_data["timestamp"]) if isinstance(stock_data["timestamp"], str) else stock_data["timestamp"],
                is_top_stock=1,
            )
            session.add(stock)
    
    # Add worst sectors
    for sector_data in worst_sectors:
        sector = SectorPerformanceSector(
            snapshot_id_worst=snapshot.id,
            sector=sector_data["sector"],
            stock_count=sector_data["stock_count"],
            avg_change_pct=sector_data["avg_change_pct"],
            is_top_sector=0,
        )
        session.add(sector)
        session.flush()
        
        # Add worst stocks for this sector
        for stock_data in sector_data.get("worst_stocks", []):
            stock = SectorPerformanceStock(
                snapshot_id_worst=snapshot.id,
                sector_id_worst=sector.id,
                symbol=stock_data["symbol"],
                name=stock_data["name"],
                sector=stock_data["sector"],
                close=stock_data["close"],
                prev_close=stock_data["prev_close"],
                volume=stock_data["volume"],
                change=stock_data["change"],
                change_pct=stock_data["change_pct"],
                timestamp=datetime.fromisoformat(stock_data["timestamp"]) if isinstance(stock_data["timestamp"], str) else stock_data["timestamp"],
                is_top_stock=0,
            )
            session.add(stock)
    
    # Add top stocks overall
    for stock_data in top_stocks_overall:
        stock = SectorPerformanceStock(
            snapshot_id_top=snapshot.id,
            symbol=stock_data["symbol"],
            name=stock_data["name"],
            sector=stock_data["sector"],
            close=stock_data["close"],
            prev_close=stock_data["prev_close"],
            volume=stock_data["volume"],
            change=stock_data["change"],
            change_pct=stock_data["change_pct"],
            timestamp=datetime.fromisoformat(stock_data["timestamp"]) if isinstance(stock_data["timestamp"], str) else stock_data["timestamp"],
            is_top_stock=1,
        )
        session.add(stock)
    
    # Add worst stocks overall
    for stock_data in worst_stocks_overall:
        stock = SectorPerformanceStock(
            snapshot_id_worst=snapshot.id,
            symbol=stock_data["symbol"],
            name=stock_data["name"],
            sector=stock_data["sector"],
            close=stock_data["close"],
            prev_close=stock_data["prev_close"],
            volume=stock_data["volume"],
            change=stock_data["change"],
            change_pct=stock_data["change_pct"],
            timestamp=datetime.fromisoformat(stock_data["timestamp"]) if isinstance(stock_data["timestamp"], str) else stock_data["timestamp"],
            is_top_stock=0,
        )
        session.add(stock)
    
    session.commit()
    return snapshot


def get_sector_performance(session: Session, date_str: str) -> Optional[Dict[str, Any]]:
    """Get sector performance data for a specific date."""
    snapshot = session.query(SectorPerformanceSnapshot).filter_by(date=date_str).first()
    if not snapshot:
        return None
    
    # Get top sectors
    top_sectors_data = session.query(SectorPerformanceSector).filter_by(
        snapshot_id=snapshot.id, is_top_sector=1
    ).all()
    
    top_sectors = []
    for sector in top_sectors_data:
        top_stocks = session.query(SectorPerformanceStock).filter_by(
            sector_id=sector.id, is_top_stock=1
        ).all()
        
        top_sectors.append({
            "sector": sector.sector,
            "stock_count": sector.stock_count,
            "avg_change_pct": sector.avg_change_pct,
            "top_stocks": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "sector": s.sector,
                    "close": s.close,
                    "prev_close": s.prev_close,
                    "volume": s.volume,
                    "change": s.change,
                    "change_pct": s.change_pct,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in top_stocks
            ],
            "worst_stocks": [],
        })
    
    # Get worst sectors
    worst_sectors_data = session.query(SectorPerformanceSector).filter_by(
        snapshot_id_worst=snapshot.id, is_top_sector=0
    ).all()
    
    worst_sectors = []
    for sector in worst_sectors_data:
        worst_stocks = session.query(SectorPerformanceStock).filter_by(
            sector_id_worst=sector.id, is_top_stock=0
        ).all()
        
        worst_sectors.append({
            "sector": sector.sector,
            "stock_count": sector.stock_count,
            "avg_change_pct": sector.avg_change_pct,
            "top_stocks": [],
            "worst_stocks": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "sector": s.sector,
                    "close": s.close,
                    "prev_close": s.prev_close,
                    "volume": s.volume,
                    "change": s.change,
                    "change_pct": s.change_pct,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in worst_stocks
            ],
        })
    
    # Get top stocks overall
    top_stocks_overall = session.query(SectorPerformanceStock).filter_by(
        snapshot_id_top=snapshot.id, is_top_stock=1, sector_id=None
    ).all()
    
    # Get worst stocks overall
    worst_stocks_overall = session.query(SectorPerformanceStock).filter_by(
        snapshot_id_worst=snapshot.id, is_top_stock=0, sector_id_worst=None
    ).all()
    
    return {
        "date": snapshot.date,
        "total_stocks_analyzed": snapshot.total_stocks_analyzed,
        "stocks_with_sector": snapshot.stocks_with_sector,
        "top_n": snapshot.top_n,
        "top_sectors": top_sectors,
        "worst_sectors": worst_sectors,
        "top_stocks_overall": [
            {
                "symbol": s.symbol,
                "name": s.name,
                "sector": s.sector,
                "close": s.close,
                "prev_close": s.prev_close,
                "volume": s.volume,
                "change": s.change,
                "change_pct": s.change_pct,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in top_stocks_overall
        ],
        "worst_stocks_overall": [
            {
                "symbol": s.symbol,
                "name": s.name,
                "sector": s.sector,
                "close": s.close,
                "prev_close": s.prev_close,
                "volume": s.volume,
                "change": s.change,
                "change_pct": s.change_pct,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in worst_stocks_overall
        ],
    }


def get_sector_performance_history(session: Session, limit: int = 30) -> List[Dict[str, Any]]:
    """Get sector performance history."""
    snapshots = session.query(SectorPerformanceSnapshot).order_by(SectorPerformanceSnapshot.date.desc()).limit(limit).all()
    
    history = []
    for snapshot in snapshots:
        data = get_sector_performance(session, snapshot.date)
        if data:
            history.append(data)
    
    return history

class SQLAlchemyTradeRepository(TradeRepository):
    """SQLAlchemy implementation of the TradeRepository."""
    
    def __init__(self, engine):
        self.engine = engine

    def update_trade_status(self, trade_id: str, status: str, entry_price: float) -> None:
        try:
            with Session(self.engine) as session:
                trade = session.query(SuggestedTrade).filter_by(id=trade_id).first()
                if trade:
                    trade.outcome = status
                    trade.actual_entry = entry_price
                    session.commit()
        except Exception as e:
            # Re-raise or handle domain specific exception
            raise RuntimeError(f"Database error updating trade {trade_id}: {e}")


class SQLAlchemyLogRepository(LogRepository):
    """SQLAlchemy implementation of the LogRepository."""
    
    def __init__(self, engine):
        self.engine = engine

    def log_thought(self, agent: str, message: str, symbol: Optional[str] = None, action: Optional[str] = None) -> None:
        try:
            with Session(self.engine) as session:
                log_agent_thought(session, agent, message, symbol, action)
        except Exception as e:
            # Don't fail the whole app if logging a thought fails
            pass
