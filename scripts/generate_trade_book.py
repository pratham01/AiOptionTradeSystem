import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine, SessionLocal
from trade_system.domains.market_data.infrastructure.database.models import TopGainersSnapshot, TopGainersStock, TopMoverAnalysis
from trade_system.domains.advisory.application.agent.movers_analysis_agent import MoversAnalysisAgent

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

async def run_trade_book_generation(target_date: str | None = None):
    engine = get_engine()
    
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
        
    LOGGER.info(f"Generating Detailed Trade Book for {target_date}")
    
    with SessionLocal() as session:
        # Find the TopGainersSnapshot for the date
        snapshot = session.query(TopGainersSnapshot).filter_by(date=target_date).first()
        if not snapshot:
            LOGGER.error(f"No top gainers snapshot found for {target_date}. Please run the top gainers scanner first.")
            return
            
        top_stocks = session.query(TopGainersStock).filter_by(snapshot_id=snapshot.id).order_by(TopGainersStock.change_pct.desc()).limit(5).all()
        
        if not top_stocks:
            LOGGER.error("No top stocks found in snapshot.")
            return
            
        agent = MoversAnalysisAgent(use_web_news=True)
        
        for stock in top_stocks:
            # Check if analysis already exists
            existing = session.query(TopMoverAnalysis).filter_by(stock_id=stock.id).first()
            if existing:
                LOGGER.info(f"Analysis already exists for {stock.symbol}, skipping.")
                continue
                
            LOGGER.info(f"Analyzing {stock.symbol} ({stock.change_pct}%)")
            
            record = {
                "id": stock.id,
                "symbol": stock.symbol,
                "name": stock.name,
                "change_pct": stock.change_pct,
                "volume_ratio": getattr(stock, 'volume_ratio', round(stock.volume / 100000.0, 2)), # top_gainers_stocks doesn't have volume_ratio
                "volume_divergence": getattr(stock, 'volume_divergence', 'N/A'),
                "timestamp": stock.timestamp
            }
            
            analysis_dict = await agent.analyze_mover(record)
            
            db_analysis = TopMoverAnalysis(
                stock_id=stock.id,
                catalyst_analysis=analysis_dict["catalyst_analysis"],
                volume_behavior=analysis_dict["volume_behavior"],
                predictive_factors=analysis_dict["predictive_factors"],
                technical_setup=analysis_dict["technical_setup"]
            )
            session.add(db_analysis)
            session.commit()
            
            LOGGER.info(f"Saved analysis for {stock.symbol}")
            
    LOGGER.info("Detailed Trade Book generation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="YYYY-MM-DD", default=None)
    args = parser.parse_args()
    
    asyncio.run(run_trade_book_generation(args.date))
