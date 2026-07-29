"""
One-time backfill script — writes 15m OHLCV data for missed dates (Jul 15, 16, 17) to the DB.

Run once after a valid Fyers token is present:
    PYTHONPATH=src python scripts/backfill_missing_dates.py
"""
import sys
import time
import logging
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy import FyersBrokerClient, FyersAuthService
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import Ohlcv15m

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backfill")


def backfill_date(broker, engine, symbols: list[str], target_date: date) -> int:
    """Fetch and store 15m OHLCV data for all symbols for a single date."""
    date_str = target_date.strftime("%Y-%m-%d")
    logger.info(f"📅 Backfilling {len(symbols)} symbols for {date_str}...")

    total_added = 0
    failed = 0

    for idx, symbol in enumerate(symbols):
        if idx > 0 and idx % 30 == 0:
            logger.info(f"  Progress: {idx}/{len(symbols)} — {total_added} rows written so far.")

        try:
            df = broker.fetch_history(
                symbol=symbol,
                resolution="15",
                range_from=date_str,
                range_to=date_str,
                date_format="1"
            )
            if df is None or df.empty:
                continue

            records = []
            for _, row in df.iterrows():
                dt = pd.to_datetime(row['timestamp'], format="mixed").to_pydatetime().replace(tzinfo=None)
                records.append({
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                })

            if not records:
                continue

            with Session(engine) as session:
                stmt = insert(Ohlcv15m).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'timestamp'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'volume': stmt.excluded.volume
                    }
                )
                session.execute(stmt)
                session.commit()

            total_added += len(records)

            # Throttle: 0.6s per symbol = ~2 req/sec, well within Fyers limit
            time.sleep(0.6)

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ["limit", "429", "too many"])
            if is_rate_limit:
                logger.warning(f"  ⚠️  Rate limit hit on {symbol}. Pausing 60s...")
                time.sleep(60)
            else:
                logger.error(f"  ❌ Failed {symbol}: {e}")
                failed += 1
                time.sleep(1)

    logger.info(f"✅ {date_str}: {total_added} rows written, {failed} symbols failed.\n")
    return total_added


def main():
    settings = Settings.load()
    engine = get_engine()
    
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    if not token:
        logger.error("❌ No valid Fyers token found. Please regenerate your access token first.")
        sys.exit(1)

    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id
    )

    symbols = get_fo_universe() + settings.index_symbols
    logger.info(f"Backfilling {len(symbols)} symbols...")

    # Dates to backfill — skip weekends
    today = date.today()
    # Find all weekdays in the last 7 days that are missing from ohlcv_15m
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT date(timestamp) FROM ohlcv_15m ORDER BY date(timestamp) DESC LIMIT 10"
        )).fetchall()
    existing_dates = {r[0] for r in rows}
    
    # Check last 10 weekdays
    dates_to_fill = []
    check_date = today - timedelta(days=1)
    for _ in range(10):
        if check_date.weekday() < 5:  # Mon-Fri
            date_str = check_date.strftime("%Y-%m-%d")
            if date_str not in existing_dates:
                dates_to_fill.append(check_date)
        check_date -= timedelta(days=1)

    if not dates_to_fill:
        logger.info("✅ No missing dates found. Database is up to date!")
        return

    logger.info(f"Found {len(dates_to_fill)} missing dates: {[str(d) for d in sorted(dates_to_fill)]}")

    grand_total = 0
    for target_date in sorted(dates_to_fill):
        grand_total += backfill_date(broker, engine, symbols, target_date)
        # Small pause between dates to avoid sustained rate limit pressure
        time.sleep(5)

    logger.info(f"🎉 Backfill complete. Total rows written: {grand_total}")


if __name__ == "__main__":
    main()
