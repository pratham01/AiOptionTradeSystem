import streamlit as st
import pandas as pd
from datetime import datetime

from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.db_write_worker import DbWriteWorker
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

settings = Settings.load()

st.title("🔌 Broker & Infrastructure Health")

# 1. Broker Manager Status
st.header("Broker API Status")
try:
    manager = get_broker_manager(settings)
    health_status = manager.get_health_status()
    
    if not health_status:
        st.warning("No brokers configured or initialized.")
    else:
        cols = st.columns(len(health_status))
        for i, (broker_name, status) in enumerate(health_status.items()):
            with cols[i]:
                st.subheader(f"{broker_name.upper()} Broker")
                is_healthy = status.get("is_healthy", False)
                authenticated = status.get("authenticated", False)
                
                st.metric(
                    "Status", 
                    "✅ Healthy" if is_healthy else "❌ Unhealthy",
                    delta="Authenticated" if authenticated else "Not Authenticated",
                    delta_color="normal" if authenticated else "inverse"
                )
                
                st.text(f"Failures: {status.get('failure_count', 0)}")
                if status.get("last_success"):
                    st.text(f"Last Success: {status['last_success']}")
                if status.get("last_failure"):
                    st.error(f"Last Failure: {status['last_failure']}")
                    
    st.info(f"Primary Broker: **{manager.primary_broker_name.upper()}** | Backup Broker: **{manager.backup_broker_name.upper() if manager.backup_broker_name else 'None'}**")
except Exception as e:
    st.error(f"Failed to fetch broker health: {e}")

st.markdown("---")

# 2. Database Write Worker Status
st.header("Database Write Queue")
# Since the queue is in-memory in the LiveMarketDataService, we can't easily 
# extract its live stats from a separate dashboard process unless we store it.
# We will display static config for now. In a full production system, 
# we would export metrics via Redis or a DB table.
st.write(
    "SQLite writes are now processed asynchronously via `DbWriteWorker`. "
    "This prevents the WebSocket tick thread from blocking during database flushes."
)

st.markdown("""
**Optimizations Enabled:**
- `PRAGMA journal_mode=WAL` (Write-Ahead Logging for concurrent reads)
- Background queue-based writes
- Single-statement `INSERT OR REPLACE` for batch candle upserts
- Bounded in-memory candle buffers (`.tail(1500)`)
""")

st.markdown("---")

# 3. Database Data Freshness
st.header("📅 Database Data Freshness")
st.caption("Shows how current the data in each OHLCV table is. Stale data = empty/outdated charts.")

try:
    from sqlalchemy import text
    from datetime import date as _date, timedelta as _timedelta

    engine = get_engine()
    today = _date.today()

    key_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
    tables = {
        "1-Min (ohlcv_1m)": "ohlcv_1m",
        "15-Min (ohlcv_15m)": "ohlcv_15m",
        "Daily (ohlcv_daily)": "ohlcv_daily",
    }

    freshness_rows = []
    any_stale = False
    with engine.connect() as conn:
        for label, table in tables.items():
            for sym in key_symbols:
                row = conn.execute(
                    text(f"SELECT MAX(date(timestamp)) FROM {table} WHERE symbol = :sym"),
                    {"sym": sym}
                ).fetchone()
                last_date_str = row[0] if row and row[0] else None
                if last_date_str:
                    last_date = _date.fromisoformat(last_date_str)
                    gap = (today - last_date).days
                else:
                    last_date = None
                    gap = None

                is_ok = gap is not None and gap <= 3  # allow for weekends
                if not is_ok:
                    any_stale = True

                freshness_rows.append({
                    "Table": label,
                    "Symbol": sym.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "").replace("-EQ", ""),
                    "Last Date": last_date_str or "❌ No data",
                    "Days Behind": gap if gap is not None else "N/A",
                    "Status": "✅ OK" if is_ok else "⚠️ STALE",
                })

    df_freshness = pd.DataFrame(freshness_rows)
    st.dataframe(df_freshness, use_container_width=True)

    if any_stale:
        st.warning(
            "⚠️ Some tables are missing recent data. "
            "Start the live bot (`python -m trade_system live`) to trigger automatic backfill, "
            "or run `python -m trade_system.interfaces.live.data_sync_service` manually."
        )
    else:
        st.success("✅ All data tables are current.")

except Exception as e:
    st.error(f"Failed to check data freshness: {e}")

