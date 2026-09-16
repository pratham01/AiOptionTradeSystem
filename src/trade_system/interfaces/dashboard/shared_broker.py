"""Shared Fyers broker singleton for all dashboard pages.

This avoids each dashboard page creating its own broker instance and 
re-authenticating independently. The broker is cached for 120 seconds
via st.cache_resource.
"""

import logging
import streamlit as st

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

LOGGER = logging.getLogger(__name__)


@st.cache_resource(ttl=120)
def get_cached_broker():
    """Return a shared, cached FyersBroker instance for dashboard use.
    
    Cached for 120 seconds to avoid repeated auth calls.
    All dashboard pages should import and use this function instead of
    defining their own get_broker().
    
    Token resolution order:
    1. FyersAuthService.get_valid_token() (checks cache, env, then TOTP).
    2. Fallback: run the standalone TOTP script directly.
    3. Last resort: use the raw env token even if stale (historical data still works).
    """
    try:
        settings = Settings.load()
        auth_service = FyersAuthService(settings)
        token = auth_service.get_valid_token()
        
        if not token:
            # Fallback: try running the TOTP script directly
            LOGGER.warning("FyersAuthService returned no token. Trying standalone TOTP script...")
            token = _run_totp_script_fallback(settings)
        
        if not token:
            # Last resort: use whatever is in the env (may be stale but allows DB queries)
            LOGGER.warning("All token refresh methods failed. Using raw env token as last resort.")
            token = settings.fyers_access_token
            
        if not token:
            LOGGER.error("No Fyers token available from any source.")
            return None
        
        broker = FyersBroker(
            client_id=settings.fyers.client_id,
            access_token=token,
            user_id=settings.fyers.user_id,
            authenticator=auth_service.authenticator
        )
        
        try:
            if not broker.authenticate():
                # Force once-per-day TOTP refresh if cached token fails
                new_token = auth_service.get_valid_token(force_refresh=True)
                if new_token:
                    broker.access_token = new_token
                    if not broker.authenticate():
                        LOGGER.warning("Broker auth failed after forced refresh. Proceeding with existing token.")
                else:
                    LOGGER.warning("Forced token refresh returned None. Proceeding with existing token.")
        except Exception as e:
            LOGGER.warning(f"Fyers authentication error (proceeding anyway): {e}")
        
        return broker
    except Exception as e:
        LOGGER.error(f"Fyers broker creation failed: {e}")
        return None


def _run_totp_script_fallback(settings) -> str | None:
    """Run the standalone TOTP auth script as a subprocess fallback."""
    import subprocess, sys, json
    from pathlib import Path
    
    script_path = Path(__file__).resolve().parents[4] / "scripts" / "authenticate_fyers_totp.py"
    if not script_path.exists():
        LOGGER.warning("TOTP script not found at %s", script_path)
        return None
    
    try:
        root_path = Path(__file__).resolve().parents[4]
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(root_path),
            capture_output=True,
            text=True,
            timeout=180,  # Allow up to 3 minutes for retries
        )
        if result.returncode == 0:
            # Re-read the cached token that the script wrote
            token_path = settings.fyers.token_path
            if token_path.exists():
                payload = json.loads(token_path.read_text())
                return payload.get("access_token")
    except Exception as e:
        LOGGER.warning("TOTP script fallback failed: %s", e)
    
    return None

@st.cache_data(ttl=5)
def fetch_live_quotes(symbols):
    """Fetch live quotes — reads from live state file first, falls back to broker API."""
    from trade_system.shared.live_state import LiveStateReader
    from trade_system.domains.trading.domain.ports.broker import MarketQuote
    from datetime import datetime

    # 1. Try reading from the live state file (written by the live bot)
    reader = LiveStateReader()
    quotes = {}
    remaining_symbols = list(symbols)
    if reader.is_fresh(max_age_seconds=30):
        states = reader.get_current_prices(remaining_symbols)
        if states:
            for sym, s in states.items():
                if sym not in remaining_symbols:
                    continue
                exchange = sym.split(":")[0] if ":" in sym else "NSE"
                quotes[sym] = MarketQuote(
                    symbol=sym,
                    exchange=exchange,
                    last_price=s.ltp,
                    open=s.open,
                    high=s.high,
                    low=s.low,
                    close=s.close,
                    previous_close=s.previous_close,
                    volume=s.volume,
                    change=s.change,
                    change_percent=s.change_percent,
                    timestamp=datetime.now(),
                    bid=0.0,
                    ask=0.0,
                    bid_qty=0,
                    ask_qty=0,
                    ltp=s.ltp,
                )
            remaining_symbols = [s for s in symbols if s not in quotes]
            if not remaining_symbols:
                return quotes

    # 2. Fallback to file cache
    import json
    from datetime import datetime, timedelta
    from pathlib import Path
    from trade_system.domains.trading.domain.ports.broker import MarketQuote
    
    cache_path = Path("data/live_quotes_cache.json")
    cached_quotes = {}
    use_cache = False
    
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            cached_time_str = cache_data.get("timestamp")
            if cached_time_str:
                cached_time = datetime.fromisoformat(cached_time_str)
                # Fresh if less than 60 seconds old
                if datetime.now() - cached_time < timedelta(seconds=60):
                    use_cache = True
                
                for sym, q_dict in cache_data.get("quotes", {}).items():
                    cached_quotes[sym] = MarketQuote(
                        symbol=q_dict["symbol"],
                        exchange=q_dict["exchange"],
                        last_price=float(q_dict["last_price"]),
                        open=float(q_dict["open"]),
                        high=float(q_dict["high"]),
                        low=float(q_dict["low"]),
                        close=float(q_dict["close"]),
                        previous_close=float(q_dict["previous_close"]),
                        volume=int(q_dict["volume"]),
                        change=float(q_dict["change"]),
                        change_percent=float(q_dict["change_percent"]),
                        timestamp=datetime.fromisoformat(q_dict["timestamp"]),
                        bid=float(q_dict.get("bid", 0.0)),
                        ask=float(q_dict.get("ask", 0.0)),
                        bid_qty=int(q_dict.get("bid_qty", 0)),
                        ask_qty=int(q_dict.get("ask_qty", 0)),
                        ltp=float(q_dict.get("ltp", 0.0))
                    )
        except Exception:
            pass
            
    if use_cache:
        for s in remaining_symbols:
            if s in cached_quotes:
                quotes[s] = cached_quotes[s]
        remaining_symbols = [s for s in symbols if s not in quotes]
        if not remaining_symbols:
            return quotes

    # 3. If only a small delta is missing (or everything if no cache), fetch broker manager
    try:
        from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager, reset_broker_manager
        from trade_system.shared.config import Settings
        
        settings = Settings.load()
        manager = get_broker_manager(settings)
        
        fyers_broker_health = manager.brokers.get("fyers")
        if fyers_broker_health:
            current_token = settings.fyers.access_token
            if getattr(fyers_broker_health.broker, 'access_token', None) != current_token:
                reset_broker_manager()
                manager = get_broker_manager(settings)
                
        # Only query broker for remaining missing symbols
        symbols_to_query = remaining_symbols if remaining_symbols else symbols
        new_quotes = {}
        try:
            new_quotes = manager.get_quotes(symbols_to_query)
        except Exception:
            reset_broker_manager()
            settings = Settings.load()
            manager = get_broker_manager(settings)
            new_quotes = manager.get_quotes(symbols_to_query)
            
        quotes.update(new_quotes)
        # Merge all into cached_quotes for persistent file cache
        cached_quotes.update(quotes)
        
        if cached_quotes:
            serialized_quotes = {}
            for sym, q in cached_quotes.items():
                serialized_quotes[sym] = {
                    "symbol": q.symbol,
                    "exchange": q.exchange,
                    "last_price": q.last_price,
                    "open": q.open,
                    "high": q.high,
                    "low": q.low,
                    "close": q.close,
                    "previous_close": q.previous_close,
                    "volume": q.volume,
                    "change": q.change,
                    "change_percent": q.change_percent,
                    "timestamp": q.timestamp.isoformat() if hasattr(q.timestamp, 'isoformat') else str(q.timestamp),
                    "bid": q.bid,
                    "ask": q.ask,
                    "bid_qty": q.bid_qty,
                    "ask_qty": q.ask_qty,
                    "ltp": q.ltp
                }
            cache_payload = {
                "timestamp": datetime.now().isoformat(),
                "quotes": serialized_quotes
            }
            try:
                cache_path.parent.mkdir(exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache_payload, f)
            except Exception:
                pass
                
        return quotes
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error fetching quotes: {e}")
        return {}
