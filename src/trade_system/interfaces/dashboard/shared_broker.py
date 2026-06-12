"""Shared Fyers broker singleton for all dashboard pages.

This avoids each dashboard page creating its own broker instance and 
re-authenticating independently. The broker is cached for 120 seconds
via st.cache_resource.
"""

import logging
import streamlit as st

from trade_system.config import Settings
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

LOGGER = logging.getLogger(__name__)


@st.cache_resource(ttl=120)
def get_cached_broker():
    """Return a shared, cached FyersBroker instance for dashboard use.
    
    Cached for 120 seconds to avoid repeated auth calls.
    All dashboard pages should import and use this function instead of
    defining their own get_broker().
    """
    try:
        settings = Settings.load()
        auth_service = FyersAuthService(settings)
        token = auth_service.get_valid_token()
        
        if not token:
            LOGGER.warning("Fyers token could not be retrieved.")
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
                        return None
                else:
                    return None
        except Exception as e:
            LOGGER.error(f"Fyers authentication error: {e}")
            return None
        
        return broker
    except Exception as e:
        LOGGER.error(f"Fyers broker creation failed: {e}")
        return None
