import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager, reset_broker_manager

# Reset manager
reset_broker_manager()

settings = Settings.load()
# Let's inspect settings and verify if the TOTP details are loaded
print("=== Settings Check ===")
print(f"Client ID: {settings.fyers.client_id}")
print(f"User ID: {settings.fyers.user_id}")
print(f"PIN length: {len(settings.fyers.pin) if settings.fyers.pin else 0}")
print(f"TOTP length: {len(settings.fyers.totp_secret) if settings.fyers.totp_secret else 0}")
print(f"Access Token length: {len(settings.fyers.access_token) if settings.fyers.access_token else 0}")

# Create manager
manager = get_broker_manager(settings)
print("\n=== Broker Manager health ===")
print(manager.get_health_status())

# Get Fyers Broker and test authentication
print("\n=== Testing Fyers Broker Authentication ===")
fyers_broker = manager.get_broker("fyers")
print(f"Is authenticated: {fyers_broker.is_authenticated}")
print(f"Authenticator object: {fyers_broker.authenticator}")

# Force authenticate with fake expired token to see if it triggers TOTP refresh
print("\n=== Forcing authenticate with expired token ===")
fyers_broker.access_token = "ALHT0QX10K-100:eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_expired_token"
fyers_broker._authenticated = False

try:
    auth_success = fyers_broker.authenticate()
    print(f"Authentication success: {auth_success}")
    print(f"New access token prefix: {fyers_broker.access_token[:25]}...")
except Exception as e:
    logger.exception("Forced authentication failed")
