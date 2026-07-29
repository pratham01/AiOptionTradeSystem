from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import json
import logging

# Ensure the project root is importable so we can reach the trade_system package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator

LOGGER = logging.getLogger(__name__)

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    settings = Settings.load()
    settings.ensure_directories()

    authenticator = FyersAuthenticator(settings)
    try:
        token = authenticator.generate_access_token()
    except Exception as exc:  # pragma: no cover - hard to simulate failures
        LOGGER.error("Failed to refresh FYERS TOTPs token (check network/credentials).", exc_info=exc)
        return 1

    LOGGER.info("FYERS TOTPs authentication succeeded.")
    print(token)

    authenticator.update_env_file(token)

    settings.fyers_token_path.write_text(json.dumps({"access_token": token, "source": "totp"}, indent=2))
    LOGGER.info("Cached the refreshed token to %s", settings.fyers_token_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
