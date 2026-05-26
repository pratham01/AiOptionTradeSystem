from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.interfaces.live.runtime import run_live_trading_bot

if __name__ == "__main__":
    raise SystemExit(run_live_trading_bot())
