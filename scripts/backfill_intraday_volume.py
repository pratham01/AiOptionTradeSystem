from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from datetime import date

import pandas as pd

from trade_system.domains.trading.infrastructure.brokers.legacy import FyersAuthService, FyersBrokerClient
from trade_system.shared.config import Settings
from trade_system.interfaces.live.collector import _merge_intraday_3min_bars, resample_to_timeframe
from trade_system.shared.utils.logging_utils import configure_logging

def backfill_day(symbol: str, trade_date: date) -> tuple[Path, Path, int]:
    settings = Settings.load()
    configure_logging(settings.log_level)
    token = FyersAuthService(settings).read_cached_token()
    if not token:
        raise RuntimeError("No FYERS token found. Run `trade-system auth` first.")

    broker = FyersBrokerClient(settings, token)
    minute_history = broker.fetch_history(
        symbol=symbol,
        resolution="1",
        range_from=trade_date.isoformat(),
        range_to=trade_date.isoformat(),
    )
    if minute_history.empty:
        raise RuntimeError(f"No FYERS 1-minute history returned for {symbol} on {trade_date.isoformat()}")

    if "symbol" in minute_history.columns:
        minute_history = minute_history.drop(columns=["symbol"])
    minute_history = minute_history.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    minute_path = settings.data_dir / f"{symbol.replace(':', '_')}_1min_{trade_date.isoformat()}.csv"
    minute_history.to_csv(minute_path, index=False)

    yearly_path = settings.data_dir / f"{symbol.replace(':', '_')}_3min_{trade_date.year}.csv"
    existing = pd.read_csv(yearly_path, parse_dates=["timestamp"]) if yearly_path.exists() else pd.DataFrame()
    existing = existing[[c for c in existing.columns if c in ["timestamp", "open", "high", "low", "close", "volume"]]]
    existing = existing[existing["timestamp"].dt.date != trade_date] if not existing.empty else existing

    today_bars = resample_to_timeframe(
        minute_history.set_index("timestamp")[["open", "high", "low", "close", "volume"]],
        3,
    ).reset_index()
    merged = _merge_intraday_3min_bars(existing=existing, today_bars=today_bars)
    merged.to_csv(yearly_path, index=False)
    return minute_path, yearly_path, len(today_bars)

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NSE:NIFTY50-INDEX"
    trade_date = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today()
    minute_path, yearly_path, count = backfill_day(symbol, trade_date)
    print(f"Backfilled {symbol} for {trade_date.isoformat()} with {count} three-minute bars")
    print(f"1-minute file: {minute_path}")
    print(f"3-minute yearly file: {yearly_path}")
