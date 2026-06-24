from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from trade_system.config import Settings
from trade_system.infrastructure.data.nse_universe import NSE_UNIVERSE
from trade_system.infrastructure.brokers.fyers.client import FyersBroker

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GainerRow:
    symbol: str
    close: float
    change_pct: float
    volume: int


class BrokerTopGainersAgent:
    """Fetches market quotes and returns top gainers.

    Multi-agent style: each worker fetches one symbol batch in parallel.
    """

    def __init__(
        self,
        broker: FyersBroker,
        symbols: list[str] | None = None,
        workers: int = 1,
        batch_size: int = 200,
    ) -> None:
        self.broker = broker
        self.symbols = symbols or NSE_UNIVERSE
        self.workers = max(1, workers)
        self.batch_size = max(1, batch_size)

    def _symbol_batches(self) -> list[list[str]]:
        return [self.symbols[i:i + self.batch_size] for i in range(0, len(self.symbols), self.batch_size)]

    def _fetch_batch(self, symbols: list[str]) -> list[GainerRow]:
        rows: list[GainerRow] = []
        quotes = self.broker.get_quotes(symbols)
        for symbol, quote in quotes.items():
            if quote.last_price <= 0:
                continue
            rows.append(
                GainerRow(
                    symbol=symbol,
                    close=quote.last_price,
                    change_pct=quote.change_percent,
                    volume=quote.volume,
                )
            )
        return rows

    def top_gainers(self, top_n: int = 10) -> list[GainerRow]:
        batches = self._symbol_batches()
        all_rows: list[GainerRow] = []

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._fetch_batch, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    all_rows.extend(future.result())
                except Exception as exc:
                    LOGGER.warning("Batch fetch failed: %s", exc)

        deduped: dict[str, GainerRow] = {}
        for row in all_rows:
            existing = deduped.get(row.symbol)
            if existing is None or row.change_pct > existing.change_pct:
                deduped[row.symbol] = row

        ranked = sorted(deduped.values(), key=lambda x: x.change_pct, reverse=True)
        return ranked[:top_n]

    def top_and_worst(self, n: int = 5) -> tuple[list[GainerRow], list[GainerRow]]:
        batches = self._symbol_batches()
        all_rows: list[GainerRow] = []

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._fetch_batch, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    all_rows.extend(future.result())
                except Exception as exc:
                    LOGGER.warning("Batch fetch failed: %s", exc)

        deduped: dict[str, GainerRow] = {}
        for row in all_rows:
            existing = deduped.get(row.symbol)
            if existing is None or row.change_pct > existing.change_pct:
                deduped[row.symbol] = row

        ranked = sorted(deduped.values(), key=lambda x: x.change_pct, reverse=True)
        if not ranked:
            return [], []
        
        return ranked[:n], ranked[-n:][::-1]


def create_fyers_broker() -> FyersBroker:
    settings = Settings.load()
    token_path = Path(".secrets/fyers_token.json")
    if not token_path.exists():
        raise FileNotFoundError("Token not found at .secrets/fyers_token.json. Run auth first.")

    token_data = json.loads(token_path.read_text())
    access_token = token_data.get("access_token", "")
    if not access_token:
        raise RuntimeError("Missing access_token in .secrets/fyers_token.json")

    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=access_token,
        user_id=settings.fyers.user_id,
    )
    if not broker.authenticate():
        raise RuntimeError("Fyers authentication failed. Refresh token and retry.")
    return broker


def _fmt_vol(vol: int) -> str:
    """Format volume in Indian notation: Cr / L / K."""
    if vol >= 1_00_00_000:
        return f"{vol / 1_00_00_000:.1f}Cr"
    elif vol >= 1_00_000:
        return f"{vol / 1_00_000:.1f}L"
    elif vol >= 1_000:
        return f"{vol / 1_000:.0f}K"
    return str(vol)


def format_top_gainers_table(rows: list[GainerRow], title_prefix: str = "Top") -> str:
    lines = []
    lines.append(f"{title_prefix} {len(rows)} stocks @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("#  SYMBOL               PRICE      CHANGE%      VOLUME")
    for i, row in enumerate(rows, start=1):
        clean = row.symbol.replace("NSE:", "").replace("-EQ", "")
        vol_str = _fmt_vol(row.volume)
        lines.append(f"{i:>2} {clean:<20} {row.close:>8.2f} {row.change_pct:>10.2f}% {vol_str:>10}")
    return "\n".join(lines)
