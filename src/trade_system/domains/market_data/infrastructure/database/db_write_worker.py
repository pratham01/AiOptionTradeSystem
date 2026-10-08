"""
DbWriteWorker — Background thread that drains a queue and writes candles to SQLite.

Usage:
    worker = DbWriteWorker(engine)
    worker.start()

    # From tick thread (non-blocking):
    worker.enqueue("NSE:NIFTY50-INDEX", "1", candle_records)

    # On shutdown:
    worker.stop()  # Drains remaining items, then exits
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from trade_system.domains.market_data.infrastructure.database.repository import upsert_market_data_fast

LOGGER = logging.getLogger(__name__)

# Sentinel to signal the worker to stop
_STOP = object()


class DbWriteWorker:
    """
    Decouples DB I/O from the websocket tick thread.

    The tick thread calls enqueue() which is non-blocking (O(1) in-memory push).
    A background daemon thread pulls items off the queue and commits them to SQLite.

    Metrics available via .stats():
      - queued: items waiting
      - written: total candles written since start
      - errors: failed write count
      - avg_batch_ms: rolling average write latency (ms)
    """

    def __init__(self, engine, maxsize: int = 10_000) -> None:
        self._engine = engine
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._written = 0
        self._errors = 0
        self._total_write_ms = 0.0
        self._write_count = 0
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background write thread."""
        self._thread = threading.Thread(
            target=self._run, name="DbWriteWorker", daemon=True
        )
        self._thread.start()
        LOGGER.info("DbWriteWorker started (queue maxsize=%d)", self._q.maxsize)

    def stop(self, timeout: float = 10.0) -> None:
        """
        Signal stop and wait up to `timeout` seconds for the queue to drain.
        """
        LOGGER.info("DbWriteWorker stopping — draining %d items...", self._q.qsize())
        self._q.put(_STOP)
        if self._thread:
            self._thread.join(timeout=timeout)
        LOGGER.info("DbWriteWorker stopped. total_written=%d errors=%d", self._written, self._errors)

    def enqueue(self, symbol: str, resolution: str, candles: list[dict[str, Any]]) -> None:
        """
        Non-blocking enqueue. If the queue is full, drops and logs a warning
        (this prevents the tick thread from ever blocking on I/O).
        """
        try:
            self._q.put_nowait((symbol, resolution, candles))
        except queue.Full:
            LOGGER.warning(
                "DbWriteWorker queue full (%d items) — dropping %d candles for %s",
                self._q.qsize(), len(candles), symbol,
            )

    def stats(self) -> dict[str, Any]:
        """Return current performance metrics."""
        with self._lock:
            avg_ms = (
                self._total_write_ms / self._write_count
                if self._write_count > 0
                else 0.0
            )
            return {
                "queue_depth": self._q.qsize(),
                "total_written": self._written,
                "total_errors": self._errors,
                "avg_write_ms": round(avg_ms, 2),
            }

    # ── Worker Loop ─────────────────────────────────────────────────────────────

    # Micro-batch settings
    _BATCH_WINDOW_MS = 500  # Accumulate items for up to this many ms before writing
    _BATCH_MAX_ITEMS = 50   # Flush immediately if this many items are queued

    def _run(self) -> None:
        """Main worker loop with micro-batching for reduced write amplification."""
        batch: list[tuple] = []
        batch_deadline = 0.0

        while True:
            # Calculate timeout: either remaining batch window or 1s idle
            if batch:
                remaining = max(0, batch_deadline - time.perf_counter())
                timeout = min(remaining, 0.1)
            else:
                timeout = 1.0

            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                # Timeout: flush whatever we have
                if batch:
                    self._process_batch(batch)
                    batch = []
                continue

            if item is _STOP:
                # Flush remaining batch, then drain queue
                if batch:
                    self._process_batch(batch)
                self._drain_remaining()
                break

            batch.append(item)
            if not batch_deadline:
                batch_deadline = time.perf_counter() + (self._BATCH_WINDOW_MS / 1000.0)

            # Flush if batch is large enough or deadline passed
            if len(batch) >= self._BATCH_MAX_ITEMS or time.perf_counter() >= batch_deadline:
                self._process_batch(batch)
                batch = []
                batch_deadline = 0.0

    def _drain_remaining(self) -> None:
        """Process everything left in the queue on shutdown."""
        batch: list[tuple] = []
        while not self._q.empty():
            try:
                item = self._q.get_nowait()
                if item is not _STOP:
                    batch.append(item)
            except queue.Empty:
                break
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: list[tuple]) -> None:
        """Process a batch of items in a single pass."""
        for item in batch:
            self._process_item(item)

    def _process_item(self, item: tuple) -> None:
        symbol, resolution, candles = item
        t0 = time.perf_counter()
        try:
            count = upsert_market_data_fast(self._engine, symbol, resolution, candles)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._written += count
                self._total_write_ms += elapsed_ms
                self._write_count += 1
        except Exception as exc:
            with self._lock:
                self._errors += 1
            LOGGER.error(
                "DbWriteWorker failed to write %d candles for %s/%s: %s",
                len(candles), symbol, resolution, exc,
            )

