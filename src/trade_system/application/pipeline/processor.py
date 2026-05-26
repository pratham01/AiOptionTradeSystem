"""Data pipeline for real-time and batch processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from trade_system.core.events.bus import EventBus, EventType, MarketEvent

LOGGER = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base for pipeline stages."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._next: PipelineStage | None = None

    def then(self, stage: PipelineStage) -> PipelineStage:
        """Chain another stage."""
        self._next = stage
        return stage

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data and return transformed data."""
        pass

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute this stage and any chained stages."""
        result = self.process(data)
        if self._next:
            return self._next.run(result)
        return result


class DataPipeline:
    """Manages data processing pipelines with event integration."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        buffer_size: int = 1000,
    ) -> None:
        self.event_bus = event_bus
        self.buffer_size = buffer_size
        self._stages: list[PipelineStage] = []
        self._buffer: list[dict[str, Any]] = []
        self._running = False

    def add_stage(self, stage: PipelineStage) -> "DataPipeline":
        """Add a processing stage."""
        if self._stages:
            self._stages[-1].then(stage)
        self._stages.append(stage)
        LOGGER.debug("Added pipeline stage: %s", stage.name)
        return self

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data through all stages."""
        if not self._stages:
            return data
        return self._stages[0].run(data)

    def on_market_data(self, event: MarketEvent) -> None:
        """Handle market data event."""
        self._buffer.append({
            "symbol": event.symbol,
            "timestamp": event.timestamp,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        })

        # Process when buffer is full
        if len(self._buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Process buffered data."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        result = self.process(df)

        # Emit results if event bus is connected
        if self.event_bus:
            for _, row in result.iterrows():
                # Create appropriate event based on output
                pass

        self._buffer = []

    async def start(self) -> None:
        """Start the pipeline (subscribe to events)."""
        self._running = True
        if self.event_bus:
            # Subscribe to market data events
            pass  # Event bus subscription would go here
        LOGGER.info("Pipeline started with %d stages", len(self._stages))

    async def stop(self) -> None:
        """Stop the pipeline and flush remaining data."""
        self._running = False
        self._flush_buffer()
        LOGGER.info("Pipeline stopped")


class BatchPipeline(DataPipeline):
    """Pipeline optimized for batch historical processing."""

    def __init__(self) -> None:
        super().__init__(event_bus=None, buffer_size=10000)

    def process_chunked(
        self,
        data: pd.DataFrame,
        chunk_size: int = 1000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Process large DataFrame in chunks with progress tracking."""
        total_rows = len(data)
        processed_rows = 0
        results: list[pd.DataFrame] = []

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = data.iloc[start:end]
            result = self.process(chunk)
            results.append(result)

            processed_rows += len(chunk)
            if progress_callback:
                progress_callback(processed_rows, total_rows)

        return pd.concat(results, ignore_index=True)
