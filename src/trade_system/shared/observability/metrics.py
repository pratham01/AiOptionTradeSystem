"""Lightweight in-process metrics collector.

Thread-safe counters, histograms, and gauges — no external dependencies.
Can export to Prometheus text format or JSON for health endpoints.

Usage:
    from trade_system.shared.observability.metrics import METRICS

    METRICS.increment("ticks_processed_total")
    METRICS.observe("tick_latency_ms", 2.4)
    METRICS.set_gauge("active_symbols", 212)

    print(METRICS.to_prometheus())
    print(METRICS.to_dict())
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class MetricsCollector:
    """Thread-safe in-process metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._start_time = time.monotonic()

    # ── Counters ────────────────────────────────────────────────

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a counter by value (default 1)."""
        with self._lock:
            self._counters[name] += value

    def get_counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    # ── Gauges ──────────────────────────────────────────────────

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to an absolute value."""
        with self._lock:
            self._gauges[name] = value

    def get_gauge(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, 0.0)

    # ── Histograms ──────────────────────────────────────────────

    def observe(self, name: str, value: float, max_samples: int = 1000) -> None:
        """Record an observation in a histogram (keeps last N samples)."""
        with self._lock:
            samples = self._histograms[name]
            samples.append(value)
            if len(samples) > max_samples:
                # Keep only the most recent samples
                self._histograms[name] = samples[-max_samples:]

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get summary statistics for a histogram."""
        with self._lock:
            samples = self._histograms.get(name, [])
        if not samples:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
        sorted_s = sorted(samples)
        n = len(sorted_s)
        return {
            "count": n,
            "min": round(sorted_s[0], 3),
            "max": round(sorted_s[-1], 3),
            "avg": round(sum(sorted_s) / n, 3),
            "p50": round(sorted_s[int(n * 0.5)], 3),
            "p95": round(sorted_s[min(int(n * 0.95), n - 1)], 3),
            "p99": round(sorted_s[min(int(n * 0.99), n - 1)], 3),
        }

    # ── Export ──────────────────────────────────────────────────

    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._start_time, 1)

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as a JSON-friendly dictionary."""
        with self._lock:
            result: dict[str, Any] = {
                "uptime_seconds": self.uptime_seconds(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }
        # Histogram stats computed outside lock
        for name in list(self._histograms.keys()):
            result["histograms"][name] = self.get_histogram_stats(name)
        return result

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
        lines.append(f"# TYPE uptime_seconds gauge")
        lines.append(f"uptime_seconds {self.uptime_seconds()}")

        for name in sorted(self._histograms.keys()):
            stats = self.get_histogram_stats(name)
            lines.append(f"# TYPE {name} summary")
            lines.append(f'{name}_count {stats["count"]}')
            lines.append(f'{name}_avg {stats["avg"]}')
            lines.append(f'{name}_p95 {stats["p95"]}')
            lines.append(f'{name}_p99 {stats["p99"]}')

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.monotonic()


# Global singleton instance
METRICS = MetricsCollector()
