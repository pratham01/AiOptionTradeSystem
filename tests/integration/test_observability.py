"""Integration tests for the observability stack — metrics, health, and health server."""

import json
import threading
import time
import urllib.request

import pytest

from trade_system.shared.observability.metrics import MetricsCollector
from trade_system.shared.observability.health import HealthChecker


class TestMetricsCollector:
    """Test the in-process metrics collector."""

    def test_counter_increment(self):
        m = MetricsCollector()
        m.increment("test_counter")
        m.increment("test_counter")
        m.increment("test_counter", 5)
        assert m.get_counter("test_counter") == 7.0

    def test_gauge_set(self):
        m = MetricsCollector()
        m.set_gauge("active_symbols", 42)
        assert m.get_gauge("active_symbols") == 42
        m.set_gauge("active_symbols", 100)
        assert m.get_gauge("active_symbols") == 100

    def test_histogram_observe(self):
        m = MetricsCollector()
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.observe("latency_ms", val)
        stats = m.get_histogram_stats("latency_ms")
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["avg"] == 3.0

    def test_histogram_empty(self):
        m = MetricsCollector()
        stats = m.get_histogram_stats("nonexistent")
        assert stats["count"] == 0

    def test_to_dict(self):
        m = MetricsCollector()
        m.increment("requests_total", 10)
        m.set_gauge("db_size_mb", 2800.5)
        d = m.to_dict()
        assert d["counters"]["requests_total"] == 10
        assert d["gauges"]["db_size_mb"] == 2800.5
        assert "uptime_seconds" in d

    def test_to_prometheus_format(self):
        m = MetricsCollector()
        m.increment("http_requests_total", 100)
        m.set_gauge("uptime_gauge", 3600)
        text = m.to_prometheus()
        assert "http_requests_total 100" in text
        assert "uptime_gauge 3600" in text

    def test_thread_safety(self):
        """Stress test: concurrent increments from multiple threads."""
        m = MetricsCollector()
        errors = []

        def worker(n):
            try:
                for _ in range(1000):
                    m.increment("concurrent_counter")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert m.get_counter("concurrent_counter") == 10000.0

    def test_reset(self):
        m = MetricsCollector()
        m.increment("a", 5)
        m.set_gauge("b", 10)
        m.observe("c", 1.0)
        m.reset()
        assert m.get_counter("a") == 0
        assert m.get_gauge("b") == 0
        assert m.get_histogram_stats("c")["count"] == 0


class TestHealthChecker:
    """Test the health checker."""

    def test_check_all_returns_structure(self):
        checker = HealthChecker()
        result = checker.check_all()
        assert "status" in result
        assert "timestamp" in result
        assert "uptime_seconds" in result
        assert "components" in result

    def test_custom_check_healthy(self):
        checker = HealthChecker()
        checker.register_check("test_component", lambda: {"status": "healthy"})
        result = checker.check_all()
        assert result["components"]["test_component"]["status"] == "healthy"

    def test_custom_check_unhealthy_degrades_overall(self):
        checker = HealthChecker()
        checker.register_check("failing", lambda: {"status": "unhealthy", "error": "down"})
        result = checker.check_all()
        assert result["status"] == "degraded"

    def test_check_exception_caught(self):
        checker = HealthChecker()
        checker.register_check("broken", lambda: 1/0)
        result = checker.check_all()
        assert result["components"]["broken"]["status"] == "unhealthy"
        assert "division by zero" in result["components"]["broken"]["error"]

    def test_is_healthy_boolean(self):
        checker = HealthChecker()
        # With only DB check (may or may not be healthy in test env)
        result = checker.is_healthy()
        assert isinstance(result, bool)


class TestHealthServer:
    """Test the HTTP health server."""

    def test_health_server_starts_and_responds(self):
        """Test that the health server starts and responds to /health."""
        from trade_system.interfaces.live.health_server import start_health_server

        # Use a random high port to avoid conflicts
        server = start_health_server(port=19090)
        if server is None:
            pytest.skip("Port 19090 already in use")

        try:
            time.sleep(0.3)  # Give server time to start

            # Test /health endpoint
            req = urllib.request.Request("http://localhost:19090/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "status" in data

            # Test /metrics endpoint
            req = urllib.request.Request("http://localhost:19090/metrics")
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200
                body = resp.read().decode()
                assert "uptime_seconds" in body

            # Test 404 for unknown paths
            try:
                req = urllib.request.Request("http://localhost:19090/unknown")
                urllib.request.urlopen(req, timeout=5)
                assert False, "Should have gotten 404"
            except urllib.error.HTTPError as e:
                assert e.code == 404

        finally:
            server.shutdown()
