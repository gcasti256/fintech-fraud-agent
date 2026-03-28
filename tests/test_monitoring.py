"""Tests for monitoring — metrics and dashboard."""

from __future__ import annotations

from fraud_agent.monitoring.dashboard import MetricsDashboard, _format_uptime
from fraud_agent.monitoring.metrics import MetricsCollector


class TestMetricsCollector:
    def test_record_latency(self):
        collector = MetricsCollector()
        collector.record_scoring_latency(42.0)
        collector.record_scoring_latency(58.0)
        summary = collector.get_summary()
        assert summary["avg_latency_ms"] == 50.0

    def test_record_decision(self):
        collector = MetricsCollector()
        collector.record_decision("HIGH", True)
        collector.record_decision("LOW", False)
        summary = collector.get_summary()
        assert summary["total_scored"] == 2
        assert summary["fraud_rate"] == 0.5
        assert summary["risk_distribution"]["HIGH"] == 1
        assert summary["risk_distribution"]["LOW"] == 1

    def test_increment_counter(self):
        collector = MetricsCollector()
        collector.increment_counter("test_counter")
        collector.increment_counter("test_counter")
        summary = collector.get_summary()
        assert summary["counters"]["test_counter"] == 2

    def test_counter_with_labels(self):
        collector = MetricsCollector()
        collector.increment_counter("rule_fired", {"rule": "velocity"})
        summary = collector.get_summary()
        assert "rule_fired{rule=velocity}" in summary["counters"]

    def test_reset(self):
        collector = MetricsCollector()
        collector.record_scoring_latency(100.0)
        collector.record_decision("HIGH", True)
        collector.reset()
        summary = collector.get_summary()
        assert summary["total_scored"] == 0
        assert summary["avg_latency_ms"] == 0.0

    def test_percentile(self):
        data = list(range(1, 101))
        p95 = MetricsCollector._percentile(data, 95)
        assert p95 == 95

    def test_percentile_empty(self):
        assert MetricsCollector._percentile([], 95) == 0.0


class TestMetricsDashboard:
    def test_render_text(self):
        collector = MetricsCollector()
        collector.record_scoring_latency(42.0)
        collector.record_decision("HIGH", True)
        dashboard = MetricsDashboard(collector)
        text = dashboard.render_text()
        assert "Overview" in text
        assert "Risk Distribution" in text

    def test_to_dict(self):
        collector = MetricsCollector()
        dashboard = MetricsDashboard(collector)
        data = dashboard.to_dict()
        assert "total_scored" in data
        assert "uptime_display" in data


class TestFormatUptime:
    def test_seconds_only(self):
        assert _format_uptime(45.0) == "45s"

    def test_minutes_and_seconds(self):
        assert _format_uptime(125.0) == "2m 5s"

    def test_hours(self):
        assert _format_uptime(3661.0) == "1h 1m 1s"
