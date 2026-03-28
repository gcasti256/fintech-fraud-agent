"""In-memory metrics collector for the fraud detection agent.

All methods are thread-safe.  Use :class:`MetricsCollector` as a process-level
singleton (or inject it as a dependency) to accumulate scoring telemetry and
expose it for dashboards and health checks.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import Any


class MetricsCollector:
    """Thread-safe, in-memory metrics store.

    Collects latency samples, decision outcomes, throughput events, and
    arbitrary named counters.  All data lives in process memory; use
    :meth:`reset` to clear between test runs or rolling windows.

    Example::

        collector = MetricsCollector()
        collector.record_scoring_latency(42.3)
        collector.record_decision("HIGH", is_fraud=True)
        print(collector.get_summary())
    """

    def __init__(self) -> None:
        self._lock = Lock()

        # Latency samples in milliseconds.
        self._latencies: list[float] = []

        # Decision counters keyed by risk_level string.
        self._risk_distribution: dict[str, int] = defaultdict(int)

        # Fraud / total counts for rate calculation.
        self._total_scored: int = 0
        self._total_fraud: int = 0

        # Throughput: list of epoch timestamps (seconds) for each scored event.
        self._throughput_timestamps: list[float] = []

        # Generic named counters (keyed by a stable string key).
        self._counters: dict[str, int] = defaultdict(int)

        # Wall-clock start time for uptime reporting.
        self._start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_scoring_latency(self, latency_ms: float) -> None:
        """Record a single end-to-end scoring latency sample.

        Args:
            latency_ms: Elapsed time in milliseconds for one scoring request.
        """
        with self._lock:
            self._latencies.append(latency_ms)

    def record_decision(self, risk_level: str, is_fraud: bool) -> None:
        """Record the outcome of a fraud scoring decision.

        Updates total scored count, fraud count, and per-risk-level distribution.

        Args:
            risk_level: The :class:`~fraud_agent.data.schemas.RiskLevel` string
                value (e.g. ``"HIGH"``).
            is_fraud: Whether the transaction was classified as fraudulent.
        """
        with self._lock:
            self._total_scored += 1
            self._risk_distribution[risk_level.upper()] += 1
            if is_fraud:
                self._total_fraud += 1

    def record_throughput(self) -> None:
        """Record that one transaction was processed at the current moment.

        Call this once per scored transaction.  The collector stores the
        wall-clock timestamp so that :meth:`get_summary` can compute a
        rolling events-per-second rate.
        """
        with self._lock:
            self._throughput_timestamps.append(time.monotonic())
            # Discard events older than 60 seconds to bound memory usage.
            cutoff = time.monotonic() - 60.0
            self._throughput_timestamps = [t for t in self._throughput_timestamps if t >= cutoff]

    def increment_counter(self, name: str, labels: dict[str, Any] | None = None) -> None:
        """Increment a named counter by one.

        ``labels`` are currently unused in the in-memory store but accepted for
        API compatibility with future label-aware backends (e.g. Prometheus).

        Args:
            name: Counter name (e.g. ``"rule_velocity_triggered"``).
            labels: Optional key-value labels (stored on the key for future use).
        """
        key = name
        if labels:
            # Encode labels into the key so different label sets are distinct.
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{name}{{{label_str}}}"

        with self._lock:
            self._counters[key] += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a snapshot of all collected metrics.

        Percentiles are computed from a sorted copy of the latency list using
        the nearest-rank method.

        Returns:
            Dictionary with the following keys:

            - ``total_scored`` (:class:`int`): Total transactions evaluated.
            - ``avg_latency_ms`` (:class:`float`): Mean scoring latency.
            - ``p95_latency_ms`` (:class:`float`): 95th-percentile latency.
            - ``p99_latency_ms`` (:class:`float`): 99th-percentile latency.
            - ``throughput_per_second`` (:class:`float`): Events/s in last 60 s.
            - ``fraud_rate`` (:class:`float`): Fraction of scored txns flagged.
            - ``risk_distribution`` (:class:`dict`): Counts per risk level.
            - ``counters`` (:class:`dict`): Named counter values.
            - ``uptime_seconds`` (:class:`float`): Seconds since construction.
        """
        with self._lock:
            latencies = sorted(self._latencies)
            n = len(latencies)

            avg_latency = sum(latencies) / n if n else 0.0
            p95_latency = self._percentile(latencies, 95) if n else 0.0
            p99_latency = self._percentile(latencies, 99) if n else 0.0

            now = time.monotonic()
            cutoff = now - 60.0
            recent = [t for t in self._throughput_timestamps if t >= cutoff]
            throughput = len(recent) / 60.0 if recent else 0.0

            fraud_rate = self._total_fraud / self._total_scored if self._total_scored else 0.0

            uptime = now - self._start_time

            return {
                "total_scored": self._total_scored,
                "avg_latency_ms": round(avg_latency, 3),
                "p95_latency_ms": round(p95_latency, 3),
                "p99_latency_ms": round(p99_latency, 3),
                "throughput_per_second": round(throughput, 4),
                "fraud_rate": round(fraud_rate, 6),
                "risk_distribution": dict(self._risk_distribution),
                "counters": dict(self._counters),
                "uptime_seconds": round(uptime, 2),
            }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated metrics and restart the uptime clock.

        Useful between test runs or when implementing rolling-window semantics
        at a higher level.
        """
        with self._lock:
            self._latencies.clear()
            self._risk_distribution = defaultdict(int)
            self._total_scored = 0
            self._total_fraud = 0
            self._throughput_timestamps.clear()
            self._counters = defaultdict(int)
            self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(sorted_data: list[float], pct: int) -> float:
        """Compute a percentile from a pre-sorted list using the nearest-rank method.

        Args:
            sorted_data: A sorted (ascending) list of numeric values.
            pct: The desired percentile (0–100 inclusive).

        Returns:
            The value at the requested percentile, or ``0.0`` for an empty list.
        """
        if not sorted_data:
            return 0.0
        n = len(sorted_data)
        # Nearest-rank formula: ceil(pct / 100 * n) clamped to [1, n].
        rank = max(1, min(n, int(pct / 100.0 * n + 0.5)))
        return sorted_data[rank - 1]
