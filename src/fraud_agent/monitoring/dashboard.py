"""Text-based metrics dashboard for the fraud detection agent.

Renders a human-readable summary of collected metrics using the
:mod:`rich` library.  Suitable for CLI health checks, operator consoles,
and stdout logging.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fraud_agent.monitoring.metrics import MetricsCollector


class MetricsDashboard:
    """Renders a text dashboard from a :class:`MetricsCollector` snapshot.

    Args:
        collector: The metrics collector instance to read from.

    Example::

        dashboard = MetricsDashboard(collector)
        print(dashboard.render_text())
    """

    def __init__(self, collector: MetricsCollector) -> None:
        self._collector = collector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_text(self) -> str:
        """Render a rich text dashboard as a string.

        Builds a panel containing two tables:

        1. **Overview** — throughput, latency percentiles, fraud rate, uptime.
        2. **Risk Distribution** — per-risk-level transaction counts.

        A third table for non-empty counter values is appended when present.

        Returns:
            A string with ANSI escape sequences suitable for terminal output.
            Use ``rich.console.Console(force_terminal=False)`` output if plain
            text is required.
        """
        summary = self._collector.get_summary()
        console = Console(record=True, highlight=False)

        # --- Overview table ------------------------------------------------
        overview = Table(
            title="Overview",
            show_header=True,
            header_style="bold cyan",
            min_width=45,
        )
        overview.add_column("Metric", style="dim", no_wrap=True)
        overview.add_column("Value", justify="right")

        uptime_s = summary["uptime_seconds"]
        uptime_display = _format_uptime(uptime_s)

        overview.add_row("Transactions Scored", str(summary["total_scored"]))
        overview.add_row(
            "Throughput",
            f"{summary['throughput_per_second']:.2f} txn/s",
        )
        overview.add_row("Avg Latency", f"{summary['avg_latency_ms']:.1f} ms")
        overview.add_row("p95 Latency", f"{summary['p95_latency_ms']:.1f} ms")
        overview.add_row("p99 Latency", f"{summary['p99_latency_ms']:.1f} ms")
        overview.add_row(
            "Fraud Rate",
            f"{summary['fraud_rate'] * 100:.2f}%",
        )
        overview.add_row("Uptime", uptime_display)

        # --- Risk distribution table ---------------------------------------
        risk_table = Table(
            title="Risk Distribution",
            show_header=True,
            header_style="bold cyan",
            min_width=30,
        )
        risk_table.add_column("Risk Level", style="dim", no_wrap=True)
        risk_table.add_column("Count", justify="right")

        risk_dist: dict[str, int] = summary["risk_distribution"]
        total = sum(risk_dist.values()) or 1  # avoid division by zero

        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            count = risk_dist.get(level, 0)
            pct = count / total * 100
            colour = {"CRITICAL": "red", "HIGH": "yellow", "MEDIUM": "blue", "LOW": "green"}.get(
                level, "white"
            )
            risk_table.add_row(
                f"[{colour}]{level}[/{colour}]",
                f"{count} ({pct:.1f}%)",
            )

        # --- Counters table (optional) ------------------------------------
        counters: dict[str, int] = summary["counters"]

        # --- Composite panel ----------------------------------------------
        console.print(
            Panel(overview, title="[bold]Fraud Detection Agent[/bold]", border_style="bright_blue")
        )
        console.print(Panel(risk_table, border_style="bright_blue"))

        if counters:
            counter_table = Table(
                title="Counters",
                show_header=True,
                header_style="bold cyan",
                min_width=40,
            )
            counter_table.add_column("Name", style="dim")
            counter_table.add_column("Value", justify="right")

            for name, value in sorted(counters.items()):
                counter_table.add_row(name, str(value))

            console.print(Panel(counter_table, border_style="bright_blue"))

        return console.export_text()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all metrics.

        Delegates directly to :meth:`MetricsCollector.get_summary` and
        adds a human-readable ``uptime_display`` field.

        Returns:
            Dictionary suitable for JSON serialisation.
        """
        summary = self._collector.get_summary()
        summary["uptime_display"] = _format_uptime(summary["uptime_seconds"])
        return summary


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        String in the form ``"Xh Ym Zs"`` (hours/minutes omitted when zero).
    """
    secs = int(seconds)
    hours, remainder = divmod(secs, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)
