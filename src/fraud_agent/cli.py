"""CLI for the fraud detection agent system."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date, datetime
from decimal import Decimal

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fraud_agent import __version__

console = Console()

_RISK_COLORS: dict[str, str] = {
    "LOW": "green",
    "MEDIUM": "yellow",
    "HIGH": "red",
    "CRITICAL": "bold red",
}


def _dict_to_transaction(data: dict):
    """Convert a raw dict to a Transaction domain object."""
    from fraud_agent.data.schemas import Location, Transaction, TransactionChannel

    return Transaction(
        id=data.get("id", str(uuid.uuid4())),
        timestamp=datetime.fromisoformat(data["timestamp"])
        if "timestamp" in data
        else datetime.now(),
        amount=Decimal(str(data.get("amount", 0))),
        currency=data.get("currency", "USD"),
        merchant_name=data.get("merchant_name", "Unknown"),
        merchant_category_code=data.get("merchant_category_code", "5999"),
        card_last_four=data.get("card_last_four", "0000"),
        account_id=data.get("account_id", "ACC-0001-0001"),
        location=Location(
            **data.get(
                "location",
                {
                    "city": "New York",
                    "country": "US",
                    "latitude": 40.7128,
                    "longitude": -74.006,
                },
            )
        ),
        channel=TransactionChannel(data.get("channel", "ONLINE")),
        is_international=data.get("is_international", False),
    )


def _default_account(account_id: str, avg_amount: float = 75.0):
    """Build a default account for CLI scoring commands."""
    from fraud_agent.data.schemas import Account, Location

    return Account(
        id=account_id,
        holder_name="Account Holder",
        average_transaction_amount=Decimal(str(avg_amount)),
        typical_location=Location(
            city="New York", country="US", latitude=40.7128, longitude=-74.006
        ),
        account_open_date=date(2020, 1, 1),
        transaction_history_count=100,
    )


@click.group()
@click.version_option(version=__version__)
def cli():
    """Fraud Detection Agent — Agentic fraud analysis for financial services."""
    pass


@cli.command()
@click.option("--transaction", "-t", required=True, help="Transaction JSON string")
@click.option("--account-avg", "-a", default=75.0, help="Account average transaction amount")
def score(transaction: str, account_avg: float):
    """Score a single transaction for fraud."""
    from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
    from fraud_agent.agents.state import TransactionContext
    from fraud_agent.guardrails.pii_masker import PIIMasker

    try:
        txn_data = json.loads(transaction)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        sys.exit(1)

    txn = _dict_to_transaction(txn_data)
    account = _default_account(txn.account_id, account_avg)

    orchestrator = FraudDetectionOrchestrator()
    context = TransactionContext(transaction=txn, account=account)
    decision = orchestrator.analyze_transaction(context)

    masker = PIIMasker()

    color = _RISK_COLORS.get(decision.risk_level.value, "white")

    console.print(
        Panel.fit(
            f"[bold]Transaction:[/bold] {masker.mask_account_id(decision.transaction_id)}\n"
            f"[bold]Risk Level:[/bold] [{color}]{decision.risk_level.value}[/{color}]\n"
            f"[bold]Fraud Score:[/bold] {decision.fraud_score:.4f}\n"
            f"[bold]Is Fraud:[/bold] {'[red]YES[/red]' if decision.is_fraud else '[green]NO[/green]'}\n"
            f"[bold]Confidence:[/bold] {decision.confidence:.2%}\n"
            f"[bold]Action:[/bold] {decision.recommended_action}\n"
            f"[bold]Agent Trace:[/bold] {' → '.join(decision.agent_trace or [])}\n\n"
            f"[bold]Explanation:[/bold]\n{masker.mask_text(decision.explanation)}",
            title="Fraud Analysis Result",
        )
    )

    if decision.rules_triggered:
        table = Table(title="Rules Triggered")
        table.add_column("Rule", style="bold")
        for rule in decision.rules_triggered:
            table.add_row(rule)
        console.print(table)


@cli.command()
@click.option(
    "--file",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with transactions",
)
@click.option("--account-avg", "-a", default=75.0, help="Account average transaction amount")
def batch(file: str, account_avg: float):
    """Score a batch of transactions from a JSON file."""
    from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
    from fraud_agent.agents.state import TransactionContext

    with open(file) as f:
        transactions_data = json.load(f)

    if not isinstance(transactions_data, list):
        transactions_data = [transactions_data]

    orchestrator = FraudDetectionOrchestrator()
    results = []

    with console.status("Scoring transactions..."):
        for txn_data in transactions_data:
            txn = _dict_to_transaction(txn_data)
            account = _default_account(txn.account_id, account_avg)
            context = TransactionContext(transaction=txn, account=account)
            decision = orchestrator.analyze_transaction(context)
            results.append(decision)

    table = Table(title=f"Batch Results ({len(results)} transactions)")
    table.add_column("Transaction", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Risk", justify="center")
    table.add_column("Fraud?", justify="center")
    table.add_column("Action")

    for d in results:
        color = _RISK_COLORS.get(d.risk_level.value, "white")
        table.add_row(
            d.transaction_id[:12] + "...",
            f"{d.fraud_score:.4f}",
            f"[{color}]{d.risk_level.value}[/{color}]",
            "[red]YES[/red]" if d.is_fraud else "[green]NO[/green]",
            d.recommended_action,
        )

    console.print(table)

    if results:
        flagged = sum(1 for d in results if d.is_fraud)
        console.print(
            f"\n[bold]Total:[/bold] {len(results)} | "
            f"[bold]Flagged:[/bold] {flagged} | "
            f"[bold]Rate:[/bold] {flagged / len(results):.1%}"
        )


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the REST API server."""
    import uvicorn

    console.print(f"[bold green]Starting REST API server on {host}:{port}[/bold green]")
    uvicorn.run(
        "fraud_agent.api.rest:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@cli.command(name="grpc")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=50051, help="Port to listen on")
def grpc_serve(host: str, port: int):
    """Start the gRPC server."""
    from fraud_agent.api.grpc_server import serve as grpc_start

    console.print(f"[bold green]Starting gRPC server on {host}:{port}[/bold green]")
    server = grpc_start(host=host, port=port)
    console.print("[green]gRPC server running. Press Ctrl+C to stop.[/green]")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)
        console.print("\n[yellow]gRPC server stopped.[/yellow]")


@cli.command()
@click.option("--count", "-n", default=100, help="Number of transactions to generate")
@click.option("--fraud-rate", "-r", default=0.05, help="Fraction of fraudulent transactions")
@click.option("--output", "-o", default=None, help="Output file (default: stdout)")
@click.option("--seed", "-s", default=42, help="Random seed")
def generate(count: int, fraud_rate: float, output: str | None, seed: int):
    """Generate synthetic transaction data."""
    from fraud_agent.data.generator import TransactionGenerator

    generator = TransactionGenerator(seed=seed)
    transactions = generator.generate_batch(count=count, fraud_rate=fraud_rate)

    data = [json.loads(t.model_dump_json()) for t in transactions]

    if output:
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"[green]Generated {count} transactions → {output}[/green]")
    else:
        click.echo(json.dumps(data, indent=2, default=str))


@cli.command()
@click.option(
    "--file",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with labelled transactions",
)
@click.option("--account-avg", "-a", default=75.0, help="Account average transaction amount")
def evaluate(file: str, account_avg: float):
    """Run evaluation on a labelled dataset and report accuracy metrics."""
    import time

    from fraud_agent.scoring.engine import ScoringEngine

    with open(file) as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        dataset = [dataset]

    engine = ScoringEngine()
    tp = fp = tn = fn = 0
    total_latency = 0.0

    for entry in dataset:
        label = entry.get(
            "is_fraud", (entry.get("metadata") or {}).get("fraud_pattern") is not None
        )
        txn = _dict_to_transaction(entry)
        account = _default_account(txn.account_id, account_avg)

        start = time.monotonic()
        decision = engine.score_transaction(txn, account)
        total_latency += (time.monotonic() - start) * 1000

        predicted = decision.is_fraud
        if label and predicted:
            tp += 1
        elif label and not predicted:
            fn += 1
        elif not label and predicted:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total samples", str(total))
    table.add_row("True positives", str(tp))
    table.add_row("False positives", str(fp))
    table.add_row("True negatives", str(tn))
    table.add_row("False negatives", str(fn))
    table.add_row("Accuracy", f"{accuracy:.2%}")
    table.add_row("Precision", f"{precision:.2%}")
    table.add_row("Recall", f"{recall:.2%}")
    table.add_row("F1 Score", f"{f1:.2%}")
    table.add_row("Avg latency", f"{total_latency / total:.1f}ms" if total else "N/A")

    console.print(table)


@cli.command()
def dashboard():
    """Show the scoring performance metrics dashboard."""
    from fraud_agent.monitoring.dashboard import MetricsDashboard
    from fraud_agent.monitoring.metrics import MetricsCollector

    collector = MetricsCollector()
    dash = MetricsDashboard(collector)
    console.print(dash.render_text())


@cli.group()
def patterns():
    """Manage fraud pattern knowledge base."""
    pass


@patterns.command(name="list")
def list_patterns():
    """List all fraud patterns in the knowledge base."""
    from fraud_agent.data.knowledge_base import FraudKnowledgeBase

    kb = FraudKnowledgeBase()
    all_patterns = kb.get_patterns()

    table = Table(title=f"Fraud Patterns ({len(all_patterns)})")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Category")
    table.add_column("Risk Level")
    table.add_column("Description", max_width=50)

    for p in all_patterns:
        color = _RISK_COLORS.get(p["risk_level"].upper(), "white")
        table.add_row(
            p["id"],
            p["name"],
            p["category"],
            f"[{color}]{p['risk_level']}[/{color}]",
            p["description"][:50] + "..." if len(p["description"]) > 50 else p["description"],
        )

    console.print(table)


@cli.command()
def metrics():
    """Show scoring performance metrics (alias for dashboard)."""
    from fraud_agent.monitoring.dashboard import MetricsDashboard
    from fraud_agent.monitoring.metrics import MetricsCollector

    collector = MetricsCollector()
    dash = MetricsDashboard(collector)
    console.print(dash.render_text())


if __name__ == "__main__":
    cli()
