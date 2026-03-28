#!/usr/bin/env python3
"""Analyze a batch of transactions — demonstrates batch scoring with metrics."""

from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
from fraud_agent.agents.state import TransactionContext
from fraud_agent.data.generator import TransactionGenerator
from fraud_agent.monitoring.metrics import MetricsCollector

import time


def main():
    generator = TransactionGenerator(seed=42)
    orchestrator = FraudDetectionOrchestrator()
    metrics = MetricsCollector()

    # Generate 20 transactions with 10% fraud rate
    accounts = [generator.generate_account() for _ in range(5)]
    transactions = []
    for i in range(20):
        account = accounts[i % len(accounts)]
        is_fraud = i % 10 == 0  # Every 10th transaction is fraud
        txn = generator.generate_transaction(account, is_fraud=is_fraud)
        transactions.append((txn, account))

    print(f"Scoring {len(transactions)} transactions...\n")

    results = []
    for txn, account in transactions:
        context = TransactionContext(transaction=txn, account=account)

        start = time.monotonic()
        decision = orchestrator.analyze_transaction(context)
        latency = (time.monotonic() - start) * 1000

        metrics.record_scoring_latency(latency)
        metrics.record_decision(decision.risk_level.value, decision.is_fraud)
        results.append(decision)

    # Print results
    print(f"{'ID':<15} {'Score':>7} {'Risk':<10} {'Fraud?':<8} {'Action'}")
    print("-" * 60)
    for d in results:
        print(
            f"{d.transaction_id[:12]+'...':<15} "
            f"{d.fraud_score:>7.4f} "
            f"{d.risk_level.value:<10} "
            f"{'YES' if d.is_fraud else 'NO':<8} "
            f"{d.recommended_action}"
        )

    # Print metrics
    summary = metrics.get_summary()
    print(f"\n{'='*60}")
    print(f"SCORING METRICS")
    print(f"{'='*60}")
    print(f"Total scored:     {summary['total_scored']}")
    print(f"Avg latency:      {summary['avg_latency_ms']:.1f}ms")
    print(f"P95 latency:      {summary['p95_latency_ms']:.1f}ms")
    print(f"Fraud rate:       {summary['fraud_rate']:.1%}")
    print(f"Risk distribution: {summary['risk_distribution']}")


if __name__ == "__main__":
    main()
