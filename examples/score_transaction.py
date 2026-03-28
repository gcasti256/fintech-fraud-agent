#!/usr/bin/env python3
"""Score a single transaction for fraud — demonstrates the full agent pipeline."""

from datetime import datetime, date
from decimal import Decimal

from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
from fraud_agent.agents.state import TransactionContext
from fraud_agent.data.schemas import (
    Account,
    Location,
    Transaction,
    TransactionChannel,
)
from fraud_agent.guardrails.pii_masker import PIIMasker


def main():
    # Suspicious transaction: large amount, international, online, gambling merchant
    transaction = Transaction(
        id="txn-example-001",
        timestamp=datetime.now(),
        amount=Decimal("4999.99"),
        currency="USD",
        merchant_name="Lucky Stars Casino",
        merchant_category_code="7995",  # Gambling
        card_last_four="4567",
        account_id="ACC-8821-4567",
        location=Location(city="Macau", country="MO", latitude=22.1987, longitude=113.5439),
        channel=TransactionChannel.ONLINE,
        is_international=True,
    )

    account = Account(
        id="ACC-8821-4567",
        holder_name="Jane Doe",
        average_transaction_amount=Decimal("85.00"),
        typical_location=Location(
            city="San Francisco", country="US", latitude=37.7749, longitude=-122.4194
        ),
        account_open_date=date(2021, 6, 15),
        transaction_history_count=342,
    )

    # Run the fraud detection pipeline
    orchestrator = FraudDetectionOrchestrator()
    context = TransactionContext(transaction=transaction, account=account)
    decision = orchestrator.analyze_transaction(context)

    # Mask PII in output
    masker = PIIMasker()

    print("=" * 60)
    print("FRAUD ANALYSIS RESULT")
    print("=" * 60)
    print(f"Transaction ID: {masker.mask_account_id(decision.transaction_id)}")
    print(f"Risk Level:     {decision.risk_level.value}")
    print(f"Fraud Score:    {decision.fraud_score:.4f}")
    print(f"Is Fraud:       {decision.is_fraud}")
    print(f"Confidence:     {decision.confidence:.2%}")
    print(f"Action:         {decision.recommended_action}")
    print(f"Agent Trace:    {' → '.join(decision.agent_trace or [])}")
    print(f"\nExplanation:\n{masker.mask_text(decision.explanation)}")

    if decision.rules_triggered:
        print(f"\nRules Triggered:")
        for rule in decision.rules_triggered:
            print(f"  - {rule}")


if __name__ == "__main__":
    main()
