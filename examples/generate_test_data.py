#!/usr/bin/env python3
"""Generate synthetic transaction data for testing and development."""

import json

from fraud_agent.data.generator import TransactionGenerator


def main():
    generator = TransactionGenerator(seed=42)

    # Generate 100 transactions with 5% fraud rate
    transactions = generator.generate_batch(count=100, fraud_rate=0.05)

    data = [json.loads(t.model_dump_json()) for t in transactions]

    # Write to file
    with open("sample_transactions.json", "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Print summary
    fraud_count = sum(1 for t in transactions if t.is_international)
    print(f"Generated {len(transactions)} transactions")
    print(f"International: {fraud_count}")
    print(f"Written to: sample_transactions.json")

    # Show first few
    print("\nFirst 5 transactions:")
    for t in transactions[:5]:
        print(f"  {t.id[:12]}... | ${t.amount:>8} | {t.merchant_name:<25} | {t.location.city}")


if __name__ == "__main__":
    main()
