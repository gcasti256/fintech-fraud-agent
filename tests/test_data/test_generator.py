"""Tests for fraud_agent.data.generator.TransactionGenerator.

Covers account generation, normal/fraud transaction generation,
batch generation, seed reproducibility, and field-level sanity.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from fraud_agent.data.generator import TransactionGenerator
from fraud_agent.data.schemas import Account, Location, Transaction, TransactionChannel


@pytest.fixture
def gen() -> TransactionGenerator:
    """A deterministic generator seeded at 42."""
    return TransactionGenerator(seed=42)


@pytest.fixture
def account(gen: TransactionGenerator) -> Account:
    """One synthetic account from the seeded generator."""
    return gen.generate_account()


class TestGenerateAccount:
    def test_generate_account(self):
        """generate_account() returns a valid Account with realistic data."""
        gen = TransactionGenerator(seed=42)
        acc = gen.generate_account()
        assert isinstance(acc, Account)
        assert acc.id and len(acc.id) > 0
        # Name is 'First Last' — two non-empty parts
        parts = acc.holder_name.split()
        assert len(parts) == 2
        assert all(len(p) > 0 for p in parts)
        # Typical location is always a US city in the generator
        assert acc.typical_location.country == "US"
        assert acc.average_transaction_amount > Decimal("0")
        assert acc.account_open_date < date.today()
        assert acc.transaction_history_count >= 0

    def test_generate_account_unique_ids(self):
        """Two consecutive accounts get distinct UUIDs."""
        gen = TransactionGenerator(seed=42)
        a1 = gen.generate_account()
        a2 = gen.generate_account()
        assert a1.id != a2.id

    def test_generate_account_location_is_location(self):
        """typical_location is a Location instance."""
        gen = TransactionGenerator(seed=42)
        acc = gen.generate_account()
        assert isinstance(acc.typical_location, Location)


class TestGenerateNormalTransaction:
    def test_generate_normal_transaction(self, gen: TransactionGenerator, account: Account):
        """generate_transaction(is_fraud=False) returns a valid Transaction."""
        txn = gen.generate_transaction(account, is_fraud=False)
        assert isinstance(txn, Transaction)
        assert txn.account_id == account.id
        assert txn.currency == "USD"

    def test_generate_normal_transaction_amount_in_range(
        self, gen: TransactionGenerator, account: Account
    ):
        """Normal amounts are clamped to [1.00, 5000.00]."""
        for _ in range(20):
            txn = gen.generate_transaction(account, is_fraud=False)
            assert Decimal("1.00") <= txn.amount <= Decimal("5000.00")

    def test_generate_normal_transaction_not_international(
        self, gen: TransactionGenerator, account: Account
    ):
        """Normal domestic transactions always have is_international=False."""
        results = [gen.generate_transaction(account, is_fraud=False) for _ in range(30)]
        assert all(not txn.is_international for txn in results)

    def test_generate_normal_transaction_channel_is_enum(
        self, gen: TransactionGenerator, account: Account
    ):
        """Channel field is always a valid TransactionChannel."""
        for _ in range(10):
            txn = gen.generate_transaction(account, is_fraud=False)
            assert isinstance(txn.channel, TransactionChannel)


class TestGenerateFraudTransaction:
    def test_generate_fraud_transaction(self, gen: TransactionGenerator, account: Account):
        """generate_transaction(is_fraud=True) returns a Transaction."""
        txn = gen.generate_transaction(account, is_fraud=True)
        assert isinstance(txn, Transaction)
        assert txn.metadata is not None
        assert "fraud_pattern" in txn.metadata

    def test_generate_fraud_transaction_amount_positive(
        self, gen: TransactionGenerator, account: Account
    ):
        """Fraud transaction amounts are always > 0."""
        for _ in range(20):
            txn = gen.generate_transaction(account, is_fraud=True)
            assert txn.amount > Decimal("0")

    def test_generate_fraud_transaction_anomalous_amount(self):
        """Amount-anomaly fraud transactions reach 5x+ the account average in enough tries."""
        gen = TransactionGenerator(seed=7)
        acc = gen.generate_account()
        max_ratio = 0.0
        for _ in range(50):
            txn = gen.generate_transaction(acc, is_fraud=True)
            ratio = float(txn.amount) / float(acc.average_transaction_amount)
            max_ratio = max(max_ratio, ratio)
        # The amount_anomaly pattern is 5-20x; should appear within 50 samples
        assert max_ratio >= 3.0


class TestGenerateBatch:
    def test_generate_batch_count(self, gen: TransactionGenerator):
        """generate_batch returns exactly the requested count."""
        batch = gen.generate_batch(count=50)
        assert len(batch) == 50

    def test_generate_batch_returns_transactions(self, gen: TransactionGenerator):
        """Every element in the batch is a Transaction."""
        batch = gen.generate_batch(count=10)
        assert all(isinstance(t, Transaction) for t in batch)

    def test_generate_batch_fraud_rate(self):
        """Batch fraud count matches round(count * fraud_rate) exactly."""
        gen = TransactionGenerator(seed=42)
        count = 200
        fraud_rate = 0.10
        batch = gen.generate_batch(count=count, fraud_rate=fraud_rate)
        fraud_count = sum(1 for t in batch if t.metadata and "fraud_pattern" in t.metadata)
        expected = round(count * fraud_rate)
        assert fraud_count == expected

    def test_generate_batch_fraud_rate_zero(self, gen: TransactionGenerator):
        """fraud_rate=0.0 produces no fraudulent transactions."""
        batch = gen.generate_batch(count=20, fraud_rate=0.0)
        fraud_count = sum(1 for t in batch if t.metadata and "fraud_pattern" in t.metadata)
        assert fraud_count == 0

    def test_generate_batch_fraud_rate_one(self, gen: TransactionGenerator):
        """fraud_rate=1.0 makes all transactions fraudulent."""
        batch = gen.generate_batch(count=20, fraud_rate=1.0)
        fraud_count = sum(1 for t in batch if t.metadata and "fraud_pattern" in t.metadata)
        assert fraud_count == 20

    def test_generate_batch_invalid_fraud_rate(self, gen: TransactionGenerator):
        """fraud_rate outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            gen.generate_batch(count=10, fraud_rate=1.5)
        with pytest.raises(ValueError):
            gen.generate_batch(count=10, fraud_rate=-0.1)


class TestSeedReproducibility:
    def test_seed_reproducibility(self):
        """Same seed produces identical Account output."""
        g1 = TransactionGenerator(seed=99)
        g2 = TransactionGenerator(seed=99)
        a1 = g1.generate_account()
        a2 = g2.generate_account()
        assert a1.holder_name == a2.holder_name
        assert a1.average_transaction_amount == a2.average_transaction_amount
        assert a1.typical_location == a2.typical_location

    def test_seed_reproducibility_batch(self):
        """Same seed produces the same batch MCC sequence."""
        g1 = TransactionGenerator(seed=7)
        g2 = TransactionGenerator(seed=7)
        b1 = g1.generate_batch(count=10)
        b2 = g2.generate_batch(count=10)
        assert [t.amount for t in b1] == [t.amount for t in b2]
        assert [t.merchant_name for t in b1] == [t.merchant_name for t in b2]

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        g1 = TransactionGenerator(seed=1)
        g2 = TransactionGenerator(seed=2)
        b1 = g1.generate_batch(count=10)
        b2 = g2.generate_batch(count=10)
        amounts1 = [t.amount for t in b1]
        amounts2 = [t.amount for t in b2]
        assert amounts1 != amounts2


class TestFieldsPopulated:
    def test_transaction_fields_populated(self, gen: TransactionGenerator, account: Account):
        """All required Transaction string fields are non-empty and correct format."""
        txn = gen.generate_transaction(account, is_fraud=False)
        assert txn.id and len(txn.id) > 0
        assert txn.merchant_name and len(txn.merchant_name) > 0
        assert len(txn.merchant_category_code) == 4
        assert txn.merchant_category_code.isdigit()
        assert len(txn.card_last_four) == 4
        assert txn.card_last_four.isdigit()
        assert txn.account_id == account.id
        assert txn.currency == "USD"
        assert isinstance(txn.location, Location)
        assert isinstance(txn.channel, TransactionChannel)

    def test_amounts_are_positive(self, gen: TransactionGenerator, account: Account):
        """All generated transaction amounts are strictly positive."""
        for is_fraud in [False, True]:
            for _ in range(10):
                txn = gen.generate_transaction(account, is_fraud=is_fraud)
                assert txn.amount > Decimal("0"), (
                    f"Non-positive amount {txn.amount} for is_fraud={is_fraud}"
                )

    def test_timestamp_is_timezone_aware(self, gen: TransactionGenerator, account: Account):
        """Generated transaction timestamps are always timezone-aware."""
        txn = gen.generate_transaction(account, is_fraud=False)
        assert txn.timestamp.tzinfo is not None
