"""Shared pytest fixtures for the fraud detection agent test suite."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from fraud_agent.data.schemas import (
    Account,
    Location,
    Transaction,
    TransactionChannel,
)


@pytest.fixture
def new_york_location() -> Location:
    return Location(city="New York", country="US", latitude=40.7128, longitude=-74.0060)


@pytest.fixture
def london_location() -> Location:
    return Location(city="London", country="GB", latitude=51.5074, longitude=-0.1278)


@pytest.fixture
def tokyo_location() -> Location:
    return Location(city="Tokyo", country="JP", latitude=35.6762, longitude=139.6503)


@pytest.fixture
def default_account(new_york_location) -> Account:
    return Account(
        id="ACC-TEST-0001",
        holder_name="Test User",
        average_transaction_amount=Decimal("75.00"),
        typical_location=new_york_location,
        account_open_date=date(2020, 1, 1),
        transaction_history_count=100,
    )


@pytest.fixture
def high_spender_account(new_york_location) -> Account:
    return Account(
        id="ACC-TEST-0002",
        holder_name="High Spender",
        average_transaction_amount=Decimal("500.00"),
        typical_location=new_york_location,
        account_open_date=date(2018, 6, 15),
        transaction_history_count=500,
    )


@pytest.fixture
def normal_transaction(new_york_location) -> Transaction:
    return Transaction(
        id="txn-normal-001",
        timestamp=datetime(2024, 6, 15, 14, 30, 0, tzinfo=UTC),
        amount=Decimal("42.50"),
        currency="USD",
        merchant_name="Whole Foods Market",
        merchant_category_code="5411",
        card_last_four="1234",
        account_id="ACC-TEST-0001",
        location=new_york_location,
        channel=TransactionChannel.IN_STORE,
        is_international=False,
    )


@pytest.fixture
def suspicious_transaction(london_location) -> Transaction:
    return Transaction(
        id="txn-suspicious-001",
        timestamp=datetime(2024, 6, 15, 3, 0, 0, tzinfo=UTC),
        amount=Decimal("4999.99"),
        currency="USD",
        merchant_name="Lucky Stars Casino",
        merchant_category_code="7995",
        card_last_four="5678",
        account_id="ACC-TEST-0001",
        location=london_location,
        channel=TransactionChannel.ONLINE,
        is_international=True,
    )


@pytest.fixture
def micro_transaction(new_york_location) -> Transaction:
    return Transaction(
        id="txn-micro-001",
        timestamp=datetime(2024, 6, 15, 14, 30, 0, tzinfo=UTC),
        amount=Decimal("0.99"),
        currency="USD",
        merchant_name="Test Store",
        merchant_category_code="5999",
        card_last_four="9999",
        account_id="ACC-TEST-0001",
        location=new_york_location,
        channel=TransactionChannel.ONLINE,
        is_international=False,
    )


@pytest.fixture
def large_recent_transaction(new_york_location) -> Transaction:
    return Transaction(
        id="txn-large-001",
        timestamp=datetime(2024, 6, 15, 14, 0, 0, tzinfo=UTC),
        amount=Decimal("999.99"),
        currency="USD",
        merchant_name="Electronics Store",
        merchant_category_code="5732",
        card_last_four="9999",
        account_id="ACC-TEST-0001",
        location=new_york_location,
        channel=TransactionChannel.ONLINE,
        is_international=False,
    )


def make_transaction(**overrides) -> Transaction:
    defaults = {
        "id": "test-txn-001",
        "timestamp": datetime(2024, 6, 15, 14, 30),
        "amount": Decimal("50.00"),
        "currency": "USD",
        "merchant_name": "Test Store",
        "merchant_category_code": "5411",
        "card_last_four": "1234",
        "account_id": "ACC-0001-1234",
        "location": Location(city="New York", country="US", latitude=40.7128, longitude=-74.006),
        "channel": TransactionChannel.IN_STORE,
        "is_international": False,
    }
    defaults.update(overrides)
    return Transaction(**defaults)


def make_account(**overrides) -> Account:
    defaults = {
        "id": "ACC-0001-1234",
        "holder_name": "Test User",
        "average_transaction_amount": Decimal("75.00"),
        "typical_location": Location(
            city="New York", country="US", latitude=40.7128, longitude=-74.006
        ),
        "account_open_date": date(2020, 1, 1),
        "transaction_history_count": 100,
    }
    defaults.update(overrides)
    return Account(**defaults)


def make_recent_transactions(
    base_ts: datetime, count: int, location: Location
) -> list[Transaction]:
    """Helper to create a list of recent transactions for velocity tests."""
    from datetime import timedelta

    txns = []
    for i in range(count):
        txns.append(
            Transaction(
                id=f"txn-recent-{i:03d}",
                timestamp=base_ts - timedelta(minutes=i),
                amount=Decimal("25.00"),
                currency="USD",
                merchant_name=f"Store {i}",
                merchant_category_code="5411",
                card_last_four="1111",
                account_id="ACC-TEST-0001",
                location=location,
                channel=TransactionChannel.IN_STORE,
                is_international=False,
            )
        )
    return txns


@pytest.fixture
def tmp_db_path():
    """Provide a temporary database path that is cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def db(tmp_db_path):
    from fraud_agent.db import Database

    database = Database(db_path=tmp_db_path)
    yield database
    database.close()
