"""Tests for the LangGraph orchestrator."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
from fraud_agent.agents.state import TransactionContext
from fraud_agent.data.schemas import (
    Account,
    FraudDecision,
    Location,
    RiskLevel,
    Transaction,
    TransactionChannel,
)


def make_transaction(**overrides):
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


def make_account(**overrides):
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


def _make_context(
    amount=50.0,
    mcc="5411",
    international=False,
    city="New York",
    country="US",
    lat=40.7128,
    lon=-74.006,
):
    txn = make_transaction(
        amount=Decimal(str(amount)),
        merchant_category_code=mcc,
        is_international=international,
        location=Location(city=city, country=country, latitude=lat, longitude=lon),
        channel=TransactionChannel.IN_STORE if not international else TransactionChannel.ONLINE,
    )
    acct = make_account()
    return TransactionContext(transaction=txn, account=acct)


@pytest.fixture(scope="module")
def orch():
    """Shared orchestrator — expensive to build due to LangGraph + retriever."""
    return FraudDetectionOrchestrator()


class TestOrchestrator:
    def test_orchestrator_normal_transaction(self, orch):
        """Normal (low-amount, domestic, in-store) transaction → auto-approve."""
        ctx = _make_context(amount=50.0)
        decision = orch.analyze_transaction(ctx)
        assert isinstance(decision, FraudDecision)
        assert decision.is_fraud is False

    def test_orchestrator_suspicious_transaction(self, orch):
        """Suspicious (high-amount, international, high-risk MCC) → full pipeline runs."""
        ctx = _make_context(
            amount=5000.0,
            mcc="7995",
            international=True,
            city="Macau",
            country="MO",
            lat=22.19,
            lon=113.54,
        )
        decision = orch.analyze_transaction(ctx)
        assert isinstance(decision, FraudDecision)
        assert decision.fraud_score > 0.3
        # Full pipeline should include at minimum triage + analyze + decide
        assert len(decision.agent_trace or []) >= 2

    def test_orchestrator_returns_fraud_decision(self, orch):
        """analyze_transaction always returns a FraudDecision instance."""
        ctx = _make_context()
        decision = orch.analyze_transaction(ctx)
        assert isinstance(decision, FraudDecision)
        assert decision.transaction_id == "test-txn-001"

    def test_orchestrator_batch(self, orch):
        """analyze_batch returns exactly as many decisions as contexts provided."""
        contexts = [
            _make_context(amount=50.0),
            _make_context(amount=100.0),
            _make_context(amount=200.0),
        ]
        results = orch.analyze_batch(contexts)
        assert len(results) == 3
        assert all(isinstance(r, FraudDecision) for r in results)

    def test_orchestrator_agent_trace_populated(self, orch):
        """Returned decision has a non-empty agent_trace."""
        ctx = _make_context()
        decision = orch.analyze_transaction(ctx)
        assert decision.agent_trace is not None
        assert len(decision.agent_trace) >= 1

    def test_orchestrator_decision_has_explanation(self, orch):
        """Returned decision has a non-empty explanation."""
        ctx = _make_context()
        decision = orch.analyze_transaction(ctx)
        assert isinstance(decision.explanation, str)
        assert len(decision.explanation) > 0

    def test_orchestrator_decision_risk_level_valid(self, orch):
        """Risk level in returned decision is one of the valid enum values."""
        ctx = _make_context()
        decision = orch.analyze_transaction(ctx)
        valid = {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL}
        assert decision.risk_level in valid

    def test_orchestrator_fraud_score_in_range(self, orch):
        """Fraud score is always in [0, 1]."""
        for amount in (10.0, 100.0, 5000.0):
            ctx = _make_context(amount=amount)
            decision = orch.analyze_transaction(ctx)
            assert 0.0 <= decision.fraud_score <= 1.0
