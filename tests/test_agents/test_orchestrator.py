"""Tests for the LangGraph orchestrator."""

from decimal import Decimal

import pytest
from conftest import make_account, make_transaction

from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
from fraud_agent.agents.state import TransactionContext
from fraud_agent.data.schemas import FraudDecision, Location, RiskLevel, TransactionChannel


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
    return FraudDetectionOrchestrator()


class TestOrchestrator:
    def test_orchestrator_normal_transaction(self, orch):
        ctx = _make_context(amount=50.0)
        decision = orch.analyze_transaction(ctx)
        assert isinstance(decision, FraudDecision)
        assert decision.is_fraud is False

    def test_orchestrator_suspicious_transaction(self, orch):
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
        assert len(decision.agent_trace or []) >= 2

    def test_orchestrator_returns_fraud_decision(self, orch):
        ctx = _make_context()
        decision = orch.analyze_transaction(ctx)
        assert isinstance(decision, FraudDecision)
        assert decision.transaction_id == "test-txn-001"

    def test_orchestrator_batch(self, orch):
        contexts = [
            _make_context(amount=50.0),
            _make_context(amount=100.0),
            _make_context(amount=200.0),
        ]
        results = orch.analyze_batch(contexts)
        assert len(results) == 3
        assert all(isinstance(r, FraudDecision) for r in results)

    def test_orchestrator_agent_trace_populated(self, orch):
        decision = orch.analyze_transaction(_make_context())
        assert decision.agent_trace is not None
        assert len(decision.agent_trace) >= 1

    def test_orchestrator_decision_has_explanation(self, orch):
        decision = orch.analyze_transaction(_make_context())
        assert isinstance(decision.explanation, str)
        assert len(decision.explanation) > 0

    def test_orchestrator_decision_risk_level_valid(self, orch):
        decision = orch.analyze_transaction(_make_context())
        valid = {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL}
        assert decision.risk_level in valid

    def test_orchestrator_fraud_score_in_range(self, orch):
        for amount in (10.0, 100.0, 5000.0):
            decision = orch.analyze_transaction(_make_context(amount=amount))
            assert 0.0 <= decision.fraud_score <= 1.0
