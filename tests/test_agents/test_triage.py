"""Tests for TriageAgent and AgentState routing logic."""

from decimal import Decimal

import pytest
from conftest import make_account, make_transaction

from fraud_agent.agents.state import AgentState
from fraud_agent.agents.triage import TriageAgent
from fraud_agent.data.schemas import Location, TransactionChannel


def _build_state(transaction, account):
    return AgentState(
        transaction=transaction.model_dump(mode="json"),
        account=account.model_dump(mode="json"),
    )


@pytest.fixture()
def triage():
    return TriageAgent()


@pytest.fixture()
def normal_state():
    txn = make_transaction(amount=Decimal("50.00"))
    acc = make_account(average_transaction_amount=Decimal("75.00"))
    return _build_state(txn, acc)


@pytest.fixture()
def suspicious_state():
    txn = make_transaction(
        amount=Decimal("9500.00"),
        channel=TransactionChannel.ONLINE,
        is_international=True,
        merchant_category_code="7995",
        location=Location(city="Macau", country="MO", latitude=22.19, longitude=113.54),
    )
    acc = make_account(average_transaction_amount=Decimal("75.00"))
    return _build_state(txn, acc)


class TestTriageAgent:
    def test_triage_low_risk(self, triage, normal_state):
        result = triage.run(normal_state)
        assert result.initial_risk_level == "LOW"

    def test_triage_high_risk(self, triage, suspicious_state):
        result = triage.run(suspicious_state)
        assert result.initial_risk_level in ("MEDIUM", "HIGH", "CRITICAL")

    def test_triage_updates_state(self, triage, normal_state):
        result = triage.run(normal_state)
        assert isinstance(result.initial_risk_score, float)
        assert 0.0 <= result.initial_risk_score <= 1.0
        assert isinstance(result.triage_explanation, str)
        assert len(result.triage_explanation) > 0

    def test_triage_agent_trace(self, triage, normal_state):
        result = triage.run(normal_state)
        assert "triage" in result.agent_trace

    def test_triage_rules_triggered_list(self, triage, normal_state):
        result = triage.run(normal_state)
        assert isinstance(result.rules_triggered, list)

    def test_triage_score_in_bounds(self, triage, suspicious_state):
        result = triage.run(suspicious_state)
        assert 0.0 <= result.initial_risk_score <= 1.0

    def test_triage_preserves_existing_trace(self, triage, normal_state):
        normal_state.agent_trace = ["previous_agent"]
        result = triage.run(normal_state)
        assert "previous_agent" in result.agent_trace
        assert "triage" in result.agent_trace


class TestShouldAnalyze:
    @pytest.mark.parametrize(
        "level,expected",
        [("HIGH", "analyze"), ("CRITICAL", "analyze"), ("MEDIUM", "analyze"), ("LOW", "approve")],
    )
    def test_should_analyze(self, level, expected):
        state = AgentState(initial_risk_level=level)
        assert TriageAgent.should_analyze(state) == expected
