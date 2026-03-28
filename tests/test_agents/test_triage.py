"""Tests for TriageAgent and AgentState routing logic."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from fraud_agent.agents.state import AgentState
from fraud_agent.agents.triage import TriageAgent
from fraud_agent.data.schemas import (
    Account,
    Location,
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


def _build_state(transaction: Transaction, account: Account) -> AgentState:
    """Construct an AgentState from domain objects for test convenience."""
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
    """High-amount international online transaction — should trigger high risk."""
    txn = make_transaction(
        amount=Decimal("9500.00"),
        channel=TransactionChannel.ONLINE,
        is_international=True,
        merchant_category_code="7995",
        location=Location(city="Macau", country="MO", latitude=22.19, longitude=113.54),
    )
    acc = make_account(average_transaction_amount=Decimal("75.00"))
    return _build_state(txn, acc)


# ---------------------------------------------------------------------------
# Basic triage output
# ---------------------------------------------------------------------------


class TestTriageAgent:
    def test_triage_low_risk(self, triage, normal_state):
        """Normal transaction gets LOW initial risk level."""
        result = triage.run(normal_state)
        assert result.initial_risk_level == "LOW"

    def test_triage_high_risk(self, triage, suspicious_state):
        """Suspicious transaction gets an elevated (non-LOW) risk level."""
        result = triage.run(suspicious_state)
        assert result.initial_risk_level in ("MEDIUM", "HIGH", "CRITICAL")

    def test_triage_updates_state(self, triage, normal_state):
        """State fields initial_risk_score and triage_explanation are populated."""
        result = triage.run(normal_state)
        assert isinstance(result.initial_risk_score, float)
        assert 0.0 <= result.initial_risk_score <= 1.0
        assert isinstance(result.triage_explanation, str)
        assert len(result.triage_explanation) > 0

    def test_triage_agent_trace(self, triage, normal_state):
        """'triage' is appended to agent_trace after running."""
        result = triage.run(normal_state)
        assert "triage" in result.agent_trace

    def test_triage_rules_triggered_list(self, triage, normal_state):
        """rules_triggered is a list (may be empty for low-risk)."""
        result = triage.run(normal_state)
        assert isinstance(result.rules_triggered, list)

    def test_triage_score_in_bounds(self, triage, suspicious_state):
        """Fraud score is in [0, 1] for suspicious transaction."""
        result = triage.run(suspicious_state)
        assert 0.0 <= result.initial_risk_score <= 1.0

    def test_triage_preserves_existing_trace(self, triage, normal_state):
        """Triage appends to an existing agent trace rather than overwriting it."""
        normal_state.agent_trace = ["previous_agent"]
        result = triage.run(normal_state)
        assert "previous_agent" in result.agent_trace
        assert "triage" in result.agent_trace


# ---------------------------------------------------------------------------
# Routing via should_analyze
# ---------------------------------------------------------------------------


class TestShouldAnalyze:
    def test_should_analyze_high(self):
        """HIGH risk routes to 'analyze'."""
        state = AgentState(initial_risk_level="HIGH")
        assert TriageAgent.should_analyze(state) == "analyze"

    def test_should_analyze_critical(self):
        """CRITICAL risk routes to 'analyze'."""
        state = AgentState(initial_risk_level="CRITICAL")
        assert TriageAgent.should_analyze(state) == "analyze"

    def test_should_analyze_medium(self):
        """MEDIUM risk routes to 'analyze'."""
        state = AgentState(initial_risk_level="MEDIUM")
        assert TriageAgent.should_analyze(state) == "analyze"

    def test_should_analyze_low(self):
        """LOW risk routes to 'approve'."""
        state = AgentState(initial_risk_level="LOW")
        assert TriageAgent.should_analyze(state) == "approve"
