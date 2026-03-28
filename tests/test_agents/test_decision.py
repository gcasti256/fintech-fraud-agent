"""Tests for DecisionAgent: scoring, risk classification, explanations, actions."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from fraud_agent.agents.decision import DecisionAgent
from fraud_agent.agents.state import AgentState
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


def _low_risk_state() -> AgentState:
    return AgentState(
        transaction={"id": "txn-low-001"},
        initial_risk_level="LOW",
        initial_risk_score=0.05,
        rules_triggered=[],
        anomaly_flags=[],
        pattern_matches=[],
    )


def _high_risk_state() -> AgentState:
    return AgentState(
        transaction={"id": "txn-high-001"},
        initial_risk_level="HIGH",
        initial_risk_score=0.85,
        rules_triggered=["velocity_rule", "amount_rule", "geo_rule", "time_rule"],
        anomaly_flags=[
            "Amount is 15x account average",
            "High velocity: 5 transactions in 10 minutes",
            "Transaction location significantly far from typical",
            "Transaction during high-risk nighttime hours (2-5 AM)",
            "High-risk merchant category code",
        ],
        pattern_matches=[
            {
                "pattern_id": "FP-001",
                "pattern_name": "Card Testing",
                "description": "desc",
                "risk_level": "HIGH",
                "indicators": [],
            },
            {
                "pattern_id": "FP-002",
                "pattern_name": "Account Takeover",
                "description": "desc",
                "risk_level": "CRITICAL",
                "indicators": [],
            },
            {
                "pattern_id": "FP-003",
                "pattern_name": "Velocity Abuse",
                "description": "desc",
                "risk_level": "HIGH",
                "indicators": [],
            },
        ],
    )


@pytest.fixture()
def agent():
    return DecisionAgent()


class TestDecisionAgent:
    def test_decision_low_risk(self, agent):
        """Low-risk state produces is_fraud=False and a low fraud score."""
        state = _low_risk_state()
        result = agent.run(state)
        assert result.is_fraud is False
        assert result.final_fraud_score < 0.6

    def test_decision_high_risk(self, agent):
        """High-risk state produces is_fraud=True and a high fraud score."""
        state = _high_risk_state()
        result = agent.run(state)
        assert result.is_fraud is True
        assert result.final_fraud_score >= 0.6

    def test_decision_builds_explanation(self, agent):
        """Explanation is a non-empty string containing score information."""
        state = _low_risk_state()
        result = agent.run(state)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0
        assert "score" in result.explanation.lower()

    def test_decision_explanation_mentions_verdict(self, agent):
        """Explanation states FRAUDULENT or LEGITIMATE."""
        state = _high_risk_state()
        result = agent.run(state)
        assert "FRAUDULENT" in result.explanation or "LEGITIMATE" in result.explanation

    def test_decision_recommended_action(self, agent):
        """Low-risk state gets 'approve' recommended action."""
        state = _low_risk_state()
        result = agent.run(state)
        assert result.recommended_action == "approve"

    def test_decision_recommended_action_high_risk(self, agent):
        """High-risk state gets a blocking action."""
        state = _high_risk_state()
        result = agent.run(state)
        assert result.recommended_action in ("block_and_review", "block_and_alert")

    def test_decision_agent_trace(self, agent):
        """'decision' is appended to agent_trace after running."""
        state = AgentState(transaction={"id": "txn-trace-001"})
        result = agent.run(state)
        assert "decision" in result.agent_trace

    def test_decision_preserves_existing_trace(self, agent):
        """DecisionAgent appends to existing trace entries."""
        state = AgentState(transaction={"id": "txn-trace-002"})
        state.agent_trace = ["triage", "analyzer"]
        result = agent.run(state)
        assert "triage" in result.agent_trace
        assert "analyzer" in result.agent_trace
        assert "decision" in result.agent_trace

    def test_decision_confidence_in_range(self, agent):
        """Confidence is in (0, 1) for any state."""
        for state in (_low_risk_state(), _high_risk_state()):
            result = agent.run(state)
            assert 0.0 < result.confidence < 1.0

    def test_decision_fraud_score_in_range(self, agent):
        """Final fraud score is clamped to [0, 1]."""
        for state in (_low_risk_state(), _high_risk_state()):
            result = agent.run(state)
            assert 0.0 <= result.final_fraud_score <= 1.0

    def test_decision_final_risk_level_set(self, agent):
        """final_risk_level is one of the valid risk level strings."""
        valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for state in (_low_risk_state(), _high_risk_state()):
            result = agent.run(state)
            assert result.final_risk_level in valid_levels
