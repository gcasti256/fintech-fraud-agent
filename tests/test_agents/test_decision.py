"""Tests for DecisionAgent: scoring, risk classification, explanations, actions."""

import pytest

from fraud_agent.agents.decision import DecisionAgent
from fraud_agent.agents.state import AgentState


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
        result = agent.run(_low_risk_state())
        assert result.is_fraud is False
        assert result.final_fraud_score < 0.6

    def test_decision_high_risk(self, agent):
        result = agent.run(_high_risk_state())
        assert result.is_fraud is True
        assert result.final_fraud_score >= 0.6

    def test_decision_builds_explanation(self, agent):
        result = agent.run(_low_risk_state())
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0
        assert "score" in result.explanation.lower()

    def test_decision_explanation_mentions_verdict(self, agent):
        result = agent.run(_high_risk_state())
        assert "FRAUDULENT" in result.explanation or "LEGITIMATE" in result.explanation

    def test_decision_recommended_action(self, agent):
        result = agent.run(_low_risk_state())
        assert result.recommended_action == "approve"

    def test_decision_recommended_action_high_risk(self, agent):
        result = agent.run(_high_risk_state())
        assert result.recommended_action in ("block_and_review", "block_and_alert")

    def test_decision_agent_trace(self, agent):
        result = agent.run(AgentState(transaction={"id": "txn-trace-001"}))
        assert "decision" in result.agent_trace

    def test_decision_preserves_existing_trace(self, agent):
        state = AgentState(transaction={"id": "txn-trace-002"})
        state.agent_trace = ["triage", "analyzer"]
        result = agent.run(state)
        assert "triage" in result.agent_trace
        assert "analyzer" in result.agent_trace
        assert "decision" in result.agent_trace

    def test_decision_confidence_in_range(self, agent):
        for state in (_low_risk_state(), _high_risk_state()):
            result = agent.run(state)
            assert 0.0 < result.confidence < 1.0

    def test_decision_fraud_score_in_range(self, agent):
        for state in (_low_risk_state(), _high_risk_state()):
            result = agent.run(state)
            assert 0.0 <= result.final_fraud_score <= 1.0

    def test_decision_final_risk_level_set(self, agent):
        valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for state in (_low_risk_state(), _high_risk_state()):
            result = agent.run(state)
            assert result.final_risk_level in valid_levels
