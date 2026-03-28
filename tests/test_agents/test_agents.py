"""Tests for agent pipeline (triage, analyzer, decision, orchestrator)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from fraud_agent.agents.analyzer import AnalysisAgent
from fraud_agent.agents.decision import DecisionAgent
from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
from fraud_agent.agents.state import AgentState, TransactionContext
from fraud_agent.agents.triage import TriageAgent
from fraud_agent.data.schemas import (
    Account,
    FraudDecision,
    Location,
    RiskLevel,
    Transaction,
    TransactionChannel,
)


@pytest.fixture
def ny_loc():
    return Location(city="New York", country="US", latitude=40.7128, longitude=-74.0060)


@pytest.fixture
def account(ny_loc):
    return Account(
        id="ACC-TEST-0001",
        holder_name="Test User",
        average_transaction_amount=Decimal("75.00"),
        typical_location=ny_loc,
        account_open_date=date(2020, 1, 1),
        transaction_history_count=100,
    )


@pytest.fixture
def normal_txn(ny_loc):
    return Transaction(
        id="txn-agent-normal",
        timestamp=datetime(2024, 6, 15, 14, 0, 0, tzinfo=UTC),
        amount=Decimal("42.50"),
        currency="USD",
        merchant_name="Grocery Store",
        merchant_category_code="5411",
        card_last_four="1234",
        account_id="ACC-TEST-0001",
        location=ny_loc,
        channel=TransactionChannel.IN_STORE,
        is_international=False,
    )


@pytest.fixture
def suspicious_txn():
    return Transaction(
        id="txn-agent-suspicious",
        timestamp=datetime(2024, 6, 15, 3, 0, 0, tzinfo=UTC),
        amount=Decimal("4999.99"),
        currency="USD",
        merchant_name="Lucky Casino",
        merchant_category_code="7995",
        card_last_four="5678",
        account_id="ACC-TEST-0001",
        location=Location(city="London", country="GB", latitude=51.5074, longitude=-0.1278),
        channel=TransactionChannel.ONLINE,
        is_international=True,
    )


class TestTriageAgent:
    def test_triage_low_risk(self, normal_txn, account):
        agent = TriageAgent()
        state = AgentState(
            transaction=normal_txn.model_dump(mode="json"),
            account=account.model_dump(mode="json"),
        )
        result = agent.run(state)
        assert result.initial_risk_level == "LOW"
        assert "triage" in result.agent_trace

    def test_triage_higher_risk(self, suspicious_txn, account):
        agent = TriageAgent()
        state = AgentState(
            transaction=suspicious_txn.model_dump(mode="json"),
            account=account.model_dump(mode="json"),
        )
        result = agent.run(state)
        assert result.initial_risk_level in ("MEDIUM", "HIGH", "CRITICAL")
        assert result.initial_risk_score > 0.0

    def test_should_analyze_routing(self):
        low_state = AgentState(initial_risk_level="LOW")
        assert TriageAgent.should_analyze(low_state) == "approve"

        high_state = AgentState(initial_risk_level="HIGH")
        assert TriageAgent.should_analyze(high_state) == "analyze"


class TestAnalysisAgent:
    def test_analysis_detects_anomalies(self, suspicious_txn, account):
        agent = AnalysisAgent()
        state = AgentState(
            transaction=suspicious_txn.model_dump(mode="json"),
            account=account.model_dump(mode="json"),
            initial_risk_level="HIGH",
            initial_risk_score=0.7,
        )
        result = agent.run(state)
        assert len(result.anomaly_flags) > 0
        assert "analyzer" in result.agent_trace

    def test_analysis_retrieves_patterns(self, suspicious_txn, account):
        agent = AnalysisAgent()
        state = AgentState(
            transaction=suspicious_txn.model_dump(mode="json"),
            account=account.model_dump(mode="json"),
        )
        result = agent.run(state)
        assert len(result.pattern_matches) > 0

    def test_analysis_extracts_features(self, normal_txn, account):
        agent = AnalysisAgent()
        state = AgentState(
            transaction=normal_txn.model_dump(mode="json"),
            account=account.model_dump(mode="json"),
        )
        result = agent.run(state)
        assert "amount_ratio" in result.feature_analysis


class TestDecisionAgent:
    def test_decision_low_risk(self):
        agent = DecisionAgent()
        state = AgentState(
            transaction={"id": "txn-001"},
            initial_risk_level="LOW",
            initial_risk_score=0.1,
            anomaly_flags=[],
            pattern_matches=[],
            rules_triggered=[],
        )
        result = agent.run(state)
        assert result.final_risk_level in ("LOW", "MEDIUM")
        assert "decision" in result.agent_trace

    def test_decision_high_risk(self):
        agent = DecisionAgent()
        state = AgentState(
            transaction={"id": "txn-002"},
            initial_risk_level="HIGH",
            initial_risk_score=0.85,
            anomaly_flags=["anomaly1", "anomaly2", "anomaly3"],
            pattern_matches=[
                {"pattern_name": "Card Testing", "pattern_id": "FP-001"},
                {"pattern_name": "Account Takeover", "pattern_id": "FP-002"},
            ],
            rules_triggered=["velocity_rule", "amount_rule", "merchant_rule"],
        )
        result = agent.run(state)
        assert result.is_fraud
        assert result.final_fraud_score > 0.5

    def test_confidence_calculation(self):
        agent = DecisionAgent()
        state = AgentState(
            transaction={"id": "txn-003"},
            initial_risk_level="MEDIUM",
            initial_risk_score=0.5,
            anomaly_flags=["a1"],
            pattern_matches=[{"pattern_name": "P1", "pattern_id": "FP-001"}],
            rules_triggered=["r1"],
        )
        result = agent.run(state)
        assert 0.1 <= result.confidence <= 0.99


class TestOrchestrator:
    def test_full_pipeline_normal(self, normal_txn, account):
        orchestrator = FraudDetectionOrchestrator()
        context = TransactionContext(transaction=normal_txn, account=account)
        decision = orchestrator.analyze_transaction(context)
        assert isinstance(decision, FraudDecision)
        assert decision.risk_level == RiskLevel.LOW
        assert not decision.is_fraud
        assert len(decision.agent_trace) >= 1

    def test_full_pipeline_suspicious(self, suspicious_txn, account):
        orchestrator = FraudDetectionOrchestrator()
        context = TransactionContext(transaction=suspicious_txn, account=account)
        decision = orchestrator.analyze_transaction(context)
        assert isinstance(decision, FraudDecision)
        assert decision.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert len(decision.agent_trace) >= 2

    def test_batch_analysis(self, normal_txn, suspicious_txn, account):
        orchestrator = FraudDetectionOrchestrator()
        contexts = [
            TransactionContext(transaction=normal_txn, account=account),
            TransactionContext(transaction=suspicious_txn, account=account),
        ]
        results = orchestrator.analyze_batch(contexts)
        assert len(results) == 2
