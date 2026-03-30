"""Tests for AnalysisAgent: feature extraction, anomaly detection, pattern matching."""

from decimal import Decimal

import pytest
from conftest import make_account, make_transaction

from fraud_agent.agents.analyzer import AnalysisAgent
from fraud_agent.agents.state import AgentState
from fraud_agent.data.schemas import Account, Transaction


def _build_state(txn: Transaction, acc: Account) -> AgentState:
    return AgentState(
        transaction=txn.model_dump(mode="json"),
        account=acc.model_dump(mode="json"),
    )


@pytest.fixture(scope="module")
def analyzer():
    return AnalysisAgent()


class TestAnalysisAgent:
    def test_analyzer_updates_state(self, analyzer):
        state = _build_state(make_transaction(), make_account())
        result = analyzer.run(state)
        assert isinstance(result.feature_analysis, dict)
        assert result.feature_analysis
        assert isinstance(result.pattern_matches, list)

    def test_analyzer_detects_anomalies(self, analyzer):
        txn = make_transaction(amount=Decimal("5000.00"))
        acc = make_account(average_transaction_amount=Decimal("75.00"))
        state = _build_state(txn, acc)
        result = analyzer.run(state)
        amount_flags = [f for f in result.anomaly_flags if "amount" in f.lower() or "Amount" in f]
        assert len(amount_flags) >= 1

    def test_analyzer_agent_trace(self, analyzer):
        state = _build_state(make_transaction(), make_account())
        result = analyzer.run(state)
        assert "analyzer" in result.agent_trace

    def test_analyzer_builds_analysis_summary(self, analyzer):
        txn = make_transaction(amount=Decimal("5000.00"), merchant_category_code="7995")
        state = _build_state(txn, make_account())
        result = analyzer.run(state)
        assert isinstance(result.analysis_summary, str)
        assert len(result.analysis_summary) > 0

    def test_analyzer_pattern_matches_structure(self, analyzer):
        state = _build_state(make_transaction(), make_account())
        result = analyzer.run(state)
        for match in result.pattern_matches:
            assert "pattern_id" in match
            assert "pattern_name" in match
            assert "description" in match
            assert "risk_level" in match

    def test_analyzer_anomaly_flags_are_strings(self, analyzer):
        txn = make_transaction(amount=Decimal("3000.00"))
        state = _build_state(txn, make_account())
        result = analyzer.run(state)
        for flag in result.anomaly_flags:
            assert isinstance(flag, str)

    def test_analyzer_preserves_existing_trace(self, analyzer):
        state = _build_state(make_transaction(), make_account())
        state.agent_trace = ["triage"]
        result = analyzer.run(state)
        assert "triage" in result.agent_trace
        assert "analyzer" in result.agent_trace
