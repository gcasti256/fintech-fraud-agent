"""Tests for AnalysisAgent: feature extraction, anomaly detection, pattern matching."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from fraud_agent.agents.analyzer import AnalysisAgent
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


def _build_state(txn: Transaction, acc: Account) -> AgentState:
    return AgentState(
        transaction=txn.model_dump(mode="json"),
        account=acc.model_dump(mode="json"),
    )


@pytest.fixture(scope="module")
def analyzer():
    """Module-scoped analyzer — FraudPatternRetriever is expensive to build."""
    return AnalysisAgent()


class TestAnalysisAgent:
    def test_analyzer_updates_state(self, analyzer):
        """After running, feature_analysis and pattern_matches are populated."""
        txn = make_transaction()
        acc = make_account()
        state = _build_state(txn, acc)
        result = analyzer.run(state)
        assert isinstance(result.feature_analysis, dict)
        assert result.feature_analysis  # non-empty
        assert isinstance(result.pattern_matches, list)

    def test_analyzer_detects_anomalies(self, analyzer):
        """High-amount transaction triggers an amount-related anomaly flag."""
        txn = make_transaction(amount=Decimal("5000.00"))
        acc = make_account(average_transaction_amount=Decimal("75.00"))
        state = _build_state(txn, acc)
        result = analyzer.run(state)
        # Amount is ~66x the account average — should trigger an anomaly
        amount_flags = [f for f in result.anomaly_flags if "amount" in f.lower() or "Amount" in f]
        assert len(amount_flags) >= 1

    def test_analyzer_agent_trace(self, analyzer):
        """'analyzer' is appended to agent_trace after running."""
        txn = make_transaction()
        acc = make_account()
        state = _build_state(txn, acc)
        result = analyzer.run(state)
        assert "analyzer" in result.agent_trace

    def test_analyzer_builds_analysis_summary(self, analyzer):
        """analysis_summary is populated (non-empty string)."""
        txn = make_transaction(amount=Decimal("5000.00"), merchant_category_code="7995")
        acc = make_account()
        state = _build_state(txn, acc)
        result = analyzer.run(state)
        assert isinstance(result.analysis_summary, str)
        assert len(result.analysis_summary) > 0

    def test_analyzer_pattern_matches_structure(self, analyzer):
        """Each pattern_match dict contains required keys."""
        txn = make_transaction()
        acc = make_account()
        state = _build_state(txn, acc)
        result = analyzer.run(state)
        for match in result.pattern_matches:
            assert "pattern_id" in match
            assert "pattern_name" in match
            assert "description" in match
            assert "risk_level" in match

    def test_analyzer_anomaly_flags_are_strings(self, analyzer):
        """All anomaly flags are strings."""
        txn = make_transaction(amount=Decimal("3000.00"))
        acc = make_account()
        state = _build_state(txn, acc)
        result = analyzer.run(state)
        for flag in result.anomaly_flags:
            assert isinstance(flag, str)

    def test_analyzer_preserves_existing_trace(self, analyzer):
        """Analyzer appends to existing trace without clobbering earlier entries."""
        txn = make_transaction()
        acc = make_account()
        state = _build_state(txn, acc)
        state.agent_trace = ["triage"]
        result = analyzer.run(state)
        assert "triage" in result.agent_trace
        assert "analyzer" in result.agent_trace
