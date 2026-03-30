"""Tests for ScoringEngine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from conftest import make_account, make_transaction

from fraud_agent.data.schemas import FraudDecision, RiskLevel, TransactionChannel
from fraud_agent.scoring.engine import ScoringEngine
from fraud_agent.scoring.rules import AmountRule, MerchantRule, VelocityRule


class TestScoringEngine:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = ScoringEngine()

    def test_score_normal_transaction(self):
        decision = self.engine.score_transaction(make_transaction(), make_account())
        assert decision.risk_level == RiskLevel.LOW
        assert decision.is_fraud is False
        assert decision.recommended_action == "ALLOW"
        assert 0.0 <= decision.fraud_score <= 1.0

    def test_score_normal_transaction_conftest(self, normal_transaction, default_account):
        decision = self.engine.score_transaction(normal_transaction, default_account)
        assert decision.risk_level == RiskLevel.LOW
        assert not decision.is_fraud
        assert 0.0 <= decision.fraud_score <= 1.0
        assert decision.recommended_action == "ALLOW"

    def test_score_high_risk_transaction(self):
        account = make_account(average_transaction_amount=Decimal("50.00"))
        txn = make_transaction(
            amount=Decimal("5000.00"),
            merchant_category_code="7995",
            timestamp=datetime(2024, 6, 15, 3, 0, tzinfo=UTC),
            channel=TransactionChannel.ONLINE,
            is_international=True,
        )
        decision = self.engine.score_transaction(txn, account)
        assert decision.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert decision.fraud_score > 0.0

    def test_score_high_risk_conftest(self, suspicious_transaction, default_account):
        decision = self.engine.score_transaction(suspicious_transaction, default_account)
        assert decision.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert 0.0 <= decision.fraud_score <= 1.0
        assert len(decision.rules_triggered) > 0

    def test_score_returns_fraud_decision(self):
        result = self.engine.score_transaction(make_transaction(), make_account())
        assert isinstance(result, FraudDecision)

    def test_decision_fields_populated(self):
        txn = make_transaction()
        decision = self.engine.score_transaction(txn, make_account())
        assert decision.transaction_id == txn.id
        assert decision.risk_level in RiskLevel
        assert isinstance(decision.is_fraud, bool)
        assert 0.0 <= decision.fraud_score <= 1.0
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.explanation) > 0
        assert isinstance(decision.rules_triggered, list)
        assert decision.recommended_action in ("ALLOW", "REQUEST_OTP", "REVIEW", "BLOCK")

    def test_decision_has_explanation(self, normal_transaction, default_account):
        decision = self.engine.score_transaction(normal_transaction, default_account)
        assert len(decision.explanation) > 0

    def test_confidence_in_range(self, normal_transaction, default_account):
        decision = self.engine.score_transaction(normal_transaction, default_account)
        assert 0.0 <= decision.confidence <= 1.0

    def test_score_batch(self):
        account = make_account()
        pairs = [(make_transaction(id=f"txn-{i}"), account) for i in range(5)]
        results = self.engine.score_batch(pairs)
        assert len(results) == 5
        assert all(isinstance(r, FraudDecision) for r in results)

    def test_score_batch_count(self, normal_transaction, suspicious_transaction, default_account):
        batch = [
            (normal_transaction, default_account),
            (suspicious_transaction, default_account),
        ]
        results = self.engine.score_batch(batch)
        assert len(results) == 2
        assert results[1].fraud_score >= results[0].fraud_score

    def test_score_batch_preserves_order(self):
        account = make_account()
        txns = [
            make_transaction(id=f"ordered-{i}", amount=Decimal(str(10 * (i + 1)))) for i in range(4)
        ]
        pairs = [(t, account) for t in txns]
        results = self.engine.score_batch(pairs)
        for i, (decision, (txn, _)) in enumerate(zip(results, pairs)):
            assert decision.transaction_id == txn.id, f"Order mismatch at index {i}"

    def test_score_batch_empty(self):
        results = self.engine.score_batch([])
        assert results == []

    def test_custom_rules(self):
        engine = ScoringEngine(rules=[AmountRule()])
        account = make_account(average_transaction_amount=Decimal("50.00"))
        txn = make_transaction(amount=Decimal("500.00"))
        decision = engine.score_transaction(txn, account)
        assert isinstance(decision, FraudDecision)
        assert "amount_rule" in decision.rules_triggered

    def test_custom_rules_empty(self):
        engine = ScoringEngine(rules=[])
        decision = engine.score_transaction(make_transaction(), make_account())
        assert isinstance(decision, FraudDecision)
        assert decision.rules_triggered == []

    def test_custom_rules_merchant_only(self):
        engine = ScoringEngine(rules=[MerchantRule()])
        account = make_account()
        gambling = engine.score_transaction(
            make_transaction(merchant_category_code="7995"), account
        )
        grocery = engine.score_transaction(make_transaction(merchant_category_code="5411"), account)
        assert "merchant_rule" in gambling.rules_triggered
        assert "merchant_rule" not in grocery.rules_triggered

    def test_rules_triggered_populated(self):
        engine = ScoringEngine()
        account = make_account(average_transaction_amount=Decimal("50.00"))
        txn = make_transaction(
            amount=Decimal("500.00"),
            merchant_category_code="7995",
            timestamp=datetime(2024, 6, 15, 3, 0, tzinfo=UTC),
        )
        decision = engine.score_transaction(txn, account)
        assert len(decision.rules_triggered) > 0
        assert "merchant_rule" in decision.rules_triggered

    def test_rules_triggered_empty_for_clean_transaction(self):
        decision = self.engine.score_transaction(make_transaction(), make_account())
        assert "amount_rule" not in decision.rules_triggered
        assert "merchant_rule" not in decision.rules_triggered

    def test_rules_triggered_with_velocity(self):
        engine = ScoringEngine(rules=[VelocityRule(threshold=3, window_minutes=10)])
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        txn = make_transaction(id="current", timestamp=base_ts)
        recent = [
            make_transaction(id=f"v-{i}", timestamp=base_ts - timedelta(minutes=i + 1))
            for i in range(5)
        ]
        decision = engine.score_transaction(txn, make_account(), recent)
        assert "velocity_rule" in decision.rules_triggered

    def test_recommended_action_allow_for_low(self):
        decision = self.engine.score_transaction(make_transaction(), make_account())
        assert decision.risk_level == RiskLevel.LOW
        assert decision.recommended_action == "ALLOW"
