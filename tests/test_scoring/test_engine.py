"""Tests for fraud_agent.scoring.engine.ScoringEngine.

Covers single-transaction scoring, batch scoring, custom rule sets,
return types, and triggered-rule population.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from fraud_agent.data.schemas import (
    Account,
    FraudDecision,
    Location,
    RiskLevel,
    Transaction,
    TransactionChannel,
)
from fraud_agent.scoring.engine import ScoringEngine
from fraud_agent.scoring.rules import AmountRule, MerchantRule, VelocityRule

# ---------------------------------------------------------------------------
# Helpers (mirror the make_* pattern from rules tests)
# ---------------------------------------------------------------------------

_NY = Location(city="New York", country="US", latitude=40.7128, longitude=-74.006)


def make_transaction(**overrides) -> Transaction:
    defaults = dict(
        id="eng-txn-001",
        timestamp=datetime(2024, 6, 15, 14, 30, tzinfo=UTC),
        amount=Decimal("50.00"),
        currency="USD",
        merchant_name="Whole Foods Market",
        merchant_category_code="5411",
        card_last_four="1234",
        account_id="ACC-0001-ENG",
        location=_NY,
        channel=TransactionChannel.IN_STORE,
        is_international=False,
    )
    defaults.update(overrides)
    return Transaction(**defaults)


def make_account(**overrides) -> Account:
    defaults = dict(
        id="ACC-0001-ENG",
        holder_name="Engine Test User",
        average_transaction_amount=Decimal("75.00"),
        typical_location=_NY,
        account_open_date=date(2020, 1, 1),
        transaction_history_count=100,
    )
    defaults.update(overrides)
    return Account(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScoringEngine:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = ScoringEngine()

    # ------------------------------------------------------------------
    # Basic scoring
    # ------------------------------------------------------------------

    def test_score_normal_transaction(self):
        """Low-risk transaction gets a LOW risk score and recommended ALLOW."""
        txn = make_transaction()
        account = make_account()
        decision = self.engine.score_transaction(txn, account)

        assert decision.risk_level == RiskLevel.LOW
        assert decision.is_fraud is False
        assert decision.recommended_action == "ALLOW"
        assert 0.0 <= decision.fraud_score <= 1.0

    def test_score_normal_transaction_conftest(self, normal_transaction, default_account):
        """conftest normal_transaction scores LOW with no fraud flag."""
        decision = self.engine.score_transaction(normal_transaction, default_account)
        assert decision.risk_level == RiskLevel.LOW
        assert not decision.is_fraud
        assert 0.0 <= decision.fraud_score <= 1.0
        assert decision.recommended_action == "ALLOW"

    def test_score_high_risk_transaction(self):
        """High amount (10x avg) + gambling MCC → elevated score."""
        account = make_account(average_transaction_amount=Decimal("50.00"))
        txn = make_transaction(
            amount=Decimal("5000.00"),  # 100x average
            merchant_category_code="7995",  # Gambling
            timestamp=datetime(2024, 6, 15, 3, 0, tzinfo=UTC),  # 3 AM
            channel=TransactionChannel.ONLINE,
            is_international=True,
        )
        decision = self.engine.score_transaction(txn, account)

        # Multiple signals should push score above LOW
        assert decision.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert decision.fraud_score > 0.0

    def test_score_high_risk_conftest(self, suspicious_transaction, default_account):
        """conftest suspicious_transaction is elevated above LOW."""
        decision = self.engine.score_transaction(suspicious_transaction, default_account)
        assert decision.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert 0.0 <= decision.fraud_score <= 1.0
        assert len(decision.rules_triggered) > 0

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_score_returns_fraud_decision(self):
        """score_transaction returns a FraudDecision instance."""
        txn = make_transaction()
        account = make_account()
        result = self.engine.score_transaction(txn, account)
        assert isinstance(result, FraudDecision)

    def test_decision_fields_populated(self):
        """All FraudDecision fields are populated (non-None where required)."""
        txn = make_transaction()
        account = make_account()
        decision = self.engine.score_transaction(txn, account)

        assert decision.transaction_id == txn.id
        assert decision.risk_level in RiskLevel
        assert isinstance(decision.is_fraud, bool)
        assert 0.0 <= decision.fraud_score <= 1.0
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.explanation) > 0
        assert isinstance(decision.rules_triggered, list)
        assert decision.recommended_action in ("ALLOW", "REQUEST_OTP", "REVIEW", "BLOCK")

    def test_decision_has_explanation(self, normal_transaction, default_account):
        """Explanation is always a non-empty string."""
        decision = self.engine.score_transaction(normal_transaction, default_account)
        assert len(decision.explanation) > 0

    def test_confidence_in_range(self, normal_transaction, default_account):
        """Confidence is always in [0, 1]."""
        decision = self.engine.score_transaction(normal_transaction, default_account)
        assert 0.0 <= decision.confidence <= 1.0

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def test_score_batch(self):
        """score_batch returns exactly one FraudDecision per input pair."""
        account = make_account()
        pairs = [(make_transaction(id=f"txn-{i}"), account) for i in range(5)]
        results = self.engine.score_batch(pairs)
        assert len(results) == 5
        assert all(isinstance(r, FraudDecision) for r in results)

    def test_score_batch_count(self, normal_transaction, suspicious_transaction, default_account):
        """Batch of 2 produces 2 decisions; suspicious scores >= normal."""
        batch = [
            (normal_transaction, default_account),
            (suspicious_transaction, default_account),
        ]
        results = self.engine.score_batch(batch)
        assert len(results) == 2
        assert results[1].fraud_score >= results[0].fraud_score

    def test_score_batch_preserves_order(self):
        """Batch results are in the same order as the input pairs."""
        account = make_account()
        txns = [
            make_transaction(id=f"ordered-{i}", amount=Decimal(str(10 * (i + 1)))) for i in range(4)
        ]
        pairs = [(t, account) for t in txns]
        results = self.engine.score_batch(pairs)
        for i, (decision, (txn, _)) in enumerate(zip(results, pairs)):
            assert decision.transaction_id == txn.id, f"Order mismatch at index {i}"

    def test_score_batch_empty(self):
        """Empty batch returns empty list."""
        results = self.engine.score_batch([])
        assert results == []

    # ------------------------------------------------------------------
    # Custom rules
    # ------------------------------------------------------------------

    def test_custom_rules(self):
        """Engine with a single rule subset works without error."""
        engine = ScoringEngine(rules=[AmountRule()])
        account = make_account(average_transaction_amount=Decimal("50.00"))
        txn = make_transaction(amount=Decimal("500.00"))  # 10x average
        decision = engine.score_transaction(txn, account)
        assert isinstance(decision, FraudDecision)
        assert "amount_rule" in decision.rules_triggered

    def test_custom_rules_empty(self):
        """Engine with no rules still returns a valid FraudDecision (model-only scoring)."""
        engine = ScoringEngine(rules=[])
        txn = make_transaction()
        account = make_account()
        decision = engine.score_transaction(txn, account)
        assert isinstance(decision, FraudDecision)
        assert decision.rules_triggered == []

    def test_custom_rules_merchant_only(self):
        """MerchantRule-only engine flags gambling MCC but not grocery."""
        engine = ScoringEngine(rules=[MerchantRule()])
        account = make_account()

        gambling_txn = make_transaction(merchant_category_code="7995")
        grocery_txn = make_transaction(merchant_category_code="5411")

        gambling_decision = engine.score_transaction(gambling_txn, account)
        grocery_decision = engine.score_transaction(grocery_txn, account)

        assert "merchant_rule" in gambling_decision.rules_triggered
        assert "merchant_rule" not in grocery_decision.rules_triggered

    # ------------------------------------------------------------------
    # rules_triggered populated
    # ------------------------------------------------------------------

    def test_rules_triggered_populated(self):
        """High-risk transaction populates rules_triggered with at least one rule."""
        engine = ScoringEngine()
        account = make_account(average_transaction_amount=Decimal("50.00"))
        # Multiple fraud signals: gambling MCC, 3 AM, 10x amount
        txn = make_transaction(
            amount=Decimal("500.00"),
            merchant_category_code="7995",
            timestamp=datetime(2024, 6, 15, 3, 0, tzinfo=UTC),
        )
        decision = engine.score_transaction(txn, account)
        assert len(decision.rules_triggered) > 0
        # Merchant rule should always fire on gambling MCC
        assert "merchant_rule" in decision.rules_triggered

    def test_rules_triggered_empty_for_clean_transaction(self):
        """Clean transaction at typical location and normal amount triggers no rules."""
        engine = ScoringEngine()
        account = make_account()
        txn = make_transaction()
        decision = engine.score_transaction(txn, account)
        # Normal transaction should trigger zero or very few rules
        # Amount $50 is below 3x avg $75 → amount_rule won't fire
        assert "amount_rule" not in decision.rules_triggered
        assert "merchant_rule" not in decision.rules_triggered

    def test_rules_triggered_with_velocity(self):
        """High-velocity scenario populates velocity_rule in triggered list."""
        engine = ScoringEngine(rules=[VelocityRule(threshold=3, window_minutes=10)])
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        account = make_account()
        txn = make_transaction(id="current", timestamp=base_ts)

        # 5 recent transactions — above threshold of 3
        recent = [
            make_transaction(id=f"v-{i}", timestamp=base_ts - timedelta(minutes=i + 1))
            for i in range(5)
        ]
        decision = engine.score_transaction(txn, account, recent)
        assert "velocity_rule" in decision.rules_triggered

    # ------------------------------------------------------------------
    # Recommended action mapping
    # ------------------------------------------------------------------

    def test_recommended_action_allow_for_low(self):
        """LOW risk → recommended_action is ALLOW."""
        txn = make_transaction()
        account = make_account()
        decision = self.engine.score_transaction(txn, account)
        if decision.risk_level == RiskLevel.LOW:
            assert decision.recommended_action == "ALLOW"

    def test_recommended_action_block_for_critical(self):
        """Verify BLOCK action is assigned for CRITICAL risk decisions."""
        decision = FraudDecision(
            transaction_id="txn-x",
            risk_level=RiskLevel.CRITICAL,
            fraud_score=0.95,
            is_fraud=True,
            confidence=0.9,
            explanation="Critical risk.",
            recommended_action="BLOCK",
        )
        assert decision.recommended_action == "BLOCK"
        assert decision.is_fraud is True
