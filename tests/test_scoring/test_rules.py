"""Tests for fraud detection rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from conftest import make_account, make_recent_transactions, make_transaction

from fraud_agent.data.schemas import Location, Transaction, TransactionChannel
from fraud_agent.scoring.rules import (
    AmountRule,
    GeographicRule,
    MerchantRule,
    NewMerchantRule,
    TestingRule,
    TimeRule,
    VelocityRule,
    _haversine,
)


class TestHaversine:
    def test_same_point(self):
        assert _haversine(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0)

    def test_known_distance(self):
        dist = _haversine(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < dist < 5600


class TestVelocityRule:
    def test_triggers(self):
        rule = VelocityRule(threshold=5, window_minutes=10)
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        current_txn = make_transaction(id="current", timestamp=base_ts)
        recent = [
            make_transaction(id=f"r-{i}", timestamp=base_ts - timedelta(minutes=i + 1))
            for i in range(6)
        ]
        fired, score, explanation = rule.evaluate(current_txn, make_account(), recent)
        assert fired is True
        assert score == 0.9
        assert "transactions" in explanation

    def test_no_trigger(self):
        rule = VelocityRule(threshold=5, window_minutes=10)
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        current_txn = make_transaction(id="current", timestamp=base_ts)
        recent = make_recent_transactions(base_ts, 3, current_txn.location)
        fired, score, _ = rule.evaluate(current_txn, make_account(), recent)
        assert fired is False
        assert score == 0.0

    def test_no_recent_transactions(self):
        rule = VelocityRule()
        fired, score, _ = rule.evaluate(make_transaction(), make_account(), None)
        assert fired is False
        assert score == 0.0

    def test_excludes_current_transaction(self):
        rule = VelocityRule(threshold=5, window_minutes=10)
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        current_txn = make_transaction(id="current", timestamp=base_ts)
        recent = [current_txn] + [
            make_transaction(id=f"r-{i}", timestamp=base_ts - timedelta(minutes=i + 1))
            for i in range(4)
        ]
        fired, _, _ = rule.evaluate(current_txn, make_account(), recent)
        assert fired is False

    def test_name(self):
        assert VelocityRule().name == "velocity_rule"

    def test_above_threshold_via_conftest(
        self, normal_transaction, default_account, new_york_location
    ):
        rule = VelocityRule(threshold=5)
        recent = make_recent_transactions(normal_transaction.timestamp, 8, new_york_location)
        triggered, score, explanation = rule.evaluate(normal_transaction, default_account, recent)
        assert triggered
        assert score == 0.9
        assert "transactions" in explanation


class TestAmountRule:
    def test_triggers(self):
        rule = AmountRule(multiplier_threshold=3.0)
        account = make_account(average_transaction_amount=Decimal("50.00"))
        txn = make_transaction(amount=Decimal("250.00"))
        fired, score, explanation = rule.evaluate(txn, account, None)
        assert fired is True
        assert score > 0.0
        assert "account average" in explanation

    def test_no_trigger(self):
        rule = AmountRule(multiplier_threshold=3.0)
        account = make_account(average_transaction_amount=Decimal("100.00"))
        txn = make_transaction(amount=Decimal("200.00"))
        fired, score, _ = rule.evaluate(txn, account, None)
        assert fired is False
        assert score == 0.0

    def test_exactly_at_threshold(self):
        rule = AmountRule(multiplier_threshold=3.0)
        acct = make_account(average_transaction_amount=Decimal("100.00"))
        txn = make_transaction(amount=Decimal("300.00"))
        fired, _, _ = rule.evaluate(txn, acct)
        assert fired is False

    def test_risk_scales_with_ratio(self):
        rule = AmountRule(multiplier_threshold=3.0)
        acct = make_account(average_transaction_amount=Decimal("100.00"))
        _, low_score, _ = rule.evaluate(make_transaction(amount=Decimal("400.00")), acct)
        _, high_score, _ = rule.evaluate(make_transaction(amount=Decimal("900.00")), acct)
        assert high_score > low_score

    def test_very_high_capped(self):
        rule = AmountRule(multiplier_threshold=3.0)
        acct = make_account(average_transaction_amount=Decimal("10.00"))
        fired, score, _ = rule.evaluate(make_transaction(amount=Decimal("100000.00")), acct)
        assert fired is True
        assert score <= 1.0

    def test_name(self):
        assert AmountRule().name == "amount_rule"

    def test_normal_amount_with_conftest(self, normal_transaction, default_account):
        rule = AmountRule()
        triggered, _, _ = rule.evaluate(normal_transaction, default_account)
        assert not triggered


class TestGeographicRule:
    def test_triggers(self):
        rule = GeographicRule(distance_threshold_miles=500.0, time_threshold_hours=2.0)
        account = make_account(
            typical_location=Location(
                city="New York", country="US", latitude=40.7128, longitude=-74.006
            )
        )
        txn = make_transaction(
            location=Location(
                city="Los Angeles", country="US", latitude=34.0522, longitude=-118.2437
            )
        )
        fired, score, explanation = rule.evaluate(txn, account, recent_transactions=None)
        assert fired is True
        assert score == 0.85
        assert "mi from typical" in explanation

    def test_no_trigger(self):
        rule = GeographicRule(distance_threshold_miles=500.0)
        account = make_account(
            typical_location=Location(
                city="New York", country="US", latitude=40.7128, longitude=-74.006
            )
        )
        txn = make_transaction(
            location=Location(city="New York", country="US", latitude=40.7130, longitude=-74.0062)
        )
        fired, _, _ = rule.evaluate(txn, account, None)
        assert fired is False

    def test_elapsed_suppresses(self, default_account, london_location):
        rule = GeographicRule(distance_threshold_miles=500.0, time_threshold_hours=2.0)
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        txn = make_transaction(id="current", timestamp=base_ts, location=london_location)
        prior = make_transaction(id="prior", timestamp=base_ts - timedelta(hours=3))
        fired, _, _ = rule.evaluate(txn, default_account, [prior])
        assert not fired

    def test_impossible_travel_fires(self, default_account, london_location):
        rule = GeographicRule(distance_threshold_miles=500.0, time_threshold_hours=2.0)
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        txn = make_transaction(id="current", timestamp=base_ts, location=london_location)
        prior = make_transaction(id="prior", timestamp=base_ts - timedelta(minutes=30))
        fired, score, _ = rule.evaluate(txn, default_account, [prior])
        assert fired is True
        assert score > 0.0

    def test_close_location_with_conftest(self, normal_transaction, default_account):
        rule = GeographicRule()
        triggered, _, _ = rule.evaluate(normal_transaction, default_account)
        assert not triggered

    def test_name(self):
        assert GeographicRule().name == "geographic_rule"


class TestTimeRule:
    def test_triggers(self):
        rule = TimeRule(start_hour=2, end_hour=5)
        txn = make_transaction(timestamp=datetime(2024, 6, 15, 3, 0, tzinfo=UTC))
        fired, score, explanation = rule.evaluate(txn, make_account())
        assert fired is True
        assert score == 0.3
        assert "03:00" in explanation

    def test_no_trigger(self):
        rule = TimeRule(start_hour=2, end_hour=5)
        txn = make_transaction(timestamp=datetime(2024, 6, 15, 14, 0, tzinfo=UTC))
        fired, score, _ = rule.evaluate(txn, make_account())
        assert fired is False
        assert score == 0.0

    def test_boundary_start(self):
        rule = TimeRule(start_hour=2, end_hour=5)
        txn = make_transaction(timestamp=datetime(2024, 6, 15, 2, 0, tzinfo=UTC))
        fired, _, _ = rule.evaluate(txn, make_account())
        assert fired is True

    def test_boundary_end(self):
        rule = TimeRule(start_hour=2, end_hour=5)
        txn = make_transaction(timestamp=datetime(2024, 6, 15, 5, 59, tzinfo=UTC))
        fired, _, _ = rule.evaluate(txn, make_account())
        assert fired is True

    def test_naive_timestamp(self):
        rule = TimeRule(start_hour=2, end_hour=5)
        txn = make_transaction(timestamp=datetime(2024, 6, 15, 3, 0))
        fired, _, _ = rule.evaluate(txn, make_account())
        assert fired is True

    def test_name(self):
        assert TimeRule().name == "time_rule"

    def test_daytime_no_trigger_with_conftest(self, normal_transaction, default_account):
        rule = TimeRule()
        triggered, _, _ = rule.evaluate(normal_transaction, default_account)
        assert not triggered

    def test_nighttime_triggers_with_conftest(self, suspicious_transaction, default_account):
        rule = TimeRule()
        triggered, score, explanation = rule.evaluate(suspicious_transaction, default_account)
        assert triggered
        assert score == 0.3
        assert "unusual hour" in explanation


class TestMerchantRule:
    def test_triggers(self):
        rule = MerchantRule()
        txn = make_transaction(merchant_category_code="7995")
        fired, score, explanation = rule.evaluate(txn, make_account())
        assert fired is True
        assert score == 0.5
        assert "Gambling" in explanation

    def test_no_trigger(self):
        rule = MerchantRule()
        txn = make_transaction(merchant_category_code="5411")
        fired, score, _ = rule.evaluate(txn, make_account())
        assert fired is False
        assert score == 0.0

    def test_crypto_triggers(self):
        rule = MerchantRule()
        txn = make_transaction(merchant_category_code="6051")
        fired, _, explanation = rule.evaluate(txn, make_account())
        assert fired is True
        assert "Cryptocurrency" in explanation

    def test_wire_transfer_triggers(self):
        rule = MerchantRule()
        txn = make_transaction(merchant_category_code="4829")
        fired, _, _ = rule.evaluate(txn, make_account())
        assert fired is True

    def test_with_conftest_suspicious(self, suspicious_transaction, default_account):
        rule = MerchantRule()
        triggered, score, explanation = rule.evaluate(suspicious_transaction, default_account)
        assert triggered
        assert score == 0.5
        assert "Gambling" in explanation

    def test_with_conftest_normal(self, normal_transaction, default_account):
        rule = MerchantRule()
        triggered, _, _ = rule.evaluate(normal_transaction, default_account)
        assert not triggered

    def test_name(self):
        assert MerchantRule().name == "merchant_rule"


class TestTestingRule:
    def test_triggers(self):
        rule = TestingRule(
            small_amount_threshold=2.0, large_amount_threshold=500.0, lookback_hours=1.0
        )
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        micro_txn = make_transaction(id="micro", timestamp=base_ts, amount=Decimal("1.00"))
        large_txn = make_transaction(
            id="large", timestamp=base_ts - timedelta(minutes=30), amount=Decimal("750.00")
        )
        fired, score, explanation = rule.evaluate(micro_txn, make_account(), [large_txn])
        assert fired is True
        assert score == 0.95
        assert "Card testing" in explanation

    def test_no_trigger(self):
        rule = TestingRule(
            small_amount_threshold=2.0, large_amount_threshold=500.0, lookback_hours=1.0
        )
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        normal_txn = make_transaction(id="normal", timestamp=base_ts, amount=Decimal("50.00"))
        large_txn = make_transaction(
            id="large", timestamp=base_ts - timedelta(minutes=10), amount=Decimal("800.00")
        )
        fired, score, _ = rule.evaluate(normal_txn, make_account(), [large_txn])
        assert fired is False
        assert score == 0.0

    def test_large_outside_window(self):
        rule = TestingRule(
            small_amount_threshold=2.0, large_amount_threshold=500.0, lookback_hours=1.0
        )
        base_ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        micro_txn = make_transaction(id="micro", timestamp=base_ts, amount=Decimal("0.99"))
        old_large = make_transaction(
            id="old-large", timestamp=base_ts - timedelta(hours=2), amount=Decimal("750.00")
        )
        fired, _, _ = rule.evaluate(micro_txn, make_account(), [old_large])
        assert fired is False

    def test_no_recent_transactions(self):
        rule = TestingRule()
        txn = make_transaction(amount=Decimal("0.99"))
        fired, score, _ = rule.evaluate(txn, make_account(), None)
        assert fired is False
        assert score == 0.0

    def test_with_conftest(self, micro_transaction, default_account, large_recent_transaction):
        rule = TestingRule()
        triggered, score, explanation = rule.evaluate(
            micro_transaction, default_account, [large_recent_transaction]
        )
        assert triggered
        assert score == 0.95
        assert "Card testing" in explanation

    def test_name(self):
        assert TestingRule().name == "testing_rule"


class TestNewMerchantRule:
    def test_triggers(self):
        rule = NewMerchantRule(amount_threshold=200.0)
        txn = make_transaction(
            id="new-txn", amount=Decimal("350.00"), merchant_name="Brand New Electronics"
        )
        history = [
            make_transaction(id=f"h-{i}", merchant_name=f"Known Merchant {i}") for i in range(5)
        ]
        fired, score, explanation = rule.evaluate(txn, make_account(), history)
        assert fired is True
        assert score == 0.4
        assert "First-time merchant" in explanation

    def test_no_trigger_small_amount(self):
        rule = NewMerchantRule(amount_threshold=200.0)
        txn = make_transaction(amount=Decimal("150.00"), merchant_name="New Small Shop")
        history = [make_transaction(id=f"h-{i}", merchant_name=f"K{i}") for i in range(3)]
        fired, score, _ = rule.evaluate(txn, make_account(), history)
        assert fired is False
        assert score == 0.0

    def test_no_trigger_known_merchant(self):
        rule = NewMerchantRule(amount_threshold=200.0)
        known = "Regular Superstore"
        txn = make_transaction(amount=Decimal("500.00"), merchant_name=known)
        history = [make_transaction(id="hist-1", merchant_name=known)]
        fired, _, _ = rule.evaluate(txn, make_account(), history)
        assert fired is False

    def test_no_history_triggers(self):
        rule = NewMerchantRule(amount_threshold=200.0)
        txn = make_transaction(amount=Decimal("999.00"), merchant_name="Completely New Merchant")
        fired, score, _ = rule.evaluate(txn, make_account(), recent_transactions=None)
        assert fired is True
        assert score > 0.0

    def test_exactly_at_threshold(self):
        rule = NewMerchantRule(amount_threshold=200.0)
        txn = make_transaction(amount=Decimal("200.00"), merchant_name="New Store")
        fired, _, _ = rule.evaluate(txn, make_account(), None)
        assert fired is False

    def test_with_conftest(self, default_account, new_york_location):
        rule = NewMerchantRule(amount_threshold=200.0)
        txn = Transaction(
            id="txn-newmerch-001",
            timestamp=datetime(2024, 6, 15, 14, 0, tzinfo=UTC),
            amount=Decimal("350.00"),
            currency="USD",
            merchant_name="Brand New Store Never Seen",
            merchant_category_code="5999",
            card_last_four="1234",
            account_id="ACC-0001-0001",
            location=new_york_location,
            channel=TransactionChannel.ONLINE,
            is_international=False,
        )
        triggered, score, explanation = rule.evaluate(txn, default_account, [])
        assert triggered
        assert score == 0.4
        assert "First-time merchant" in explanation

    def test_name(self):
        assert NewMerchantRule().name == "new_merchant_rule"
