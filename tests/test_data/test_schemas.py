"""Tests for fraud_agent.data.schemas — all Pydantic models.

Comprehensive test coverage for Location, Transaction, TransactionChannel,
RiskLevel, Account, and FraudDecision.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fraud_agent.data.schemas import (
    Account,
    FraudDecision,
    Location,
    RiskLevel,
    Transaction,
    TransactionChannel,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _loc(**kw) -> Location:
    base = dict(city="New York", country="US", latitude=40.7128, longitude=-74.006)
    base.update(kw)
    return Location(**base)


def _txn(**kw) -> Transaction:
    base = dict(
        id="txn-abc-001",
        timestamp=datetime(2024, 6, 15, 14, 30, tzinfo=UTC),
        amount=Decimal("125.00"),
        currency="USD",
        merchant_name="Test Merchant",
        merchant_category_code="5411",
        card_last_four="1234",
        account_id="ACC-0001",
        location=_loc(),
        channel=TransactionChannel.IN_STORE,
        is_international=False,
    )
    base.update(kw)
    return Transaction(**base)


def _acct(**kw) -> Account:
    base = dict(
        id="ACC-0001",
        holder_name="Jane Doe",
        average_transaction_amount=Decimal("75.00"),
        typical_location=_loc(),
        account_open_date=date(2019, 3, 15),
        transaction_history_count=250,
    )
    base.update(kw)
    return Account(**base)


def _decision(**kw) -> FraudDecision:
    base = dict(
        transaction_id="txn-abc-001",
        risk_level=RiskLevel.LOW,
        fraud_score=0.1,
        is_fraud=False,
        confidence=0.85,
        explanation="No fraud signals detected.",
        rules_triggered=[],
        recommended_action="ALLOW",
    )
    base.update(kw)
    return FraudDecision(**base)


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


class TestLocation:
    def test_location_creation(self):
        """Basic Location creation with all required fields."""
        loc = _loc(city="Chicago", country="US", latitude=41.8781, longitude=-87.6298)
        assert loc.city == "Chicago"
        assert loc.country == "US"
        assert loc.latitude == 41.8781
        assert loc.longitude == -87.6298

    def test_country_uppercase(self):
        """country_uppercase validator normalises country to uppercase."""
        loc = _loc(city="London", country="gb", latitude=51.5074, longitude=-0.1278)
        assert loc.country == "GB"

    def test_invalid_latitude(self):
        """Latitude outside [-90, 90] raises ValidationError."""
        with pytest.raises(ValidationError):
            _loc(latitude=91.0)
        with pytest.raises(ValidationError):
            _loc(latitude=-91.0)

    def test_invalid_longitude(self):
        """Longitude outside [-180, 180] raises ValidationError."""
        with pytest.raises(ValidationError):
            _loc(longitude=181.0)
        with pytest.raises(ValidationError):
            _loc(longitude=-181.0)

    def test_frozen(self):
        """Location model is frozen — mutation raises an error."""
        loc = _loc()
        with pytest.raises(Exception):
            loc.city = "Boston"  # type: ignore[misc]

    def test_country_min_length(self):
        """Country code shorter than 2 chars is rejected."""
        with pytest.raises(ValidationError):
            _loc(country="U")

    def test_country_max_length(self):
        """Country code longer than 2 chars is rejected."""
        with pytest.raises(ValidationError):
            _loc(country="USA")

    def test_boundary_latitude(self):
        """Boundary latitudes ±90 are valid."""
        n = _loc(latitude=90.0)
        s = _loc(latitude=-90.0)
        assert n.latitude == 90.0
        assert s.latitude == -90.0

    def test_boundary_longitude(self):
        """Boundary longitudes ±180 are valid."""
        e = _loc(longitude=180.0)
        w = _loc(longitude=-180.0)
        assert e.longitude == 180.0
        assert w.longitude == -180.0


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------


class TestTransaction:
    def test_transaction_creation(self):
        """Create a valid Transaction with all fields populated."""
        txn = _txn(metadata={"session": "abc"}, is_international=True)
        assert txn.id == "txn-abc-001"
        assert txn.amount == Decimal("125.00")
        assert txn.currency == "USD"
        assert txn.merchant_name == "Test Merchant"
        assert txn.merchant_category_code == "5411"
        assert txn.card_last_four == "1234"
        assert txn.account_id == "ACC-0001"
        assert txn.channel == TransactionChannel.IN_STORE
        assert txn.is_international is True
        assert txn.metadata == {"session": "abc"}

    def test_transaction_default_values(self):
        """Optional fields have correct defaults."""
        txn = _txn()
        assert txn.currency == "USD"
        assert txn.is_international is False
        assert txn.metadata is None

    def test_transaction_channel_enum(self):
        """All four TransactionChannel values can be used in a Transaction."""
        for channel in TransactionChannel:
            txn = _txn(channel=channel)
            assert txn.channel == channel

    def test_currency_uppercase(self):
        """currency_uppercase validator normalises currency to uppercase."""
        txn = _txn(currency="eur")
        assert txn.currency == "EUR"

    def test_naive_timestamp_gets_utc(self):
        """Naive timestamp is made UTC-aware by the validator."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        txn = _txn(timestamp=naive)
        assert txn.timestamp.tzinfo is not None
        assert txn.timestamp.tzinfo == UTC

    def test_amount_must_be_positive(self):
        """Amount of 0 or below raises ValidationError."""
        with pytest.raises(ValidationError):
            _txn(amount=Decimal("0"))
        with pytest.raises(ValidationError):
            _txn(amount=Decimal("-5.00"))

    def test_mcc_pattern_too_short(self):
        """MCC shorter than 4 digits is rejected."""
        with pytest.raises(ValidationError):
            _txn(merchant_category_code="54")

    def test_mcc_pattern_too_long(self):
        """MCC longer than 4 digits is rejected."""
        with pytest.raises(ValidationError):
            _txn(merchant_category_code="54111")

    def test_mcc_pattern_non_digit(self):
        """Non-digit MCC is rejected."""
        with pytest.raises(ValidationError):
            _txn(merchant_category_code="ABCD")

    def test_card_last_four_too_short(self):
        """card_last_four shorter than 4 digits is rejected."""
        with pytest.raises(ValidationError):
            _txn(card_last_four="123")

    def test_card_last_four_too_long(self):
        """card_last_four longer than 4 digits is rejected."""
        with pytest.raises(ValidationError):
            _txn(card_last_four="12345")

    def test_transaction_serialization(self):
        """model_dump() and model_dump_json() roundtrip preserves key fields."""
        txn = _txn()
        data = txn.model_dump()
        assert data["id"] == "txn-abc-001"
        assert data["currency"] == "USD"

        json_str = txn.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "txn-abc-001"
        assert parsed["merchant_category_code"] == "5411"
        assert parsed["card_last_four"] == "1234"


# ---------------------------------------------------------------------------
# TransactionChannel enum
# ---------------------------------------------------------------------------


class TestTransactionChannelEnum:
    def test_transaction_channel_enum_online(self):
        assert TransactionChannel.ONLINE == "ONLINE"

    def test_transaction_channel_enum_in_store(self):
        assert TransactionChannel.IN_STORE == "IN_STORE"

    def test_transaction_channel_enum_atm(self):
        assert TransactionChannel.ATM == "ATM"

    def test_transaction_channel_enum_mobile(self):
        assert TransactionChannel.MOBILE == "MOBILE"

    def test_transaction_channel_all_values(self):
        """All four channels are represented."""
        values = {c.value for c in TransactionChannel}
        assert values == {"ONLINE", "IN_STORE", "ATM", "MOBILE"}


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------


class TestRiskLevelEnum:
    def test_risk_level_low(self):
        assert RiskLevel.LOW == "LOW"

    def test_risk_level_medium(self):
        assert RiskLevel.MEDIUM == "MEDIUM"

    def test_risk_level_high(self):
        assert RiskLevel.HIGH == "HIGH"

    def test_risk_level_critical(self):
        assert RiskLevel.CRITICAL == "CRITICAL"

    def test_risk_level_all_values(self):
        """All four risk levels are represented."""
        values = {r.value for r in RiskLevel}
        assert values == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# FraudDecision
# ---------------------------------------------------------------------------


class TestFraudDecision:
    def test_fraud_decision_creation(self):
        """Create a FraudDecision with all explicit fields populated."""
        decision = FraudDecision(
            transaction_id="txn-abc-001",
            risk_level=RiskLevel.HIGH,
            fraud_score=0.75,
            is_fraud=True,
            confidence=0.9,
            explanation="High-risk merchant and unusual amount detected.",
            rules_triggered=["amount_rule", "merchant_rule"],
            recommended_action="REVIEW",
            agent_trace=["triage_agent", "decision_agent"],
        )
        assert decision.transaction_id == "txn-abc-001"
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.fraud_score == 0.75
        assert decision.is_fraud is True
        assert decision.confidence == 0.9
        assert "amount_rule" in decision.rules_triggered
        assert decision.recommended_action == "REVIEW"
        assert decision.agent_trace == ["triage_agent", "decision_agent"]

    def test_fraud_decision_defaults(self):
        """timestamp is auto-set; agent_trace defaults to None; rules_triggered defaults to []."""
        decision = _decision()
        assert decision.timestamp is not None
        assert isinstance(decision.timestamp, datetime)
        assert decision.agent_trace is None
        assert decision.rules_triggered == []

    def test_fraud_decision_timestamp_auto_utc(self):
        """Auto-generated timestamp is timezone-aware."""
        decision = _decision()
        assert decision.timestamp.tzinfo is not None

    def test_fraud_score_bounds_too_high(self):
        """fraud_score > 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            _decision(fraud_score=1.1)

    def test_fraud_score_bounds_negative(self):
        """Negative fraud_score raises ValidationError."""
        with pytest.raises(ValidationError):
            _decision(fraud_score=-0.01)

    def test_confidence_bounds(self):
        """confidence outside [0, 1] raises ValidationError."""
        with pytest.raises(ValidationError):
            _decision(confidence=1.5)
        with pytest.raises(ValidationError):
            _decision(confidence=-0.1)

    def test_critical_requires_high_score(self):
        """CRITICAL risk_level requires fraud_score >= 0.7."""
        with pytest.raises(ValidationError):
            _decision(risk_level=RiskLevel.CRITICAL, fraud_score=0.65)

    def test_low_requires_low_score(self):
        """LOW risk_level requires fraud_score <= 0.3."""
        with pytest.raises(ValidationError):
            _decision(risk_level=RiskLevel.LOW, fraud_score=0.5)

    def test_critical_valid_high_score(self):
        """CRITICAL with fraud_score >= 0.7 is accepted."""
        decision = _decision(risk_level=RiskLevel.CRITICAL, fraud_score=0.95, is_fraud=True)
        assert decision.risk_level == RiskLevel.CRITICAL

    def test_medium_risk_no_strict_constraint(self):
        """MEDIUM has no strict score constraint beyond [0, 1]."""
        decision = _decision(risk_level=RiskLevel.MEDIUM, fraud_score=0.5, is_fraud=False)
        assert decision.risk_level == RiskLevel.MEDIUM

    def test_fraud_decision_serialization(self):
        """model_dump_json() roundtrip preserves all fields."""
        decision = _decision(
            rules_triggered=["velocity_rule"],
            agent_trace=["triage"],
        )
        json_str = decision.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["transaction_id"] == "txn-abc-001"
        assert parsed["rules_triggered"] == ["velocity_rule"]
        assert parsed["agent_trace"] == ["triage"]


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


class TestAccount:
    def test_account_creation(self):
        """Account with all fields is created correctly."""
        account = _acct()
        assert account.id == "ACC-0001"
        assert account.holder_name == "Jane Doe"
        assert account.average_transaction_amount == Decimal("75.00")
        assert account.account_open_date == date(2019, 3, 15)
        assert account.transaction_history_count == 250

    def test_account_average_amount_zero_rejected(self):
        """average_transaction_amount of 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _acct(average_transaction_amount=Decimal("0"))

    def test_account_average_amount_negative_rejected(self):
        """Negative average_transaction_amount raises ValidationError."""
        with pytest.raises(ValidationError):
            _acct(average_transaction_amount=Decimal("-10.00"))

    def test_account_transaction_history_count_negative_rejected(self):
        """Negative transaction_history_count raises ValidationError."""
        with pytest.raises(ValidationError):
            _acct(transaction_history_count=-1)

    def test_account_zero_history_count_allowed(self):
        """A brand-new account with zero transactions is valid."""
        account = _acct(transaction_history_count=0)
        assert account.transaction_history_count == 0

    def test_account_typical_location_is_location(self):
        """typical_location is a Location instance."""
        account = _acct()
        assert isinstance(account.typical_location, Location)
