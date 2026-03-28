"""Tests for AuditLogger: decisions, access events, overrides, chain integrity."""

from datetime import date, datetime
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
from fraud_agent.guardrails.audit_logger import AuditLogger


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


def make_decision(**overrides) -> FraudDecision:
    """Helper to build a minimal valid FraudDecision."""
    defaults = {
        "transaction_id": "test-txn-001",
        "risk_level": RiskLevel.LOW,
        "fraud_score": 0.1,
        "is_fraud": False,
        "confidence": 0.8,
        "explanation": "Normal transaction.",
        "rules_triggered": [],
        "recommended_action": "ALLOW",
        "agent_trace": ["triage", "decision"],
    }
    defaults.update(overrides)
    return FraudDecision(**defaults)


@pytest.fixture()
def logger(tmp_path):
    """AuditLogger writing to a temp file so tests don't pollute the filesystem."""
    log_file = str(tmp_path / "audit.log")
    return AuditLogger(log_path=log_file)


# ---------------------------------------------------------------------------
# log_decision
# ---------------------------------------------------------------------------


class TestLogDecision:
    def test_log_decision(self, logger):
        """log_decision stores an entry and returns a dict with a hash."""
        decision = make_decision()
        masked_txn = {"id": "test-txn-001", "amount": "50.00"}
        entry = logger.log_decision(decision, masked_txn)

        assert isinstance(entry, dict)
        assert entry["event_type"] == "fraud_decision"
        assert "hash" in entry
        assert len(entry["hash"]) == 64  # SHA-256 hex digest

    def test_log_decision_contains_transaction_id(self, logger):
        """Logged decision entry contains the transaction ID."""
        decision = make_decision(transaction_id="txn-xyz-999")
        entry = logger.log_decision(decision, {})
        assert entry["data"]["transaction_id"] == "txn-xyz-999"

    def test_log_decision_contains_risk_level(self, logger):
        """Logged decision entry contains the risk level string."""
        decision = make_decision(risk_level=RiskLevel.HIGH, fraud_score=0.75, is_fraud=True)
        entry = logger.log_decision(decision, {})
        assert entry["data"]["risk_level"] == "HIGH"

    def test_log_decision_contains_fraud_flag(self, logger):
        """Logged entry reflects is_fraud correctly."""
        decision = make_decision(risk_level=RiskLevel.HIGH, fraud_score=0.75, is_fraud=True)
        entry = logger.log_decision(decision, {})
        assert entry["data"]["is_fraud"] is True

    def test_log_decision_has_timestamp(self, logger):
        """Logged entry includes a timestamp string."""
        decision = make_decision()
        entry = logger.log_decision(decision, {})
        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], str)


# ---------------------------------------------------------------------------
# log_access
# ---------------------------------------------------------------------------


class TestLogAccess:
    def test_log_access(self, logger):
        """log_access creates a data_access entry with correct fields."""
        entry = logger.log_access("user-001", "transaction:txn-123", "read")
        assert entry["event_type"] == "data_access"
        assert entry["data"]["user_id"] == "user-001"
        assert entry["data"]["resource"] == "transaction:txn-123"
        assert entry["data"]["action"] == "read"

    def test_log_access_has_hash(self, logger):
        """log_access entry contains a SHA-256 hash."""
        entry = logger.log_access("user-002", "account:acc-456", "write")
        assert "hash" in entry
        assert len(entry["hash"]) == 64


# ---------------------------------------------------------------------------
# log_override
# ---------------------------------------------------------------------------


class TestLogOverride:
    def test_log_override(self, logger):
        """log_override creates a decision_override entry with correct fields."""
        entry = logger.log_override(
            decision_id="txn-789",
            original="ALLOW",
            override_to="BLOCK",
            reason="Manual review found suspicious pattern.",
            user_id="analyst-01",
        )
        assert entry["event_type"] == "decision_override"
        assert entry["data"]["decision_id"] == "txn-789"
        assert entry["data"]["original"] == "ALLOW"
        assert entry["data"]["override_to"] == "BLOCK"
        assert entry["data"]["user_id"] == "analyst-01"

    def test_log_override_has_hash(self, logger):
        """log_override entry contains a SHA-256 hash."""
        entry = logger.log_override("txn-111", "ALLOW", "BLOCK", "reason", "user-1")
        assert len(entry["hash"]) == 64


# ---------------------------------------------------------------------------
# log_system_event
# ---------------------------------------------------------------------------


class TestLogSystemEvent:
    def test_log_system_event(self, logger):
        """log_system_event creates a system_* entry with the provided details."""
        entry = logger.log_system_event("startup", {"version": "0.1.0", "env": "prod"})
        assert entry["event_type"] == "system_startup"
        assert entry["data"]["version"] == "0.1.0"

    def test_log_system_event_has_hash(self, logger):
        """log_system_event entry contains a SHA-256 hash."""
        entry = logger.log_system_event("shutdown", {})
        assert len(entry["hash"]) == 64


# ---------------------------------------------------------------------------
# Chain integrity
# ---------------------------------------------------------------------------


class TestChainIntegrity:
    def test_chain_integrity_empty(self, logger):
        """Empty chain is always valid."""
        assert logger.verify_chain_integrity() is True

    def test_chain_integrity(self, logger):
        """Chain is valid after multiple entries are logged."""
        decision = make_decision()
        logger.log_decision(decision, {})
        logger.log_access("user-1", "resource", "read")
        logger.log_override("txn-1", "ALLOW", "BLOCK", "reason", "user-1")
        assert logger.verify_chain_integrity() is True

    def test_chain_integrity_tampered(self, logger):
        """Tampering with an entry's data breaks chain integrity."""
        decision = make_decision()
        logger.log_decision(decision, {})
        logger.log_access("user-1", "resource", "read")

        # Tamper: change the hash of the first entry directly
        logger._entries[0]["hash"] = "0" * 64

        assert logger.verify_chain_integrity() is False

    def test_hash_chain(self, logger):
        """Each entry's hash depends on the previous entry's hash (chain structure)."""
        logger.log_access("u1", "r1", "read")
        logger.log_access("u2", "r2", "write")

        entry0 = logger._entries[0]
        entry1 = logger._entries[1]

        # entry1's previous_hash must equal entry0's hash
        assert entry1["previous_hash"] == entry0["hash"]

    def test_hash_chain_first_entry_previous_is_genesis(self, logger):
        """The first entry in the chain references 'genesis' as its previous hash."""
        logger.log_access("u1", "r1", "read")
        assert logger._entries[0]["previous_hash"] == "genesis"


# ---------------------------------------------------------------------------
# get_audit_trail
# ---------------------------------------------------------------------------


class TestGetAuditTrail:
    def test_get_audit_trail(self, logger):
        """Retrieves only entries related to the given transaction_id."""
        dec1 = make_decision(transaction_id="txn-AAA")
        dec2 = make_decision(transaction_id="txn-BBB", fraud_score=0.2)
        logger.log_decision(dec1, {})
        logger.log_decision(dec2, {})
        logger.log_access("user-1", "resource", "read")

        trail = logger.get_audit_trail("txn-AAA")
        assert len(trail) == 1
        assert trail[0]["data"]["transaction_id"] == "txn-AAA"

    def test_get_audit_trail_empty_for_unknown(self, logger):
        """Returns empty list when no entry matches the transaction_id."""
        logger.log_access("user-1", "resource", "read")
        trail = logger.get_audit_trail("txn-NONEXISTENT")
        assert trail == []

    def test_get_audit_trail_override_included(self, logger):
        """Override entries with matching decision_id are included in the trail."""
        logger.log_override("txn-OVER", "ALLOW", "BLOCK", "reason", "analyst-1")
        trail = logger.get_audit_trail("txn-OVER")
        assert len(trail) == 1
        assert trail[0]["event_type"] == "decision_override"
