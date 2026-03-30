"""Tests for AuditLogger: decisions, access events, overrides, chain integrity."""

import pytest

from fraud_agent.data.schemas import FraudDecision, RiskLevel
from fraud_agent.guardrails.audit_logger import AuditLogger


def make_decision(**overrides) -> FraudDecision:
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
    log_file = str(tmp_path / "audit.log")
    return AuditLogger(log_path=log_file)


class TestLogDecision:
    def test_log_decision(self, logger):
        decision = make_decision()
        entry = logger.log_decision(decision, {"id": "test-txn-001", "amount": "50.00"})

        assert isinstance(entry, dict)
        assert entry["event_type"] == "fraud_decision"
        assert "hash" in entry
        assert len(entry["hash"]) == 64

    def test_log_decision_contains_transaction_id(self, logger):
        decision = make_decision(transaction_id="txn-xyz-999")
        entry = logger.log_decision(decision, {})
        assert entry["data"]["transaction_id"] == "txn-xyz-999"

    def test_log_decision_contains_risk_level(self, logger):
        decision = make_decision(risk_level=RiskLevel.HIGH, fraud_score=0.75, is_fraud=True)
        entry = logger.log_decision(decision, {})
        assert entry["data"]["risk_level"] == "HIGH"

    def test_log_decision_contains_fraud_flag(self, logger):
        decision = make_decision(risk_level=RiskLevel.HIGH, fraud_score=0.75, is_fraud=True)
        entry = logger.log_decision(decision, {})
        assert entry["data"]["is_fraud"] is True

    def test_log_decision_has_timestamp(self, logger):
        entry = logger.log_decision(make_decision(), {})
        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], str)


class TestLogAccess:
    def test_log_access(self, logger):
        entry = logger.log_access("user-001", "transaction:txn-123", "read")
        assert entry["event_type"] == "data_access"
        assert entry["data"]["user_id"] == "user-001"
        assert entry["data"]["resource"] == "transaction:txn-123"
        assert entry["data"]["action"] == "read"

    def test_log_access_has_hash(self, logger):
        entry = logger.log_access("user-002", "account:acc-456", "write")
        assert "hash" in entry
        assert len(entry["hash"]) == 64


class TestLogOverride:
    def test_log_override(self, logger):
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
        entry = logger.log_override("txn-111", "ALLOW", "BLOCK", "reason", "user-1")
        assert len(entry["hash"]) == 64


class TestLogSystemEvent:
    def test_log_system_event(self, logger):
        entry = logger.log_system_event("startup", {"version": "0.1.0", "env": "prod"})
        assert entry["event_type"] == "system_startup"
        assert entry["data"]["version"] == "0.1.0"

    def test_log_system_event_has_hash(self, logger):
        entry = logger.log_system_event("shutdown", {})
        assert len(entry["hash"]) == 64


class TestChainIntegrity:
    def test_chain_integrity_empty(self, logger):
        assert logger.verify_chain_integrity() is True

    def test_chain_integrity(self, logger):
        logger.log_decision(make_decision(), {})
        logger.log_access("user-1", "resource", "read")
        logger.log_override("txn-1", "ALLOW", "BLOCK", "reason", "user-1")
        assert logger.verify_chain_integrity() is True

    def test_chain_integrity_tampered(self, logger):
        logger.log_decision(make_decision(), {})
        logger.log_access("user-1", "resource", "read")
        logger._entries[0]["hash"] = "0" * 64
        assert logger.verify_chain_integrity() is False

    def test_hash_chain(self, logger):
        logger.log_access("u1", "r1", "read")
        logger.log_access("u2", "r2", "write")
        entry0 = logger._entries[0]
        entry1 = logger._entries[1]
        assert entry1["previous_hash"] == entry0["hash"]

    def test_hash_chain_first_entry_previous_is_genesis(self, logger):
        logger.log_access("u1", "r1", "read")
        assert logger._entries[0]["previous_hash"] == "genesis"


class TestGetAuditTrail:
    def test_get_audit_trail(self, logger):
        logger.log_decision(make_decision(transaction_id="txn-AAA"), {})
        logger.log_decision(make_decision(transaction_id="txn-BBB", fraud_score=0.2), {})
        logger.log_access("user-1", "resource", "read")

        trail = logger.get_audit_trail("txn-AAA")
        assert len(trail) == 1
        assert trail[0]["data"]["transaction_id"] == "txn-AAA"

    def test_get_audit_trail_empty_for_unknown(self, logger):
        logger.log_access("user-1", "resource", "read")
        trail = logger.get_audit_trail("txn-NONEXISTENT")
        assert trail == []

    def test_get_audit_trail_override_included(self, logger):
        logger.log_override("txn-OVER", "ALLOW", "BLOCK", "reason", "analyst-1")
        trail = logger.get_audit_trail("txn-OVER")
        assert len(trail) == 1
        assert trail[0]["event_type"] == "decision_override"
