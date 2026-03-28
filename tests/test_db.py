"""Tests for the database layer."""

from __future__ import annotations

import pytest

from fraud_agent.data.schemas import FraudDecision, RiskLevel
from fraud_agent.db import Database


class TestDatabase:
    def test_save_and_get_decision(self, db):
        decision = FraudDecision(
            transaction_id="txn-db-001",
            risk_level=RiskLevel.HIGH,
            fraud_score=0.85,
            is_fraud=True,
            confidence=0.9,
            explanation="Test decision",
            rules_triggered=["velocity_rule", "amount_rule"],
            recommended_action="BLOCK",
        )
        db.save_decision(decision)

        rows = db.get_decisions_by_transaction("txn-db-001")
        assert len(rows) == 1
        row = rows[0]
        assert row["transaction_id"] == "txn-db-001"
        assert row["risk_level"] == "HIGH"
        assert row["is_fraud"] is True
        assert "velocity_rule" in row["rules_triggered"]

    def test_get_decisions_pagination(self, db):
        for i in range(5):
            decision = FraudDecision(
                transaction_id=f"txn-page-{i:03d}",
                risk_level=RiskLevel.LOW,
                fraud_score=0.1,
                is_fraud=False,
                confidence=0.9,
                explanation="Test",
                recommended_action="ALLOW",
            )
            db.save_decision(decision)

        page1 = db.get_decisions(limit=2, offset=0)
        assert len(page1) == 2

        page2 = db.get_decisions(limit=2, offset=2)
        assert len(page2) == 2

    def test_get_decision_not_found(self, db):
        result = db.get_decision("nonexistent-id")
        assert result is None

    def test_save_and_get_metric(self, db):
        db.save_metric("test_metric", 42.5, {"level": "HIGH"})
        metrics = db.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0]["metric_value"] == 42.5
        assert metrics[0]["labels"]["level"] == "HIGH"

    def test_get_metrics_limit(self, db):
        for i in range(10):
            db.save_metric("counter", float(i))

        metrics = db.get_metrics("counter", limit=5)
        assert len(metrics) == 5

    def test_close(self, tmp_db_path):
        database = Database(db_path=tmp_db_path)
        database.close()
        # After close, operations should raise
        with pytest.raises(Exception):
            database.get_decisions()
