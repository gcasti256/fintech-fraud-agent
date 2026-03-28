"""Immutable audit trail for fraud detection decisions and data access."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import structlog

from fraud_agent.data.schemas import FraudDecision

logger = structlog.get_logger(__name__)


class AuditLogger:
    """Structured, tamper-evident audit logger with chain hashing.

    Each log entry includes a SHA-256 hash of the previous entry's hash
    concatenated with the current entry's data, forming an integrity chain.
    """

    def __init__(self, log_path: str = "audit.log") -> None:
        self.log_path = log_path
        self._entries: list[dict[str, Any]] = []
        self._last_hash = "genesis"

    def log_decision(self, decision: FraudDecision, masked_transaction: dict) -> dict:
        """Log a fraud scoring decision.

        Args:
            decision: The fraud decision to log.
            masked_transaction: PII-masked transaction data.

        Returns:
            The created audit log entry.
        """
        entry = self._create_entry(
            event_type="fraud_decision",
            data={
                "transaction_id": decision.transaction_id,
                "risk_level": decision.risk_level.value,
                "fraud_score": decision.fraud_score,
                "is_fraud": decision.is_fraud,
                "confidence": decision.confidence,
                "explanation": decision.explanation,
                "rules_triggered": decision.rules_triggered,
                "recommended_action": decision.recommended_action,
                "agent_trace": decision.agent_trace or [],
                "masked_transaction": masked_transaction,
            },
        )

        logger.info(
            "audit.decision_logged",
            transaction_id=decision.transaction_id,
            risk_level=decision.risk_level.value,
        )
        return entry

    def log_access(self, user_id: str, resource: str, action: str) -> dict:
        """Log a data access event.

        Args:
            user_id: ID of the user accessing data.
            resource: The resource being accessed.
            action: The action performed (read, write, delete).

        Returns:
            The created audit log entry.
        """
        entry = self._create_entry(
            event_type="data_access",
            data={
                "user_id": user_id,
                "resource": resource,
                "action": action,
            },
        )

        logger.info("audit.access_logged", user_id=user_id, resource=resource, action=action)
        return entry

    def log_override(
        self,
        decision_id: str,
        original: str,
        override_to: str,
        reason: str,
        user_id: str,
    ) -> dict:
        """Log a manual decision override.

        Args:
            decision_id: ID of the decision being overridden.
            original: Original decision value.
            override_to: New decision value.
            reason: Reason for the override.
            user_id: ID of the user performing the override.

        Returns:
            The created audit log entry.
        """
        entry = self._create_entry(
            event_type="decision_override",
            data={
                "decision_id": decision_id,
                "original": original,
                "override_to": override_to,
                "reason": reason,
                "user_id": user_id,
            },
        )

        logger.warning(
            "audit.override_logged",
            decision_id=decision_id,
            original=original,
            override_to=override_to,
            user_id=user_id,
        )
        return entry

    def log_system_event(self, event_type: str, details: dict) -> dict:
        """Log a system event.

        Args:
            event_type: Type of system event (startup, shutdown, error, etc.).
            details: Event details.

        Returns:
            The created audit log entry.
        """
        entry = self._create_entry(
            event_type=f"system_{event_type}",
            data=details,
        )

        logger.info("audit.system_event", event_type=event_type)
        return entry

    def get_audit_trail(self, transaction_id: str) -> list[dict]:
        """Retrieve all audit entries related to a transaction.

        Args:
            transaction_id: The transaction ID to search for.

        Returns:
            List of audit entries related to the transaction.
        """
        return [
            entry
            for entry in self._entries
            if entry.get("data", {}).get("transaction_id") == transaction_id
            or entry.get("data", {}).get("decision_id") == transaction_id
        ]

    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the audit log chain.

        Returns:
            True if the chain is intact, False if tampered.
        """
        if not self._entries:
            return True

        prev_hash = "genesis"
        for entry in self._entries:
            expected_hash = self._compute_hash(prev_hash, entry["data"])
            if entry["hash"] != expected_hash:
                logger.error(
                    "audit.chain_integrity_violation",
                    entry_index=self._entries.index(entry),
                )
                return False
            prev_hash = entry["hash"]

        return True

    def _create_entry(self, event_type: str, data: dict) -> dict:
        """Create a new audit log entry with chain hash."""
        entry_hash = self._compute_hash(self._last_hash, data)

        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "data": data,
            "hash": entry_hash,
            "previous_hash": self._last_hash,
        }

        self._entries.append(entry)
        self._last_hash = entry_hash

        self._write_to_file(entry)
        return entry

    @staticmethod
    def _compute_hash(previous_hash: str, data: dict) -> str:
        """Compute SHA-256 hash for chain integrity."""
        payload = previous_hash + json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _write_to_file(self, entry: dict) -> None:
        """Append entry to the audit log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            # In-memory only if file write fails
            pass
