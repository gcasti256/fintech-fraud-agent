"""SQLite database layer for persisting fraud decisions, audit logs, and metrics.

All I/O is synchronous and uses the standard-library :mod:`sqlite3` module.
The :class:`Database` class is thread-safe by default because it uses
``check_same_thread=False`` and WAL journal mode, but callers should still
avoid sharing a single instance across multiple threads without external
serialisation for write-heavy workloads.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from fraud_agent.data.schemas import FraudDecision


class Database:
    """Thin SQLite wrapper for the fraud detection agent.

    Creates all necessary tables on first use.  All public methods are safe to
    call immediately after construction.

    Args:
        db_path: Filesystem path for the SQLite database file.  Defaults to
            ``"fraud_agent.db"`` in the current working directory.
    """

    def __init__(self, db_path: str = "fraud_agent.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL mode gives better concurrent read performance.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        """Create all required tables if they do not already exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id                  TEXT        PRIMARY KEY,
                    transaction_id      TEXT        NOT NULL,
                    risk_level          TEXT        NOT NULL,
                    fraud_score         REAL        NOT NULL,
                    is_fraud            BOOLEAN     NOT NULL,
                    confidence          REAL        NOT NULL,
                    explanation         TEXT        NOT NULL,
                    rules_triggered     TEXT        NOT NULL DEFAULT '[]',
                    recommended_action  TEXT        NOT NULL,
                    created_at          TIMESTAMP   NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_transaction_id
                    ON decisions(transaction_id);

                CREATE INDEX IF NOT EXISTS idx_decisions_created_at
                    ON decisions(created_at);

                CREATE TABLE IF NOT EXISTS audit_log (
                    id          INTEGER     PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT        NOT NULL,
                    event_type  TEXT        NOT NULL,
                    data        TEXT        NOT NULL DEFAULT '{}',
                    hash        TEXT        NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_audit_log_event_type
                    ON audit_log(event_type);

                CREATE TABLE IF NOT EXISTS metrics (
                    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT        NOT NULL,
                    metric_name     TEXT        NOT NULL,
                    metric_value    REAL        NOT NULL,
                    labels          TEXT        NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_name
                    ON metrics(metric_name);
                """
            )

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def save_decision(self, decision: FraudDecision) -> None:
        """Persist a :class:`~fraud_agent.data.schemas.FraudDecision` to the database.

        A new UUID is generated for the primary-key ``id`` column so that
        multiple evaluations of the same transaction can coexist.

        Args:
            decision: The fraud decision to store.

        Raises:
            sqlite3.DatabaseError: On any database-level failure.
        """
        decision_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO decisions
                    (id, transaction_id, risk_level, fraud_score, is_fraud,
                     confidence, explanation, rules_triggered,
                     recommended_action, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    decision.transaction_id,
                    decision.risk_level.value,
                    decision.fraud_score,
                    int(decision.is_fraud),
                    decision.confidence,
                    decision.explanation,
                    json.dumps(decision.rules_triggered),
                    decision.recommended_action,
                    created_at,
                ),
            )

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        """Retrieve a single decision by its primary key.

        Args:
            decision_id: The UUID string primary key of the decision row.

        Returns:
            A dictionary of column name → value, or ``None`` if not found.
        """
        row = self._conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()

        if row is None:
            return None

        return self._deserialise_decision(row)

    def get_decisions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return a paginated list of decisions ordered by creation time (newest first).

        Args:
            limit: Maximum number of rows to return.
            offset: Number of rows to skip (for pagination).

        Returns:
            List of decision dictionaries.
        """
        rows = self._conn.execute(
            "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        return [self._deserialise_decision(r) for r in rows]

    def get_decisions_by_transaction(self, transaction_id: str) -> list[dict[str, Any]]:
        """Return all decisions associated with a specific transaction.

        Args:
            transaction_id: The transaction identifier to look up.

        Returns:
            List of decision dictionaries, ordered newest first.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM decisions
            WHERE transaction_id = ?
            ORDER BY created_at DESC
            """,
            (transaction_id,),
        ).fetchall()

        return [self._deserialise_decision(r) for r in rows]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def save_metric(
        self,
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        """Record a single metric observation.

        Args:
            name: Metric name (e.g. ``"scoring_latency_ms"``).
            value: Numeric value of the observation.
            labels: Optional key-value labels for dimensionality (e.g. risk level).
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO metrics (timestamp, metric_name, metric_value, labels)
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    name,
                    value,
                    json.dumps(labels or {}),
                ),
            )

    def get_metrics(self, name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve recent observations for a named metric.

        Args:
            name: The metric name to query.
            limit: Maximum number of rows to return (most recent first).

        Returns:
            List of dicts with keys: ``id``, ``timestamp``, ``metric_name``,
            ``metric_value``, ``labels``.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM metrics
            WHERE metric_name = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (name, limit),
        ).fetchall()

        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "metric_name": r["metric_name"],
                "metric_value": r["metric_value"],
                "labels": json.loads(r["labels"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection.

        After calling this method the :class:`Database` instance must not be
        used again.
        """
        self._conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialise_decision(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a raw SQLite row into a plain Python dictionary.

        Deserialises the JSON ``rules_triggered`` column and casts the
        ``is_fraud`` integer back to a boolean.

        Args:
            row: A :class:`sqlite3.Row` from the ``decisions`` table.

        Returns:
            Dictionary with native Python types.
        """
        data = dict(row)
        data["is_fraud"] = bool(data["is_fraud"])
        data["rules_triggered"] = json.loads(data.get("rules_triggered") or "[]")
        return data
