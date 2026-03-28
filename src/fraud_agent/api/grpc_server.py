"""gRPC server for the fraud scoring service."""

from __future__ import annotations

import uuid
from concurrent import futures
from datetime import date, datetime
from decimal import Decimal

import grpc
import structlog

from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
from fraud_agent.agents.state import TransactionContext
from fraud_agent.data.schemas import Account, Location, Transaction, TransactionChannel
from fraud_agent.db import Database
from fraud_agent.monitoring.metrics import MetricsCollector

logger = structlog.get_logger(__name__)

# Import generated protobuf modules — these are generated from the .proto file
# by running: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. fraud_scoring.proto
try:
    from fraud_agent.proto import fraud_scoring_pb2, fraud_scoring_pb2_grpc
except ImportError:
    fraud_scoring_pb2 = None
    fraud_scoring_pb2_grpc = None
    logger.warning("grpc.proto_not_compiled", hint="Run 'fraud-agent compile-proto' first")


class FraudScoringServicer:
    """gRPC servicer implementing the FraudScoringService."""

    def __init__(
        self,
        orchestrator: FraudDetectionOrchestrator | None = None,
        db: Database | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.orchestrator = orchestrator or FraudDetectionOrchestrator()
        self.db = db or Database()
        self.metrics = metrics or MetricsCollector()

    def ScoreTransaction(self, request, context):
        """Score a single transaction for fraud."""
        import time

        start = time.monotonic()

        try:
            transaction = self._proto_to_transaction(request)
            account = self._build_default_account(request)
            tx_context = TransactionContext(transaction=transaction, account=account)

            decision = self.orchestrator.analyze_transaction(tx_context)
            latency = (time.monotonic() - start) * 1000

            self.metrics.record_scoring_latency(latency)
            self.metrics.record_decision(decision.risk_level.value, decision.is_fraud)
            self.db.save_decision(decision)

            return fraud_scoring_pb2.FraudScore(
                transaction_id=decision.transaction_id,
                risk_level=decision.risk_level.value,
                fraud_score=decision.fraud_score,
                is_fraud=decision.is_fraud,
                confidence=decision.confidence,
                explanation=decision.explanation,
                rules_triggered=decision.rules_triggered,
                recommended_action=decision.recommended_action,
            )
        except Exception as e:
            logger.error("grpc.score_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Scoring error: {e}")
            return fraud_scoring_pb2.FraudScore()

    def BatchScore(self, request, context):
        """Score a batch of transactions."""
        scores = []
        for txn in request.transactions:
            score = self.ScoreTransaction(txn, context)
            scores.append(score)

        flagged = sum(1 for s in scores if s.is_fraud)
        avg = sum(s.fraud_score for s in scores) / len(scores) if scores else 0.0

        return fraud_scoring_pb2.BatchResponse(
            scores=scores,
            total_processed=len(scores),
            flagged_count=flagged,
            average_score=avg,
        )

    def GetDecision(self, request, context):
        """Retrieve a previous fraud decision."""
        rows = self.db.get_decisions_by_transaction(request.transaction_id)
        if not rows:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Decision not found")
            return fraud_scoring_pb2.DecisionDetail()

        row = rows[0]
        score = fraud_scoring_pb2.FraudScore(
            transaction_id=row.get("transaction_id", ""),
            risk_level=row.get("risk_level", "LOW"),
            fraud_score=row.get("fraud_score", 0.0),
            is_fraud=row.get("is_fraud", False),
            confidence=row.get("confidence", 0.0),
            explanation=row.get("explanation", ""),
            rules_triggered=row.get("rules_triggered", []),
            recommended_action=row.get("recommended_action", ""),
        )

        return fraud_scoring_pb2.DecisionDetail(
            score=score,
            agent_trace=[],
            analysis_summary=row.get("explanation", ""),
        )

    def _proto_to_transaction(self, request) -> Transaction:
        """Convert protobuf TransactionRequest to domain Transaction."""
        loc = request.location if request.location else None
        return Transaction(
            id=request.id or str(uuid.uuid4()),
            timestamp=datetime.fromisoformat(request.timestamp)
            if request.timestamp
            else datetime.now(),
            amount=Decimal(str(request.amount)),
            currency=request.currency or "USD",
            merchant_name=request.merchant_name,
            merchant_category_code=request.merchant_category_code or "5999",
            card_last_four=request.card_last_four or "0000",
            account_id=request.account_id or "ACC-0001-0001",
            location=Location(
                city=loc.city if loc else "New York",
                country=loc.country if loc else "US",
                latitude=loc.latitude if loc else 40.7128,
                longitude=loc.longitude if loc else -74.0060,
            ),
            channel=TransactionChannel(request.channel)
            if request.channel
            else TransactionChannel.ONLINE,
            is_international=request.is_international,
        )

    def _build_default_account(self, request) -> Account:
        """Build default account for gRPC request."""
        loc = request.location if request.location else None
        return Account(
            id=request.account_id or "ACC-0001-0001",
            holder_name="Account Holder",
            average_transaction_amount=Decimal("75.00"),
            typical_location=Location(
                city=loc.city if loc else "New York",
                country=loc.country if loc else "US",
                latitude=loc.latitude if loc else 40.7128,
                longitude=loc.longitude if loc else -74.0060,
            ),
            account_open_date=date(2020, 1, 1),
            transaction_history_count=100,
        )


def serve(host: str = "0.0.0.0", port: int = 50051) -> grpc.Server:
    """Start the gRPC server.

    Args:
        host: Host to bind to.
        port: Port to listen on.

    Returns:
        The running gRPC server instance.
    """
    if fraud_scoring_pb2_grpc is None:
        raise RuntimeError(
            "gRPC proto files not compiled. Run: "
            "python -m grpc_tools.protoc "
            "-Isrc/fraud_agent/proto "
            "--python_out=src/fraud_agent/proto "
            "--grpc_python_out=src/fraud_agent/proto "
            "src/fraud_agent/proto/fraud_scoring.proto"
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fraud_scoring_pb2_grpc.add_FraudScoringServiceServicer_to_server(FraudScoringServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    logger.info("grpc.server_started", host=host, port=port)
    return server
