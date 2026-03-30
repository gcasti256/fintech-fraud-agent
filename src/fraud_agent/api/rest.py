"""FastAPI REST API for the fraud detection service."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal

import structlog
from fastapi import FastAPI, HTTPException, Query

from fraud_agent import __version__
from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
from fraud_agent.agents.state import TransactionContext
from fraud_agent.api.schemas import (
    AccountRequest,
    BatchRequest,
    BatchResponse,
    DecisionResponse,
    FraudScoreResponse,
    HealthResponse,
    MetricsResponse,
    PatternResponse,
    TransactionRequest,
)
from fraud_agent.data.knowledge_base import FraudKnowledgeBase
from fraud_agent.data.schemas import Account, Location, Transaction, TransactionChannel
from fraud_agent.db import Database
from fraud_agent.guardrails.pii_masker import PIIMasker
from fraud_agent.monitoring.metrics import MetricsCollector

logger = structlog.get_logger(__name__)

# Module-level singletons initialized at startup
_orchestrator: FraudDetectionOrchestrator | None = None
_db: Database | None = None
_metrics: MetricsCollector | None = None
_masker: PIIMasker | None = None
_kb: FraudKnowledgeBase | None = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize and tear down resources."""
    global _orchestrator, _db, _metrics, _masker, _kb, _start_time

    _orchestrator = FraudDetectionOrchestrator()
    _db = Database()
    _metrics = MetricsCollector()
    _masker = PIIMasker()
    _kb = FraudKnowledgeBase()
    _start_time = time.time()

    logger.info("api.startup", version=__version__)
    yield

    if _db:
        _db.close()
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Fraud Detection Agent API",
        description="Agentic fraud detection system with multi-agent orchestration",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=__version__,
            uptime_seconds=round(time.time() - _start_time, 2),
            components={
                "orchestrator": "ready" if _orchestrator else "not_initialized",
                "database": "ready" if _db else "not_initialized",
            },
        )

    @app.post("/api/v1/score", response_model=FraudScoreResponse, tags=["scoring"])
    async def score_transaction(request: TransactionRequest):
        """Score a single transaction for fraud."""
        if not (_orchestrator and _metrics and _masker and _db):
            raise HTTPException(status_code=503, detail="Service not initialized")

        start = time.monotonic()

        transaction = _build_transaction(request)
        account = _build_default_account(request)
        context = TransactionContext(
            transaction=transaction,
            account=account,
        )

        decision = _orchestrator.analyze_transaction(context)
        latency = (time.monotonic() - start) * 1000

        _metrics.record_scoring_latency(latency)
        _metrics.record_decision(decision.risk_level.value, decision.is_fraud)
        _db.save_decision(decision)

        return FraudScoreResponse(
            transaction_id=_masker.mask_account_id(decision.transaction_id),
            risk_level=decision.risk_level.value,
            fraud_score=round(decision.fraud_score, 4),
            is_fraud=decision.is_fraud,
            confidence=round(decision.confidence, 4),
            explanation=_masker.mask_text(decision.explanation),
            rules_triggered=decision.rules_triggered,
            recommended_action=decision.recommended_action,
        )

    @app.post("/api/v1/batch", response_model=BatchResponse, tags=["scoring"])
    async def score_batch(request: BatchRequest):
        """Score a batch of transactions."""
        if not (_orchestrator and _metrics and _db):
            raise HTTPException(status_code=503, detail="Service not initialized")

        scores = []
        for txn_req in request.transactions:
            transaction = _build_transaction(txn_req)
            account = (
                _build_account_from_request(request.account)
                if request.account
                else _build_default_account(txn_req)
            )
            context = TransactionContext(transaction=transaction, account=account)

            start = time.monotonic()
            decision = _orchestrator.analyze_transaction(context)
            latency = (time.monotonic() - start) * 1000

            _metrics.record_scoring_latency(latency)
            _metrics.record_decision(decision.risk_level.value, decision.is_fraud)
            _db.save_decision(decision)

            scores.append(
                FraudScoreResponse(
                    transaction_id=decision.transaction_id,
                    risk_level=decision.risk_level.value,
                    fraud_score=round(decision.fraud_score, 4),
                    is_fraud=decision.is_fraud,
                    confidence=round(decision.confidence, 4),
                    explanation=decision.explanation,
                    rules_triggered=decision.rules_triggered,
                    recommended_action=decision.recommended_action,
                )
            )

        flagged = sum(1 for s in scores if s.is_fraud)
        avg_score = sum(s.fraud_score for s in scores) / len(scores) if scores else 0.0

        return BatchResponse(
            scores=scores,
            total_processed=len(scores),
            flagged_count=flagged,
            average_score=round(avg_score, 4),
        )

    @app.get("/api/v1/decisions", response_model=list[DecisionResponse], tags=["decisions"])
    async def list_decisions(
        limit: int = Query(default=50, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        """List recent fraud decisions."""
        if not _db:
            raise HTTPException(status_code=503, detail="Service not initialized")
        rows = _db.get_decisions(limit=limit, offset=offset)
        return [_row_to_decision_response(r) for r in rows]

    @app.get(
        "/api/v1/decisions/{transaction_id}",
        response_model=DecisionResponse,
        tags=["decisions"],
    )
    async def get_decision(transaction_id: str):
        """Get decision detail for a specific transaction."""
        if not _db:
            raise HTTPException(status_code=503, detail="Service not initialized")
        rows = _db.get_decisions_by_transaction(transaction_id)
        if not rows:
            raise HTTPException(status_code=404, detail="Decision not found")
        return _row_to_decision_response(rows[0])

    @app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["monitoring"])
    async def get_metrics():
        """Get scoring performance metrics."""
        if not _metrics:
            raise HTTPException(status_code=503, detail="Service not initialized")
        summary = _metrics.get_summary()
        return MetricsResponse(**summary)

    @app.get("/api/v1/patterns", response_model=list[PatternResponse], tags=["knowledge"])
    async def list_patterns():
        """List known fraud patterns from the knowledge base."""
        if not _kb:
            raise HTTPException(status_code=503, detail="Service not initialized")
        patterns = _kb.get_patterns()
        return [
            PatternResponse(
                id=p["id"],
                name=p["name"],
                description=p["description"],
                indicators=p["indicators"],
                risk_level=p["risk_level"],
                category=p["category"],
            )
            for p in patterns
        ]

    return app


def _build_transaction(request: TransactionRequest) -> Transaction:
    """Convert API request to Transaction domain model."""
    return Transaction(
        id=request.id or str(uuid.uuid4()),
        timestamp=request.timestamp or datetime.now(),
        amount=Decimal(str(request.amount)),
        currency=request.currency,
        merchant_name=request.merchant_name,
        merchant_category_code=request.merchant_category_code,
        card_last_four=request.card_last_four,
        account_id=request.account_id,
        location=Location(
            city=request.location.city,
            country=request.location.country,
            latitude=request.location.latitude,
            longitude=request.location.longitude,
        ),
        channel=TransactionChannel(request.channel),
        is_international=request.is_international,
    )


def _build_default_account(request: TransactionRequest) -> Account:
    """Build a default account for single-transaction scoring."""
    return Account(
        id=request.account_id,
        holder_name="Account Holder",
        average_transaction_amount=Decimal("75.00"),
        typical_location=Location(
            city=request.location.city,
            country=request.location.country,
            latitude=request.location.latitude,
            longitude=request.location.longitude,
        ),
        account_open_date=date(2020, 1, 1),
        transaction_history_count=100,
    )


def _build_account_from_request(req: AccountRequest) -> Account:
    """Build account from batch request."""
    return Account(
        id=req.id,
        holder_name=req.holder_name,
        average_transaction_amount=Decimal(str(req.average_transaction_amount)),
        typical_location=Location(
            city=req.typical_location.city,
            country=req.typical_location.country,
            latitude=req.typical_location.latitude,
            longitude=req.typical_location.longitude,
        ),
        account_open_date=date(2020, 1, 1),
        transaction_history_count=req.transaction_history_count,
    )


def _row_to_decision_response(row: dict) -> DecisionResponse:
    """Convert database row to API response."""
    return DecisionResponse(
        transaction_id=row.get("transaction_id", ""),
        risk_level=row.get("risk_level", "LOW"),
        fraud_score=row.get("fraud_score", 0.0),
        is_fraud=bool(row.get("is_fraud", False)),
        confidence=row.get("confidence", 0.0),
        explanation=row.get("explanation", ""),
        rules_triggered=row.get("rules_triggered", []),
        recommended_action=row.get("recommended_action", ""),
        created_at=row.get("created_at"),
    )
