"""Tests for the FastAPI REST API."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from fraud_agent.api import rest
from fraud_agent.api.rest import create_app


@pytest.fixture(autouse=True)
def init_singletons():
    """Manually initialize the module-level singletons that the lifespan
    would normally set up, since httpx ASGITransport doesn't run the lifespan."""
    import time

    from fraud_agent.agents.orchestrator import FraudDetectionOrchestrator
    from fraud_agent.data.knowledge_base import FraudKnowledgeBase
    from fraud_agent.db import Database
    from fraud_agent.guardrails.pii_masker import PIIMasker
    from fraud_agent.monitoring.metrics import MetricsCollector

    rest._orchestrator = FraudDetectionOrchestrator()
    rest._db = Database(db_path=":memory:")
    rest._metrics = MetricsCollector()
    rest._masker = PIIMasker()
    rest._kb = FraudKnowledgeBase()
    rest._start_time = time.time()

    yield

    if rest._db:
        rest._db.close()
    rest._orchestrator = None
    rest._db = None
    rest._metrics = None
    rest._masker = None
    rest._kb = None


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestScoreEndpoint:
    @pytest.mark.asyncio
    async def test_score_transaction(self, client):
        payload = {
            "amount": 42.50,
            "merchant_name": "Grocery Store",
            "merchant_category_code": "5411",
            "card_last_four": "1234",
            "account_id": "ACC-0001-0001",
            "channel": "IN_STORE",
        }
        resp = await client.post("/api/v1/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "fraud_score" in data
        assert "risk_level" in data
        assert "is_fraud" in data
        assert 0.0 <= data["fraud_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_suspicious(self, client):
        payload = {
            "amount": 5000.00,
            "merchant_name": "Casino",
            "merchant_category_code": "7995",
            "card_last_four": "5678",
            "channel": "ONLINE",
            "is_international": True,
        }
        resp = await client.post("/api/v1/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["fraud_score"] > 0


class TestBatchEndpoint:
    @pytest.mark.asyncio
    async def test_batch_scoring(self, client):
        payload = {
            "transactions": [
                {
                    "amount": 25.00,
                    "merchant_name": "Store A",
                    "channel": "IN_STORE",
                },
                {
                    "amount": 50.00,
                    "merchant_name": "Store B",
                    "channel": "ONLINE",
                },
            ]
        }
        resp = await client.post("/api/v1/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_processed"] == 2
        assert len(data["scores"]) == 2


class TestDecisionsEndpoint:
    @pytest.mark.asyncio
    async def test_list_decisions(self, client):
        resp = await client.get("/api/v1/decisions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_decision_not_found(self, client):
        resp = await client.get("/api/v1/decisions/nonexistent-id")
        assert resp.status_code == 404


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics(self, client):
        resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_scored" in data


class TestPatternsEndpoint:
    @pytest.mark.asyncio
    async def test_list_patterns(self, client):
        resp = await client.get("/api/v1/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert "name" in data[0]

    @pytest.mark.asyncio
    async def test_pattern_structure(self, client):
        """Each pattern has required fields: id, name, description, risk_level, category."""
        resp = await client.get("/api/v1/patterns")
        data = resp.json()
        for pattern in data:
            assert "id" in pattern
            assert "name" in pattern
            assert "description" in pattern
            assert "risk_level" in pattern
            assert "category" in pattern


class TestScoreValidation:
    @pytest.mark.asyncio
    async def test_score_invalid_request(self, client):
        """POST /api/v1/score with missing required fields returns HTTP 422."""
        resp = await client.post("/api/v1/score", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_response_has_recommended_action(self, client):
        """Score response includes a recommended_action string."""
        payload = {"amount": 50.0, "merchant_name": "Test Shop"}
        resp = await client.post("/api/v1/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_action" in data
        assert isinstance(data["recommended_action"], str)


class TestMetricsDetails:
    @pytest.mark.asyncio
    async def test_metrics_latency_fields(self, client):
        """Metrics response includes latency statistics."""
        resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "avg_latency_ms" in data
        assert "p95_latency_ms" in data
