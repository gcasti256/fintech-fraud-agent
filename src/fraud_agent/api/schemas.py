"""API request and response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class LocationRequest(BaseModel):
    """Location in an API request."""

    city: str = "New York"
    country: str = "US"
    latitude: float = 40.7128
    longitude: float = -74.0060


class TransactionRequest(BaseModel):
    """Single transaction scoring request."""

    id: str | None = None
    timestamp: datetime | None = None
    amount: float
    currency: str = "USD"
    merchant_name: str
    merchant_category_code: str = "5999"
    card_last_four: str = "0000"
    account_id: str = "ACC-0001-0001"
    location: LocationRequest = Field(default_factory=LocationRequest)
    channel: str = "ONLINE"
    is_international: bool = False


class AccountRequest(BaseModel):
    """Account information for scoring context."""

    id: str = "ACC-0001-0001"
    holder_name: str = "Account Holder"
    average_transaction_amount: float = 75.0
    typical_location: LocationRequest = Field(default_factory=LocationRequest)
    transaction_history_count: int = 100


class BatchRequest(BaseModel):
    """Batch transaction scoring request."""

    transactions: list[TransactionRequest]
    account: AccountRequest | None = None


class FraudScoreResponse(BaseModel):
    """Single fraud score response."""

    transaction_id: str
    risk_level: str
    fraud_score: float
    is_fraud: bool
    confidence: float
    explanation: str
    rules_triggered: list[str]
    recommended_action: str


class BatchResponse(BaseModel):
    """Batch scoring response."""

    scores: list[FraudScoreResponse]
    total_processed: int
    flagged_count: int
    average_score: float


class DecisionResponse(BaseModel):
    """Decision detail response."""

    transaction_id: str
    risk_level: str
    fraud_score: float
    is_fraud: bool
    confidence: float
    explanation: str
    rules_triggered: list[str]
    recommended_action: str
    agent_trace: list[str] = Field(default_factory=list)
    created_at: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    uptime_seconds: float
    components: dict[str, str] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Scoring metrics response."""

    total_scored: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_second: float
    fraud_rate: float
    risk_distribution: dict[str, int]


class PatternResponse(BaseModel):
    """Fraud pattern from knowledge base."""

    id: str
    name: str
    description: str
    indicators: list[str]
    risk_level: str
    category: str
