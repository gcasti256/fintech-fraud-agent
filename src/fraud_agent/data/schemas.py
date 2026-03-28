"""Pydantic v2 data models for the fraud detection system.

Defines the core domain objects: transactions, accounts, locations,
and fraud decision outputs used throughout the agent pipeline.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TransactionChannel(str, Enum):
    """Channel through which a transaction was initiated."""

    ONLINE = "ONLINE"
    IN_STORE = "IN_STORE"
    ATM = "ATM"
    MOBILE = "MOBILE"


class RiskLevel(str, Enum):
    """Tiered risk classification for a transaction or fraud decision."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Location(BaseModel):
    """Geographic location associated with a transaction or account."""

    city: str = Field(..., description="City name.")
    country: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code (e.g. 'US', 'GB').",
        min_length=2,
        max_length=2,
    )
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS-84 latitude.")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS-84 longitude.")

    model_config = {"frozen": True}

    @field_validator("country")
    @classmethod
    def country_uppercase(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()


class Transaction(BaseModel):
    """A single card or account transaction presented for fraud scoring.

    All monetary amounts are stored as :class:`~decimal.Decimal` to avoid
    floating-point rounding errors that are unacceptable in financial contexts.
    """

    id: str = Field(..., description="Unique transaction identifier (UUID).")
    timestamp: datetime = Field(..., description="UTC datetime at which the transaction occurred.")
    amount: Decimal = Field(..., gt=Decimal("0"), description="Transaction amount.")
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code.",
    )
    merchant_name: str = Field(..., description="Display name of the merchant.")
    merchant_category_code: str = Field(
        ...,
        pattern=r"^\d{4}$",
        description="ISO 18245 Merchant Category Code (4 digits).",
    )
    card_last_four: str = Field(
        ...,
        pattern=r"^\d{4}$",
        description="Last four digits of the payment card.",
    )
    account_id: str = Field(..., description="Owning account identifier.")
    location: Location = Field(..., description="Geographic location of the transaction.")
    channel: TransactionChannel = Field(
        ..., description="Channel through which the transaction was made."
    )
    is_international: bool = Field(
        default=False,
        description="True when the merchant country differs from the card-issuing country.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary key-value metadata for enrichment or debugging.",
    )

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        """Normalise currency code to uppercase."""
        return v.upper()

    @field_validator("timestamp")
    @classmethod
    def timestamp_has_timezone(cls, v: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware; assume UTC if naive."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class Account(BaseModel):
    """Profile of a cardholder account used as baseline context for scoring.

    The behavioural fields (``average_transaction_amount``,
    ``typical_location``, ``transaction_history_count``) are derived from
    historical spend data and updated periodically by the analytics pipeline.
    """

    id: str = Field(..., description="Unique account identifier.")
    holder_name: str = Field(..., description="Full legal name of the account holder.")
    average_transaction_amount: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Rolling average of the account's transaction amounts.",
    )
    typical_location: Location = Field(
        ..., description="The most common geographic location for this account."
    )
    account_open_date: date = Field(
        ..., description="Calendar date on which the account was opened."
    )
    transaction_history_count: int = Field(
        ..., ge=0, description="Total number of transactions on record for this account."
    )


class FraudDecision(BaseModel):
    """Output produced by the fraud detection agent for a single transaction.

    ``fraud_score`` is a continuous probability in [0, 1]; ``is_fraud`` is the
    binary classification derived from that score plus any hard rules.
    ``agent_trace`` records the ordered list of agents/tools that participated
    in reaching the decision, enabling full auditability.
    """

    transaction_id: str = Field(..., description="ID of the :class:`Transaction` being evaluated.")
    risk_level: RiskLevel = Field(
        ..., description="Categorical risk tier assigned to the transaction."
    )
    fraud_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous fraud probability estimate in [0, 1].",
    )
    is_fraud: bool = Field(
        ..., description="Binary fraud classification derived from score and rules."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the fraud classification in [0, 1].",
    )
    explanation: str = Field(
        ...,
        description="Human-readable narrative explaining the fraud decision.",
    )
    rules_triggered: list[str] = Field(
        default_factory=list,
        description="Names of deterministic rules that fired during evaluation.",
    )
    recommended_action: str = Field(
        ...,
        description=("Operational recommendation, e.g. 'BLOCK', 'REVIEW', 'ALLOW', 'REQUEST_OTP'."),
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC datetime at which the decision was produced.",
    )
    agent_trace: list[str] | None = Field(
        default=None,
        description=(
            "Ordered list of agent/tool identifiers that participated in "
            "producing this decision, for auditability."
        ),
    )

    @model_validator(mode="after")
    def validate_risk_score_consistency(self) -> "FraudDecision":
        """Verify that ``risk_level`` and ``fraud_score`` are broadly consistent.

        This is a soft consistency check; hard overrides (e.g. rule-based
        blocks) may legitimately produce HIGH risk at a lower fraud score.
        """
        score = self.fraud_score
        level = self.risk_level

        if level == RiskLevel.CRITICAL and score < 0.7:
            raise ValueError(f"CRITICAL risk_level requires fraud_score >= 0.7, got {score:.3f}")
        if level == RiskLevel.LOW and score > 0.3:
            raise ValueError(f"LOW risk_level requires fraud_score <= 0.3, got {score:.3f}")
        return self
