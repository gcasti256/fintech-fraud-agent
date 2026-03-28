"""Shared agent state definitions for the fraud detection pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from fraud_agent.data.schemas import Account, FraudDecision, RiskLevel, Transaction


class TransactionContext(BaseModel):
    """Context for a transaction being analyzed."""

    transaction: Transaction
    account: Account
    recent_transactions: list[Transaction] = Field(default_factory=list)


class AgentState(BaseModel):
    """Shared state passed between agents in the LangGraph pipeline.

    This TypedDict-compatible model flows through the graph, accumulating
    analysis from each agent in the pipeline.
    """

    # Input
    transaction: dict = Field(default_factory=dict)
    account: dict = Field(default_factory=dict)
    recent_transactions: list[dict] = Field(default_factory=list)

    # Triage output
    initial_risk_level: str = "LOW"
    initial_risk_score: float = 0.0
    triage_explanation: str = ""
    rules_triggered: list[str] = Field(default_factory=list)

    # Analysis output
    pattern_matches: list[dict] = Field(default_factory=list)
    feature_analysis: dict = Field(default_factory=dict)
    anomaly_flags: list[str] = Field(default_factory=list)
    analysis_summary: str = ""

    # Decision output
    final_risk_level: str = "LOW"
    final_fraud_score: float = 0.0
    is_fraud: bool = False
    confidence: float = 0.0
    explanation: str = ""
    recommended_action: str = "approve"

    # Trace
    agent_trace: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def to_decision(self) -> FraudDecision:
        """Convert final state to a FraudDecision."""
        risk_map = {
            "LOW": RiskLevel.LOW,
            "MEDIUM": RiskLevel.MEDIUM,
            "HIGH": RiskLevel.HIGH,
            "CRITICAL": RiskLevel.CRITICAL,
        }
        return FraudDecision(
            transaction_id=self.transaction.get("id", ""),
            risk_level=risk_map.get(self.final_risk_level, RiskLevel.LOW),
            fraud_score=self.final_fraud_score,
            is_fraud=self.is_fraud,
            confidence=self.confidence,
            explanation=self.explanation,
            rules_triggered=self.rules_triggered,
            recommended_action=self.recommended_action,
            agent_trace=self.agent_trace,
        )
