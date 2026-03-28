"""Analysis agent — deep pattern matching and anomaly detection."""

from __future__ import annotations

import structlog

from fraud_agent.agents.state import AgentState
from fraud_agent.data.schemas import Account, Transaction
from fraud_agent.rag.retriever import FraudPatternRetriever
from fraud_agent.scoring.features import FeatureExtractor

logger = structlog.get_logger(__name__)


class AnalysisAgent:
    """Deep analysis agent that examines transactions flagged by triage.

    Performs:
    - RAG lookup against fraud pattern knowledge base
    - Feature extraction and anomaly analysis
    - Pattern correlation across recent transaction history
    """

    def __init__(
        self,
        retriever: FraudPatternRetriever | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        self.retriever = retriever or FraudPatternRetriever()
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def run(self, state: AgentState) -> AgentState:
        """Execute deep analysis on the transaction.

        Args:
            state: Current agent pipeline state with triage results.

        Returns:
            Updated state with analysis results.
        """
        logger.info("analysis_agent.start", transaction_id=state.transaction.get("id"))

        try:
            transaction = Transaction.model_validate(state.transaction)
            account = Account.model_validate(state.account)
            recent = [Transaction.model_validate(t) for t in state.recent_transactions]

            # Extract features for detailed analysis
            features = self.feature_extractor.extract(transaction, account, recent)
            state.feature_analysis = features

            # Identify anomalies from features
            anomalies = self._detect_anomalies(features)
            state.anomaly_flags = anomalies

            # RAG lookup for matching fraud patterns
            patterns = self.retriever.retrieve_for_transaction(transaction)
            state.pattern_matches = [
                {
                    "pattern_id": p.get("id", ""),
                    "pattern_name": p.get("name", ""),
                    "description": p.get("description", ""),
                    "risk_level": p.get("risk_level", ""),
                    "indicators": p.get("indicators", []),
                }
                for p in patterns
            ]

            # Build analysis summary
            state.analysis_summary = self._build_summary(
                features, anomalies, patterns, state.rules_triggered
            )
            state.agent_trace = state.agent_trace + ["analyzer"]

            logger.info(
                "analysis_agent.complete",
                transaction_id=transaction.id,
                anomaly_count=len(anomalies),
                pattern_matches=len(patterns),
            )

        except Exception as e:
            logger.error("analysis_agent.error", error=str(e))
            state.errors = state.errors + [f"Analysis error: {e}"]
            state.analysis_summary = f"Analysis incomplete due to error: {e}"
            state.agent_trace = state.agent_trace + ["analyzer(error)"]

        return state

    def _detect_anomalies(self, features: dict[str, float]) -> list[str]:
        """Identify anomalous features that deviate from normal patterns."""
        anomalies = []

        if features.get("amount_ratio", 0) > 3.0:
            ratio = features["amount_ratio"]
            anomalies.append(f"Amount is {ratio:.1f}x account average")

        if features.get("velocity_10min", 0) > 3:
            count = int(features["velocity_10min"])
            anomalies.append(f"High velocity: {count} transactions in 10 minutes")

        if features.get("distance_from_typical", 0) > 0.5:
            anomalies.append("Transaction location significantly far from typical")

        if features.get("is_nighttime", 0) > 0:
            anomalies.append("Transaction during high-risk nighttime hours (2-5 AM)")

        if features.get("high_risk_mcc", 0) > 0:
            anomalies.append("High-risk merchant category code")

        if features.get("channel_risk", 0) > 0.5:
            anomalies.append("Elevated channel risk (online/card-not-present)")

        if features.get("amount_zscore", 0) > 2.0:
            zscore = features["amount_zscore"]
            anomalies.append(f"Amount z-score of {zscore:.1f} (>2 standard deviations)")

        return anomalies

    def _build_summary(
        self,
        features: dict[str, float],
        anomalies: list[str],
        patterns: list[dict],
        rules_triggered: list[str],
    ) -> str:
        """Build a human-readable analysis summary."""
        parts = []

        if anomalies:
            parts.append(f"Detected {len(anomalies)} anomalies: {'; '.join(anomalies)}.")

        if patterns:
            pattern_names = [p.get("name", "unknown") for p in patterns[:3]]
            parts.append(f"Matching fraud patterns: {', '.join(pattern_names)}.")

        if rules_triggered:
            parts.append(f"Rules triggered: {', '.join(rules_triggered)}.")

        risk_factors = sum(
            1
            for key in ["amount_ratio", "velocity_10min", "distance_from_typical", "high_risk_mcc"]
            if features.get(key, 0) > 0.5
        )
        parts.append(f"Risk factors identified: {risk_factors}/4.")

        return " ".join(parts) if parts else "No significant anomalies detected."
