"""Decision agent — renders final fraud/not-fraud verdict with explanation."""

from __future__ import annotations

import structlog

from fraud_agent.agents.state import AgentState

logger = structlog.get_logger(__name__)


# Action recommendations based on risk assessment
ACTION_MAP = {
    "CRITICAL": "block_and_alert",
    "HIGH": "block_and_review",
    "MEDIUM": "flag_for_review",
    "LOW": "approve",
}


class DecisionAgent:
    """Final decision agent that synthesizes triage and analysis into a verdict.

    Produces:
    - Binary fraud/not-fraud classification
    - Confidence score (0-1)
    - Human-readable explanation
    - Recommended action (approve, flag_for_review, block_and_review, block_and_alert)
    """

    def __init__(
        self,
        high_risk_threshold: float = 0.7,
        fraud_threshold: float = 0.6,
    ) -> None:
        self.high_risk_threshold = high_risk_threshold
        self.fraud_threshold = fraud_threshold

    def run(self, state: AgentState) -> AgentState:
        """Render final fraud decision.

        Combines triage score, analysis findings, and pattern matches
        to produce a final verdict.

        Args:
            state: Pipeline state with triage and analysis results.

        Returns:
            Updated state with final decision.
        """
        logger.info("decision_agent.start", transaction_id=state.transaction.get("id"))

        try:
            # Calculate composite fraud score
            fraud_score = self._calculate_composite_score(state)

            # Determine risk level
            risk_level = self._determine_risk_level(fraud_score, state)

            # Determine if fraudulent
            is_fraud = fraud_score >= self.fraud_threshold

            # Calculate confidence based on evidence strength
            confidence = self._calculate_confidence(state)

            # Build explanation
            explanation = self._build_explanation(state, fraud_score, is_fraud)

            # Determine action
            recommended_action = ACTION_MAP.get(risk_level, "approve")

            # Update state
            state.final_risk_level = risk_level
            state.final_fraud_score = round(fraud_score, 4)
            state.is_fraud = is_fraud
            state.confidence = round(confidence, 4)
            state.explanation = explanation
            state.recommended_action = recommended_action
            state.agent_trace = state.agent_trace + ["decision"]

            logger.info(
                "decision_agent.complete",
                transaction_id=state.transaction.get("id"),
                is_fraud=is_fraud,
                fraud_score=fraud_score,
                confidence=confidence,
                action=recommended_action,
            )

        except Exception as e:
            logger.error("decision_agent.error", error=str(e))
            state.errors = state.errors + [f"Decision error: {e}"]
            state.final_risk_level = state.initial_risk_level
            state.final_fraud_score = state.initial_risk_score
            state.is_fraud = state.initial_risk_score >= self.fraud_threshold
            state.confidence = 0.5
            state.explanation = f"Decision based on triage only (analysis error): {e}"
            state.recommended_action = ACTION_MAP.get(state.initial_risk_level, "flag_for_review")
            state.agent_trace = state.agent_trace + ["decision(error)"]

        return state

    def _calculate_composite_score(self, state: AgentState) -> float:
        """Calculate weighted composite fraud score from all evidence."""
        triage_score = state.initial_risk_score
        anomaly_count = len(state.anomaly_flags)
        pattern_count = len(state.pattern_matches)
        rule_count = len(state.rules_triggered)

        # Base: triage score (40% weight)
        composite = triage_score * 0.4

        # Anomaly contribution (25% weight)
        anomaly_factor = min(anomaly_count / 5.0, 1.0)
        composite += anomaly_factor * 0.25

        # Pattern match contribution (20% weight)
        pattern_factor = min(pattern_count / 3.0, 1.0)
        composite += pattern_factor * 0.20

        # Rule trigger contribution (15% weight)
        rule_factor = min(rule_count / 4.0, 1.0)
        composite += rule_factor * 0.15

        return min(max(composite, 0.0), 1.0)

    def _determine_risk_level(self, fraud_score: float, state: AgentState) -> str:
        """Map fraud score to risk level, considering additional context."""
        # Critical indicators override score
        critical_rules = {"card_testing_pattern", "geographic_impossibility"}
        if any(r.lower().replace(" ", "_") in critical_rules for r in state.rules_triggered):
            if fraud_score > 0.5:
                return "CRITICAL"

        if fraud_score >= 0.8:
            return "CRITICAL"
        elif fraud_score >= 0.6:
            return "HIGH"
        elif fraud_score >= 0.4:
            return "MEDIUM"
        return "LOW"

    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate decision confidence based on evidence quantity and consistency."""
        evidence_count = (
            len(state.rules_triggered) + len(state.anomaly_flags) + len(state.pattern_matches)
        )

        # More evidence = higher confidence (up to a point)
        evidence_factor = min(evidence_count / 6.0, 1.0)

        # Consistency: if triage and analysis agree, higher confidence
        triage_high = state.initial_risk_level in ("HIGH", "CRITICAL")
        analysis_high = len(state.anomaly_flags) >= 2

        if triage_high == analysis_high:
            consistency_bonus = 0.15
        else:
            consistency_bonus = -0.1

        # Errors reduce confidence
        error_penalty = len(state.errors) * 0.15

        confidence = 0.5 + (evidence_factor * 0.35) + consistency_bonus - error_penalty
        return min(max(confidence, 0.1), 0.99)

    def _build_explanation(self, state: AgentState, fraud_score: float, is_fraud: bool) -> str:
        """Build a comprehensive human-readable explanation."""
        parts = []

        verdict = "FRAUDULENT" if is_fraud else "LEGITIMATE"
        parts.append(f"Transaction assessed as {verdict} (score: {fraud_score:.2f}).")

        if state.rules_triggered:
            parts.append(f"Triggered rules: {', '.join(state.rules_triggered)}.")

        if state.anomaly_flags:
            parts.append(f"Anomalies: {'; '.join(state.anomaly_flags[:3])}.")

        if state.pattern_matches:
            names = [p["pattern_name"] for p in state.pattern_matches[:2]]
            parts.append(f"Matching known patterns: {', '.join(names)}.")

        if (
            state.analysis_summary
            and state.analysis_summary != "No significant anomalies detected."
        ):
            parts.append(f"Analysis: {state.analysis_summary}")

        return " ".join(parts)
