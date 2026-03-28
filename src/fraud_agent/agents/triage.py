"""Triage agent — performs initial risk assessment and routing."""

from __future__ import annotations

import structlog

from fraud_agent.agents.state import AgentState
from fraud_agent.data.schemas import Account, Transaction
from fraud_agent.scoring.engine import ScoringEngine

logger = structlog.get_logger(__name__)


class TriageAgent:
    """First-pass agent that quickly scores transactions and routes them.

    Routes to:
    - LOW risk: auto-approve
    - MEDIUM risk: forward to AnalysisAgent for deeper inspection
    - HIGH/CRITICAL risk: forward to AnalysisAgent with urgency flag
    """

    def __init__(self, scoring_engine: ScoringEngine | None = None) -> None:
        self.scoring_engine = scoring_engine or ScoringEngine()

    def run(self, state: AgentState) -> AgentState:
        """Execute triage on the transaction.

        Args:
            state: Current agent pipeline state.

        Returns:
            Updated state with triage results.
        """
        logger.info("triage_agent.start", transaction_id=state.transaction.get("id"))

        try:
            transaction = Transaction.model_validate(state.transaction)
            account = Account.model_validate(state.account)
            recent = [Transaction.model_validate(t) for t in state.recent_transactions]

            decision = self.scoring_engine.score_transaction(transaction, account, recent)

            state.initial_risk_level = decision.risk_level.value
            state.initial_risk_score = decision.fraud_score
            state.triage_explanation = decision.explanation
            state.rules_triggered = decision.rules_triggered
            state.agent_trace = state.agent_trace + ["triage"]

            logger.info(
                "triage_agent.complete",
                transaction_id=transaction.id,
                risk_level=decision.risk_level.value,
                score=decision.fraud_score,
            )

        except Exception as e:
            logger.error("triage_agent.error", error=str(e))
            state.errors = state.errors + [f"Triage error: {e}"]
            state.initial_risk_level = "HIGH"
            state.initial_risk_score = 0.75
            state.triage_explanation = f"Error during triage, defaulting to HIGH risk: {e}"
            state.agent_trace = state.agent_trace + ["triage(error)"]

        return state

    @staticmethod
    def should_analyze(state: AgentState) -> str:
        """Routing function for LangGraph conditional edges.

        Returns:
            "analyze" if transaction needs deeper analysis, "approve" if low risk.
        """
        if state.initial_risk_level in ("HIGH", "CRITICAL", "MEDIUM"):
            return "analyze"
        return "approve"
