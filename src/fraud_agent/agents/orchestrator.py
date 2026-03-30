"""LangGraph orchestrator — wires triage, analysis, and decision agents into a StateGraph."""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from fraud_agent.agents.analyzer import AnalysisAgent
from fraud_agent.agents.decision import DecisionAgent
from fraud_agent.agents.state import AgentState, TransactionContext
from fraud_agent.agents.triage import TriageAgent
from fraud_agent.data.schemas import FraudDecision
from fraud_agent.rag.retriever import FraudPatternRetriever
from fraud_agent.scoring.engine import ScoringEngine
from fraud_agent.scoring.features import FeatureExtractor

logger = structlog.get_logger(__name__)


class FraudDetectionOrchestrator:
    """Multi-agent fraud detection pipeline built on LangGraph.

    Flow:
        Transaction → Triage Agent (quick risk score)
            → LOW: auto-approve, log
            → MEDIUM/HIGH/CRITICAL: Analyzer Agent (deep pattern matching + RAG)
                → Decision Agent (final verdict with confidence + explanation)
    """

    def __init__(
        self,
        scoring_engine: ScoringEngine | None = None,
        retriever: FraudPatternRetriever | None = None,
    ) -> None:
        self.triage = TriageAgent(scoring_engine=scoring_engine)
        self.analyzer = AnalysisAgent(
            retriever=retriever,
            feature_extractor=FeatureExtractor(),
        )
        self.decision = DecisionAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph StateGraph for the fraud detection pipeline."""
        workflow = StateGraph(AgentState)

        workflow.add_node("triage", self.triage.run)
        workflow.add_node("analyze", self.analyzer.run)
        workflow.add_node("decide", self.decision.run)
        workflow.add_node("auto_approve", self._auto_approve)

        workflow.set_entry_point("triage")

        workflow.add_conditional_edges(
            "triage",
            TriageAgent.should_analyze,
            {
                "analyze": "analyze",
                "approve": "auto_approve",
            },
        )

        workflow.add_edge("analyze", "decide")
        workflow.add_edge("decide", END)
        workflow.add_edge("auto_approve", END)

        return workflow.compile()

    @staticmethod
    def _auto_approve(state: AgentState) -> AgentState:
        """Auto-approve low-risk transactions."""
        state.final_risk_level = "LOW"
        state.final_fraud_score = state.initial_risk_score
        state.is_fraud = False
        state.confidence = 0.9
        state.explanation = (
            f"Transaction auto-approved. Triage score: {state.initial_risk_score:.2f}. "
            f"{state.triage_explanation}"
        )
        state.recommended_action = "approve"
        state.agent_trace = state.agent_trace + ["auto_approve"]
        return state

    def analyze_transaction(self, context: TransactionContext) -> FraudDecision:
        """Run the full fraud detection pipeline on a transaction.

        Args:
            context: Transaction context with transaction, account, and history.

        Returns:
            FraudDecision with the final verdict.
        """
        logger.info(
            "orchestrator.start",
            transaction_id=context.transaction.id,
        )

        initial_state = AgentState(
            transaction=context.transaction.model_dump(mode="json"),
            account=context.account.model_dump(mode="json"),
            recent_transactions=[t.model_dump(mode="json") for t in context.recent_transactions],
        )

        # Run the graph
        final_state_dict = self.graph.invoke(initial_state.model_dump())
        final_state = AgentState.model_validate(final_state_dict)

        decision = final_state.to_decision()

        logger.info(
            "orchestrator.complete",
            transaction_id=context.transaction.id,
            is_fraud=decision.is_fraud,
            risk_level=decision.risk_level.value,
            agent_trace=decision.agent_trace,
        )

        return decision

    def analyze_batch(self, contexts: list[TransactionContext]) -> list[FraudDecision]:
        """Analyze a batch of transactions sequentially.

        Args:
            contexts: List of transaction contexts.

        Returns:
            List of fraud decisions.
        """
        return [self.analyze_transaction(ctx) for ctx in contexts]
