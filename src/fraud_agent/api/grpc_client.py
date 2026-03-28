"""gRPC client for testing the fraud scoring service."""

from __future__ import annotations

from datetime import datetime

import grpc
import structlog

logger = structlog.get_logger(__name__)

try:
    from fraud_agent.proto import fraud_scoring_pb2, fraud_scoring_pb2_grpc
except ImportError:
    fraud_scoring_pb2 = None
    fraud_scoring_pb2_grpc = None


class FraudScoringClient:
    """gRPC client for the fraud scoring service."""

    def __init__(self, host: str = "localhost", port: int = 50051) -> None:
        if fraud_scoring_pb2 is None:
            raise RuntimeError("gRPC proto files not compiled")

        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = fraud_scoring_pb2_grpc.FraudScoringServiceStub(self.channel)

    def score_transaction(
        self,
        amount: float,
        merchant_name: str,
        merchant_category_code: str = "5999",
        card_last_four: str = "0000",
        account_id: str = "ACC-0001-0001",
        city: str = "New York",
        country: str = "US",
        latitude: float = 40.7128,
        longitude: float = -74.0060,
        channel: str = "ONLINE",
        is_international: bool = False,
    ) -> dict:
        """Score a single transaction.

        Returns:
            Dict with fraud score results.
        """
        location = fraud_scoring_pb2.Location(
            city=city, country=country, latitude=latitude, longitude=longitude
        )

        request = fraud_scoring_pb2.TransactionRequest(
            timestamp=datetime.now().isoformat(),
            amount=amount,
            merchant_name=merchant_name,
            merchant_category_code=merchant_category_code,
            card_last_four=card_last_four,
            account_id=account_id,
            location=location,
            channel=channel,
            is_international=is_international,
        )

        response = self.stub.ScoreTransaction(request)
        return {
            "transaction_id": response.transaction_id,
            "risk_level": response.risk_level,
            "fraud_score": response.fraud_score,
            "is_fraud": response.is_fraud,
            "confidence": response.confidence,
            "explanation": response.explanation,
            "rules_triggered": list(response.rules_triggered),
            "recommended_action": response.recommended_action,
        }

    def batch_score(self, transactions: list[dict]) -> dict:
        """Score a batch of transactions.

        Args:
            transactions: List of transaction dicts.

        Returns:
            Batch scoring results.
        """
        requests = []
        for txn in transactions:
            loc = txn.get("location", {})
            location = fraud_scoring_pb2.Location(
                city=loc.get("city", "New York"),
                country=loc.get("country", "US"),
                latitude=loc.get("latitude", 40.7128),
                longitude=loc.get("longitude", -74.0060),
            )
            requests.append(
                fraud_scoring_pb2.TransactionRequest(
                    amount=txn.get("amount", 0),
                    merchant_name=txn.get("merchant_name", ""),
                    merchant_category_code=txn.get("merchant_category_code", "5999"),
                    channel=txn.get("channel", "ONLINE"),
                    location=location,
                )
            )

        response = self.stub.BatchScore(fraud_scoring_pb2.BatchRequest(transactions=requests))
        return {
            "total_processed": response.total_processed,
            "flagged_count": response.flagged_count,
            "average_score": response.average_score,
            "scores": [
                {
                    "transaction_id": s.transaction_id,
                    "risk_level": s.risk_level,
                    "fraud_score": s.fraud_score,
                    "is_fraud": s.is_fraud,
                }
                for s in response.scores
            ],
        }

    def get_decision(self, transaction_id: str) -> dict:
        """Get a previous decision.

        Args:
            transaction_id: The transaction ID to look up.

        Returns:
            Decision details.
        """
        response = self.stub.GetDecision(
            fraud_scoring_pb2.DecisionRequest(transaction_id=transaction_id)
        )
        return {
            "score": {
                "risk_level": response.score.risk_level,
                "fraud_score": response.score.fraud_score,
                "is_fraud": response.score.is_fraud,
            },
            "agent_trace": list(response.agent_trace),
            "analysis_summary": response.analysis_summary,
        }

    def close(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()
