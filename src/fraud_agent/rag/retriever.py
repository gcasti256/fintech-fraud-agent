"""High-level fraud pattern retriever.

Composes :class:`SimpleEmbedding` and :class:`VectorStore` into a domain-aware
retriever that is pre-loaded with every pattern from :class:`FraudKnowledgeBase`
at construction time.  Callers use :meth:`retrieve` or the more convenient
:meth:`retrieve_for_transaction` which automatically builds a rich query string
from a :class:`Transaction` object.
"""

from __future__ import annotations

from fraud_agent.data.knowledge_base import FraudKnowledgeBase
from fraud_agent.data.schemas import Transaction

from .embeddings import SimpleEmbedding
from .store import VectorStore


class FraudPatternRetriever:
    """Retriever pre-populated with the fraud knowledge base.

    At construction the full catalogue is embedded and indexed into an
    in-memory :class:`VectorStore`.  All subsequent retrieve calls are
    pure in-process numpy operations — no I/O, no external calls.

    Example::

        retriever = FraudPatternRetriever()
        patterns = retriever.retrieve("ATM card skimming withdrawal", top_k=3)
        for p in patterns:
            print(p["name"], p["risk_level"])
    """

    def __init__(self) -> None:
        """Construct the retriever and index the knowledge base.

        The embedding dimension is set to 256 for this retriever to give
        enough capacity for the ~20 fraud pattern descriptions.
        """
        self._embedding = SimpleEmbedding(dimension=256)
        self._store = VectorStore(embedding=self._embedding)

        # Load and index knowledge base documents.
        kb = FraudKnowledgeBase()
        documents = kb.get_patterns()

        # Build vocabulary from all descriptions for richer IDF weighting.
        descriptions = [doc["description"] for doc in documents]
        self._embedding._build_vocab(descriptions)

        self._store.add_documents(documents, text_field="description")

    # ------------------------------------------------------------------
    # Public retrieval interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the most relevant fraud patterns for a free-text query.

        Args:
            query: Natural-language description of the suspicious activity
                or transaction features to match against.
            top_k: Number of patterns to return.  Defaults to 3.

        Returns:
            List of fraud pattern dicts sorted by descending relevance.
            Each dict contains the fields defined in
            :class:`~fraud_agent.data.knowledge_base.FraudPattern`.
        """
        results = self._store.search(query, top_k=top_k)
        return [doc for doc, _score in results]

    def retrieve_with_scores(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        """Retrieve patterns with their similarity scores.

        Like :meth:`retrieve` but also returns the cosine similarity score
        for each pattern, which callers can use as a confidence signal.

        Args:
            query: Free-text query string.
            top_k: Number of results.

        Returns:
            List of ``(pattern_dict, similarity_score)`` tuples.
        """
        return self._store.search(query, top_k=top_k)

    def retrieve_for_transaction(self, transaction: Transaction, top_k: int = 3) -> list[dict]:
        """Retrieve fraud patterns relevant to a specific transaction.

        Builds a rich natural-language query from the transaction's features
        (amount, MCC, channel, location, international flag) and delegates to
        :meth:`retrieve`.

        Args:
            transaction: The :class:`~fraud_agent.data.schemas.Transaction`
                object to retrieve patterns for.
            top_k: Number of patterns to return.

        Returns:
            List of relevant fraud pattern dicts.
        """
        query = self._build_transaction_query(transaction)
        return self.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transaction_query(self, transaction: Transaction) -> str:
        """Convert a transaction into a descriptive natural-language query.

        The query encodes the most discriminative features for fraud pattern
        matching: amount band, MCC, channel, country, and international flag.
        Additional metadata fields (if present) are appended to enrich the
        query for patterns that reference specific merchant names or custom
        risk signals.

        Args:
            transaction: Incoming transaction.

        Returns:
            Multi-sentence string describing the transaction for embedding.
        """
        parts: list[str] = []

        # Amount band — rough categorical descriptor.
        amount = float(transaction.amount)
        if amount < 10:
            amount_desc = "micro transaction under ten dollars"
        elif amount < 100:
            amount_desc = "small transaction under one hundred dollars"
        elif amount < 1_000:
            amount_desc = "medium transaction under one thousand dollars"
        elif amount < 10_000:
            amount_desc = "large transaction under ten thousand dollars"
        else:
            amount_desc = "very large transaction over ten thousand dollars"

        parts.append(f"Transaction of {amount_desc} at merchant {transaction.merchant_name}.")

        # MCC.
        parts.append(f"Merchant category code {transaction.merchant_category_code}.")

        # Channel.
        parts.append(f"Transaction channel is {transaction.channel.value.lower()}.")

        # Location.
        parts.append(
            f"Transaction location: {transaction.location.city}, {transaction.location.country}."
        )

        # International flag.
        if transaction.is_international:
            parts.append(
                "This is an international transaction occurring outside the "
                "cardholder home country."
            )

        # Currency if non-USD.
        if transaction.currency != "USD":
            parts.append(
                f"Transaction currency is {transaction.currency}, "
                "indicating a foreign currency purchase."
            )

        # Hour of day — off-hours transactions are higher risk.
        hour = transaction.timestamp.hour
        if hour < 6 or hour >= 22:
            parts.append(
                "Transaction occurred outside normal business hours, "
                "which may indicate automated or unauthorised activity."
            )

        # Extra metadata signals.
        if transaction.metadata:
            for key, value in transaction.metadata.items():
                if isinstance(value, str) and value:
                    parts.append(f"{key.replace('_', ' ')}: {value}.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._store)
        return f"FraudPatternRetriever(n_patterns={n})"
