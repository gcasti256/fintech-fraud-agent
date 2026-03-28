"""In-memory vector store for fraud pattern retrieval.

Stores document embeddings as a dense numpy matrix and retrieves the
top-k most similar documents using cosine similarity.  All operations
are O(n) in the number of stored documents, which is acceptable for
the small fraud pattern knowledge base (< 1 000 entries).

No persistence layer is required — the store is re-populated from the
knowledge base on every process startup, which takes milliseconds.
"""

from __future__ import annotations

import numpy as np

from .embeddings import SimpleEmbedding


class VectorStore:
    """In-memory cosine-similarity document store.

    Documents are arbitrary dicts.  During indexing, one field (by default
    ``"description"``) is selected as the embedding source.  All other fields
    are preserved and returned as-is on retrieval.

    Attributes:
        embedding: The :class:`SimpleEmbedding` instance used for all
            embed operations.

    Example::

        emb = SimpleEmbedding(dimension=128)
        store = VectorStore(embedding=emb)
        store.add_documents([
            {"description": "card skimming at ATM", "name": "card_skimming"},
            {"description": "online account takeover", "name": "ato"},
        ])
        results = store.search("suspicious ATM withdrawal", top_k=1)
        doc, score = results[0]
        assert doc["name"] == "card_skimming"
    """

    def __init__(self, embedding: SimpleEmbedding) -> None:
        """Initialise an empty vector store.

        Args:
            embedding: Embedding instance used to encode documents and queries.
        """
        self.embedding = embedding
        self._documents: list[dict] = []
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[dict], text_field: str = "description") -> None:
        """Embed and store a batch of documents.

        Documents are appended to any already-stored documents; calling this
        method multiple times accumulates entries.  The embedding matrix is
        rebuilt after each call.

        Args:
            documents: List of dicts.  Each must contain *text_field*.
            text_field: Key within each document dict whose value is used as
                the embedding source.  Defaults to ``"description"``.

        Raises:
            ValueError: If *documents* is empty or any entry is missing
                *text_field*.
        """
        if not documents:
            raise ValueError("documents must be a non-empty list")

        for i, doc in enumerate(documents):
            if text_field not in doc:
                raise ValueError(f"Document at index {i} is missing required field '{text_field}'")

        texts = [str(doc[text_field]) for doc in documents]
        new_embeddings = self.embedding.embed_batch(texts)

        self._documents.extend(documents)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Retrieve the top-k most similar documents for *query*.

        Similarity is measured by cosine similarity between the query
        embedding and each stored document embedding.  Results are sorted
        by descending similarity score.

        Args:
            query: Free-text search query.
            top_k: Maximum number of results to return.  Clamped to the
                number of stored documents if larger.

        Returns:
            List of ``(document, similarity_score)`` tuples, sorted from
            most to least similar.  Returns an empty list if the store is
            empty.

        Raises:
            ValueError: If *top_k* is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if self._embeddings is None or len(self._documents) == 0:
            return []

        query_vec = self.embedding.embed(query)
        similarities = self._cosine_similarity(query_vec, self._embeddings)

        # Clamp top_k to the number of available documents.
        k = min(top_k, len(self._documents))

        # Use argpartition for O(n) partial sort, then sort the top-k slice.
        if k == len(self._documents):
            top_indices = np.argsort(similarities)[::-1]
        else:
            partition_idx = np.argpartition(similarities, -k)[-k:]
            top_indices = partition_idx[np.argsort(similarities[partition_idx])[::-1]]

        return [(self._documents[int(idx)], float(similarities[idx])) for idx in top_indices]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between *query_vec* and every row of *doc_vecs*.

        Because :class:`SimpleEmbedding` always produces L2-normalised vectors,
        cosine similarity reduces to a simple dot product:

            cos(q, d) = q · d  (when ||q|| == ||d|| == 1)

        For robustness, we still normalise both sides here so that the store
        remains correct even if called with externally-produced vectors.

        Args:
            query_vec: 1-D float array of shape ``(dimension,)``.
            doc_vecs: 2-D float array of shape ``(n_docs, dimension)``.

        Returns:
            1-D float array of shape ``(n_docs,)`` with similarity scores
            in ``[-1, 1]``.  Degenerate zero vectors yield a score of 0.0.
        """
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-12:
            return np.zeros(len(doc_vecs), dtype=np.float64)

        q_unit = query_vec / query_norm

        # Row-wise norms; add epsilon to avoid division by zero.
        doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms < 1e-12, 1e-12, doc_norms)
        d_unit = doc_vecs / doc_norms

        return d_unit @ q_unit

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of stored documents."""
        return len(self._documents)

    def __repr__(self) -> str:
        n = len(self._documents)
        dim = self.embedding.dimension
        return f"VectorStore(n_docs={n}, embedding_dim={dim})"
