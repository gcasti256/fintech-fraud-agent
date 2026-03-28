"""RAG (Retrieval-Augmented Generation) sub-package for the fraud detection system.

Provides lightweight, dependency-free vector retrieval over the fraud pattern
knowledge base.  No external API calls or heavy ML models are required —
embeddings are produced via a hash-based TF-IDF approximation using only numpy.

Public API
----------
SimpleEmbedding
    Hash-based TF-IDF embedding with L2 normalisation.
VectorStore
    In-memory cosine-similarity document store.
FraudPatternRetriever
    High-level retriever pre-loaded with the fraud knowledge base.
"""

from .embeddings import SimpleEmbedding
from .retriever import FraudPatternRetriever
from .store import VectorStore

__all__ = ["FraudPatternRetriever", "SimpleEmbedding", "VectorStore"]
