"""Tests for RAG components: SimpleEmbedding, VectorStore, FraudPatternRetriever."""

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest

from fraud_agent.data.schemas import (
    Account,
    Location,
    Transaction,
    TransactionChannel,
)
from fraud_agent.rag.embeddings import SimpleEmbedding
from fraud_agent.rag.retriever import FraudPatternRetriever
from fraud_agent.rag.store import VectorStore


def make_transaction(**overrides):
    defaults = {
        "id": "test-txn-001",
        "timestamp": datetime(2024, 6, 15, 14, 30),
        "amount": Decimal("50.00"),
        "currency": "USD",
        "merchant_name": "Test Store",
        "merchant_category_code": "5411",
        "card_last_four": "1234",
        "account_id": "ACC-0001-1234",
        "location": Location(city="New York", country="US", latitude=40.7128, longitude=-74.006),
        "channel": TransactionChannel.IN_STORE,
        "is_international": False,
    }
    defaults.update(overrides)
    return Transaction(**defaults)


def make_account(**overrides):
    defaults = {
        "id": "ACC-0001-1234",
        "holder_name": "Test User",
        "average_transaction_amount": Decimal("75.00"),
        "typical_location": Location(
            city="New York", country="US", latitude=40.7128, longitude=-74.006
        ),
        "account_open_date": date(2020, 1, 1),
        "transaction_history_count": 100,
    }
    defaults.update(overrides)
    return Account(**defaults)


# ---------------------------------------------------------------------------
# SimpleEmbedding tests
# ---------------------------------------------------------------------------


class TestSimpleEmbedding:
    def test_simple_embedding_creates_vector(self):
        """embed() returns a numpy array of the correct dimension."""
        emb = SimpleEmbedding(dimension=128)
        vec = emb.embed("card skimming ATM withdrawal suspicious")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (128,)

    def test_simple_embedding_default_dimension(self):
        """Default dimension is 128."""
        emb = SimpleEmbedding()
        assert emb.dimension == 128
        vec = emb.embed("hello world")
        assert vec.shape == (128,)

    def test_simple_embedding_custom_dimension(self):
        """Custom dimension is respected."""
        emb = SimpleEmbedding(dimension=256)
        vec = emb.embed("test text")
        assert vec.shape == (256,)

    def test_simple_embedding_unit_norm(self):
        """Embedding vectors are L2-normalised (unit norm)."""
        emb = SimpleEmbedding(dimension=128)
        vec = emb.embed("fraud detection velocity")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_simple_embedding_consistency(self):
        """The same text always produces the same embedding."""
        emb = SimpleEmbedding(dimension=128)
        text = "card skimming at ATM machine"
        vec1 = emb.embed(text)
        vec2 = emb.embed(text)
        np.testing.assert_array_equal(vec1, vec2)

    def test_simple_embedding_different_texts(self):
        """Different texts produce different embedding vectors."""
        emb = SimpleEmbedding(dimension=128)
        vec1 = emb.embed("card skimming ATM fraud")
        vec2 = emb.embed("account takeover credential stuffing phishing")
        assert not np.array_equal(vec1, vec2)

    def test_simple_embedding_invalid_dimension(self):
        """Dimension < 1 raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be >= 1"):
            SimpleEmbedding(dimension=0)

    def test_simple_embedding_embed_batch_shape(self):
        """embed_batch returns a 2-D array with the correct shape."""
        emb = SimpleEmbedding(dimension=64)
        texts = ["first text", "second text", "third text"]
        matrix = emb.embed_batch(texts)
        assert matrix.shape == (3, 64)

    def test_simple_embedding_embed_batch_empty(self):
        """embed_batch with empty list returns a zero-row matrix."""
        emb = SimpleEmbedding(dimension=64)
        matrix = emb.embed_batch([])
        assert matrix.shape == (0, 64)


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------


class TestVectorStore:
    @pytest.fixture()
    def store(self):
        emb = SimpleEmbedding(dimension=128)
        return VectorStore(embedding=emb)

    @pytest.fixture()
    def populated_store(self, store):
        docs = [
            {
                "description": "card skimming ATM cash withdrawal suspicious",
                "name": "card_skimming",
            },
            {"description": "account takeover credential stuffing phishing", "name": "ato"},
            {"description": "velocity rapid micro transactions card testing", "name": "velocity"},
            {"description": "geographic impossibility travel far distance", "name": "geo"},
            {
                "description": "online purchase fraud international transaction",
                "name": "online_fraud",
            },
        ]
        store.add_documents(docs)
        return store

    def test_vector_store_add_and_search(self, populated_store):
        """add_documents + search returns relevant results."""
        results = populated_store.search("card skimming ATM", top_k=2)
        assert len(results) == 2
        # Each result is a (doc, score) tuple
        doc, score = results[0]
        assert "name" in doc
        assert isinstance(score, float)

    def test_vector_store_top_k(self, populated_store):
        """top_k parameter limits the number of returned results."""
        for k in (1, 2, 3):
            results = populated_store.search("fraud transaction", top_k=k)
            assert len(results) == k

    def test_vector_store_top_k_clamped_to_store_size(self, populated_store):
        """top_k larger than store size returns all stored docs."""
        results = populated_store.search("fraud", top_k=100)
        assert len(results) == len(populated_store)

    def test_vector_store_scores_in_range(self, populated_store):
        """Cosine similarity scores are in [-1, 1]."""
        results = populated_store.search("fraud detection", top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_vector_store_relevance_ordering(self, populated_store):
        """Results are sorted by descending similarity score."""
        results = populated_store.search("ATM card skimming", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_vector_store_add_documents_empty_raises(self, store):
        """add_documents raises ValueError for an empty list."""
        with pytest.raises(ValueError, match="non-empty"):
            store.add_documents([])

    def test_vector_store_add_documents_missing_field_raises(self, store):
        """add_documents raises ValueError when text_field is absent."""
        with pytest.raises(ValueError, match="missing required field"):
            store.add_documents([{"name": "no_description"}])

    def test_vector_store_empty_search_returns_empty(self, store):
        """Searching an empty store returns an empty list."""
        results = store.search("anything", top_k=5)
        assert results == []

    def test_vector_store_len(self, populated_store):
        """__len__ returns the number of indexed documents."""
        assert len(populated_store) == 5

    def test_vector_store_top_k_invalid_raises(self, populated_store):
        """top_k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            populated_store.search("fraud", top_k=0)


# ---------------------------------------------------------------------------
# FraudPatternRetriever tests
# ---------------------------------------------------------------------------


class TestFraudPatternRetriever:
    @pytest.fixture(scope="class")
    def retriever(self):
        """Single shared retriever instance (expensive to build)."""
        return FraudPatternRetriever()

    def test_retriever_initialization(self, retriever):
        """FraudPatternRetriever loads patterns at construction."""
        assert len(retriever._store) > 0

    def test_retriever_repr(self, retriever):
        """repr includes the pattern count."""
        r = repr(retriever)
        assert "FraudPatternRetriever" in r
        assert "n_patterns=" in r

    def test_retriever_retrieve(self, retriever):
        """retrieve() returns dicts; velocity query returns velocity-related pattern."""
        results = retriever.retrieve("velocity rapid micro transactions card testing", top_k=3)
        assert len(results) == 3
        for item in results:
            assert isinstance(item, dict)
            assert "name" in item
            assert "description" in item

    def test_retriever_retrieve_velocity_related(self, retriever):
        """A velocity-focused query surfaces a pattern with 'velocity' or 'testing' in name/desc."""
        results = retriever.retrieve("multiple rapid transactions velocity card testing", top_k=5)
        names_and_descs = " ".join(
            (r.get("name", "") + " " + r.get("description", "")).lower() for r in results
        )
        assert any(kw in names_and_descs for kw in ("velocity", "testing", "rapid", "micro"))

    def test_retriever_retrieve_for_transaction(self, retriever):
        """retrieve_for_transaction returns patterns for a high-risk transaction."""
        txn = make_transaction(
            amount=Decimal("5000.00"),
            channel=TransactionChannel.ONLINE,
            is_international=True,
        )
        results = retriever.retrieve_for_transaction(txn, top_k=3)
        assert len(results) == 3
        for item in results:
            assert "name" in item
            assert "risk_level" in item

    def test_retriever_retrieve_with_scores(self, retriever):
        """retrieve_with_scores returns (doc, float) tuples."""
        results = retriever.retrieve_with_scores("card fraud", top_k=2)
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(doc, dict)
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0
