"""Tests for RAG components: SimpleEmbedding, VectorStore, FraudPatternRetriever."""

from decimal import Decimal

import numpy as np
import pytest
from conftest import make_transaction

from fraud_agent.data.schemas import TransactionChannel
from fraud_agent.rag.embeddings import SimpleEmbedding
from fraud_agent.rag.retriever import FraudPatternRetriever
from fraud_agent.rag.store import VectorStore


class TestSimpleEmbedding:
    def test_creates_vector(self):
        emb = SimpleEmbedding(dimension=128)
        vec = emb.embed("card skimming ATM withdrawal suspicious")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (128,)

    def test_default_dimension(self):
        emb = SimpleEmbedding()
        assert emb.dimension == 128
        vec = emb.embed("hello world")
        assert vec.shape == (128,)

    def test_custom_dimension(self):
        emb = SimpleEmbedding(dimension=256)
        vec = emb.embed("test text")
        assert vec.shape == (256,)

    def test_unit_norm(self):
        emb = SimpleEmbedding(dimension=128)
        vec = emb.embed("fraud detection velocity")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_consistency(self):
        emb = SimpleEmbedding(dimension=128)
        text = "card skimming at ATM machine"
        vec1 = emb.embed(text)
        vec2 = emb.embed(text)
        np.testing.assert_array_equal(vec1, vec2)

    def test_different_texts(self):
        emb = SimpleEmbedding(dimension=128)
        vec1 = emb.embed("card skimming ATM fraud")
        vec2 = emb.embed("account takeover credential stuffing phishing")
        assert not np.array_equal(vec1, vec2)

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="dimension must be >= 1"):
            SimpleEmbedding(dimension=0)

    def test_embed_batch_shape(self):
        emb = SimpleEmbedding(dimension=64)
        texts = ["first text", "second text", "third text"]
        matrix = emb.embed_batch(texts)
        assert matrix.shape == (3, 64)

    def test_embed_batch_empty(self):
        emb = SimpleEmbedding(dimension=64)
        matrix = emb.embed_batch([])
        assert matrix.shape == (0, 64)


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

    def test_add_and_search(self, populated_store):
        results = populated_store.search("card skimming ATM", top_k=2)
        assert len(results) == 2
        doc, score = results[0]
        assert "name" in doc
        assert isinstance(score, float)

    def test_top_k(self, populated_store):
        for k in (1, 2, 3):
            results = populated_store.search("fraud transaction", top_k=k)
            assert len(results) == k

    def test_top_k_clamped_to_store_size(self, populated_store):
        results = populated_store.search("fraud", top_k=100)
        assert len(results) == len(populated_store)

    def test_scores_in_range(self, populated_store):
        results = populated_store.search("fraud detection", top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_relevance_ordering(self, populated_store):
        results = populated_store.search("ATM card skimming", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_documents_empty_raises(self, store):
        with pytest.raises(ValueError, match="non-empty"):
            store.add_documents([])

    def test_add_documents_missing_field_raises(self, store):
        with pytest.raises(ValueError, match="missing required field"):
            store.add_documents([{"name": "no_description"}])

    def test_empty_search_returns_empty(self, store):
        results = store.search("anything", top_k=5)
        assert results == []

    def test_len(self, populated_store):
        assert len(populated_store) == 5

    def test_top_k_invalid_raises(self, populated_store):
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            populated_store.search("fraud", top_k=0)


class TestFraudPatternRetriever:
    @pytest.fixture(scope="class")
    def retriever(self):
        return FraudPatternRetriever()

    def test_initialization(self, retriever):
        assert len(retriever._store) > 0

    def test_repr(self, retriever):
        r = repr(retriever)
        assert "FraudPatternRetriever" in r
        assert "n_patterns=" in r

    def test_retrieve(self, retriever):
        results = retriever.retrieve("velocity rapid micro transactions card testing", top_k=3)
        assert len(results) == 3
        for item in results:
            assert isinstance(item, dict)
            assert "name" in item
            assert "description" in item

    def test_retrieve_velocity_related(self, retriever):
        results = retriever.retrieve("multiple rapid transactions velocity card testing", top_k=5)
        names_and_descs = " ".join(
            (r.get("name", "") + " " + r.get("description", "")).lower() for r in results
        )
        assert any(kw in names_and_descs for kw in ("velocity", "testing", "rapid", "micro"))

    def test_retrieve_for_transaction(self, retriever):
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

    def test_retrieve_with_scores(self, retriever):
        results = retriever.retrieve_with_scores("card fraud", top_k=2)
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(doc, dict)
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0
