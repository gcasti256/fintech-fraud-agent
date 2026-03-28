"""Tests for RAG components — embeddings, vector store, retriever."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from fraud_agent.data.schemas import Location, Transaction, TransactionChannel
from fraud_agent.rag.embeddings import SimpleEmbedding
from fraud_agent.rag.retriever import FraudPatternRetriever
from fraud_agent.rag.store import VectorStore


class TestSimpleEmbedding:
    def test_embed_shape(self):
        emb = SimpleEmbedding(dimension=128)
        vec = emb.embed("test text")
        assert vec.shape == (128,)

    def test_embed_normalized(self):
        emb = SimpleEmbedding(dimension=128)
        vec = emb.embed("card skimming ATM withdrawal suspicious")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_empty_text_zero_vector(self):
        emb = SimpleEmbedding(dimension=64)
        vec = emb.embed("")
        assert np.all(vec == 0.0)

    def test_deterministic(self):
        emb = SimpleEmbedding(dimension=128)
        v1 = emb.embed("fraud detection")
        v2 = emb.embed("fraud detection")
        np.testing.assert_array_equal(v1, v2)

    def test_embed_batch(self):
        emb = SimpleEmbedding(dimension=64)
        result = emb.embed_batch(["text one", "text two", "text three"])
        assert result.shape == (3, 64)

    def test_embed_batch_empty(self):
        emb = SimpleEmbedding(dimension=64)
        result = emb.embed_batch([])
        assert result.shape == (0, 64)

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            SimpleEmbedding(dimension=0)

    def test_build_vocab(self):
        emb = SimpleEmbedding(dimension=128)
        emb._build_vocab(["hello world", "world test"])
        assert "hello" in emb._vocab
        assert "world" in emb._vocab


class TestVectorStore:
    def test_add_and_search(self):
        emb = SimpleEmbedding(dimension=128)
        store = VectorStore(embedding=emb)
        docs = [
            {"description": "card skimming at ATM", "name": "skimming"},
            {"description": "online account takeover", "name": "ato"},
            {"description": "velocity abuse rapid transactions", "name": "velocity"},
        ]
        store.add_documents(docs)
        results = store.search("ATM card skimming withdrawal", top_k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert doc["name"] == "skimming"
        assert score > 0

    def test_empty_store_search(self):
        emb = SimpleEmbedding(dimension=64)
        store = VectorStore(embedding=emb)
        results = store.search("anything", top_k=3)
        assert results == []

    def test_top_k_limit(self):
        emb = SimpleEmbedding(dimension=64)
        store = VectorStore(embedding=emb)
        docs = [{"description": f"document {i}"} for i in range(10)]
        store.add_documents(docs)
        results = store.search("document", top_k=3)
        assert len(results) == 3

    def test_invalid_top_k(self):
        emb = SimpleEmbedding(dimension=64)
        store = VectorStore(embedding=emb)
        with pytest.raises(ValueError, match="top_k"):
            store.search("q", top_k=0)

    def test_empty_documents_raises(self):
        emb = SimpleEmbedding(dimension=64)
        store = VectorStore(embedding=emb)
        with pytest.raises(ValueError, match="non-empty"):
            store.add_documents([])

    def test_len(self):
        emb = SimpleEmbedding(dimension=64)
        store = VectorStore(embedding=emb)
        assert len(store) == 0
        store.add_documents([{"description": "test"}])
        assert len(store) == 1


class TestFraudPatternRetriever:
    def test_retrieve(self):
        retriever = FraudPatternRetriever()
        results = retriever.retrieve("card testing micro charge", top_k=3)
        assert len(results) == 3
        assert all("id" in r for r in results)

    def test_retrieve_with_scores(self):
        retriever = FraudPatternRetriever()
        results = retriever.retrieve_with_scores("ATM skimming", top_k=2)
        assert len(results) == 2
        doc, score = results[0]
        assert "name" in doc
        assert isinstance(score, float)

    def test_retrieve_for_transaction(self):
        retriever = FraudPatternRetriever()
        txn = Transaction(
            id="txn-rag-001",
            timestamp=datetime(2024, 6, 15, 3, 0, 0, tzinfo=UTC),
            amount=Decimal("5000.00"),
            currency="USD",
            merchant_name="CryptoSwap",
            merchant_category_code="6051",
            card_last_four="1234",
            account_id="ACC-001",
            location=Location(city="NYC", country="US", latitude=40.7, longitude=-74.0),
            channel=TransactionChannel.ONLINE,
            is_international=False,
        )
        results = retriever.retrieve_for_transaction(txn, top_k=3)
        assert len(results) == 3

    def test_repr(self):
        retriever = FraudPatternRetriever()
        assert "FraudPatternRetriever" in repr(retriever)
