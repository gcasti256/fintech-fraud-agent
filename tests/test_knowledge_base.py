"""Tests for the fraud knowledge base."""

from __future__ import annotations

from fraud_agent.data.knowledge_base import FraudKnowledgeBase


class TestFraudKnowledgeBase:
    def test_get_patterns(self):
        kb = FraudKnowledgeBase()
        patterns = kb.get_patterns()
        assert len(patterns) == 16

    def test_get_pattern_by_id(self):
        kb = FraudKnowledgeBase()
        pattern = kb.get_pattern_by_id("FP-001")
        assert pattern is not None
        assert pattern["name"] == "Card Testing"

    def test_get_pattern_not_found(self):
        kb = FraudKnowledgeBase()
        assert kb.get_pattern_by_id("FP-999") is None

    def test_search_patterns(self):
        kb = FraudKnowledgeBase()
        results = kb.search_patterns("card testing")
        assert len(results) >= 1
        assert any(p["id"] == "FP-001" for p in results)

    def test_search_empty_query(self):
        kb = FraudKnowledgeBase()
        results = kb.search_patterns("")
        assert len(results) == 16

    def test_get_patterns_by_category(self):
        kb = FraudKnowledgeBase()
        results = kb.get_patterns_by_category("card_testing")
        assert len(results) >= 1

    def test_get_patterns_by_risk_level(self):
        kb = FraudKnowledgeBase()
        critical = kb.get_patterns_by_risk_level("CRITICAL")
        assert len(critical) >= 1
        assert all(p["risk_level"] == "CRITICAL" for p in critical)

    def test_get_patterns_by_mcc(self):
        kb = FraudKnowledgeBase()
        results = kb.get_patterns_by_mcc("6051")
        assert len(results) >= 1

    def test_custom_patterns(self):
        custom = [{"id": "CUSTOM-001", "name": "Test", "description": "test pattern"}]
        kb = FraudKnowledgeBase(patterns=custom)
        assert len(kb.get_patterns()) == 1

    def test_returns_copy(self):
        kb = FraudKnowledgeBase()
        p1 = kb.get_patterns()
        p2 = kb.get_patterns()
        assert p1 is not p2
