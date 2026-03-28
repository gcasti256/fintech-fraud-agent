"""Tests for scoring models."""

from __future__ import annotations

import pytest

from fraud_agent.scoring.models import EnsembleModel, RuleBasedModel, ScoringModel


class TestRuleBasedModel:
    def test_score_zero_features(self):
        model = RuleBasedModel()
        features = {k: 0.0 for k in RuleBasedModel.DEFAULT_WEIGHTS}
        score = model.score(features)
        assert score == pytest.approx(0.0)

    def test_score_all_high_features(self):
        model = RuleBasedModel()
        features = {
            "amount_ratio": 10.0,  # normalized to 1.0
            "is_international": 1.0,
            "is_nighttime": 1.0,
            "high_risk_mcc": 1.0,
            "velocity_10min": 1.0,
            "distance_from_typical": 1.0,
            "channel_risk": 1.0,
        }
        score = model.score(features)
        assert score == pytest.approx(1.0)

    def test_score_in_range(self):
        model = RuleBasedModel()
        features = {
            "amount_ratio": 2.0,
            "is_international": 0.0,
            "is_nighttime": 0.0,
            "high_risk_mcc": 1.0,
            "velocity_10min": 0.3,
            "distance_from_typical": 0.0,
            "channel_risk": 0.7,
        }
        score = model.score(features)
        assert 0.0 <= score <= 1.0

    def test_amount_ratio_normalization(self):
        model = RuleBasedModel()
        base = {k: 0.0 for k in RuleBasedModel.DEFAULT_WEIGHTS}

        base["amount_ratio"] = 1.0
        score_1x = model.score(base)
        assert score_1x == pytest.approx(0.0)

        base["amount_ratio"] = 5.5
        score_5x = model.score(base)
        assert score_5x > 0.0

    def test_custom_weights(self):
        custom = {"velocity_10min": 1.0}
        model = RuleBasedModel(weights=custom)
        features = {"velocity_10min": 0.5}
        score = model.score(features)
        assert score == pytest.approx(0.5)

    def test_missing_feature_defaults_zero(self):
        model = RuleBasedModel()
        score = model.score({})
        assert score == 0.0


class TestEnsembleModel:
    def test_single_model(self):
        base = RuleBasedModel()
        ensemble = EnsembleModel(models=[base])
        features = {k: 0.0 for k in RuleBasedModel.DEFAULT_WEIGHTS}
        score = ensemble.score(features)
        assert score == pytest.approx(base.score(features))

    def test_weighted_average(self):
        class FixedScore(ScoringModel):
            def __init__(self, val: float):
                self._val = val

            def score(self, features):
                return self._val

        m1 = FixedScore(0.2)
        m2 = FixedScore(0.8)
        ensemble = EnsembleModel(models=[m1, m2], weights=[1.0, 1.0])
        assert ensemble.score({}) == pytest.approx(0.5)

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            EnsembleModel(models=[])

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError, match="Length of weights"):
            EnsembleModel(models=[RuleBasedModel()], weights=[1.0, 2.0])

    def test_weight_normalization(self):
        class FixedScore(ScoringModel):
            def __init__(self, val: float):
                self._val = val

            def score(self, features):
                return self._val

        m1 = FixedScore(1.0)
        m2 = FixedScore(0.0)
        ensemble = EnsembleModel(models=[m1, m2], weights=[3.0, 1.0])
        assert ensemble.score({}) == pytest.approx(0.75)
