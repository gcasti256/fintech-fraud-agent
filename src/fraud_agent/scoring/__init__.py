"""Fraud scoring engine public API.

This package exposes the complete scoring pipeline:

* :class:`ScoringEngine` — the main entry point; wires features + rules + model.
* :class:`FeatureExtractor` — converts raw domain objects to a feature vector.
* Rule classes — deterministic fraud signal detectors.
* Model classes — statistical / ML fraud probability estimators.

Typical usage::

    from fraud_agent.scoring import ScoringEngine

    engine = ScoringEngine()
    decision = engine.score_transaction(transaction, account, recent_transactions)
"""

from .engine import ScoringEngine
from .features import FeatureExtractor
from .models import EnsembleModel, RuleBasedModel, ScoringModel
from .rules import (
    AmountRule,
    FraudRule,
    GeographicRule,
    MerchantRule,
    NewMerchantRule,
    TestingRule,
    TimeRule,
    VelocityRule,
)

__all__ = [
    # Engine
    "ScoringEngine",
    # Feature extraction
    "FeatureExtractor",
    # Models
    "ScoringModel",
    "RuleBasedModel",
    "EnsembleModel",
    # Rules
    "FraudRule",
    "VelocityRule",
    "AmountRule",
    "GeographicRule",
    "TimeRule",
    "MerchantRule",
    "TestingRule",
    "NewMerchantRule",
]
