"""Scoring model abstractions for the fraud detection engine.

This module defines the :class:`ScoringModel` protocol and two concrete
implementations:

* :class:`RuleBasedModel` — a transparent, interpretable weighted-sum model
  that operates directly on extracted features.  Suitable for audit and
  explainability requirements common in financial services.

* :class:`EnsembleModel` — a meta-model that combines the outputs of multiple
  :class:`ScoringModel` instances via a weighted average, enabling progressive
  model stacking as richer models (e.g. gradient-boosted trees, neural nets)
  are added to the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ScoringModel(ABC):
    """Abstract base class for fraud scoring models.

    All models accept a pre-extracted feature dictionary (produced by
    :class:`~fraud_agent.scoring.features.FeatureExtractor`) and return a
    single continuous probability score.
    """

    @abstractmethod
    def score(self, features: dict[str, float]) -> float:
        """Compute a fraud probability from the supplied feature vector.

        Args:
            features: A mapping of feature name to normalised float value,
                as produced by :class:`~fraud_agent.scoring.features.FeatureExtractor`.

        Returns:
            A fraud probability in ``[0, 1]``.  Values closer to 1 indicate
            higher confidence in fraudulent activity.
        """
        ...


# ---------------------------------------------------------------------------
# Rule-based (weighted sum) model
# ---------------------------------------------------------------------------


class RuleBasedModel(ScoringModel):
    """Interpretable weighted-sum model operating on extracted features.

    The model computes a weighted linear combination of a curated feature
    subset and clamps the result to ``[0, 1]``.  Weights are calibrated to
    reflect the empirical predictive power of each feature based on industry
    research and internal benchmarks.

    Feature weights
    ---------------
    =====================  ======  ===========================================
    Feature                Weight  Rationale
    =====================  ======  ===========================================
    amount_ratio           0.25    Strongly predictive; over-sized transactions
    is_international       0.10    Cross-border adds meaningful risk
    is_nighttime           0.15    Off-hours correlation with automated fraud
    high_risk_mcc          0.15    Category-level risk signal
    velocity_10min         0.20    Rapid burst is a primary fraud indicator
    distance_from_typical  0.10    Geographic anomaly
    channel_risk           0.05    Card-not-present uplift
    =====================  ======  ===========================================

    The ``amount_ratio`` contribution is also normalised before weighting: a
    ratio of 1.0 (equal to average) contributes 0.0, while a ratio of 10.0
    (ten times average) contributes 1.0, capped at 1.0.
    """

    #: Default feature weights.  Keys must match the output of FeatureExtractor.
    DEFAULT_WEIGHTS: dict[str, float] = {
        "amount_ratio": 0.25,
        "is_international": 0.10,
        "is_nighttime": 0.15,
        "high_risk_mcc": 0.15,
        "velocity_10min": 0.20,
        "distance_from_typical": 0.10,
        "channel_risk": 0.05,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialise with optional custom weights.

        Args:
            weights: Override the default feature weights.  Must sum
                approximately to 1.0 for the output to remain well-calibrated,
                though this is not strictly enforced.
        """
        self._weights: dict[str, float] = weights or dict(self.DEFAULT_WEIGHTS)

    def score(self, features: dict[str, float]) -> float:
        """Compute the weighted-sum fraud probability.

        The ``amount_ratio`` feature is normalised to ``[0, 1]`` before
        weighting (capped at 10× account average = 1.0).  All other features
        are expected to already lie in ``[0, 1]``.

        Args:
            features: Feature dictionary from :class:`~fraud_agent.scoring.features.FeatureExtractor`.

        Returns:
            A fraud probability clamped to ``[0, 1]``.
        """
        raw_score = 0.0

        for feature_name, weight in self._weights.items():
            raw_value = features.get(feature_name, 0.0)

            # Normalise amount_ratio: ratio of 1 → 0, ratio of 10+ → 1.
            if feature_name == "amount_ratio":
                normalised = min(max(raw_value - 1.0, 0.0) / 9.0, 1.0)
            else:
                # All other features are already in [0, 1].
                normalised = min(max(raw_value, 0.0), 1.0)

            raw_score += weight * normalised

        return min(max(raw_score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------


class EnsembleModel(ScoringModel):
    """Meta-model that combines multiple :class:`ScoringModel` instances.

    Each member model is assigned a weight.  The ensemble score is the
    weighted average of member scores, normalised to ``[0, 1]``.

    This enables a progressive enrichment strategy: start with
    :class:`RuleBasedModel`, add a gradient-boosted tree once labelled data
    is available, and assign higher weight to the better-calibrated model
    without replacing the transparent baseline.

    Example::

        from fraud_agent.scoring.models import RuleBasedModel, EnsembleModel

        ensemble = EnsembleModel(
            models=[RuleBasedModel()],
            weights=[1.0],
        )
        fraud_prob = ensemble.score(features)
    """

    def __init__(
        self,
        models: list[ScoringModel],
        weights: list[float] | None = None,
    ) -> None:
        """Initialise the ensemble.

        Args:
            models: One or more :class:`ScoringModel` instances to combine.
                Must be non-empty.
            weights: Per-model weights.  If ``None``, all models are weighted
                equally.  Weights are automatically normalised to sum to 1.0.

        Raises:
            ValueError: If ``models`` is empty or ``weights`` length does not
                match ``models`` length.
        """
        if not models:
            raise ValueError("EnsembleModel requires at least one member model.")

        if weights is not None and len(weights) != len(models):
            raise ValueError(
                f"Length of weights ({len(weights)}) must match length of models ({len(models)})."
            )

        self._models = list(models)
        raw_weights = list(weights) if weights is not None else [1.0] * len(models)

        total = sum(raw_weights)
        if total <= 0.0:
            raise ValueError("Sum of ensemble weights must be positive.")

        self._weights: list[float] = [w / total for w in raw_weights]

    def score(self, features: dict[str, float]) -> float:
        """Compute the weighted-average fraud probability across all member models.

        Args:
            features: Feature dictionary from :class:`~fraud_agent.scoring.features.FeatureExtractor`.

        Returns:
            A fraud probability in ``[0, 1]``.
        """
        weighted_sum = 0.0
        for model, weight in zip(self._models, self._weights):
            member_score = model.score(features)
            weighted_sum += weight * min(max(member_score, 0.0), 1.0)

        return min(max(weighted_sum, 0.0), 1.0)
