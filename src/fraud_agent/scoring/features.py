"""Feature extraction for the fraud scoring engine.

This module converts raw :class:`~fraud_agent.data.schemas.Transaction` and
:class:`~fraud_agent.data.schemas.Account` objects into a flat dictionary of
normalised numerical features suitable for downstream rule evaluation and
machine-learning model inference.
"""

from __future__ import annotations

import math
from datetime import UTC, timedelta

from fraud_agent.data.schemas import Account, Transaction

# ---------------------------------------------------------------------------
# High-risk Merchant Category Codes
# ---------------------------------------------------------------------------

#: MCC codes that are statistically associated with elevated fraud rates.
HIGH_RISK_MCC: frozenset[str] = frozenset(
    [
        "7995",  # Gambling / betting
        "6051",  # Non-financial institutions (crypto / digital currency)
        "4829",  # Wire transfer / money orders
        "6012",  # Financial institutions / quasi-cash
        "5912",  # Drug stores / pharmacies (card-not-present fraud vector)
        "7273",  # Dating / escort services
        "5962",  # Direct marketing / travel
        "6211",  # Securities / commodity brokers
    ]
)

# ---------------------------------------------------------------------------
# Feature normalisation constants
# ---------------------------------------------------------------------------

#: Above this distance (km) the distance feature is clamped to 1.0.
_MAX_DISTANCE_KM: float = 20_000.0

#: Velocity counts above this value are clamped to 1.0.
_MAX_VELOCITY_10MIN: float = 20.0
_MAX_VELOCITY_1HR: float = 60.0

#: Z-score magnitude above which the feature is clamped to 1.0.
_MAX_ZSCORE: float = 10.0


class FeatureExtractor:
    """Extracts a flat dictionary of normalised numerical features.

    All returned feature values are in the range ``[0.0, 1.0]`` unless
    otherwise noted (e.g. ``amount_zscore`` may be negative for below-average
    amounts before clamping).

    Example::

        extractor = FeatureExtractor()
        features = extractor.extract(transaction, account, recent_transactions)
        fraud_prob = model.score(features)
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> dict[str, float]:
        """Extract numerical features from a transaction / account pair.

        Args:
            transaction: The transaction being evaluated.
            account: The account profile that provides behavioural baselines.
            recent_transactions: Ordered list of recent transactions for the
                same account, used to compute velocity features.  May be
                ``None`` or empty.

        Returns:
            A dictionary mapping feature name to normalised float value.
        """
        recent: list[Transaction] = recent_transactions or []

        # ----------------------------------------------------------------
        # Monetary features
        # ----------------------------------------------------------------
        avg_amount = float(account.average_transaction_amount)
        txn_amount = float(transaction.amount)

        amount_ratio = txn_amount / avg_amount if avg_amount > 0 else 1.0

        # Z-score: (x - mu) / sigma  — we approximate sigma as 0.5 * mu
        # (a common heuristic when only the mean is available).
        sigma_est = 0.5 * avg_amount if avg_amount > 0 else 1.0
        amount_zscore = (txn_amount - avg_amount) / sigma_est

        # ----------------------------------------------------------------
        # Categorical / binary features
        # ----------------------------------------------------------------
        is_international: float = 1.0 if transaction.is_international else 0.0
        high_risk_mcc: float = 1.0 if transaction.merchant_category_code in HIGH_RISK_MCC else 0.0

        # ----------------------------------------------------------------
        # Temporal features
        # ----------------------------------------------------------------
        ts = transaction.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        hour_of_day: float = ts.hour / 23.0  # normalised to [0, 1]
        is_nighttime: float = 1.0 if 2 <= ts.hour <= 5 else 0.0

        # ----------------------------------------------------------------
        # Channel risk
        # ----------------------------------------------------------------
        channel_risk: float = self._channel_risk(transaction)

        # ----------------------------------------------------------------
        # Velocity features
        # ----------------------------------------------------------------
        velocity_10min, velocity_1hr = self._compute_velocity(transaction, recent)

        # ----------------------------------------------------------------
        # Geographic features
        # ----------------------------------------------------------------
        distance_from_typical: float = self._compute_distance_feature(transaction, account)

        return {
            "amount_ratio": float(amount_ratio),
            "is_international": is_international,
            "hour_of_day": hour_of_day,
            "is_nighttime": is_nighttime,
            "high_risk_mcc": high_risk_mcc,
            "velocity_10min": velocity_10min,
            "velocity_1hr": velocity_1hr,
            "distance_from_typical": distance_from_typical,
            "amount_zscore": float(max(-_MAX_ZSCORE, min(_MAX_ZSCORE, amount_zscore))),
            "channel_risk": channel_risk,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _channel_risk(transaction: Transaction) -> float:
        """Return a risk scalar for the transaction channel.

        Channel risk reflects the relative ease of impersonation/interception
        for each payment channel type.

        Returns:
            0.0  for IN_STORE (card-present, lowest risk)
            0.3  for ATM
            0.5  for MOBILE
            0.7  for ONLINE (card-not-present, highest risk)
        """
        from fraud_agent.data.schemas import TransactionChannel

        mapping: dict[TransactionChannel, float] = {
            TransactionChannel.IN_STORE: 0.0,
            TransactionChannel.ATM: 0.3,
            TransactionChannel.MOBILE: 0.5,
            TransactionChannel.ONLINE: 0.7,
        }
        return mapping.get(transaction.channel, 0.5)

    @staticmethod
    def _compute_velocity(
        transaction: Transaction,
        recent: list[Transaction],
    ) -> tuple[float, float]:
        """Compute normalised transaction velocity features.

        Args:
            transaction: The current transaction (defines the reference time).
            recent: List of prior transactions for the same account.

        Returns:
            A ``(velocity_10min, velocity_1hr)`` tuple, each normalised to
            ``[0, 1]`` by clamping against ``_MAX_VELOCITY_10MIN`` and
            ``_MAX_VELOCITY_1HR`` respectively.
        """
        if not recent:
            return 0.0, 0.0

        ref_ts = transaction.timestamp
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.replace(tzinfo=UTC)

        cutoff_10min = ref_ts - timedelta(minutes=10)
        cutoff_1hr = ref_ts - timedelta(hours=1)

        count_10min = 0
        count_1hr = 0

        for t in recent:
            t_ts = t.timestamp
            if t_ts.tzinfo is None:
                t_ts = t_ts.replace(tzinfo=UTC)

            # Exclude the current transaction itself (same id)
            if t.id == transaction.id:
                continue

            if t_ts >= cutoff_10min:
                count_10min += 1
            if t_ts >= cutoff_1hr:
                count_1hr += 1

        v10 = min(float(count_10min) / _MAX_VELOCITY_10MIN, 1.0)
        v1h = min(float(count_1hr) / _MAX_VELOCITY_1HR, 1.0)
        return v10, v1h

    @staticmethod
    def _compute_distance_feature(
        transaction: Transaction,
        account: Account,
    ) -> float:
        """Compute normalised distance from the account's typical location.

        Returns:
            A value in ``[0, 1]``, where 1.0 represents ``_MAX_DISTANCE_KM``
            or more.  Returns 0.0 when location data is unavailable.
        """
        typ_loc = account.typical_location
        txn_loc = transaction.location

        if typ_loc is None or txn_loc is None:
            return 0.0

        distance_km = FeatureExtractor._haversine(
            typ_loc.latitude,
            typ_loc.longitude,
            txn_loc.latitude,
            txn_loc.longitude,
        )
        return min(distance_km / _MAX_DISTANCE_KM, 1.0)

    @staticmethod
    def _haversine(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Compute the great-circle distance between two points on Earth.

        Uses the Haversine formula which is numerically stable for small
        distances and accurate to within ~0.3% for most terrestrial pairs.

        Args:
            lat1: Latitude of point 1 in decimal degrees.
            lon1: Longitude of point 1 in decimal degrees.
            lat2: Latitude of point 2 in decimal degrees.
            lon2: Longitude of point 2 in decimal degrees.

        Returns:
            Distance in kilometres.
        """
        _R = 6_371.0  # Earth's mean radius in km

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return _R * c
