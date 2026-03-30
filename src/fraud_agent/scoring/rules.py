"""Deterministic fraud detection rules for the scoring engine.

Each rule is a self-contained, auditable unit that inspects a transaction and
its surrounding context and returns a structured verdict.  Rules are designed
to be cheap (no I/O, pure computation) and composable — the
:class:`~fraud_agent.scoring.engine.ScoringEngine` runs them in parallel and
aggregates their outputs.

All rule return types are ``tuple[bool, float, str]``:

* ``triggered`` — whether the rule condition was met.
* ``risk_contribution`` — a value in ``[0, 1]`` indicating how strongly this
  rule believes fraud is occurring.  0.0 means no contribution; 1.0 means
  near-certain fraud according to this rule.
* ``explanation`` — a human-readable string describing the finding.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import UTC, timedelta

from fraud_agent.data.schemas import Account, Transaction

#: Mapping of high-risk Merchant Category Codes to human-readable names.
HIGH_RISK_MCC: dict[str, str] = {
    "7995": "Gambling / Betting",
    "6051": "Cryptocurrency / Digital Currency",
    "4829": "Wire Transfer / Money Orders",
    "6012": "Financial Institutions / Quasi-Cash",
    "5912": "Drug Stores / Pharmacies",
    "7273": "Dating / Escort Services",
    "5962": "Direct Marketing – Travel",
    "6211": "Securities / Commodity Brokers",
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in **kilometres** between two WGS-84 points.

    Args:
        lat1: Latitude of point 1 (decimal degrees).
        lon1: Longitude of point 1 (decimal degrees).
        lat2: Latitude of point 2 (decimal degrees).
        lon2: Longitude of point 2 (decimal degrees).

    Returns:
        Distance in kilometres.
    """
    _R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return _R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


_KM_TO_MILES: float = 0.621371


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FraudRule(ABC):
    """Abstract base class for all deterministic fraud rules.

    Subclasses must implement :meth:`name` and :meth:`evaluate`.

    Design contract:

    * Rules are **stateless** — all context is passed via :meth:`evaluate`.
    * Rules must **never raise** — return ``(False, 0.0, "")`` on error.
    * ``risk_contribution`` must lie in ``[0, 1]``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable machine-readable identifier for this rule (e.g. ``"velocity_rule"``)."""
        ...

    @abstractmethod
    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Evaluate the rule against the supplied transaction context.

        Args:
            transaction: The transaction being assessed.
            account: The account profile providing behavioural baselines.
            recent_transactions: Recent transactions for velocity / pattern
                analysis.  May be ``None`` or empty.

        Returns:
            A three-tuple ``(triggered, risk_contribution, explanation)``.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------


class VelocityRule(FraudRule):
    """Detects abnormally high transaction velocity within a short window.

    Triggers when more than ``threshold`` distinct transactions are observed
    within the preceding ``window_minutes`` minutes.  This pattern is a strong
    indicator of card-present cloning or automated fraud attacks.
    """

    def __init__(
        self,
        threshold: int = 5,
        window_minutes: int = 10,
    ) -> None:
        """Initialise the velocity rule.

        Args:
            threshold: Maximum number of transactions allowed in the window
                before the rule triggers.
            window_minutes: Size of the lookback window in minutes.
        """
        self._threshold = threshold
        self._window_minutes = window_minutes

    @property
    def name(self) -> str:
        return "velocity_rule"

    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Trigger if more than ``threshold`` transactions fall within ``window_minutes``."""
        if not recent_transactions:
            return False, 0.0, ""

        ref_ts = transaction.timestamp
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.replace(tzinfo=UTC)

        cutoff = ref_ts - timedelta(minutes=self._window_minutes)
        count = 0
        for t in recent_transactions:
            if t.id == transaction.id:
                continue
            t_ts = t.timestamp
            if t_ts.tzinfo is None:
                t_ts = t_ts.replace(tzinfo=UTC)
            if t_ts >= cutoff:
                count += 1

        if count > self._threshold:
            explanation = f"{count} transactions in {self._window_minutes} minutes"
            return True, 0.9, explanation

        return False, 0.0, ""


class AmountRule(FraudRule):
    """Flags transactions that are significantly above the account average.

    The risk contribution scales with the ratio of transaction amount to
    account average, clamped to ``[0, 1]``.
    """

    def __init__(self, multiplier_threshold: float = 3.0) -> None:
        """Initialise the amount rule.

        Args:
            multiplier_threshold: The ratio above which the rule triggers
                (default 3.0× account average).
        """
        self._threshold = multiplier_threshold

    @property
    def name(self) -> str:
        return "amount_rule"

    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Trigger when transaction amount exceeds ``threshold`` × account average."""
        avg = float(account.average_transaction_amount)
        if avg <= 0:
            return False, 0.0, ""

        amount = float(transaction.amount)
        ratio = amount / avg

        if ratio > self._threshold:
            # Risk scales from threshold (→ 0.5) up to 10× (→ 1.0), clamped.
            risk = min(0.5 + (ratio - self._threshold) / (10.0 - self._threshold) * 0.5, 1.0)
            explanation = f"Amount is {ratio:.1f}x account average"
            return True, risk, explanation

        return False, 0.0, ""


class GeographicRule(FraudRule):
    """Detects physically-impossible or suspicious geographic velocity.

    Triggers when the transaction location is more than ``distance_threshold``
    miles from the account's typical location **and** the time since the last
    transaction is less than ``time_threshold_hours`` hours.  This catches
    "impossible travel" scenarios indicative of card duplication or
    account takeover.
    """

    def __init__(
        self,
        distance_threshold_miles: float = 500.0,
        time_threshold_hours: float = 2.0,
    ) -> None:
        """Initialise the geographic rule.

        Args:
            distance_threshold_miles: Minimum distance (miles) from typical
                location required to trigger.
            time_threshold_hours: Maximum elapsed time (hours) since the last
                transaction for the rule to apply.
        """
        self._distance_miles = distance_threshold_miles
        self._time_hours = time_threshold_hours

    @property
    def name(self) -> str:
        return "geographic_rule"

    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Trigger on impossible-travel or far-from-home transactions."""
        typ_loc = account.typical_location
        txn_loc = transaction.location

        if typ_loc is None or txn_loc is None:
            return False, 0.0, ""

        distance_km = _haversine(
            typ_loc.latitude,
            typ_loc.longitude,
            txn_loc.latitude,
            txn_loc.longitude,
        )
        distance_miles = distance_km * _KM_TO_MILES

        if distance_miles <= self._distance_miles:
            return False, 0.0, ""

        # Check time elapsed since most recent prior transaction
        if recent_transactions:
            ref_ts = transaction.timestamp
            if ref_ts.tzinfo is None:
                ref_ts = ref_ts.replace(tzinfo=UTC)

            prior_timestamps = []
            for t in recent_transactions:
                if t.id == transaction.id:
                    continue
                t_ts = t.timestamp
                if t_ts.tzinfo is None:
                    t_ts = t_ts.replace(tzinfo=UTC)
                prior_timestamps.append(t_ts)

            if prior_timestamps:
                most_recent = max(prior_timestamps)
                elapsed_hours = (ref_ts - most_recent).total_seconds() / 3600.0
                if elapsed_hours >= self._time_hours:
                    # Enough time has elapsed — not necessarily impossible travel
                    return False, 0.0, ""

        explanation = f"Transaction {distance_miles:.0f}mi from typical location"
        return True, 0.85, explanation


class TimeRule(FraudRule):
    """Flags transactions that occur during atypical overnight hours.

    Transactions between 02:00 and 05:00 (local time, approximated as UTC)
    carry elevated risk because most legitimate cardholders are not active at
    those hours, and automated fraud scripts often operate continuously.
    """

    def __init__(self, start_hour: int = 2, end_hour: int = 5) -> None:
        """Initialise the time rule.

        Args:
            start_hour: Hour (0–23, inclusive) at which the risky window starts.
            end_hour: Hour (0–23, inclusive) at which the risky window ends.
        """
        self._start = start_hour
        self._end = end_hour

    @property
    def name(self) -> str:
        return "time_rule"

    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Trigger for transactions occurring between ``start_hour`` and ``end_hour``."""
        ts = transaction.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        if self._start <= ts.hour <= self._end:
            explanation = f"Transaction at unusual hour {ts.strftime('%H:%M')}"
            return True, 0.3, explanation

        return False, 0.0, ""


class MerchantRule(FraudRule):
    """Flags transactions at merchants in high-risk categories.

    Certain Merchant Category Codes are disproportionately associated with
    money laundering, gambling addiction exploitation, crypto cash-outs, and
    wire-fraud schemes.
    """

    @property
    def name(self) -> str:
        return "merchant_rule"

    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Trigger when the MCC is in the high-risk catalogue."""
        mcc = transaction.merchant_category_code
        if mcc in HIGH_RISK_MCC:
            category_name = HIGH_RISK_MCC[mcc]
            explanation = f"High-risk merchant category: {category_name}"
            return True, 0.5, explanation

        return False, 0.0, ""


class TestingRule(FraudRule):
    """Detects card-testing fraud patterns.

    Card testing is a technique where fraudsters make a tiny charge (typically
    sub-$2) to verify that a stolen card is active before executing a large
    fraudulent transaction.  This rule looks for the combination of:

    1. A sub-threshold micro-charge (< ``small_amount_threshold``), AND
    2. A recent large transaction (> ``large_amount_threshold``) within the
       preceding ``lookback_hours`` hours.
    """

    def __init__(
        self,
        small_amount_threshold: float = 2.0,
        large_amount_threshold: float = 500.0,
        lookback_hours: float = 1.0,
    ) -> None:
        """Initialise the testing rule.

        Args:
            small_amount_threshold: Amount below which a charge is considered
                a "micro-charge" test (default $2.00).
            large_amount_threshold: Amount above which a prior charge is
                considered a "large" transaction (default $500.00).
            lookback_hours: Window (hours) within which to look for a prior
                large transaction (default 1 hour).
        """
        self._small = small_amount_threshold
        self._large = large_amount_threshold
        self._lookback = lookback_hours

    @property
    def name(self) -> str:
        return "testing_rule"

    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Trigger on micro-charge + recent large-charge pattern."""
        if float(transaction.amount) >= self._small:
            return False, 0.0, ""

        if not recent_transactions:
            return False, 0.0, ""

        ref_ts = transaction.timestamp
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.replace(tzinfo=UTC)

        cutoff = ref_ts - timedelta(hours=self._lookback)

        for t in recent_transactions:
            if t.id == transaction.id:
                continue
            t_ts = t.timestamp
            if t_ts.tzinfo is None:
                t_ts = t_ts.replace(tzinfo=UTC)
            if t_ts >= cutoff and float(t.amount) > self._large:
                return True, 0.95, "Card testing pattern detected"

        return False, 0.0, ""


class NewMerchantRule(FraudRule):
    """Flags first-time merchants when the transaction amount is elevated.

    Fraudsters frequently exploit newly obtained card credentials at merchants
    the cardholder has never visited before.  A first-time merchant combined
    with an above-normal transaction amount is a meaningful signal.
    """

    def __init__(self, amount_threshold: float = 200.0) -> None:
        """Initialise the new merchant rule.

        Args:
            amount_threshold: Minimum amount (in account currency) for a
                first-time merchant transaction to trigger the rule.
        """
        self._threshold = amount_threshold

    @property
    def name(self) -> str:
        return "new_merchant_rule"

    def evaluate(
        self,
        transaction: Transaction,
        account: Account,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[bool, float, str]:
        """Trigger on first-time merchant with amount above threshold."""
        amount = float(transaction.amount)
        if amount <= self._threshold:
            return False, 0.0, ""

        # Derive known merchants from account attribute (if available) or
        # from recent transaction history as a fallback.
        known_merchant_ids: set[str] = set()

        # Some Account instances may carry an explicit known_merchant_ids set.
        if hasattr(account, "known_merchant_ids") and account.known_merchant_ids:
            known_merchant_ids = set(account.known_merchant_ids)
        elif recent_transactions:
            # Build from history, excluding the current transaction.
            for t in recent_transactions:
                if t.id != transaction.id:
                    known_merchant_ids.add(t.merchant_name)

        is_new_merchant = transaction.merchant_name not in known_merchant_ids

        if is_new_merchant:
            explanation = f"First-time merchant with elevated amount ${amount:.2f}"
            return True, 0.4, explanation

        return False, 0.0, ""
