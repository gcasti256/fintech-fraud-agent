"""Tests for FeatureExtractor."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from fraud_agent.data.schemas import Account, Location, Transaction, TransactionChannel
from fraud_agent.scoring.features import FeatureExtractor
from tests.conftest import make_recent_transactions

_NY = Location(city="New York", country="US", latitude=40.7128, longitude=-74.006)
_LA = Location(city="Los Angeles", country="US", latitude=34.0522, longitude=-118.2437)


def _make_transaction(**kw) -> Transaction:
    base = dict(
        id="feat-txn-001",
        timestamp=datetime(2024, 6, 15, 14, 30, tzinfo=UTC),
        amount=Decimal("75.00"),
        currency="USD",
        merchant_name="Test Store",
        merchant_category_code="5411",
        card_last_four="1234",
        account_id="ACC-0001",
        location=_NY,
        channel=TransactionChannel.IN_STORE,
        is_international=False,
    )
    base.update(kw)
    return Transaction(**base)


def _make_account(**kw) -> Account:
    base = dict(
        id="ACC-0001",
        holder_name="Test User",
        average_transaction_amount=Decimal("75.00"),
        typical_location=_NY,
        account_open_date=date(2020, 1, 1),
        transaction_history_count=100,
    )
    base.update(kw)
    return Account(**base)


EXPECTED_KEYS = {
    "amount_ratio",
    "is_international",
    "hour_of_day",
    "is_nighttime",
    "high_risk_mcc",
    "velocity_10min",
    "velocity_1hr",
    "distance_from_typical",
    "amount_zscore",
    "channel_risk",
}


class TestFeatureExtractor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # Key presence
    # ------------------------------------------------------------------

    def test_extract_basic_features(self):
        """extract() returns a dict with exactly the expected feature keys."""
        txn = _make_transaction()
        account = _make_account()
        features = self.extractor.extract(txn, account)
        assert set(features.keys()) == EXPECTED_KEYS

    def test_extract_returns_dict(self):
        """Return type is dict[str, float]."""
        features = self.extractor.extract(_make_transaction(), _make_account())
        assert isinstance(features, dict)
        for key, val in features.items():
            assert isinstance(key, str)
            assert isinstance(val, float), f"Feature '{key}' is not a float: {val}"

    def test_extract_with_conftest_fixtures(self, normal_transaction, default_account):
        """Works correctly with conftest fixtures."""
        features = self.extractor.extract(normal_transaction, default_account)
        assert set(features.keys()) == EXPECTED_KEYS

    # ------------------------------------------------------------------
    # Amount ratio
    # ------------------------------------------------------------------

    def test_amount_ratio(self):
        """amount_ratio equals txn.amount / account.average_transaction_amount."""
        txn = _make_transaction(amount=Decimal("150.00"))
        account = _make_account(average_transaction_amount=Decimal("75.00"))
        features = self.extractor.extract(txn, account)
        expected = 150.0 / 75.0
        assert features["amount_ratio"] == pytest.approx(expected, rel=1e-4)

    def test_amount_ratio_equal_to_average(self):
        """amount_ratio of 1.0 when amount equals average."""
        txn = _make_transaction(amount=Decimal("75.00"))
        account = _make_account(average_transaction_amount=Decimal("75.00"))
        features = self.extractor.extract(txn, account)
        assert features["amount_ratio"] == pytest.approx(1.0, rel=1e-4)

    def test_amount_ratio_conftest(self, normal_transaction, default_account):
        """Correct ratio for conftest fixtures (42.50 / 75.00 ≈ 0.567)."""
        features = self.extractor.extract(normal_transaction, default_account)
        expected = float(normal_transaction.amount) / float(
            default_account.average_transaction_amount
        )
        assert features["amount_ratio"] == pytest.approx(expected, rel=1e-4)

    # ------------------------------------------------------------------
    # Nighttime detection
    # ------------------------------------------------------------------

    def test_nighttime_detection(self):
        """Transaction at 3 AM UTC is flagged as nighttime."""
        txn = _make_transaction(timestamp=datetime(2024, 6, 15, 3, 0, tzinfo=UTC))
        features = self.extractor.extract(txn, _make_account())
        assert features["is_nighttime"] == 1.0

    def test_daytime_detection(self):
        """Transaction at 2 PM UTC is not flagged as nighttime."""
        txn = _make_transaction(timestamp=datetime(2024, 6, 15, 14, 0, tzinfo=UTC))
        features = self.extractor.extract(txn, _make_account())
        assert features["is_nighttime"] == 0.0

    def test_nighttime_conftest_suspicious(self, suspicious_transaction, default_account):
        """conftest suspicious_transaction (3:00 AM) is nighttime."""
        features = self.extractor.extract(suspicious_transaction, default_account)
        assert features["is_nighttime"] == 1.0

    def test_daytime_conftest_normal(self, normal_transaction, default_account):
        """conftest normal_transaction (14:30) is not nighttime."""
        features = self.extractor.extract(normal_transaction, default_account)
        assert features["is_nighttime"] == 0.0

    def test_hour_of_day_normalised(self):
        """hour_of_day is normalised to [0, 1] (hour / 23)."""
        txn = _make_transaction(timestamp=datetime(2024, 6, 15, 23, 0, tzinfo=UTC))
        features = self.extractor.extract(txn, _make_account())
        assert features["hour_of_day"] == pytest.approx(1.0, rel=1e-4)

    # ------------------------------------------------------------------
    # MCC risk
    # ------------------------------------------------------------------

    def test_high_risk_mcc(self):
        """Gambling MCC 7995 sets high_risk_mcc = 1.0."""
        txn = _make_transaction(merchant_category_code="7995")
        features = self.extractor.extract(txn, _make_account())
        assert features["high_risk_mcc"] == 1.0

    def test_normal_mcc(self):
        """Grocery MCC 5411 sets high_risk_mcc = 0.0."""
        txn = _make_transaction(merchant_category_code="5411")
        features = self.extractor.extract(txn, _make_account())
        assert features["high_risk_mcc"] == 0.0

    def test_high_risk_mcc_conftest_suspicious(self, suspicious_transaction, default_account):
        """conftest suspicious_transaction (MCC 7995) has high_risk_mcc=1.0."""
        features = self.extractor.extract(suspicious_transaction, default_account)
        assert features["high_risk_mcc"] == 1.0

    def test_normal_mcc_conftest_normal(self, normal_transaction, default_account):
        """conftest normal_transaction (MCC 5411) has high_risk_mcc=0.0."""
        features = self.extractor.extract(normal_transaction, default_account)
        assert features["high_risk_mcc"] == 0.0

    # ------------------------------------------------------------------
    # Channel risk
    # ------------------------------------------------------------------

    def test_channel_risk_values(self):
        """ONLINE risk (0.7) > IN_STORE risk (0.0)."""
        online_txn = _make_transaction(channel=TransactionChannel.ONLINE)
        instore_txn = _make_transaction(channel=TransactionChannel.IN_STORE)
        account = _make_account()

        online_features = self.extractor.extract(online_txn, account)
        instore_features = self.extractor.extract(instore_txn, account)

        assert online_features["channel_risk"] == pytest.approx(0.7)
        assert instore_features["channel_risk"] == pytest.approx(0.0)
        assert online_features["channel_risk"] > instore_features["channel_risk"]

    def test_channel_risk_atm(self):
        """ATM channel risk is 0.3."""
        txn = _make_transaction(channel=TransactionChannel.ATM)
        features = self.extractor.extract(txn, _make_account())
        assert features["channel_risk"] == pytest.approx(0.3)

    def test_channel_risk_mobile(self):
        """MOBILE channel risk is 0.5."""
        txn = _make_transaction(channel=TransactionChannel.MOBILE)
        features = self.extractor.extract(txn, _make_account())
        assert features["channel_risk"] == pytest.approx(0.5)

    def test_channel_risk_ordering(self):
        """Channel risk ordering: IN_STORE < ATM < MOBILE < ONLINE."""
        account = _make_account()
        risks = {
            ch: self.extractor.extract(_make_transaction(channel=ch), account)["channel_risk"]
            for ch in TransactionChannel
        }
        assert risks[TransactionChannel.IN_STORE] < risks[TransactionChannel.ATM]
        assert risks[TransactionChannel.ATM] < risks[TransactionChannel.MOBILE]
        assert risks[TransactionChannel.MOBILE] < risks[TransactionChannel.ONLINE]

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    def test_velocity_no_recent(self):
        """No recent transactions → both velocity features are 0.0."""
        features = self.extractor.extract(_make_transaction(), _make_account(), [])
        assert features["velocity_10min"] == 0.0
        assert features["velocity_1hr"] == 0.0

    def test_velocity_with_recent_conftest(
        self, normal_transaction, default_account, new_york_location
    ):
        """5 recent transactions within window produces non-zero velocity_10min."""
        recent = make_recent_transactions(normal_transaction.timestamp, 5, new_york_location)
        features = self.extractor.extract(normal_transaction, default_account, recent)
        assert features["velocity_10min"] > 0.0

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def test_distance_same_location(self):
        """Transaction at account's typical location has ~0 distance feature."""
        txn = _make_transaction(location=_NY)
        account = _make_account(typical_location=_NY)
        features = self.extractor.extract(txn, account)
        assert features["distance_from_typical"] == pytest.approx(0.0, abs=0.01)

    def test_distance_far_location(self):
        """Transaction far from typical location produces non-zero distance feature."""
        # Account in NY, transaction in LA (~3900 km)
        txn = _make_transaction(location=_LA)
        account = _make_account(typical_location=_NY)
        features = self.extractor.extract(txn, account)
        assert features["distance_from_typical"] > 0.0

    def test_distance_far_conftest(self, suspicious_transaction, default_account):
        """conftest suspicious_transaction (London) is far from default NY account."""
        features = self.extractor.extract(suspicious_transaction, default_account)
        assert features["distance_from_typical"] > 0.0

    # ------------------------------------------------------------------
    # International flag
    # ------------------------------------------------------------------

    def test_international_flag_false(self):
        """Domestic transaction sets is_international=0.0."""
        txn = _make_transaction(is_international=False)
        features = self.extractor.extract(txn, _make_account())
        assert features["is_international"] == 0.0

    def test_international_flag_true(self):
        """International transaction sets is_international=1.0."""
        txn = _make_transaction(is_international=True)
        features = self.extractor.extract(txn, _make_account())
        assert features["is_international"] == 1.0

    # ------------------------------------------------------------------
    # Amount z-score
    # ------------------------------------------------------------------

    def test_amount_zscore_high_for_suspicious(self, suspicious_transaction, default_account):
        """conftest suspicious_transaction ($4999.99) has a high amount_zscore."""
        features = self.extractor.extract(suspicious_transaction, default_account)
        assert features["amount_zscore"] > 2.0
