"""Synthetic transaction and account data generator.

Produces realistic-looking financial data for development, testing, and
agent evaluation.  All random draws use a seeded :class:`numpy.random.Generator`
so that results are fully reproducible given the same seed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

import numpy as np

from .schemas import Account, Location, Transaction, TransactionChannel

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

_MERCHANT_GROCERY = [
    ("Whole Foods Market", "5411"),
    ("Kroger", "5411"),
    ("Safeway", "5411"),
    ("Trader Joe's", "5411"),
    ("Publix", "5411"),
    ("Aldi", "5411"),
    ("Costco", "5411"),
    ("Walmart Grocery", "5411"),
]

_MERCHANT_RESTAURANT = [
    ("McDonald's", "5812"),
    ("Starbucks", "5812"),
    ("Chipotle", "5812"),
    ("Subway", "5812"),
    ("Panera Bread", "5812"),
    ("Olive Garden", "5812"),
    ("Chick-fil-A", "5812"),
    ("Domino's Pizza", "5812"),
    ("DoorDash", "5812"),
    ("Uber Eats", "5812"),
]

_MERCHANT_GAS = [
    ("Shell", "5541"),
    ("Chevron", "5541"),
    ("BP", "5541"),
    ("ExxonMobil", "5541"),
    ("Valero", "5541"),
    ("Circle K", "5541"),
    ("Sunoco", "5541"),
]

_MERCHANT_DEPT = [
    ("Target", "5311"),
    ("Macy's", "5311"),
    ("Nordstrom", "5311"),
    ("Kohl's", "5311"),
    ("JCPenney", "5311"),
    ("Bloomingdale's", "5311"),
]

_MERCHANT_MISC = [
    ("Amazon", "5999"),
    ("eBay", "5999"),
    ("Etsy", "5999"),
    ("Wayfair", "5999"),
    ("Best Buy", "5732"),
    ("Apple Store", "5732"),
    ("Home Depot", "5251"),
    ("Lowe's", "5251"),
    ("CVS Pharmacy", "5912"),
    ("Walgreens", "5912"),
    ("Netflix", "7841"),
    ("Spotify", "7841"),
    ("Lyft", "4121"),
    ("Uber", "4121"),
    ("Airbnb", "7011"),
    ("Marriott Hotels", "7011"),
    ("Delta Airlines", "4511"),
    ("American Airlines", "4511"),
]

_MERCHANT_HIGH_RISK = [
    ("CryptoSwap Pro", "6051"),
    ("BitExchange", "6051"),
    ("Lucky Star Casino", "7995"),
    ("Mega Jackpot", "7995"),
    ("GlobalWire Transfer", "4829"),
    ("SwiftRemit", "4829"),
    ("CoinVault", "6051"),
    ("Vegas Online Slots", "7995"),
]

_NORMAL_MERCHANTS = (
    _MERCHANT_GROCERY + _MERCHANT_RESTAURANT + _MERCHANT_GAS + _MERCHANT_DEPT + _MERCHANT_MISC
)

# (city, country, latitude, longitude)
_LOCATIONS: list[tuple[str, str, float, float]] = [
    ("New York", "US", 40.7128, -74.0060),
    ("Los Angeles", "US", 34.0522, -118.2437),
    ("Chicago", "US", 41.8781, -87.6298),
    ("Houston", "US", 29.7604, -95.3698),
    ("Phoenix", "US", 33.4484, -112.0740),
    ("Philadelphia", "US", 39.9526, -75.1652),
    ("San Antonio", "US", 29.4241, -98.4936),
    ("San Diego", "US", 32.7157, -117.1611),
    ("Dallas", "US", 32.7767, -96.7970),
    ("San Jose", "US", 37.3382, -121.8863),
    ("Austin", "US", 30.2672, -97.7431),
    ("Seattle", "US", 47.6062, -122.3321),
    ("Denver", "US", 39.7392, -104.9903),
    ("Miami", "US", 25.7617, -80.1918),
    ("Atlanta", "US", 33.7490, -84.3880),
    ("London", "GB", 51.5074, -0.1278),
    ("Paris", "FR", 48.8566, 2.3522),
    ("Berlin", "DE", 52.5200, 13.4050),
    ("Toronto", "CA", 43.6532, -79.3832),
    ("Sydney", "AU", -33.8688, 151.2093),
    ("Tokyo", "JP", 35.6762, 139.6503),
    ("Singapore", "SG", 1.3521, 103.8198),
    ("Dubai", "AE", 25.2048, 55.2708),
    ("Mexico City", "MX", 19.4326, -99.1332),
    ("São Paulo", "BR", -23.5505, -46.6333),
    ("Lagos", "NG", 6.5244, 3.3792),
    ("Bucharest", "RO", 44.4268, 26.1025),
    ("Kiev", "UA", 50.4501, 30.5234),
    ("Minsk", "BY", 53.9006, 27.5590),
]

_US_LOCATIONS = [loc for loc in _LOCATIONS if loc[1] == "US"]
_FOREIGN_LOCATIONS = [loc for loc in _LOCATIONS if loc[1] != "US"]
_HIGH_RISK_FOREIGN = [loc for loc in _LOCATIONS if loc[1] in ("NG", "RO", "UA", "BY")]

_HOLDER_FIRST_NAMES = [
    "James",
    "Mary",
    "John",
    "Patricia",
    "Robert",
    "Jennifer",
    "Michael",
    "Linda",
    "William",
    "Barbara",
    "David",
    "Elizabeth",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Christopher",
    "Lisa",
    "Daniel",
    "Nancy",
    "Matthew",
    "Betty",
    "Anthony",
    "Margaret",
    "Donald",
    "Sandra",
]

_HOLDER_LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TransactionGenerator:
    """Generates synthetic :class:`~fraud_agent.data.schemas.Transaction` and
    :class:`~fraud_agent.data.schemas.Account` objects for testing and evaluation.

    Uses a seeded :class:`numpy.random.Generator` so that all output is fully
    reproducible given the same ``seed``.

    Parameters
    ----------
    seed:
        Integer seed for the underlying NumPy random generator.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_account(self) -> Account:
        """Create a realistic synthetic account profile.

        Returns
        -------
        Account
            A fully populated :class:`~fraud_agent.data.schemas.Account`.
        """
        first = self._choice(_HOLDER_FIRST_NAMES)
        last = self._choice(_HOLDER_LAST_NAMES)
        loc_tuple = self._choice(_US_LOCATIONS)
        location = Location(
            city=loc_tuple[0],
            country=loc_tuple[1],
            latitude=loc_tuple[2],
            longitude=loc_tuple[3],
        )

        # Average transaction amount: log-normal centred around $65
        avg_amount = Decimal(str(round(float(self._rng.lognormal(mean=4.17, sigma=0.6)), 2)))

        # Account age: 0-15 years
        days_open = int(self._rng.integers(30, 15 * 365))
        open_date = (datetime.now(UTC) - timedelta(days=days_open)).date()

        # Transaction history proportional to account age
        history_count = int(self._rng.integers(max(1, days_open // 5), max(2, days_open)))

        return Account(
            id=str(uuid.uuid4()),
            holder_name=f"{first} {last}",
            average_transaction_amount=avg_amount,
            typical_location=location,
            account_open_date=open_date,
            transaction_history_count=history_count,
        )

    def generate_transaction(self, account: Account, is_fraud: bool = False) -> Transaction:
        """Generate a single transaction for the given account.

        Parameters
        ----------
        account:
            The owning account whose profile anchors the behavioural baseline.
        is_fraud:
            When ``True`` a fraud pattern is randomly selected; otherwise a
            normal spending pattern is used.

        Returns
        -------
        Transaction
        """
        if is_fraud:
            return self._generate_fraud_transaction(account)
        return self._generate_normal_transaction(account)

    def generate_batch(
        self,
        count: int,
        fraud_rate: float = 0.05,
    ) -> list[Transaction]:
        """Generate a batch of transactions across freshly created accounts.

        Parameters
        ----------
        count:
            Total number of transactions to generate.
        fraud_rate:
            Fraction of transactions that should be fraudulent (0.0–1.0).

        Returns
        -------
        list[Transaction]
            Shuffled list of the requested transactions.
        """
        if not 0.0 <= fraud_rate <= 1.0:
            raise ValueError(f"fraud_rate must be in [0, 1], got {fraud_rate}")

        fraud_count = int(round(count * fraud_rate))
        legit_count = count - fraud_count

        # Create a pool of accounts (one per ~5–15 transactions is realistic)
        account_pool_size = max(1, count // 10)
        accounts = [self.generate_account() for _ in range(account_pool_size)]

        transactions: list[Transaction] = []

        for _ in range(legit_count):
            account = self._choice(accounts)
            transactions.append(self._generate_normal_transaction(account))

        for _ in range(fraud_count):
            account = self._choice(accounts)
            transactions.append(self._generate_fraud_transaction(account))

        # Shuffle in-place with the seeded generator
        indices = self._rng.permutation(len(transactions))
        return [transactions[i] for i in indices]

    # ------------------------------------------------------------------
    # Private helpers — normal transactions
    # ------------------------------------------------------------------

    def _generate_normal_transaction(self, account: Account) -> Transaction:
        """Produce a transaction consistent with the account's spending history.

        Amount is drawn from a log-normal distribution anchored to the account
        average.  Location is the account's typical location with small noise.
        """
        merchant_name, mcc = self._choice(_NORMAL_MERCHANTS)

        # Amount: log-normal near account average with sigma ~0.4
        avg = float(account.average_transaction_amount)
        raw_amount = float(self._rng.lognormal(mean=np.log(avg), sigma=0.4))
        # Clamp to reasonable retail range
        raw_amount = max(1.0, min(raw_amount, 5000.0))
        amount = Decimal(str(round(raw_amount, 2)))

        # Timestamp: random point in the last 90 days
        seconds_ago = int(self._rng.integers(0, 90 * 24 * 3600))
        timestamp = datetime.now(UTC) - timedelta(seconds=seconds_ago)

        # Location: usually typical; 5% chance of a US travel location
        if self._rng.random() < 0.05:
            loc_tuple = self._choice(_US_LOCATIONS)
            location = Location(
                city=loc_tuple[0],
                country=loc_tuple[1],
                latitude=loc_tuple[2],
                longitude=loc_tuple[3],
            )
            is_international = False
        else:
            # Jitter the typical location by up to ~5 km
            jitter_lat = float(self._rng.normal(0, 0.05))
            jitter_lon = float(self._rng.normal(0, 0.05))
            location = Location(
                city=account.typical_location.city,
                country=account.typical_location.country,
                latitude=round(
                    max(-90.0, min(90.0, account.typical_location.latitude + jitter_lat)),
                    4,
                ),
                longitude=round(
                    max(-180.0, min(180.0, account.typical_location.longitude + jitter_lon)),
                    4,
                ),
            )
            is_international = False

        channel = self._weighted_channel()

        return Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            amount=amount,
            currency="USD",
            merchant_name=merchant_name,
            merchant_category_code=mcc,
            card_last_four=self._card_last_four(),
            account_id=account.id,
            location=location,
            channel=channel,
            is_international=is_international,
            metadata=None,
        )

    # ------------------------------------------------------------------
    # Private helpers — fraud transactions
    # ------------------------------------------------------------------

    def _generate_fraud_transaction(self, account: Account) -> Transaction:
        """Select a random fraud pattern and generate the corresponding transaction."""
        patterns: list[
            Literal[
                "velocity_abuse",
                "geo_impossible",
                "amount_anomaly",
                "card_testing",
                "high_risk_merchant",
            ]
        ] = [
            "velocity_abuse",
            "geo_impossible",
            "amount_anomaly",
            "card_testing",
            "high_risk_merchant",
        ]
        pattern = self._choice(patterns)  # type: ignore[arg-type]

        if pattern == "velocity_abuse":
            return self._fraud_velocity_abuse(account)
        elif pattern == "geo_impossible":
            return self._fraud_geo_impossible(account)
        elif pattern == "amount_anomaly":
            return self._fraud_amount_anomaly(account)
        elif pattern == "card_testing":
            return self._fraud_card_testing(account)
        else:
            return self._fraud_high_risk_merchant(account)

    def _fraud_velocity_abuse(self, account: Account) -> Transaction:
        """Many small charges in rapid succession (velocity abuse pattern)."""
        merchant_name, mcc = self._choice(_NORMAL_MERCHANTS)

        # Small amounts typical of velocity abuse ($3 – $30)
        amount = Decimal(str(round(float(self._rng.uniform(3.0, 30.0)), 2)))

        # Very recent — within the last 15 minutes
        seconds_ago = int(self._rng.integers(0, 900))
        timestamp = datetime.now(UTC) - timedelta(seconds=seconds_ago)

        location = self._typical_location(account)

        return Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            amount=amount,
            currency="USD",
            merchant_name=merchant_name,
            merchant_category_code=mcc,
            card_last_four=self._card_last_four(),
            account_id=account.id,
            location=location,
            channel=TransactionChannel.ONLINE,
            is_international=False,
            metadata={"fraud_pattern": "velocity_abuse"},
        )

    def _fraud_geo_impossible(self, account: Account) -> Transaction:
        """Transaction from a distant foreign location (geo-impossible travel)."""
        merchant_name, mcc = self._choice(_NORMAL_MERCHANTS)

        avg = float(account.average_transaction_amount)
        amount = Decimal(str(round(float(self._rng.lognormal(mean=np.log(avg), sigma=0.5)), 2)))

        # Very recent timestamp
        seconds_ago = int(self._rng.integers(0, 3600))
        timestamp = datetime.now(UTC) - timedelta(seconds=seconds_ago)

        # Pick a high-risk or random foreign location
        if self._rng.random() < 0.5 and _HIGH_RISK_FOREIGN:
            loc_tuple = self._choice(_HIGH_RISK_FOREIGN)
        else:
            loc_tuple = self._choice(_FOREIGN_LOCATIONS)

        location = Location(
            city=loc_tuple[0],
            country=loc_tuple[1],
            latitude=loc_tuple[2],
            longitude=loc_tuple[3],
        )

        return Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            amount=amount,
            currency="USD",
            merchant_name=merchant_name,
            merchant_category_code=mcc,
            card_last_four=self._card_last_four(),
            account_id=account.id,
            location=location,
            channel=TransactionChannel.IN_STORE,
            is_international=True,
            metadata={"fraud_pattern": "geo_impossible"},
        )

    def _fraud_amount_anomaly(self, account: Account) -> Transaction:
        """Single charge 5–20x the account's average (amount anomaly)."""
        merchant_name, mcc = self._choice(_NORMAL_MERCHANTS)

        avg = float(account.average_transaction_amount)
        multiplier = float(self._rng.uniform(5.0, 20.0))
        amount = Decimal(str(round(avg * multiplier, 2)))

        seconds_ago = int(self._rng.integers(0, 7 * 24 * 3600))
        timestamp = datetime.now(UTC) - timedelta(seconds=seconds_ago)

        location = self._typical_location(account)

        return Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            amount=amount,
            currency="USD",
            merchant_name=merchant_name,
            merchant_category_code=mcc,
            card_last_four=self._card_last_four(),
            account_id=account.id,
            location=location,
            channel=self._weighted_channel(),
            is_international=False,
            metadata={"fraud_pattern": "amount_anomaly", "multiplier": round(multiplier, 2)},
        )

    def _fraud_card_testing(self, account: Account) -> Transaction:
        """Micro-charge ($0.50–$1.99) used to verify a stolen card number."""
        merchant_name, mcc = self._choice(_NORMAL_MERCHANTS)

        amount = Decimal(str(round(float(self._rng.uniform(0.50, 1.99)), 2)))

        seconds_ago = int(self._rng.integers(0, 3600))
        timestamp = datetime.now(UTC) - timedelta(seconds=seconds_ago)

        location = self._typical_location(account)

        return Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            amount=amount,
            currency="USD",
            merchant_name=merchant_name,
            merchant_category_code=mcc,
            card_last_four=self._card_last_four(),
            account_id=account.id,
            location=location,
            channel=TransactionChannel.ONLINE,
            is_international=False,
            metadata={"fraud_pattern": "card_testing"},
        )

    def _fraud_high_risk_merchant(self, account: Account) -> Transaction:
        """Purchase at a high-risk merchant (gambling, crypto, wire transfer)."""
        merchant_name, mcc = self._choice(_MERCHANT_HIGH_RISK)

        avg = float(account.average_transaction_amount)
        # Higher amounts at high-risk merchants — 1x to 8x average
        multiplier = float(self._rng.uniform(1.0, 8.0))
        amount = Decimal(str(round(avg * multiplier, 2)))

        seconds_ago = int(self._rng.integers(0, 30 * 24 * 3600))
        timestamp = datetime.now(UTC) - timedelta(seconds=seconds_ago)

        # Mix of domestic and international
        is_international = bool(self._rng.random() < 0.4)
        if is_international and _FOREIGN_LOCATIONS:
            loc_tuple = self._choice(_FOREIGN_LOCATIONS)
        else:
            loc_tuple = self._choice(_US_LOCATIONS)

        location = Location(
            city=loc_tuple[0],
            country=loc_tuple[1],
            latitude=loc_tuple[2],
            longitude=loc_tuple[3],
        )

        return Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            amount=amount,
            currency="USD",
            merchant_name=merchant_name,
            merchant_category_code=mcc,
            card_last_four=self._card_last_four(),
            account_id=account.id,
            location=location,
            channel=TransactionChannel.ONLINE,
            is_international=is_international,
            metadata={"fraud_pattern": "high_risk_merchant"},
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _choice(self, seq: list) -> object:  # type: ignore[type-arg]
        """Return a uniformly random element from *seq*."""
        idx = int(self._rng.integers(0, len(seq)))
        return seq[idx]

    def _card_last_four(self) -> str:
        """Generate a random 4-digit card suffix."""
        return str(int(self._rng.integers(1000, 9999)))

    def _weighted_channel(self) -> TransactionChannel:
        """Sample a :class:`TransactionChannel` with realistic weights.

        Weights: IN_STORE 45%, ONLINE 35%, MOBILE 15%, ATM 5%.
        """
        roll = float(self._rng.random())
        if roll < 0.45:
            return TransactionChannel.IN_STORE
        elif roll < 0.80:
            return TransactionChannel.ONLINE
        elif roll < 0.95:
            return TransactionChannel.MOBILE
        else:
            return TransactionChannel.ATM

    def _typical_location(self, account: Account) -> Location:
        """Return the account's typical location with small coordinate jitter."""
        jitter_lat = float(self._rng.normal(0, 0.02))
        jitter_lon = float(self._rng.normal(0, 0.02))
        return Location(
            city=account.typical_location.city,
            country=account.typical_location.country,
            latitude=round(
                max(-90.0, min(90.0, account.typical_location.latitude + jitter_lat)), 4
            ),
            longitude=round(
                max(-180.0, min(180.0, account.typical_location.longitude + jitter_lon)), 4
            ),
        )
