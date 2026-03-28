"""In-memory fraud pattern knowledge base.

Provides a lightweight, queryable collection of known fraud patterns used
by detection agents to match transaction characteristics against documented
fraud typologies.  For production use this would be backed by a vector store
or relational database; the current implementation keeps the full pattern list
in memory for zero-latency local inference.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------

FRAUD_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "FP-001",
        "name": "Card Testing",
        "description": (
            "Fraudsters who obtain stolen card numbers perform micro-charges "
            "($0.50–$2.00) to verify that a card is active before making "
            "large fraudulent purchases.  Targets are typically low-friction "
            "online merchants that do not require strong authentication.  "
            "Multiple cards are often tested within the same short window."
        ),
        "indicators": [
            "Transaction amount between $0.50 and $2.00",
            "Online channel with no 3DS challenge",
            "Multiple cards used at same merchant in short window",
            "New account or recently issued card",
            "Rapid succession of micro-charges",
        ],
        "risk_level": "HIGH",
        "category": "card_testing",
        "typical_amount_range": {"min": 0.50, "max": 2.00, "currency": "USD"},
        "common_mcc_codes": ["5999", "5812", "5411", "4816"],
    },
    {
        "id": "FP-002",
        "name": "Account Takeover",
        "description": (
            "An adversary gains access to a legitimate cardholder's account, "
            "typically via credential stuffing, phishing, or SIM-swap, then "
            "rapidly changes contact details and drains available credit.  "
            "Indicators include login from new device/IP, profile changes, "
            "and an immediate spike in transaction volume."
        ),
        "indicators": [
            "Login from previously unseen device or IP geolocation",
            "Email or phone number changed shortly before transactions",
            "Transactions immediately following profile update",
            "Purchases of easily liquidated goods (gift cards, electronics)",
            "High-value transactions well above account average",
        ],
        "risk_level": "CRITICAL",
        "category": "account_takeover",
        "typical_amount_range": {"min": 200.00, "max": 5000.00, "currency": "USD"},
        "common_mcc_codes": ["5732", "5999", "6051", "5411"],
    },
    {
        "id": "FP-003",
        "name": "Synthetic Identity Fraud",
        "description": (
            "Criminals construct fictional identities by combining real and "
            "fabricated personal data (e.g., valid SSN with a false name) to "
            "open new accounts.  After months of building apparent "
            "creditworthiness, they execute a 'bust-out' — maxing all "
            "available credit and disappearing."
        ),
        "indicators": [
            "New account with no or thin credit history",
            "Address or SSN associated with multiple identities",
            "Rapid credit line utilisation after account opening",
            "Frequent credit limit increase requests",
            "Multiple accounts opened in short period using similar data",
        ],
        "risk_level": "HIGH",
        "category": "synthetic_identity",
        "typical_amount_range": {"min": 500.00, "max": 10000.00, "currency": "USD"},
        "common_mcc_codes": ["5311", "5732", "5912", "5999"],
    },
    {
        "id": "FP-004",
        "name": "First-Party Fraud",
        "description": (
            "The legitimate account holder deliberately misrepresents "
            "transactions, for example by claiming goods were not received "
            "when they were, or intentionally overdrafting with no intent "
            "to repay.  Also includes deliberate default on credit products "
            "after spending."
        ),
        "indicators": [
            "High chargeback rate on account",
            "Repeated dispute claims for same merchant category",
            "Purchasing expensive items shortly before account default",
            "Disputed transactions inconsistent with stated usage pattern",
            "Claims of non-receipt concentrated at specific merchants",
        ],
        "risk_level": "MEDIUM",
        "category": "first_party",
        "typical_amount_range": {"min": 50.00, "max": 2000.00, "currency": "USD"},
        "common_mcc_codes": ["5311", "5812", "5999", "7011"],
    },
    {
        "id": "FP-005",
        "name": "Friendly Fraud (Chargeback Abuse)",
        "description": (
            "A cardholder makes a legitimate purchase, receives the goods or "
            "services, and then files a chargeback claiming the transaction "
            "was unauthorised or the item was not delivered.  This is "
            "increasingly common in e-commerce and digital goods sectors."
        ),
        "indicators": [
            "Chargeback filed for a transaction with delivery confirmation",
            "Prior history of disputed e-commerce transactions",
            "Dispute filed at the maximum allowed time window",
            "Merchant has strong fulfillment evidence",
            "Transaction matches cardholder's normal spending profile",
        ],
        "risk_level": "MEDIUM",
        "category": "friendly_fraud",
        "typical_amount_range": {"min": 20.00, "max": 500.00, "currency": "USD"},
        "common_mcc_codes": ["5999", "5732", "7841", "4816"],
    },
    {
        "id": "FP-006",
        "name": "Cross-Border Fraud",
        "description": (
            "Stolen card data is used in a foreign jurisdiction where "
            "enforcement is difficult or EMV chip adoption is lower.  "
            "Transactions appear in countries the cardholder has never "
            "visited or appear simultaneously with domestic usage, creating "
            "a geographical impossibility."
        ),
        "indicators": [
            "Transaction country differs from cardholder country",
            "Simultaneous transactions in geographically distant locations",
            "High-risk originating country (known fraud hotspots)",
            "Magstripe fallback in a country with high EMV adoption",
            "First international transaction on an established domestic account",
        ],
        "risk_level": "HIGH",
        "category": "cross_border",
        "typical_amount_range": {"min": 100.00, "max": 3000.00, "currency": "USD"},
        "common_mcc_codes": ["5311", "5812", "5732", "7011"],
    },
    {
        "id": "FP-007",
        "name": "Merchant Collusion",
        "description": (
            "A dishonest merchant (or compromised merchant terminal) "
            "processes unauthorised charges against cards that passed "
            "through their point-of-sale system.  The merchant may also "
            "inflate transaction amounts or process refunds to money-mule "
            "accounts.  Often surfaces as a cluster of fraud across many "
            "different cardholders at one merchant."
        ),
        "indicators": [
            "Multiple different cardholders reporting fraud at same merchant",
            "Refunds issued to cards that never made a purchase at the merchant",
            "Transactions at unusual hours inconsistent with merchant business hours",
            "Split transactions just below authorisation thresholds",
            "Merchant recently added to high-risk monitoring list",
        ],
        "risk_level": "CRITICAL",
        "category": "merchant_collusion",
        "typical_amount_range": {"min": 50.00, "max": 2000.00, "currency": "USD"},
        "common_mcc_codes": ["5812", "5999", "5411", "7299"],
    },
    {
        "id": "FP-008",
        "name": "Velocity Abuse",
        "description": (
            "A card or account is used to perform an abnormally high number "
            "of transactions in a short time window, often to maximise "
            "extraction before the card is blocked.  Amounts may be "
            "individually modest but aggregate quickly.  Frequently seen "
            "after account takeover events."
        ),
        "indicators": [
            "More than 5 transactions within a 30-minute window",
            "Total spend in 1 hour exceeds 3x daily average",
            "Same card used at multiple merchants in rapid succession",
            "ATM withdrawals at multiple machines within hours",
            "Online purchases across multiple merchant categories simultaneously",
        ],
        "risk_level": "HIGH",
        "category": "velocity_abuse",
        "typical_amount_range": {"min": 10.00, "max": 500.00, "currency": "USD"},
        "common_mcc_codes": ["5999", "5732", "5411", "6011"],
    },
    {
        "id": "FP-009",
        "name": "Geo-Anomaly / Impossible Travel",
        "description": (
            "Two or more transactions occur at physical locations that could "
            "not both be reached by the cardholder in the elapsed time "
            "between them given any realistic mode of transport.  Also "
            "covers transactions from high-risk geographic regions with no "
            "prior account activity in that region."
        ),
        "indicators": [
            "Two transactions > 500 km apart within 2 hours",
            "Domestic transaction followed immediately by a transaction abroad",
            "Transaction in a country the account has never transacted in",
            "In-store transaction in one city while IP login is in another",
            "Location associated with known fraud-originating regions",
        ],
        "risk_level": "HIGH",
        "category": "geo_anomaly",
        "typical_amount_range": {"min": 50.00, "max": 5000.00, "currency": "USD"},
        "common_mcc_codes": ["5311", "5812", "5732", "7011"],
    },
    {
        "id": "FP-010",
        "name": "Amount Anomaly",
        "description": (
            "A single transaction amount deviates dramatically from the "
            "account's established spending baseline — typically 5x or more "
            "the account average.  May indicate an account-takeover cash-out, "
            "an invoice-redirect scam, or misuse of a corporate card.  "
            "Must be considered alongside merchant context."
        ),
        "indicators": [
            "Transaction amount > 5x account rolling average",
            "Large round-number amounts (e.g. $5,000, $10,000)",
            "High-value purchase at atypical merchant category",
            "No prior history of high-value transactions on the account",
            "Transaction immediately preceded by a failed authorisation",
        ],
        "risk_level": "HIGH",
        "category": "amount_anomaly",
        "typical_amount_range": {"min": 500.00, "max": 50000.00, "currency": "USD"},
        "common_mcc_codes": ["5732", "5311", "6051", "5999"],
    },
    {
        "id": "FP-011",
        "name": "Phishing-Enabled Fraud",
        "description": (
            "The cardholder is deceived into revealing credentials or "
            "one-time passcodes via a fraudulent communication, enabling "
            "the attacker to authorise transactions the holder believed "
            "were for a legitimate purpose.  Social engineering element "
            "makes this harder to detect than pure card-not-present fraud."
        ),
        "indicators": [
            "OTP entered shortly after an unsolicited call or message",
            "Transaction initiated immediately after a support-line contact",
            "Unusual merchant or payee not in cardholder's history",
            "Large single transfer to a new beneficiary",
            "Transaction occurs at an unusual time for the account holder",
        ],
        "risk_level": "CRITICAL",
        "category": "phishing",
        "typical_amount_range": {"min": 200.00, "max": 20000.00, "currency": "USD"},
        "common_mcc_codes": ["6012", "4829", "6051", "6011"],
    },
    {
        "id": "FP-012",
        "name": "Card Skimming",
        "description": (
            "Physical or digital skimming devices harvest card data from ATMs, "
            "fuel pumps, or POS terminals.  The stolen data is encoded onto "
            "counterfeit cards or sold for card-not-present fraud.  "
            "Transactions typically appear at ATMs or fuel merchants followed "
            "by usage in geographically distant locations."
        ),
        "indicators": [
            "ATM or fuel-pump usage shortly before fraudulent transactions",
            "Magstripe transaction at a terminal expected to be EMV",
            "Card data used at a location distant from where it was swiped",
            "Cluster of affected cards from the same ATM or merchant",
            "Transaction immediately preceded by a balance enquiry",
        ],
        "risk_level": "HIGH",
        "category": "skimming",
        "typical_amount_range": {"min": 50.00, "max": 1000.00, "currency": "USD"},
        "common_mcc_codes": ["5541", "6011", "5411", "5812"],
    },
    {
        "id": "FP-013",
        "name": "Triangulation Fraud",
        "description": (
            "A fraudster sets up a fake online storefront at attractive prices, "
            "uses stolen credit cards to purchase real goods from legitimate "
            "retailers, and ships them to buyers who paid the fraudster.  "
            "The cardholder is unaware; the legitimate buyer receives the "
            "goods; only the retailer and card issuer suffer losses."
        ),
        "indicators": [
            "Shipping address does not match billing address",
            "Order placed with multiple different payment cards",
            "High-demand or easily-resaleable goods (gift cards, electronics)",
            "Multiple orders from the same IP to different shipping addresses",
            "Merchant reports multiple chargebacks for the same shipping address",
        ],
        "risk_level": "HIGH",
        "category": "triangulation",
        "typical_amount_range": {"min": 100.00, "max": 1500.00, "currency": "USD"},
        "common_mcc_codes": ["5732", "5999", "5411", "5311"],
    },
    {
        "id": "FP-014",
        "name": "Bust-Out Fraud",
        "description": (
            "A fraudster — possibly using real or synthetic identity — "
            "opens credit accounts, gradually builds apparent "
            "creditworthiness, secures credit limit increases, and then "
            "maxes out all credit lines before defaulting.  The pattern "
            "is characterised by sudden, rapid credit utilisation after "
            "a period of responsible usage."
        ),
        "indicators": [
            "Credit utilisation jumps from <20% to >90% within days",
            "Multiple credit products drawn down simultaneously",
            "Spending shifts to cash-equivalent or easily-liquidated categories",
            "Contact information changes shortly before bust-out",
            "Payments cease after maximum utilisation",
        ],
        "risk_level": "CRITICAL",
        "category": "bust_out",
        "typical_amount_range": {"min": 1000.00, "max": 30000.00, "currency": "USD"},
        "common_mcc_codes": ["6011", "4829", "6051", "5999"],
    },
    {
        "id": "FP-015",
        "name": "Refund Abuse",
        "description": (
            "Fraudsters exploit merchant refund policies by purchasing "
            "goods with stolen cards and then requesting refunds to "
            "different payment methods (money-mule accounts), or by "
            "fabricating return claims for goods never purchased.  "
            "Also includes return fraud where counterfeit or switched "
            "goods are returned in place of genuine items."
        ),
        "indicators": [
            "Refund requested to a different card than original payment",
            "High ratio of refunds to purchases on an account",
            "Refund amount exceeds original purchase amount",
            "Same merchant targeted repeatedly for refunds",
            "Refund requested immediately after purchase with no apparent reason",
        ],
        "risk_level": "MEDIUM",
        "category": "refund_abuse",
        "typical_amount_range": {"min": 20.00, "max": 2000.00, "currency": "USD"},
        "common_mcc_codes": ["5311", "5732", "5999", "5812"],
    },
    {
        "id": "FP-016",
        "name": "Crypto On-Ramp Fraud",
        "description": (
            "Stolen card credentials are used to purchase cryptocurrency or "
            "prepaid value via crypto exchanges and on-ramp services, "
            "enabling near-instant, irreversible conversion of stolen funds.  "
            "High-risk MCC codes (6051) at unusual hours are a primary signal.  "
            "Transactions often occur in rapid succession to bypass individual "
            "transaction limits."
        ),
        "indicators": [
            "MCC 6051 (crypto/money services) on account with no prior history",
            "Multiple small crypto purchases in rapid succession",
            "Transaction at crypto exchange outside normal account hours",
            "Purchase immediately following account credential change",
            "IP address associated with VPN or Tor exit node",
        ],
        "risk_level": "CRITICAL",
        "category": "cross_border",
        "typical_amount_range": {"min": 50.00, "max": 5000.00, "currency": "USD"},
        "common_mcc_codes": ["6051", "4829", "6012"],
    },
]


# ---------------------------------------------------------------------------
# Knowledge base class
# ---------------------------------------------------------------------------


class FraudKnowledgeBase:
    """In-memory store of fraud pattern descriptors.

    Patterns are loaded once at construction time and queried via simple
    keyword search or direct ID lookup.  For production deployments the
    ``_patterns`` list can be replaced with a database-backed source without
    changing the public API.

    Parameters
    ----------
    patterns:
        Optional list of pattern dicts to load instead of the built-in
        :data:`FRAUD_PATTERNS` catalogue.  Primarily useful for testing.
    """

    def __init__(
        self,
        patterns: list[dict[str, Any]] | None = None,
    ) -> None:
        self._patterns: list[dict[str, Any]] = patterns if patterns is not None else FRAUD_PATTERNS
        # Build an O(1) lookup by id
        self._by_id: dict[str, dict[str, Any]] = {p["id"]: p for p in self._patterns}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_patterns(self) -> list[dict[str, Any]]:
        """Return the full catalogue of fraud patterns.

        Returns
        -------
        list[dict]
            Shallow copy of the internal pattern list; modifications to the
            returned list do not affect the knowledge base state.
        """
        return list(self._patterns)

    def search_patterns(self, query: str) -> list[dict[str, Any]]:
        """Search patterns using case-insensitive keyword matching.

        The query string is tokenised on whitespace; a pattern matches if
        *all* tokens appear somewhere in its ``name``, ``description``,
        ``category``, ``indicators`` text, or ``id``.

        Parameters
        ----------
        query:
            Free-text search string (e.g. ``"geo impossible travel"``).

        Returns
        -------
        list[dict]
            Patterns that match all tokens in the query, ordered by their
            position in the catalogue (i.e. by ``id``).
        """
        tokens = [t.lower() for t in query.split() if t]
        if not tokens:
            return list(self._patterns)

        results: list[dict[str, Any]] = []
        for pattern in self._patterns:
            searchable = self._pattern_to_text(pattern).lower()
            if all(token in searchable for token in tokens):
                results.append(pattern)
        return results

    def get_pattern_by_id(self, pattern_id: str) -> dict[str, Any] | None:
        """Look up a single pattern by its unique identifier.

        Parameters
        ----------
        pattern_id:
            The pattern's ``id`` field (e.g. ``"FP-001"``).

        Returns
        -------
        dict | None
            The matching pattern dict, or ``None`` if not found.
        """
        return self._by_id.get(pattern_id)

    def get_patterns_by_category(self, category: str) -> list[dict[str, Any]]:
        """Return all patterns belonging to the specified category.

        Parameters
        ----------
        category:
            Category string to match exactly (e.g. ``"card_testing"``).

        Returns
        -------
        list[dict]
            Patterns whose ``category`` field equals *category*.
        """
        normalised = category.lower()
        return [p for p in self._patterns if p.get("category", "").lower() == normalised]

    def get_patterns_by_risk_level(self, risk_level: str) -> list[dict[str, Any]]:
        """Return all patterns at or matching the given risk level.

        Parameters
        ----------
        risk_level:
            One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, or ``"CRITICAL"``
            (case-insensitive).

        Returns
        -------
        list[dict]
        """
        normalised = risk_level.upper()
        return [p for p in self._patterns if p.get("risk_level", "").upper() == normalised]

    def get_patterns_by_mcc(self, mcc: str) -> list[dict[str, Any]]:
        """Return patterns that list a given MCC code as common.

        Parameters
        ----------
        mcc:
            A 4-digit MCC string (e.g. ``"6051"``).

        Returns
        -------
        list[dict]
        """
        return [p for p in self._patterns if mcc in p.get("common_mcc_codes", [])]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pattern_to_text(pattern: dict[str, Any]) -> str:
        """Flatten all searchable text fields of a pattern into one string."""
        parts: list[str] = [
            pattern.get("id", ""),
            pattern.get("name", ""),
            pattern.get("description", ""),
            pattern.get("category", ""),
            pattern.get("risk_level", ""),
            " ".join(pattern.get("indicators", [])),
            " ".join(pattern.get("common_mcc_codes", [])),
        ]
        return " ".join(parts)
