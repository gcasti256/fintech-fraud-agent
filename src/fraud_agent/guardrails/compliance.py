"""Compliance checks for regulatory requirements."""

from __future__ import annotations

import re

from fraud_agent.data.schemas import FraudDecision, Transaction

# Countries under comprehensive US sanctions (OFAC)
SANCTIONED_COUNTRIES = {
    "KP",  # North Korea
    "IR",  # Iran
    "SY",  # Syria
    "CU",  # Cuba
    "RU",  # Russia (select sanctions)
}

# Currency Transaction Report threshold (FinCEN)
CTR_THRESHOLD = 10_000.00

# Structuring detection — multiple transactions just under CTR threshold
STRUCTURING_LOWER = 8_000.00
STRUCTURING_UPPER = 9_999.99

# Wire transfer MCC codes
WIRE_TRANSFER_MCCS = {"4829", "6012"}

# PII regex patterns for validation
PII_PATTERNS = {
    "card_number": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
    "unmasked_account": re.compile(r"\bACC-\w{4}-\w{4}\b"),
}


class ComplianceChecker:
    """Validates transactions and decisions against regulatory requirements.

    Checks:
    - Currency Transaction Report (CTR) threshold ($10,000+)
    - Structuring detection (multiple transactions near $10k)
    - International wire transfers
    - Sanctioned country activity
    - PII handling validation
    """

    def check_transaction(self, transaction: Transaction) -> list[dict]:
        """Run all compliance checks on a transaction.

        Args:
            transaction: The transaction to check.

        Returns:
            List of compliance flags, each with type, severity, and description.
        """
        flags: list[dict] = []

        # CTR reporting check
        if float(transaction.amount) >= CTR_THRESHOLD:
            flags.append(
                {
                    "type": "ctr_reporting",
                    "severity": "high",
                    "description": (
                        f"Transaction of ${float(transaction.amount):,.2f} exceeds "
                        f"${CTR_THRESHOLD:,.2f} CTR reporting threshold"
                    ),
                    "regulation": "31 CFR 1010.311",
                }
            )

        # Structuring detection
        amount = float(transaction.amount)
        if STRUCTURING_LOWER <= amount <= STRUCTURING_UPPER:
            flags.append(
                {
                    "type": "potential_structuring",
                    "severity": "medium",
                    "description": (
                        f"Transaction of ${amount:,.2f} is within structuring range "
                        f"(${STRUCTURING_LOWER:,.2f}-${STRUCTURING_UPPER:,.2f})"
                    ),
                    "regulation": "31 USC 5324",
                }
            )

        # International wire transfer
        if (
            transaction.is_international
            and transaction.merchant_category_code in WIRE_TRANSFER_MCCS
        ):
            flags.append(
                {
                    "type": "international_wire",
                    "severity": "medium",
                    "description": (
                        f"International wire transfer to {transaction.location.country}"
                    ),
                    "regulation": "31 CFR 1010.340",
                }
            )

        # Sanctioned country
        if transaction.location.country in SANCTIONED_COUNTRIES:
            flags.append(
                {
                    "type": "sanctioned_country",
                    "severity": "critical",
                    "description": (
                        f"Transaction involves sanctioned country: {transaction.location.country}"
                    ),
                    "regulation": "OFAC SDN List / Executive Orders",
                }
            )

        return flags

    def check_decision_audit_trail(self, decision: FraudDecision) -> bool:
        """Verify a fraud decision has all required fields for audit compliance.

        Args:
            decision: The fraud decision to validate.

        Returns:
            True if the decision is audit-compliant.
        """
        required_fields = [
            decision.transaction_id,
            decision.risk_level,
            decision.explanation,
            decision.recommended_action,
        ]
        return all(f is not None and f != "" for f in required_fields)

    def validate_pii_handling(self, data: dict) -> list[str]:
        """Check if any unmasked PII is present in a data dict.

        Args:
            data: Dictionary to scan for PII.

        Returns:
            List of PII violation descriptions.
        """
        violations: list[str] = []

        def scan_value(value: object, path: str) -> None:
            if isinstance(value, str):
                for pii_type, pattern in PII_PATTERNS.items():
                    matches = pattern.findall(value)
                    for match in matches:
                        # Check if it's masked (contains ****)
                        if "****" not in match and "***" not in match:
                            violations.append(
                                f"Unmasked {pii_type} found at {path}: {match[:8]}..."
                            )
            elif isinstance(value, dict):
                for k, v in value.items():
                    scan_value(v, f"{path}.{k}")
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    scan_value(v, f"{path}[{i}]")

        scan_value(data, "root")
        return violations
