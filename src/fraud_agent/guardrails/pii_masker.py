"""PII detection and masking for the fraud detection system.

All data that flows through logging, audit trails, model prompts, or external
APIs must be sanitised by :class:`PIIMasker` before transmission.  The masker
uses compiled regex patterns to detect and replace sensitive identifiers in
both free-text and structured dict payloads.

Supported PII types
-------------------
- Payment card numbers (16-digit, hyphenated or spaced)
- US Social Security Numbers (XXX-XX-XXXX)
- Internal account IDs (ACC-XXXX-XXXX format)
- Email addresses

The masking strategy intentionally preserves partial context (first-4 / last-4
for cards, domain for emails) so that logs remain useful for debugging while
removing the identifying data that would pose a privacy risk.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from fraud_agent.data.schemas import Transaction


class PIIMasker:
    """Regex-based PII detector and masker.

    All regex patterns are compiled once at class instantiation.  The masker
    is stateless and thread-safe — a single shared instance can serve the
    entire application.

    Example::

        masker = PIIMasker()
        safe_text = masker.mask_text(
            "Card 4532-1234-5678-9012 was charged $500"
        )
        # -> "Card 4532-****-****-9012 was charged $500"
    """

    # ------------------------------------------------------------------
    # Compiled PII patterns
    # ------------------------------------------------------------------

    # 16-digit card number: accepts hyphens or spaces as separators.
    _CARD_RE: re.Pattern[str] = re.compile(r"\b(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})\b")

    # US Social Security Number: XXX-XX-XXXX (hyphens or spaces).
    _SSN_RE: re.Pattern[str] = re.compile(r"\b(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})\b")

    # Internal account ID: ACC-XXXX-XXXX (flexible separator / no separator).
    _ACCOUNT_RE: re.Pattern[str] = re.compile(r"\b(ACC)[-\s]?(\w{4})[-\s]?(\w{4})\b", re.IGNORECASE)

    # Email address (simplified RFC-5321 subset).
    _EMAIL_RE: re.Pattern[str] = re.compile(
        r"\b([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"
    )

    # ------------------------------------------------------------------
    # Per-type masking
    # ------------------------------------------------------------------

    def mask_card_number(self, card_number: str) -> str:
        """Mask a payment card number, keeping the first and last four digits.

        Accepts hyphenated (``4532-1234-5678-9012``), spaced
        (``4532 1234 5678 9012``), or plain (``4532123456789012``) formats.

        Args:
            card_number: Raw card number string.

        Returns:
            Masked string in ``XXXX-****-****-XXXX`` format, or the original
            string unchanged if it does not match a 16-digit card pattern.
        """
        match = self._CARD_RE.search(card_number)
        if not match:
            return card_number
        first4 = match.group(1)
        last4 = match.group(4)
        return f"{first4}-****-****-{last4}"

    def mask_ssn(self, ssn: str) -> str:
        """Mask a US Social Security Number, keeping only the last four digits.

        Args:
            ssn: Raw SSN string (``XXX-XX-XXXX``, ``XXX XX XXXX``, or
                ``XXXXXXXXX``).

        Returns:
            Masked string in ``***-**-XXXX`` format, or the original string
            if it does not match the SSN pattern.
        """
        match = self._SSN_RE.search(ssn)
        if not match:
            return ssn
        last4 = match.group(3)
        return f"***-**-{last4}"

    def mask_account_id(self, account_id: str) -> str:
        """Mask an internal account ID, keeping the ``ACC`` prefix and last four chars.

        Args:
            account_id: Raw account ID (e.g. ``ACC-AB12-CD34``).

        Returns:
            Masked string in ``ACC-****-XXXX`` format, or the original string
            if it does not match the account ID pattern.
        """
        match = self._ACCOUNT_RE.search(account_id)
        if not match:
            return account_id
        last4 = match.group(3)
        return f"ACC-****-{last4}"

    def mask_email(self, email: str) -> str:
        """Mask an email address, keeping the first character and the domain.

        Args:
            email: Raw email address (e.g. ``george@example.com``).

        Returns:
            Masked string in ``g***@example.com`` format, or the original
            string if it does not match the email pattern.
        """
        match = self._EMAIL_RE.search(email)
        if not match:
            return email
        local = match.group(1)
        domain = match.group(2)
        masked_local = local[0] + "***" if local else "***"
        return f"{masked_local}@{domain}"

    # ------------------------------------------------------------------
    # Text and dict masking
    # ------------------------------------------------------------------

    def mask_text(self, text: str) -> str:
        """Scan *text* for PII patterns and replace all matches in-place.

        The masking order is:
        1. Card numbers (applied first to avoid misidentifying card digits as SSNs)
        2. SSNs
        3. Account IDs
        4. Emails

        Args:
            text: Arbitrary free-text string that may contain PII.

        Returns:
            Copy of *text* with all detected PII replaced by masked tokens.
        """
        result = text

        # 1. Card numbers — must come before SSN to avoid partial matches.
        result = self._CARD_RE.sub(lambda m: f"{m.group(1)}-****-****-{m.group(4)}", result)

        # 2. SSNs.
        result = self._SSN_RE.sub(lambda m: f"***-**-{m.group(3)}", result)

        # 3. Account IDs.
        result = self._ACCOUNT_RE.sub(lambda m: f"ACC-****-{m.group(3)}", result)

        # 4. Emails.
        result = self._EMAIL_RE.sub(
            lambda m: (m.group(1)[0] + "***" if m.group(1) else "***") + "@" + m.group(2),
            result,
        )

        return result

    def mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively mask all string values in *data*.

        Creates a deep copy of *data* so the original is never mutated.
        Non-string scalar values (int, float, bool, None) are passed through
        unchanged.  Lists are recursed into element-by-element.

        Args:
            data: Arbitrary nested dict.

        Returns:
            Deep copy of *data* with all string leaves masked.
        """
        return self._mask_value(copy.deepcopy(data))

    def _mask_value(self, value: Any) -> Any:
        """Recursively mask a single value, dispatching on its type.

        Args:
            value: Any Python value.

        Returns:
            Masked equivalent.
        """
        if isinstance(value, str):
            return self.mask_text(value)
        if isinstance(value, dict):
            return {k: self._mask_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._mask_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._mask_value(item) for item in value)
        # Scalars (int, float, bool, None, Decimal, datetime, …) pass through.
        return value

    def mask_transaction(self, transaction: Transaction) -> dict[str, Any]:
        """Serialise *transaction* to a dict with all PII fields masked.

        Fields masked:
        - ``account_id`` → masked account ID
        - ``card_last_four`` → replaced with ``****``
        - ``merchant_name`` → passed through mask_text (may contain email etc.)
        - ``metadata`` → recursively masked

        All other fields (amount, timestamp, location, channel, etc.) are
        preserved verbatim as they are needed for fraud analysis.

        Args:
            transaction: The transaction to serialise and mask.

        Returns:
            Dict representation of the transaction with PII masked.
        """
        raw = transaction.model_dump(mode="json")

        # Mask account_id in place.
        raw["account_id"] = self.mask_account_id(str(raw.get("account_id", "")))

        # Replace card_last_four entirely — it provides no legitimate context
        # in a log entry (we only need to confirm the card field existed).
        raw["card_last_four"] = "****"

        # Merchant name may contain embedded contact info.
        raw["merchant_name"] = self.mask_text(str(raw.get("merchant_name", "")))

        # Recursively mask metadata which is caller-controlled and may contain
        # arbitrary PII fields.
        if raw.get("metadata"):
            raw["metadata"] = self._mask_value(raw["metadata"])

        return raw
