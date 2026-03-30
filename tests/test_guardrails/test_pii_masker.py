"""Tests for PIIMasker: card numbers, SSNs, account IDs, emails, text, dicts, transactions."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from fraud_agent.data.schemas import Location, Transaction, TransactionChannel
from fraud_agent.guardrails.pii_masker import PIIMasker


@pytest.fixture()
def masker():
    return PIIMasker()


class TestMaskCardNumber:
    def test_mask_card_number_with_dashes(self, masker):
        """Hyphenated card number: keeps first 4 and last 4."""
        result = masker.mask_card_number("4532-1234-5678-7890")
        assert result == "4532-****-****-7890"

    def test_mask_card_number_no_dashes(self, masker):
        """Plain 16-digit card number is masked correctly."""
        result = masker.mask_card_number("4532123456787890")
        assert result == "4532-****-****-7890"

    def test_mask_card_number_with_spaces(self, masker):
        """Space-separated card number is masked correctly."""
        result = masker.mask_card_number("4532 1234 5678 7890")
        assert result == "4532-****-****-7890"

    def test_mask_card_number_preserves_first_four(self, masker):
        result = masker.mask_card_number("9999-0000-1111-2222")
        assert result.startswith("9999-")

    def test_mask_card_number_preserves_last_four(self, masker):
        result = masker.mask_card_number("9999-0000-1111-2222")
        assert result.endswith("-2222")

    def test_mask_card_number_no_match_unchanged(self, masker):
        result = masker.mask_card_number("not-a-card")
        assert result == "not-a-card"


class TestMaskSSN:
    def test_mask_ssn(self, masker):
        result = masker.mask_ssn("123-45-6789")
        assert result == "***-**-6789"

    def test_mask_ssn_preserves_last_four(self, masker):
        result = masker.mask_ssn("987-65-4321")
        assert result.endswith("4321")

    def test_mask_ssn_no_match_unchanged(self, masker):
        result = masker.mask_ssn("hello world")
        assert result == "hello world"


class TestMaskAccountId:
    def test_mask_account_id(self, masker):
        result = masker.mask_account_id("ACC-8821-4567")
        assert result == "ACC-****-4567"

    def test_mask_account_id_preserves_prefix(self, masker):
        result = masker.mask_account_id("ACC-0001-1234")
        assert result.startswith("ACC-")

    def test_mask_account_id_preserves_last_four(self, masker):
        result = masker.mask_account_id("ACC-0001-1234")
        assert result.endswith("1234")

    def test_mask_account_id_no_match_unchanged(self, masker):
        result = masker.mask_account_id("random string")
        assert result == "random string"


class TestMaskEmail:
    def test_mask_email(self, masker):
        result = masker.mask_email("george@example.com")
        assert result == "g***@example.com"

    def test_mask_email_preserves_domain(self, masker):
        result = masker.mask_email("user@mybank.co.uk")
        assert "@mybank.co.uk" in result

    def test_mask_email_first_char_kept(self, masker):
        result = masker.mask_email("alice@example.org")
        assert result.startswith("a")

    def test_mask_email_no_match_unchanged(self, masker):
        result = masker.mask_email("not an email")
        assert result == "not an email"


class TestMaskText:
    def test_mask_text_with_card(self, masker):
        text = "Transaction on card 4532-1234-5678-9012 was declined."
        result = masker.mask_text(text)
        assert "1234-5678" not in result
        assert "4532" in result
        assert "9012" in result

    def test_mask_text_with_ssn(self, masker):
        text = "Customer SSN: 123-45-6789 on file."
        result = masker.mask_text(text)
        assert "123-45" not in result
        assert "6789" in result

    def test_mask_text_no_pii(self, masker):
        text = "Transaction amount was $500 at a store in New York."
        result = masker.mask_text(text)
        assert result == text

    def test_mask_text_with_account_id(self, masker):
        text = "Account ACC-8821-4567 flagged for review."
        result = masker.mask_text(text)
        assert "8821" not in result
        assert "ACC-" in result

    def test_mask_text_with_email(self, masker):
        text = "Notification sent to george@example.com."
        result = masker.mask_text(text)
        assert "george@" not in result
        assert "@example.com" in result

    def test_mask_text_multiple_pii_types(self, masker):
        text = "Card 4532-1234-5678-9012 belongs to ACC-0001-1234, email bob@test.com."
        result = masker.mask_text(text)
        assert "1234-5678" not in result
        assert "bob@" not in result


class TestMaskDict:
    def test_mask_dict(self, masker):
        data = {
            "account_id": "ACC-0001-1234",
            "email": "george@example.com",
            "nested": {"card": "4532-1234-5678-9012"},
            "amount": 500,
        }
        result = masker.mask_dict(data)
        assert "0001-1234" not in result["account_id"]
        assert result["account_id"].startswith("ACC-")
        assert "george@" not in result["email"]
        assert "1234-5678" not in result["nested"]["card"]
        assert result["amount"] == 500

    def test_mask_dict_does_not_mutate_original(self, masker):
        data = {"email": "george@example.com"}
        masker.mask_dict(data)
        assert data["email"] == "george@example.com"

    def test_mask_dict_list_values(self, masker):
        data = {"emails": ["alice@example.com", "bob@example.com"]}
        result = masker.mask_dict(data)
        for email in result["emails"]:
            assert "@example.com" in email
            assert "alice@" not in email
            assert "bob@" not in email


class TestMaskTransaction:
    def test_mask_transaction(self, masker):
        txn = Transaction(
            id="txn-pii-001",
            timestamp=datetime.now(UTC),
            amount=Decimal("50.00"),
            currency="USD",
            merchant_name="Store",
            merchant_category_code="5411",
            card_last_four="9999",
            account_id="ACC-8821-4567",
            location=Location(city="NYC", country="US", latitude=40.7, longitude=-74.0),
            channel=TransactionChannel.ONLINE,
        )
        result = masker.mask_transaction(txn)
        assert isinstance(result, dict)
        assert "8821" not in result["account_id"]
        assert result["account_id"].startswith("ACC-")
        assert result["card_last_four"] == "****"

    def test_mask_transaction_preserves_amount(self, masker):
        txn = Transaction(
            id="txn-pii-002",
            timestamp=datetime.now(UTC),
            amount=Decimal("123.45"),
            currency="USD",
            merchant_name="Test",
            merchant_category_code="5411",
            card_last_four="1234",
            account_id="ACC-AB12-CD34",
            location=Location(city="NYC", country="US", latitude=40.7, longitude=-74.0),
            channel=TransactionChannel.IN_STORE,
        )
        result = masker.mask_transaction(txn)
        assert str(result["amount"]) == "123.45"

    def test_mask_transaction_metadata_masked(self, masker):
        txn = Transaction(
            id="txn-pii-003",
            timestamp=datetime.now(UTC),
            amount=Decimal("50.00"),
            currency="USD",
            merchant_name="Store",
            merchant_category_code="5411",
            card_last_four="1234",
            account_id="ACC-AB12-CD34",
            location=Location(city="NYC", country="US", latitude=40.7, longitude=-74.0),
            channel=TransactionChannel.ONLINE,
            metadata={"contact": "alice@example.com", "ref": "ACC-0001-1234"},
        )
        result = masker.mask_transaction(txn)
        assert "alice@" not in result["metadata"]["contact"]
        assert "0001-1234" not in result["metadata"]["ref"]
