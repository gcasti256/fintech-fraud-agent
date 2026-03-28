"""Guardrails sub-package for the fraud detection system.

Provides three complementary layers of operational safety:

PIIMasker
    Regex-based PII detection and masking for card numbers, SSNs,
    account IDs, and email addresses.  Used to sanitise data before
    logging, external API calls, or model prompts.

AuditLogger
    Structured, tamper-evident JSON audit log with SHA-256 chain
    hashing for fraud decisions, data access events, manual overrides,
    and system events.

ComplianceChecker
    Rule-based compliance validator that flags transactions for CTR
    reporting (>$10k), structuring patterns, international wire
    transfers, and sanctioned-country activity.
"""

from .audit_logger import AuditLogger
from .compliance import ComplianceChecker
from .pii_masker import PIIMasker

__all__ = ["AuditLogger", "ComplianceChecker", "PIIMasker"]
