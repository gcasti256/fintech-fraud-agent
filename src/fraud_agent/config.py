"""Application configuration using pydantic-settings.

All settings are loaded from environment variables (or an optional .env file).
Callers should use :func:`get_settings` to obtain a cached instance rather than
constructing :class:`Settings` directly.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attribute names map directly to environment variable names (case-insensitive).
    For example, ``rest_host`` is populated from the ``REST_HOST`` env var.
    """

    # ---------------------------------------------------------------------------
    # API
    # ---------------------------------------------------------------------------
    rest_host: str = Field(default="0.0.0.0", description="REST API bind address.")
    rest_port: int = Field(default=8000, description="REST API port.", ge=1, le=65535)
    grpc_host: str = Field(default="0.0.0.0", description="gRPC server bind address.")
    grpc_port: int = Field(default=50051, description="gRPC server port.", ge=1, le=65535)

    # ---------------------------------------------------------------------------
    # Database
    # ---------------------------------------------------------------------------
    database_path: str = Field(
        default="fraud_agent.db",
        description="Filesystem path for the SQLite database file.",
    )

    # ---------------------------------------------------------------------------
    # Scoring thresholds
    # ---------------------------------------------------------------------------
    high_risk_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraud score at or above which a transaction is classified HIGH risk.",
    )
    medium_risk_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraud score at or above which a transaction is classified MEDIUM risk.",
    )
    low_risk_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraud score at or above which a transaction is classified LOW risk.",
    )

    # ---------------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_format: str = Field(
        default="json",
        description="Log output format: 'json' for structured logging, 'text' for human-readable.",
    )

    # ---------------------------------------------------------------------------
    # LLM (optional — agents work without it using rule-based fallback)
    # ---------------------------------------------------------------------------
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Leave empty to disable LLM-powered agents.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat completion model identifier.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier used for RAG retrieval.",
    )

    # ---------------------------------------------------------------------------
    # PII
    # ---------------------------------------------------------------------------
    pii_encryption_key: str = Field(
        default="",
        description=(
            "Fernet-compatible base64-encoded 32-byte key used to encrypt PII at rest. "
            'Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        ),
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    The instance is constructed once on first call and reused on subsequent
    calls, avoiding repeated env-var reads in hot paths.

    Returns:
        Settings: The application settings singleton.
    """
    return Settings()
