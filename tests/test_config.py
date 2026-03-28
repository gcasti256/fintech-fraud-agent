"""Tests for configuration loading."""

from __future__ import annotations

import pytest

from fraud_agent.config import Settings, get_settings


class TestSettings:
    def test_default_values(self):
        settings = Settings()
        assert settings.rest_host == "0.0.0.0"
        assert settings.rest_port == 8000
        assert settings.grpc_port == 50051
        assert settings.database_path == "fraud_agent.db"
        assert settings.high_risk_threshold == 0.8
        assert settings.medium_risk_threshold == 0.5
        assert settings.log_level == "INFO"
        assert settings.llm_model == "gpt-4o-mini"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("REST_PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.rest_port == 9000
        assert settings.log_level == "DEBUG"

    def test_port_validation(self):
        with pytest.raises(Exception):
            Settings(rest_port=0)

    def test_threshold_validation(self):
        with pytest.raises(Exception):
            Settings(high_risk_threshold=1.5)

    def test_get_settings_cached(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()
