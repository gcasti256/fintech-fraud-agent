"""Tests for the CLI."""

from __future__ import annotations

import json
import tempfile

from click.testing import CliRunner

from fraud_agent.cli import cli


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Fraud Detection Agent" in result.output

    def test_score_command(self):
        runner = CliRunner()
        txn = json.dumps(
            {
                "amount": 42.50,
                "merchant_name": "Store",
                "merchant_category_code": "5411",
                "card_last_four": "1234",
                "channel": "IN_STORE",
            }
        )
        result = runner.invoke(cli, ["score", "-t", txn])
        assert result.exit_code == 0, result.output
        assert "Fraud Analysis Result" in result.output

    def test_score_invalid_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["score", "-t", "not json"])
        assert result.exit_code != 0

    def test_generate_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "-n", "5", "-s", "42"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 5

    def test_generate_to_file(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        result = runner.invoke(cli, ["generate", "-n", "3", "-o", path])
        assert result.exit_code == 0
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_patterns_list(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["patterns", "list"])
        assert result.exit_code == 0
        assert "Fraud Patterns" in result.output

    def test_metrics_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["metrics"])
        assert result.exit_code == 0

    def test_cli_version(self):
        """--version outputs the package version (alias for test_version)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_generate(self):
        """generate creates the requested number of transactions (alias for test_generate_command)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "-n", "3", "-s", "0"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 3

    def test_cli_patterns_list(self):
        """patterns list shows fraud patterns (alias for test_patterns_list)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["patterns", "list"])
        assert result.exit_code == 0
        assert "Fraud Patterns" in result.output

    def test_generate_transaction_has_required_fields(self):
        """Generated transactions each have id, amount, merchant_name, and channel."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "-n", "2", "-s", "11"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for txn in data:
            assert "id" in txn
            assert "amount" in txn
            assert "merchant_name" in txn
            assert "channel" in txn

    def test_patterns_list_shows_pattern_names(self):
        """patterns list output includes at least one known pattern name."""
        runner = CliRunner()
        result = runner.invoke(cli, ["patterns", "list"])
        assert result.exit_code == 0
        assert any(kw in result.output for kw in ("Card", "Account", "Velocity", "Testing"))
