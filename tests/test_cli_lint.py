"""Tests for brix lint CLI command."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from typer.testing import CliRunner

from brix.cli.main import app

runner = CliRunner()


class TestLintCommand:
    def test_valid_spec_passes(self, sample_spec_dict: dict) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_spec_dict, f)
            path = f.name
        result = runner.invoke(app, ["lint", path])
        assert result.exit_code == 0
        assert "Schema validation passed" in result.output
        Path(path).unlink()

    def test_invalid_spec_fails(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"invalid": "spec"}, f)
            path = f.name
        result = runner.invoke(app, ["lint", path])
        assert result.exit_code == 2
        Path(path).unlink()

    def test_conflicting_signals_detected(self) -> None:
        spec = {
            "metadata": {"name": "conflict", "version": "1.0.0", "domain": "test"},
            "circuit_breakers": [
                {"name": "cb1", "patterns": ["shared pattern"]},
            ],
            "risk_signals": [
                {"name": "rs1", "patterns": ["shared pattern"], "weight": 0.5, "category": "registered"},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(spec, f)
            path = f.name
        result = runner.invoke(app, ["lint", path])
        assert result.exit_code == 2
        assert "shared pattern" in result.output.lower() or "conflict" in result.output.lower()
        Path(path).unlink()

    def test_balance_index_estimate_displayed(self, sample_spec_dict: dict) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_spec_dict, f)
            path = f.name
        result = runner.invoke(app, ["lint", path])
        assert "Balance Index" in result.output
        Path(path).unlink()

    def test_builtin_spec_lint(self, builtin_spec_path: Path) -> None:
        result = runner.invoke(app, ["lint", str(builtin_spec_path)])
        # Should pass (exit 0) or have warnings (exit 1), not errors (exit 2)
        assert result.exit_code in (0, 1)
        assert "Balance Index" in result.output
