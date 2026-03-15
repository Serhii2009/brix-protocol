"""Tests for brix generate-tests CLI command."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from typer.testing import CliRunner

from brix.cli.main import app

runner = CliRunner()


class TestGenerateTestsCommand:
    def test_generates_test_suite(self, sample_spec_dict: dict, tmp_path) -> None:
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(sample_spec_dict))

        output_dir = tmp_path / "output"
        result = runner.invoke(app, [
            "generate-tests", str(spec_file), "--output", str(output_dir),
        ])
        assert result.exit_code == 0

        # Check output file exists
        output_files = list(output_dir.glob("*.yaml"))
        assert len(output_files) == 1

        # Load and validate
        suite = yaml.safe_load(output_files[0].read_text())
        assert "metadata" in suite
        assert "test_cases" in suite
        assert suite["metadata"]["status"] == "draft"
        assert len(suite["test_cases"]) >= 20

    def test_generates_cb_positive_cases(self, sample_spec_dict: dict, tmp_path) -> None:
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(sample_spec_dict))
        output_dir = tmp_path / "output"
        runner.invoke(app, ["generate-tests", str(spec_file), "--output", str(output_dir)])

        suite = yaml.safe_load(list(output_dir.glob("*.yaml"))[0].read_text())
        cb_positive = [c for c in suite["test_cases"] if c["category"] == "circuit_breaker_positive"]
        assert len(cb_positive) > 0
        for case in cb_positive:
            assert case["expected_intervention"] is True
            assert case["expected_circuit_breaker"] is True

    def test_generates_cb_negative_cases(self, sample_spec_dict: dict, tmp_path) -> None:
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(sample_spec_dict))
        output_dir = tmp_path / "output"
        runner.invoke(app, ["generate-tests", str(spec_file), "--output", str(output_dir)])

        suite = yaml.safe_load(list(output_dir.glob("*.yaml"))[0].read_text())
        cb_negative = [c for c in suite["test_cases"] if c["category"] == "circuit_breaker_negative"]
        assert len(cb_negative) > 0
        for case in cb_negative:
            assert case["expected_intervention"] is False

    def test_generates_safe_passthrough(self, sample_spec_dict: dict, tmp_path) -> None:
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(sample_spec_dict))
        output_dir = tmp_path / "output"
        runner.invoke(app, ["generate-tests", str(spec_file), "--output", str(output_dir)])

        suite = yaml.safe_load(list(output_dir.glob("*.yaml"))[0].read_text())
        safe = [c for c in suite["test_cases"] if c["category"] == "safe_passthrough"]
        assert len(safe) >= 3

    def test_invalid_spec_fails(self, tmp_path) -> None:
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text("invalid: spec")
        result = runner.invoke(app, ["generate-tests", str(spec_file), "--output", str(tmp_path / "out")])
        assert result.exit_code == 2

    def test_builtin_spec_generates(self, builtin_spec_path: Path, tmp_path) -> None:
        output_dir = tmp_path / "output"
        result = runner.invoke(app, [
            "generate-tests", str(builtin_spec_path), "--output", str(output_dir),
        ])
        assert result.exit_code == 0
        output_files = list(output_dir.glob("*.yaml"))
        assert len(output_files) == 1
        suite = yaml.safe_load(output_files[0].read_text())
        assert len(suite["test_cases"]) >= 20
