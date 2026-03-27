"""Tests for spec loading and Pydantic validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from brix.regulated.core.exceptions import SpecValidationError
from brix.regulated.spec.loader import load_spec, load_spec_from_dict
from brix.regulated.spec.models import SpecModel


class TestLoadSpecFromDict:
    def test_valid_spec(self, sample_spec_dict: dict) -> None:
        spec = load_spec_from_dict(sample_spec_dict)
        assert isinstance(spec, SpecModel)
        assert spec.metadata.name == "test-spec"
        assert spec.metadata.version == "1.0.0"
        assert len(spec.circuit_breakers) == 2
        assert len(spec.risk_signals) == 4

    def test_missing_metadata_raises(self) -> None:
        with pytest.raises(SpecValidationError, match="validation failed"):
            load_spec_from_dict({"circuit_breakers": []})

    def test_missing_name_raises(self) -> None:
        with pytest.raises(SpecValidationError):
            load_spec_from_dict(
                {
                    "metadata": {"version": "1.0.0", "domain": "test"},
                }
            )

    def test_empty_cb_patterns_raises(self) -> None:
        with pytest.raises(SpecValidationError):
            load_spec_from_dict(
                {
                    "metadata": {"name": "t", "version": "1", "domain": "t"},
                    "circuit_breakers": [{"name": "bad", "patterns": []}],
                }
            )

    def test_weight_out_of_range_raises(self) -> None:
        with pytest.raises(SpecValidationError):
            load_spec_from_dict(
                {
                    "metadata": {"name": "t", "version": "1", "domain": "t"},
                    "risk_signals": [
                        {"name": "bad", "patterns": ["x"], "weight": 1.5, "category": "registered"}
                    ],
                }
            )

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(SpecValidationError):
            load_spec_from_dict(
                {
                    "metadata": {"name": "t", "version": "1", "domain": "t"},
                    "risk_signals": [
                        {"name": "bad", "patterns": ["x"], "weight": 0.5, "category": "invalid"}
                    ],
                }
            )

    def test_defaults_applied(self) -> None:
        spec = load_spec_from_dict(
            {
                "metadata": {"name": "t", "version": "1", "domain": "t"},
            }
        )
        assert spec.sampling_config.low_threshold == 0.40
        assert spec.sampling_config.high_samples == 3


class TestLoadSpecFromFile:
    def test_valid_yaml_file(self, sample_spec_dict: dict) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_spec_dict, f)
            path = Path(f.name)
        spec = load_spec(path)
        assert spec.metadata.name == "test-spec"
        path.unlink()

    def test_file_not_found(self) -> None:
        with pytest.raises(SpecValidationError, match="not found"):
            load_spec("/nonexistent/path.yaml")

    def test_invalid_yaml_syntax(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: {\n  broken")
            path = Path(f.name)
        with pytest.raises(SpecValidationError, match="Invalid YAML"):
            load_spec(path)
        path.unlink()

    def test_non_mapping_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- just\n- a\n- list\n")
            path = Path(f.name)
        with pytest.raises(SpecValidationError, match="mapping"):
            load_spec(path)
        path.unlink()

    def test_builtin_spec_loads(self, builtin_spec_path: Path) -> None:
        spec = load_spec(builtin_spec_path)
        assert spec.metadata.name == "general"
        assert len(spec.circuit_breakers) >= 3
        assert len(spec.risk_signals) >= 6
