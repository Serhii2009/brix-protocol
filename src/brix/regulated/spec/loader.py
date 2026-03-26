"""Specification loader: YAML parsing and Pydantic v2 validation.

Loads an uncertainty.yaml file from disk (or a raw dict) and returns
a fully validated SpecModel instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from brix.regulated.core.exceptions import SpecValidationError
from brix.regulated.spec.models import SpecModel


def load_spec(source: str | Path) -> SpecModel:
    """Load and validate a BRIX specification from a YAML file.

    Args:
        source: Path to an uncertainty.yaml file.

    Returns:
        A validated SpecModel instance.

    Raises:
        SpecValidationError: If the file cannot be read or fails validation.
    """
    path = Path(source)
    if not path.exists():
        raise SpecValidationError(f"Specification file not found: {path}")
    if not path.is_file():
        raise SpecValidationError(f"Specification path is not a file: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecValidationError(f"Failed to read specification file: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise SpecValidationError(f"Invalid YAML syntax: {exc}") from exc

    if not isinstance(data, dict):
        raise SpecValidationError("Specification must be a YAML mapping at the top level")

    return load_spec_from_dict(data)


def load_spec_from_dict(data: dict[str, Any]) -> SpecModel:
    """Validate a raw dictionary as a BRIX specification.

    Args:
        data: Dictionary parsed from YAML or constructed programmatically.

    Returns:
        A validated SpecModel instance.

    Raises:
        SpecValidationError: If the data fails Pydantic validation.
    """
    try:
        return SpecModel.model_validate(data)
    except ValidationError as exc:
        raise SpecValidationError(f"Specification validation failed:\n{exc}") from exc
