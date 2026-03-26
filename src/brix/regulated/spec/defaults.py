"""Default paths and constants for built-in BRIX specifications."""

from __future__ import annotations

import importlib.resources
from pathlib import Path


def get_default_spec_path() -> Path:
    """Return the filesystem path to the built-in general v1.0.0 spec.

    Uses importlib.resources to locate the spec file within the installed
    package, ensuring it works after pip install (not just from source).
    """
    ref = importlib.resources.files("brix.regulated.specs.general") / "v1.0.0.yaml"
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def get_medical_spec_path() -> Path:
    """Return the filesystem path to the built-in medical v1.0.0 spec."""
    ref = importlib.resources.files("brix.regulated.specs.medical") / "v1.0.0.yaml"
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def get_legal_spec_path() -> Path:
    """Return the filesystem path to the built-in legal v1.0.0 spec."""
    ref = importlib.resources.files("brix.regulated.specs.legal") / "v1.0.0.yaml"
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def get_finance_spec_path() -> Path:
    """Return the filesystem path to the built-in finance v1.0.0 spec."""
    ref = importlib.resources.files("brix.regulated.specs.finance") / "v1.0.0.yaml"
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def get_hr_spec_path() -> Path:
    """Return the filesystem path to the built-in HR v1.0.0 spec."""
    ref = importlib.resources.files("brix.regulated.specs.hr") / "v1.0.0.yaml"
    with importlib.resources.as_file(ref) as path:
        return Path(path)


DEFAULT_SPEC_VERSION: str = "general/v1.0.0"
MEDICAL_SPEC_PATH: str = "medical/v1.0.0"
LEGAL_SPEC_PATH: str = "legal/v1.0.0"
FINANCE_SPEC_PATH: str = "finance/v1.0.0"
HR_SPEC_PATH: str = "hr/v1.0.0"
