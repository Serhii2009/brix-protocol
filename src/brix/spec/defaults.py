"""Default paths and constants for built-in BRIX specifications."""

from __future__ import annotations

import importlib.resources
from pathlib import Path


def get_default_spec_path() -> Path:
    """Return the filesystem path to the built-in general v1.0.0 spec.

    Uses importlib.resources to locate the spec file within the installed
    package, ensuring it works after pip install (not just from source).
    """
    ref = importlib.resources.files("brix.specs.general") / "v1.0.0.yaml"
    # as_posix works for traversable; for installed packages we need the real path
    with importlib.resources.as_file(ref) as path:
        # Return a concrete Path — the context manager ensures the file exists
        return Path(path)


DEFAULT_SPEC_VERSION: str = "general/v1.0.0"
