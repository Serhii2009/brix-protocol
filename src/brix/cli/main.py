"""BRIX CLI entry point shim.

Re-exports the ``app`` from ``brix.regulated.cli.main`` so the pyproject.toml
entry point ``brix.cli.main:app`` remains valid after the regulated-module
relocation.
"""

from brix.regulated.cli.main import app

__all__ = ["app"]
