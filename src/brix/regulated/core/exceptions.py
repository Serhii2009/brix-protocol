"""BRIX regulated-module exception hierarchy.

All regulated exceptions inherit from the top-level ``brix.exceptions.BrixError``
so engineers can catch the full library family with a single except clause.
"""

from brix.exceptions import BrixError as BrixError  # re-export for backward compat


class SpecValidationError(BrixError):
    """Raised when an uncertainty.yaml specification fails validation."""


class CircuitBreakerError(BrixError):
    """Raised when a circuit breaker evaluation encounters an internal error."""


class SamplerError(BrixError):
    """Raised when the adaptive sampler fails to collect samples."""


class ClassifierError(BrixError):
    """Raised when uncertainty classification encounters an internal error."""


class RegistryError(BrixError):
    """Raised when a registry operation fails."""
