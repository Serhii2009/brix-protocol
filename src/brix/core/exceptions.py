"""BRIX exception hierarchy.

All BRIX-specific exceptions inherit from BrixError, enabling callers
to catch the full family with a single except clause when desired.
"""


class BrixError(Exception):
    """Base exception for all BRIX errors."""


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
