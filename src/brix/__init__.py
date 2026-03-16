"""BRIX — Runtime Reliability Infrastructure for LLM Pipelines.

BRIX wraps any LLM client and enforces deterministic reliability rules
defined in a declarative uncertainty.yaml specification, while measuring
the Balance Index across all interactions.
"""

from brix.core.exceptions import (
    BrixError,
    CircuitBreakerError,
    ClassifierError,
    RegistryError,
    SamplerError,
    SpecValidationError,
)
from brix.core.result import ActionTaken, StructuredResult, UncertaintyType
from brix.core.router import BrixRouter
from brix.llm.mock import MockLLMClient
from brix.llm.protocol import LLMClient
from brix.spec.defaults import (
    FINANCE_SPEC_PATH,
    HR_SPEC_PATH,
    LEGAL_SPEC_PATH,
    MEDICAL_SPEC_PATH,
)
from brix.spec.loader import load_spec, load_spec_from_dict
from brix.spec.models import SpecModel

__all__ = [
    "ActionTaken",
    "BrixError",
    "BrixRouter",
    "CircuitBreakerError",
    "ClassifierError",
    "FINANCE_SPEC_PATH",
    "HR_SPEC_PATH",
    "LEGAL_SPEC_PATH",
    "LLMClient",
    "MEDICAL_SPEC_PATH",
    "MockLLMClient",
    "RegistryError",
    "SamplerError",
    "SpecModel",
    "SpecValidationError",
    "StructuredResult",
    "UncertaintyType",
    "load_spec",
    "load_spec_from_dict",
]

__version__ = "0.2.0"
