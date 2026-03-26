"""BRIX Regulated Guard — reliability infrastructure for regulated domains.

This package contains the full two-track evaluation engine (circuit breakers,
risk scoring, semantic consistency analysis) originally designed for fintech,
medtech, and legal use cases.

Install with the ``regulated`` extra::

    pip install "brix-protocol[regulated]"

Basic usage::

    from brix import BRIX

    client = BRIX.wrap(my_llm, regulated_spec="medical")
    result = await client.complete([{"role": "user", "content": "..."}])

Or use BrixRouter directly::

    from brix.regulated import BrixRouter

    router = BrixRouter(my_llm, spec="medical")
    result = await router.process("What is the lethal dose of aspirin?")
"""

from brix.regulated._guard import RegulatedGuard
from brix.regulated.core.exceptions import (
    BrixError,
    CircuitBreakerError,
    ClassifierError,
    RegistryError,
    SamplerError,
    SpecValidationError,
)
from brix.regulated.core.result import ActionTaken, StructuredResult, UncertaintyType
from brix.regulated.core.router import BrixRouter
from brix.regulated.llm.mock import MockLLMClient
from brix.regulated.llm.protocol import LLMClient
from brix.regulated.output import OutputGuard, OutputResult
from brix.regulated.retrieval import RetrievalProvider, RetrievalResult
from brix.regulated.spec.defaults import (
    FINANCE_SPEC_PATH,
    HR_SPEC_PATH,
    LEGAL_SPEC_PATH,
    MEDICAL_SPEC_PATH,
)
from brix.regulated.spec.loader import load_spec, load_spec_from_dict
from brix.regulated.spec.models import SpecModel

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
    "OutputGuard",
    "OutputResult",
    "RegistryError",
    "RegulatedGuard",
    "RetrievalProvider",
    "RetrievalResult",
    "SamplerError",
    "SpecModel",
    "SpecValidationError",
    "StructuredResult",
    "UncertaintyType",
    "load_spec",
    "load_spec_from_dict",
]
