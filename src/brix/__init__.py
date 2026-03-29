"""BRIX — Runtime Reliability Infrastructure for LLM Pipelines.

BRIX wraps any LLM client with a configurable chain of Guards, each solving
exactly one production failure mode. One wrap() call. Zero hidden coupling.

Quick start::

    from brix import BRIX

    client = BRIX.wrap(
        openai.OpenAI(),
        regulated_spec="medical",   # activates RegulatedGuard
    )
    response = await client.complete([{"role": "user", "content": "..."}])

For regulated-domain use (fintech, medtech, legal)::

    from brix.regulated import BrixRouter

    router = BrixRouter(my_llm, spec="medical")
    result = await router.process("What is the maximum safe aspirin dose?")
"""

from brix.client import BRIX, BrixClient
from brix.exceptions import (
    BrixBudgetError,
    BrixConfigurationError,
    BrixError,
    BrixGuardBlockedError,
    BrixGuardError,
    BrixInternalError,
    BrixLoopError,
    BrixRateLimitError,
    BrixReplayError,
    BrixSchemaError,
    BrixTimeoutError,
)

# Backward-compatible re-exports from the regulated module.
# These were top-level symbols in brix <= 0.3.0 and remain available here.
from brix.regulated import (
    ActionTaken,
    BrixRouter,
    CircuitBreakerError,
    ClassifierError,
    FINANCE_SPEC_PATH,
    HR_SPEC_PATH,
    LEGAL_SPEC_PATH,
    LLMClient,
    MEDICAL_SPEC_PATH,
    MockLLMClient,
    OutputGuard,
    OutputResult,
    RegistryError,
    RegulatedGuard,
    RetrievalProvider,
    RetrievalResult,
    SamplerError,
    SpecModel,
    SpecValidationError,
    StructuredResult,
    UncertaintyType,
    load_spec,
    load_spec_from_dict,
)

__version__ = "0.5.0"

__all__ = [
    # New public API
    "BRIX",
    "BrixClient",
    # Exception hierarchy
    "BrixError",
    "BrixBudgetError",
    "BrixConfigurationError",
    "BrixGuardBlockedError",
    "BrixGuardError",
    "BrixInternalError",
    "BrixLoopError",
    "BrixRateLimitError",
    "BrixReplayError",
    "BrixSchemaError",
    "BrixTimeoutError",
    # Regulated module re-exports (backward compat)
    "ActionTaken",
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
