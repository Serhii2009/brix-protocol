"""StructuredResult and supporting enums.

Every BRIX router call returns a StructuredResult — the complete
decision artifact containing the response, classification, metrics,
and audit identifiers.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from brix.regulated.output.result import OutputResult


class UncertaintyType(StrEnum):
    """Classified uncertainty category for a processed query."""

    CERTAIN = "certain"
    EPISTEMIC = "epistemic"
    CONTRADICTORY = "contradictory"
    OPEN_ENDED = "open_ended"


class ActionTaken(StrEnum):
    """Action executed in response to the classified uncertainty type."""

    NONE = "none"
    FORCE_RETRIEVAL = "force_retrieval"
    CONFLICT_RESOLUTION = "conflict_resolution"
    DISTRIBUTION_RESPONSE = "distribution_response"


class StructuredResult(BaseModel):
    """Complete output artifact produced by the BRIX runtime for every request."""

    decision_id: UUID = Field(default_factory=uuid4, description="Unique decision identifier for audit trail")
    uncertainty_type: UncertaintyType
    subtype: str = Field(default="", description="Internal analytics subtype")
    action_taken: ActionTaken
    response: str
    circuit_breaker_hit: bool
    circuit_breaker_name: str | None = None
    signals_triggered: list[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    reliability_signal: bool = Field(description="Whether this decision contributed to reliability")
    utility_signal: bool = Field(description="Whether this decision preserved utility")
    balance_index: float = Field(ge=0.0, le=1.0, description="Running session Balance Index")
    intervention_necessary: bool
    registry_version: str
    model_compatibility_status: Literal["verified", "community", "untested", "unknown"] = "unknown"
    cost_tokens_extra: int = Field(ge=0, default=0)
    latency_ms: float = Field(ge=0.0)
    response_requires_verification: bool = Field(default=False)
    unverified_draft: str | None = Field(default=None)
    sampler_partial_failure: bool = Field(default=False)
    retrieval_executed: bool = Field(default=False)
    retrieval_failed: bool = Field(default=False)
    retrieval_sources: list[str] = Field(default_factory=list)
    output_result: OutputResult | None = Field(default=None)
