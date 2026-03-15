"""Pydantic v2 models for the uncertainty.yaml specification schema.

These models define the complete structure of a BRIX specification file,
validated at load time to catch configuration errors before runtime.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelCompatibility(BaseModel):
    """Compatibility record for a specific model family."""

    model_family: str
    min_version: str | None = None
    status: str = Field(
        default="untested",
        pattern=r"^(verified|community|untested|unknown)$",
    )
    notes: str = ""


class Metadata(BaseModel):
    """Specification metadata including identity and compatibility info."""

    name: str
    version: str
    domain: str
    description: str = ""
    model_compatibility: list[ModelCompatibility] = Field(default_factory=list)


class CircuitBreakerDef(BaseModel):
    """Definition of a single circuit breaker rule."""

    name: str
    patterns: list[str] = Field(min_length=1)
    description: str = ""
    exclude_context: list[str] = Field(default_factory=list)


class RiskSignalDef(BaseModel):
    """Definition of a single risk signal with weight and category."""

    name: str
    patterns: list[str] = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    category: str = Field(
        default="registered",
        pattern=r"^(registered|universal)$",
    )
    description: str = ""
    exclude_context: list[str] = Field(default_factory=list)


class UncertaintyActionConfig(BaseModel):
    """Action configuration for a specific uncertainty type."""

    action: str
    message_template: str = ""
    force_retrieval: bool = False


class UncertaintyTypeDef(BaseModel):
    """Definition of an uncertainty type and its associated action."""

    name: str = Field(pattern=r"^(epistemic|contradictory|open_ended)$")
    description: str = ""
    action_config: UncertaintyActionConfig


class SamplingConfig(BaseModel):
    """Configuration for the adaptive sampler tier thresholds."""

    low_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    low_samples: int = Field(default=1, ge=1)
    medium_samples: int = Field(default=2, ge=1)
    high_samples: int = Field(default=3, ge=1)
    circuit_breaker_samples: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class SpecModel(BaseModel):
    """Complete uncertainty.yaml specification model.

    Validated with Pydantic v2 at load time. Immutable after construction.
    """

    metadata: Metadata
    circuit_breakers: list[CircuitBreakerDef] = Field(default_factory=list)
    risk_signals: list[RiskSignalDef] = Field(default_factory=list)
    uncertainty_types: list[UncertaintyTypeDef] = Field(default_factory=list)
    sampling_config: SamplingConfig = Field(default_factory=SamplingConfig)
