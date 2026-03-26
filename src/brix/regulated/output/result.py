"""OutputResult — structured result of output-side response analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutputResult(BaseModel):
    """Result of output guard analysis on a model response."""

    output_blocked: bool = Field(default=False)
    output_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    output_signals_triggered: list[str] = Field(default_factory=list)
    output_block_signal: str | None = Field(default=None)
    output_risk_breakdown: dict[str, float] = Field(default_factory=dict)
    response_safe_to_display: bool = Field(default=True)
