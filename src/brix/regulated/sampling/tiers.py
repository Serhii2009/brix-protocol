"""Risk tier definitions and tier-to-sample-count mapping."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


from brix.regulated.spec.models import SamplingConfig

__all__ = ["RiskTier", "SamplingConfig", "determine_tier", "samples_for_tier"]


class RiskTier(StrEnum):
    """Risk tier determined by risk score or circuit breaker hit."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CIRCUIT_BREAKER = "circuit_breaker"


def determine_tier(
    risk_score: float, circuit_breaker_hit: bool, config: SamplingConfig
) -> RiskTier:
    """Map a risk score to a risk tier.

    Args:
        risk_score: Computed risk score (0.0–1.0).
        circuit_breaker_hit: Whether a circuit breaker fired.
        config: Sampling configuration with tier thresholds.

    Returns:
        The appropriate RiskTier.
    """
    if circuit_breaker_hit:
        return RiskTier.CIRCUIT_BREAKER
    if risk_score <= config.low_threshold:
        return RiskTier.LOW
    if risk_score <= config.medium_threshold:
        return RiskTier.MEDIUM
    return RiskTier.HIGH


def samples_for_tier(tier: RiskTier, config: SamplingConfig) -> int:
    """Return the number of samples to collect for a given tier.

    Args:
        tier: The risk tier.
        config: Sampling configuration.

    Returns:
        Number of samples to collect.
    """
    match tier:
        case RiskTier.LOW:
            return config.low_samples
        case RiskTier.MEDIUM:
            return config.medium_samples
        case RiskTier.HIGH:
            return config.high_samples
        case RiskTier.CIRCUIT_BREAKER:
            return config.circuit_breaker_samples
