"""Adaptive sampler — collects LLM response samples in parallel.

All samples are collected concurrently using asyncio.gather(), never
sequentially. The number of samples is determined by the risk tier.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from brix.regulated.core.exceptions import SamplerError
from brix.regulated.llm.protocol import LLMClient
from brix.regulated.sampling.tiers import RiskTier, SamplingConfig, determine_tier, samples_for_tier


@dataclass(frozen=True, slots=True)
class SamplerResult:
    """Result of adaptive sampling."""

    samples: list[str]
    tier: RiskTier
    force_retrieval: bool
    sample_count: int
    collected_count: int = 0
    partial_failure: bool = False


class AdaptiveSampler:
    """Collects LLM response samples based on risk tier.

    All samples are collected in parallel via asyncio.gather().
    The force_retrieval flag is set when a circuit breaker fires.
    """

    def __init__(self, llm_client: LLMClient, config: SamplingConfig) -> None:
        self._llm = llm_client
        self._config = config

    async def collect(
        self,
        query: str,
        risk_score: float,
        circuit_breaker_hit: bool,
        *,
        system: str | None = None,
    ) -> SamplerResult:
        """Collect response samples based on risk tier.

        Args:
            query: The user query to send to the LLM.
            risk_score: Computed risk score for tier determination.
            circuit_breaker_hit: Whether a circuit breaker fired.
            system: Optional system prompt for the LLM.

        Returns:
            SamplerResult with collected samples and metadata.

        Raises:
            SamplerError: If all sample collections fail.
        """
        tier = determine_tier(risk_score, circuit_breaker_hit, self._config)
        n_samples = samples_for_tier(tier, self._config)
        force_retrieval = tier == RiskTier.CIRCUIT_BREAKER

        # Collect ALL samples in parallel — never sequential
        tasks = [
            self._llm.complete(
                query,
                system=system,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            for _ in range(n_samples)
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful samples from failures
        samples = [r for r in raw if isinstance(r, str)]
        failures = [r for r in raw if isinstance(r, BaseException)]

        if len(samples) == 0:
            first_err = failures[0] if failures else Exception("No samples collected")
            raise SamplerError(f"Failed to collect samples: {first_err}") from (
                first_err if isinstance(first_err, BaseException) else None
            )

        return SamplerResult(
            samples=samples,
            tier=tier,
            force_retrieval=force_retrieval,
            sample_count=n_samples,
            collected_count=len(samples),
            partial_failure=len(failures) > 0,
        )
