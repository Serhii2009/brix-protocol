# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for the adaptive sampler — tier mapping and parallel collection."""

from __future__ import annotations

import pytest

from brix.regulated.llm.mock import MockLLMClient
from brix.regulated.sampling.sampler import AdaptiveSampler
from brix.regulated.sampling.tiers import RiskTier, SamplingConfig, determine_tier, samples_for_tier


class TestTierMapping:
    def test_low_tier(self) -> None:
        config = SamplingConfig()
        assert determine_tier(0.20, False, config) == RiskTier.LOW
        assert determine_tier(0.40, False, config) == RiskTier.LOW

    def test_medium_tier(self) -> None:
        config = SamplingConfig()
        assert determine_tier(0.41, False, config) == RiskTier.MEDIUM
        assert determine_tier(0.70, False, config) == RiskTier.MEDIUM

    def test_high_tier(self) -> None:
        config = SamplingConfig()
        assert determine_tier(0.71, False, config) == RiskTier.HIGH
        assert determine_tier(1.0, False, config) == RiskTier.HIGH

    def test_circuit_breaker_tier(self) -> None:
        config = SamplingConfig()
        assert determine_tier(0.0, True, config) == RiskTier.CIRCUIT_BREAKER
        assert determine_tier(1.0, True, config) == RiskTier.CIRCUIT_BREAKER

    def test_sample_counts(self) -> None:
        config = SamplingConfig()
        assert samples_for_tier(RiskTier.LOW, config) == 1
        assert samples_for_tier(RiskTier.MEDIUM, config) == 2
        assert samples_for_tier(RiskTier.HIGH, config) == 3
        assert samples_for_tier(RiskTier.CIRCUIT_BREAKER, config) == 3


class TestAdaptiveSampler:
    @pytest.mark.asyncio
    async def test_low_risk_one_sample(self) -> None:
        mock = MockLLMClient(default_response="Response A")
        sampler = AdaptiveSampler(mock, SamplingConfig())
        result = await sampler.collect("test query", risk_score=0.20, circuit_breaker_hit=False)
        assert result.tier == RiskTier.LOW
        assert result.sample_count == 1
        assert len(result.samples) == 1
        assert result.force_retrieval is False

    @pytest.mark.asyncio
    async def test_medium_risk_two_samples(self) -> None:
        mock = MockLLMClient(responses=["A", "B"])
        sampler = AdaptiveSampler(mock, SamplingConfig())
        result = await sampler.collect("test query", risk_score=0.50, circuit_breaker_hit=False)
        assert result.tier == RiskTier.MEDIUM
        assert result.sample_count == 2
        assert len(result.samples) == 2

    @pytest.mark.asyncio
    async def test_high_risk_three_samples(self) -> None:
        mock = MockLLMClient(responses=["A", "B", "C"])
        sampler = AdaptiveSampler(mock, SamplingConfig())
        result = await sampler.collect("test query", risk_score=0.80, circuit_breaker_hit=False)
        assert result.tier == RiskTier.HIGH
        assert result.sample_count == 3
        assert len(result.samples) == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_three_samples_force_retrieval(self) -> None:
        mock = MockLLMClient(responses=["A", "B", "C"])
        sampler = AdaptiveSampler(mock, SamplingConfig())
        result = await sampler.collect("test query", risk_score=1.0, circuit_breaker_hit=True)
        assert result.tier == RiskTier.CIRCUIT_BREAKER
        assert result.sample_count == 3
        assert result.force_retrieval is True

    @pytest.mark.asyncio
    async def test_parallel_collection(self) -> None:
        """Verify all samples are collected (mock doesn't block, but validates gather works)."""
        call_count = 0

        class CountingClient:
            async def complete(
                self_inner, prompt, *, system=None, temperature=0.7, max_tokens=1024
            ):
                nonlocal call_count
                call_count += 1
                return f"Response {call_count}"

        sampler = AdaptiveSampler(CountingClient(), SamplingConfig())
        result = await sampler.collect("test", risk_score=0.80, circuit_breaker_hit=False)
        assert len(result.samples) == 3
        assert call_count == 3
