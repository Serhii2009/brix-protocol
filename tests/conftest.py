"""Shared test fixtures for BRIX tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from brix.regulated.analysis.consistency import ConsistencyResult
from brix.regulated.llm.mock import MockLLMClient
from brix.regulated.spec.loader import load_spec, load_spec_from_dict
from brix.regulated.spec.models import SpecModel


@pytest.fixture
def sample_spec_dict() -> dict:
    """Minimal valid spec dictionary for testing."""
    return {
        "metadata": {
            "name": "test-spec",
            "version": "1.0.0",
            "domain": "testing",
            "description": "Test specification",
        },
        "circuit_breakers": [
            {
                "name": "test_cb",
                "patterns": ["lethal dose", "fatal dose"],
                "exclude_context": ["educational context", "exam question"],
            },
            {
                "name": "legal_cb",
                "patterns": ["statute of limitations", "legal requirement"],
                "exclude_context": ["academic discussion"],
            },
        ],
        "risk_signals": [
            {
                "name": "uncertainty_lang",
                "patterns": ["is it true that", "can you confirm"],
                "weight": 0.6,
                "category": "registered",
            },
            {
                "name": "factual_claims",
                "patterns": ["studies show", "research proves"],
                "weight": 0.7,
                "category": "registered",
                "exclude_context": ["opinion"],
            },
            {
                "name": "specific_numbers",
                "patterns": ["exactly", "precisely"],
                "weight": 0.5,
                "category": "universal",
                "exclude_context": ["approximate"],
            },
            {
                "name": "specific_dates",
                "patterns": ["deadline is", "due date"],
                "weight": 0.4,
                "category": "universal",
            },
        ],
        "uncertainty_types": [
            {
                "name": "epistemic",
                "action_config": {
                    "action": "force_retrieval",
                    "message_template": "Retrieval needed.",
                },
            },
            {
                "name": "contradictory",
                "action_config": {
                    "action": "conflict_resolution",
                    "message_template": "Conflict detected.",
                },
            },
            {
                "name": "open_ended",
                "action_config": {
                    "action": "distribution_response",
                    "message_template": "Multiple perspectives.",
                },
            },
        ],
        "sampling_config": {
            "low_threshold": 0.40,
            "medium_threshold": 0.70,
            "low_samples": 1,
            "medium_samples": 2,
            "high_samples": 3,
            "circuit_breaker_samples": 3,
            "temperature": 0.7,
        },
    }


@pytest.fixture
def sample_spec(sample_spec_dict: dict) -> SpecModel:
    """Validated SpecModel from sample dict."""
    return load_spec_from_dict(sample_spec_dict)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Basic MockLLMClient with a default response."""
    return MockLLMClient(default_response="This is a mock response.")


@pytest.fixture
def builtin_spec_path() -> Path:
    """Path to the built-in general v1.0.0 spec."""
    from brix.regulated.spec.defaults import get_default_spec_path

    return get_default_spec_path()


@pytest.fixture
def builtin_spec(builtin_spec_path: Path) -> SpecModel:
    """Loaded built-in spec."""
    return load_spec(builtin_spec_path)


class MockAnalyzer:
    """Mock SemanticConsistencyAnalyzer that returns configurable results."""

    def __init__(
        self,
        mean_similarity: float = 0.95,
        variance: float = 0.01,
    ) -> None:
        self._mean = mean_similarity
        self._variance = variance

    def analyze(self, samples: list[str]) -> ConsistencyResult:
        n = len(samples)
        count = max(1, n * (n - 1) // 2)
        return ConsistencyResult(
            mean_similarity=self._mean,
            variance=self._variance,
            pairwise_similarities=[self._mean] * count,
        )
